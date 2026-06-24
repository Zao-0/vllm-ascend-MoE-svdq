#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Build deterministic nonzero packed-INT4 W4A8 calibration inputs.

This probe prepares the first isolated-W4A8 validation prerequisite from the
SVDQ Appendix-1 flow. It only validates calibration tensor construction and
optional NPU memory round-trip. It does not call public grouped matmul, does
not launch the official W4A8 kernel, and does not claim numerical acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_EVIDENCE_DIR = Path("/root/workspace/lza/svdq_clean_evidence")
DEFAULT_SUMMARY_NAME = "phase_aw_w4a8_debug_calibration_summary.json"


@dataclass(frozen=True)
class CalibrationShape:
    num_experts: int = 2
    hidden_size: int = 16
    intermediate_size: int = 16
    group_size: int = 8
    num_tokens: int = 4
    top_k: int = 1

    def validate(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive.")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive for this per-group calibration probe.")
        if self.hidden_size % self.group_size != 0:
            raise ValueError("hidden_size must be divisible by group_size.")
        if self.intermediate_size % self.group_size != 0:
            raise ValueError("intermediate_size must be divisible by group_size.")
        if self.intermediate_size % 4 != 0:
            raise ValueError("intermediate_size must be divisible by 4 for int32 packed W4A8 w13.")
        if self.hidden_size % 8 != 0:
            raise ValueError("hidden_size must be divisible by 8 for int32 packed W4A8 w2.")
        if self.num_tokens <= 0 or self.top_k <= 0:
            raise ValueError("num_tokens and top_k must be positive.")


def _sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _nonzero_int4_pair_bytes(numel: int, *, seed: int) -> torch.Tensor:
    low = ((torch.arange(numel, dtype=torch.int16) + seed) % 15) + 1
    high = ((torch.arange(numel, dtype=torch.int16) * 3 + seed + 5) % 15) + 1
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.view(torch.int8)


def _pack_new_modelslim_w13(shape: CalibrationShape) -> torch.Tensor:
    raw = _nonzero_int4_pair_bytes(
        shape.num_experts * shape.intermediate_size * shape.hidden_size,
        seed=3,
    ).reshape(shape.num_experts, shape.intermediate_size, shape.hidden_size)
    post_transpose = raw.transpose(1, 2).contiguous()
    return post_transpose.view(torch.int32).contiguous()


def _pack_new_modelslim_w2(shape: CalibrationShape) -> torch.Tensor:
    raw = _nonzero_int4_pair_bytes(
        shape.num_experts * (shape.hidden_size // 2) * shape.intermediate_size,
        seed=11,
    ).reshape(shape.num_experts, shape.hidden_size // 2, shape.intermediate_size)
    post_transpose = raw.transpose(1, 2).contiguous()
    return post_transpose.view(torch.int32).contiguous()


def _pack_fp32_even_lanes(values: torch.Tensor) -> torch.Tensor:
    values_np = values.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    value_bits = values_np.view(np.uint32)
    packed_bits = np.zeros((*value_bits.shape[:-1], value_bits.shape[-1] * 2), dtype=np.uint32)
    packed_bits[..., ::2] = value_bits
    packed_int64 = np.frombuffer(packed_bits.tobytes(), dtype=np.int64).copy().reshape(value_bits.shape)
    return torch.from_numpy(packed_int64)


def _make_scale(shape: tuple[int, ...], *, base: float) -> torch.Tensor:
    count = int(np.prod(shape))
    values = torch.linspace(base, base + 0.125, steps=count, dtype=torch.float32).reshape(shape)
    return _pack_fp32_even_lanes(values)


def build_calibration_tensors(shape: CalibrationShape) -> dict[str, torch.Tensor]:
    shape.validate()
    w13_weight = _pack_new_modelslim_w13(shape)
    w2_weight = _pack_new_modelslim_w2(shape)
    w13_weight_scale = _make_scale(
        (shape.num_experts, shape.hidden_size // shape.group_size, 2 * shape.intermediate_size),
        base=0.03125,
    )
    w2_weight_scale = _make_scale(
        (shape.num_experts, shape.intermediate_size // shape.group_size, shape.hidden_size),
        base=0.046875,
    )
    w13_scale_bias = torch.linspace(
        -0.25,
        0.25,
        steps=shape.num_experts * 2 * shape.intermediate_size,
        dtype=torch.float32,
    ).reshape(shape.num_experts, 2 * shape.intermediate_size)
    w2_scale_bias = torch.linspace(
        -0.125,
        0.125,
        steps=shape.num_experts * shape.hidden_size,
        dtype=torch.float32,
    ).reshape(shape.num_experts, shape.hidden_size)
    x = torch.linspace(
        -1.0,
        1.0,
        steps=shape.num_tokens * shape.hidden_size,
        dtype=torch.float32,
    ).reshape(shape.num_tokens, shape.hidden_size).to(torch.bfloat16)
    expert_idx = (torch.arange(shape.num_tokens * shape.top_k, dtype=torch.int32) % shape.num_experts).reshape(
        shape.num_tokens, shape.top_k
    )
    probs = torch.full((shape.num_tokens, shape.top_k), 1.0 / shape.top_k, dtype=torch.float32)
    expert_token_nums = torch.zeros((1, shape.num_experts), dtype=torch.int32)
    for expert in expert_idx.flatten().tolist():
        expert_token_nums[0, int(expert)] += 1
    return {
        "x": x,
        "expert_idx": expert_idx,
        "probs": probs,
        "expert_token_nums": expert_token_nums,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_weight_scale": w13_weight_scale,
        "w2_weight_scale": w2_weight_scale,
        "w13_scale_bias": w13_scale_bias,
        "w2_scale_bias": w2_scale_bias,
    }


def _packed_int4_stats(tensor: torch.Tensor) -> dict[str, Any]:
    bytes_u8 = tensor.detach().cpu().contiguous().view(torch.uint8).flatten()
    low = bytes_u8 & 0x0F
    high = (bytes_u8 >> 4) & 0x0F
    nibble_hist = torch.bincount(torch.cat((low.flatten(), high.flatten())).to(torch.int64), minlength=16)
    return {
        "logical_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _sha256_tensor(tensor),
        "byte_count": int(bytes_u8.numel()),
        "packed_int32_count": int(tensor.numel()),
        "zero_low_nibbles": int((low == 0).sum().item()),
        "zero_high_nibbles": int((high == 0).sum().item()),
        "nonzero_nibbles": bool((low != 0).all().item() and (high != 0).all().item()),
        "nibble_histogram": [int(value) for value in nibble_hist.tolist()],
        "sampled_bytes_hex": bytes_u8[: min(32, bytes_u8.numel())].numpy().tobytes().hex(),
    }


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "logical_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
        "sha256": _sha256_tensor(tensor),
    }


def validate_calibration_tensors(tensors: dict[str, torch.Tensor], shape: CalibrationShape) -> dict[str, Any]:
    expected = {
        "x": ([shape.num_tokens, shape.hidden_size], torch.bfloat16),
        "expert_idx": ([shape.num_tokens, shape.top_k], torch.int32),
        "probs": ([shape.num_tokens, shape.top_k], torch.float32),
        "expert_token_nums": ([1, shape.num_experts], torch.int32),
        "w13_weight": ([shape.num_experts, shape.hidden_size, shape.intermediate_size // 4], torch.int32),
        "w2_weight": ([shape.num_experts, shape.intermediate_size, shape.hidden_size // 8], torch.int32),
        "w13_weight_scale": (
            [shape.num_experts, shape.hidden_size // shape.group_size, 2 * shape.intermediate_size],
            torch.int64,
        ),
        "w2_weight_scale": (
            [shape.num_experts, shape.intermediate_size // shape.group_size, shape.hidden_size],
            torch.int64,
        ),
        "w13_scale_bias": ([shape.num_experts, 2 * shape.intermediate_size], torch.float32),
        "w2_scale_bias": ([shape.num_experts, shape.hidden_size], torch.float32),
    }
    tensor_checks = {}
    for name, (expected_shape, expected_dtype) in expected.items():
        tensor = tensors[name]
        tensor_checks[name] = {
            "shape_ok": list(tensor.shape) == expected_shape,
            "dtype_ok": tensor.dtype == expected_dtype,
            "metadata": _tensor_metadata(tensor),
        }
    packed_checks = {
        "w13_weight": _packed_int4_stats(tensors["w13_weight"]),
        "w2_weight": _packed_int4_stats(tensors["w2_weight"]),
    }
    scale_nonzero = {
        "w13_weight_scale_nonzero": bool(torch.any(tensors["w13_weight_scale"] != 0).item()),
        "w2_weight_scale_nonzero": bool(torch.any(tensors["w2_weight_scale"] != 0).item()),
        "w13_scale_bias_nonzero": bool(torch.any(tensors["w13_scale_bias"] != 0).item()),
        "w2_scale_bias_nonzero": bool(torch.any(tensors["w2_scale_bias"] != 0).item()),
    }
    passed = (
        all(check["shape_ok"] and check["dtype_ok"] for check in tensor_checks.values())
        and all(check["nonzero_nibbles"] for check in packed_checks.values())
        and all(scale_nonzero.values())
        and int(tensors["expert_token_nums"].sum().item()) == shape.num_tokens * shape.top_k
    )
    return {
        "passed": passed,
        "shape": shape.__dict__,
        "official_postload_contract": {
            "quant_version": "1.0.0",
            "packing": "two int4 values per int8 in checkpoint, four int8 values viewed as one int32 post-load",
            "w13_weight_shape": "[E, hidden_size, intermediate_size // 4] int32; logical N is 2 * intermediate_size",
            "w2_weight_shape": "[E, intermediate_size, hidden_size // 8] int32; logical N is hidden_size",
            "scale_dtype": "int64 packed FP32 even lanes, matching AscendW4A8DynamicFusedMoEMethod.process_scale",
        },
        "tensor_checks": tensor_checks,
        "packed_int4_checks": packed_checks,
        "scale_nonzero_checks": scale_nonzero,
        "expert_token_total": int(tensors["expert_token_nums"].sum().item()),
        "public_grouped_matmul_used": False,
        "official_kernel_launched": False,
        "numerical_acceptance_claimed": False,
    }


def _npu_roundtrip(tensors: dict[str, torch.Tensor], *, device_id: int, require_npu: bool) -> dict[str, Any]:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        if require_npu:
            raise RuntimeError("NPU round-trip requested but torch_npu cannot be imported.") from exc
        return {"attempted": False, "skipped": True, "reason": f"torch_npu import failed: {type(exc).__name__}: {exc}"}

    if not torch.npu.is_available():
        if require_npu:
            raise RuntimeError("NPU round-trip requested but torch.npu.is_available() is false.")
        return {"attempted": False, "skipped": True, "reason": "torch.npu.is_available() is false"}

    torch.npu.set_device(device_id)
    roundtrip_checks = {}
    for name, tensor in tensors.items():
        npu_tensor = tensor.npu()
        copied_back = npu_tensor.cpu()
        roundtrip_checks[name] = {
            "exact_match": bool(torch.equal(tensor.cpu(), copied_back)),
            "device_dtype": str(npu_tensor.dtype),
            "device_shape": list(npu_tensor.shape),
        }
    torch.npu.synchronize()
    return {
        "attempted": True,
        "skipped": False,
        "device_id": device_id,
        "passed": all(check["exact_match"] for check in roundtrip_checks.values()),
        "checks": roundtrip_checks,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    shape = CalibrationShape(
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        group_size=args.group_size,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
    )
    tensors = build_calibration_tensors(shape)
    validation = validate_calibration_tensors(tensors, shape)
    npu = _npu_roundtrip(tensors, device_id=args.device_id, require_npu=args.require_npu)
    passed = bool(validation["passed"] and (not npu.get("attempted") or npu.get("passed")))
    return {
        "probe": "svdq_w4a8_debug_calibration_probe",
        "passed": passed,
        "validation": validation,
        "npu_roundtrip": npu,
        "next_required_stage": (
            "wire isolated official W4A8 debug readback op, then compare nonzero GMM1/GMM2 outputs "
            "against an unfused reference on real checkpoint data"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build_summary(args)
    os.makedirs(args.evidence_dir, exist_ok=True)
    output_path = args.evidence_dir / args.summary_name
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(output_path), "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
