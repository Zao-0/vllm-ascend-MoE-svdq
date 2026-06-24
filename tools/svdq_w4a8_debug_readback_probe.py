#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Launch the isolated official SVDQW4A8DebugReadback op.

This probe is the first real-device stage for Appendix-1 W4A8 validation. It
uses deterministic nonzero packed-INT4 calibration tensors and calls only the
official debug readback op that wraps ``dispatch_ffn_combine_w4_a8``. It does
not use public grouped-matmul compatibility APIs and does not validate
real-checkpoint GMM agreement.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svdq_w4a8_debug_calibration_probe import _packed_int4_stats  # noqa: E402

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

DEFAULT_EVIDENCE_DIR = Path("/root/workspace/lza/svdq_clean_evidence")
DEFAULT_SUMMARY_NAME = "phase_az_w4a8_debug_readback_calibration_summary.json"
CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
OFFICIAL_W4A8_DEBUG_OPAPI_SYMBOLS = (
    "aclnnSVDQW4A8DebugReadbackGetWorkspaceSize",
    "aclnnSVDQW4A8DebugReadback",
    "aclnnInnerSVDQW4A8DebugReadbackGetWorkspaceSize",
    "aclnnInnerSVDQW4A8DebugReadback",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-output-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _npu_environment(device_id: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "has_torch_npu_attr": hasattr(torch, "npu"),
        "torch_npu_imported": False,
        "torch_npu_version": None,
        "npu_available": False,
        "npu_device_count": 0,
        "selected_device": device_id,
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except Exception as exc:
        info["torch_npu_import_error"] = f"{type(exc).__name__}: {exc}"

    if hasattr(torch, "npu"):
        try:
            info["npu_available"] = bool(torch.npu.is_available())
            info["npu_device_count"] = int(torch.npu.device_count())
            if info["npu_available"] and info["npu_device_count"] > device_id:
                info["device_name"] = torch.npu.get_device_name(device_id)
                info["soc_version"] = torch.npu.get_soc_version()
        except Exception as exc:
            info["npu_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _free_tcp_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_hccl_group_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_group = _get_default_group()
    if torch.__version__ > "2.0.1":
        return str(default_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank))
    return str(default_group.get_hccl_comm_name(rank))


def _init_single_rank_hccl(device_id: int) -> dict[str, Any]:
    if dist.is_initialized():
        return {
            "initialized_here": False,
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "group": _get_hccl_group_name(dist.get_rank()),
        }

    port = _free_tcp_port() + random.randint(0, 128)
    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(backend="hccl", rank=0, world_size=1, init_method=init_method)
    return {
        "initialized_here": True,
        "backend": dist.get_backend(),
        "rank": 0,
        "world_size": 1,
        "init_method": init_method,
        "group": _get_hccl_group_name(0),
        "device_id": device_id,
    }


def _destroy_hccl_if_needed(group_info: dict[str, Any]) -> None:
    if group_info.get("initialized_here") and dist.is_initialized():
        dist.destroy_process_group()


def _official_debug_symbol_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "lib_path": str(CUSTOM_OPAPI_LIB),
        "exists": CUSTOM_OPAPI_LIB.exists(),
        "loaded": False,
        "symbols": {name: False for name in OFFICIAL_W4A8_DEBUG_OPAPI_SYMBOLS},
    }
    if not CUSTOM_OPAPI_LIB.exists():
        return status

    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = str(CUSTOM_OPAPI_LIB.parent)
    if lib_dir not in old_ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld_library_path}" if old_ld_library_path else lib_dir
    try:
        handle = ctypes.CDLL(str(CUSTOM_OPAPI_LIB), mode=ctypes.RTLD_LOCAL)
    except Exception as exc:
        status["load_error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["loaded"] = True
    status["symbols"] = {name: hasattr(handle, name) for name in OFFICIAL_W4A8_DEBUG_OPAPI_SYMBOLS}
    return status


def _pack_nonzero_int4_words(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    num_words = int(np.prod(shape))
    base = torch.arange(num_words, dtype=torch.int32)
    words = torch.zeros(num_words, dtype=torch.int32)
    for lane in range(8):
        nibble = ((base + seed + lane * 3) % 7 + 1).to(torch.int32)
        words |= nibble << (4 * lane)
    return words.reshape(shape).contiguous()


def _pack_fp32_bits_to_int64(values: torch.Tensor) -> torch.Tensor:
    value_bits = values.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).view(np.uint32)
    return torch.from_numpy(value_bits.astype(np.int64, copy=True))


def _make_scale(shape: tuple[int, ...], *, base: float) -> torch.Tensor:
    count = int(np.prod(shape))
    values = torch.linspace(base, base + 0.03125, steps=count, dtype=torch.float32).reshape(shape)
    return _pack_fp32_bits_to_int64(values)


def _build_calibration_tensors(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if args.num_experts <= 0 or args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise ValueError("num_experts, hidden_size, and intermediate_size must be positive.")
    if args.hidden_size % 8 != 0:
        raise ValueError("hidden_size must be divisible by 8 for packed W4A8 down projection.")
    if (2 * args.intermediate_size) % 8 != 0:
        raise ValueError("2 * intermediate_size must be divisible by 8 for packed W4A8 gate/up projection.")
    if args.num_tokens <= 0 or args.top_k <= 0:
        raise ValueError("num_tokens and top_k must be positive.")
    if args.max_output_size < args.num_tokens * args.top_k:
        raise ValueError("max_output_size must cover all routed token rows.")

    w13_weight = _pack_nonzero_int4_words(
        (args.num_experts, args.hidden_size, (2 * args.intermediate_size) // 8),
        seed=args.seed,
    )
    w2_weight = _pack_nonzero_int4_words(
        (args.num_experts, args.intermediate_size, args.hidden_size // 8),
        seed=args.seed + 17,
    )
    x = torch.linspace(
        -1.0,
        1.0,
        steps=args.num_tokens * args.hidden_size,
        dtype=torch.float32,
    ).reshape(args.num_tokens, args.hidden_size).to(torch.bfloat16)
    expert_idx = (
        torch.arange(args.num_tokens * args.top_k, dtype=torch.int32).reshape(args.num_tokens, args.top_k)
        % args.num_experts
    )
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool)
    return {
        "x": x,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_weight_scale": _make_scale((args.num_experts, 2 * args.intermediate_size), base=0.03125),
        "w2_weight_scale": _make_scale((args.num_experts, args.hidden_size), base=0.046875),
        "w13_scale_bias": torch.linspace(
            -0.25,
            0.25,
            steps=args.num_experts * 2 * args.intermediate_size,
            dtype=torch.float32,
        ).reshape(args.num_experts, 2 * args.intermediate_size),
        "w2_scale_bias": torch.linspace(
            -0.125,
            0.125,
            steps=args.num_experts * args.hidden_size,
            dtype=torch.float32,
        ).reshape(args.num_experts, args.hidden_size),
        "expert_idx": expert_idx,
        "probs": probs,
        "x_active_mask": x_active_mask,
    }


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "sha256": _sha256_tensor(tensor),
        "numel": int(tensor.numel()),
    }


def _float_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(cpu).all().item()) if cpu.numel() else True
    abs_cpu = torch.nan_to_num(cpu).abs()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(abs_cpu.max().item()) if abs_cpu.numel() else 0.0,
        "mean_abs": float(abs_cpu.mean().item()) if abs_cpu.numel() else 0.0,
        "nonzero": bool(torch.any(abs_cpu > 0).item()) if abs_cpu.numel() else False,
        "nan_count": int(torch.isnan(cpu).sum().item()),
        "sha256": _sha256_tensor(tensor.detach().cpu()),
        "sample": cpu.flatten()[:8].tolist(),
    }


def _move_inputs_to_device(tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device=device) for name, tensor in tensors.items()}


def _cast_packed_weights_to_official_format(npu_tensors: dict[str, torch.Tensor]) -> None:
    import torch_npu  # type: ignore[import-untyped]

    torch_npu.npu.config.allow_internal_format = True
    npu_tensors["w13_weight"] = torch_npu.npu_format_cast(npu_tensors["w13_weight"], 29)
    npu_tensors["w2_weight"] = torch_npu.npu_format_cast(npu_tensors["w2_weight"], 29)


def _has_registered_debug_op() -> bool:
    bootstrap_custom_op_env(include_vendor_lib=True)
    enable_custom_op()
    return hasattr(getattr(torch.ops, "_C_ascend", object()), "svdq_w4a8_debug_readback")


def _run_debug_readback(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    tensors = _build_calibration_tensors(args)
    calibration_metadata = {
        name: _tensor_metadata(tensor)
        for name, tensor in tensors.items()
        if name not in {"x_active_mask"}
    }
    packed_checks = {
        "w13_weight": _packed_int4_stats(tensors["w13_weight"]),
        "w2_weight": _packed_int4_stats(tensors["w2_weight"]),
    }
    npu_tensors = _move_inputs_to_device(tensors, device)
    _cast_packed_weights_to_official_format(npu_tensors)
    op = torch.ops._C_ascend.svdq_w4a8_debug_readback
    (
        out,
        expert_token_nums,
        routed_x_int8,
        routed_x_scale,
        gmm1_post_dequant,
        gmm1_hidden_prequant,
        gmm2_post_dequant,
    ) = op(
        npu_tensors["x"],
        [npu_tensors["w13_weight"]],
        [npu_tensors["w2_weight"]],
        npu_tensors["expert_idx"],
        [npu_tensors["w13_weight_scale"]],
        [npu_tensors["w2_weight_scale"]],
        [npu_tensors["w13_scale_bias"]],
        [npu_tensors["w2_scale_bias"]],
        npu_tensors["probs"],
        group,
        args.max_output_size,
        npu_tensors["x_active_mask"],
    )
    torch.npu.synchronize()

    active_rows = args.num_tokens * args.top_k
    output_stats = {
        "out": _float_stats(out),
        "routed_x_int8_active": _float_stats(routed_x_int8[:active_rows]),
        "routed_x_scale_active": _float_stats(routed_x_scale[:active_rows]),
        "gmm1_post_dequant_active": _float_stats(gmm1_post_dequant[:active_rows]),
        "gmm1_hidden_prequant_active": _float_stats(gmm1_hidden_prequant[:active_rows]),
        "gmm2_post_dequant_active": _float_stats(gmm2_post_dequant[:active_rows]),
        "expert_token_nums": {
            "shape": list(expert_token_nums.shape),
            "dtype": str(expert_token_nums.dtype),
            "values": expert_token_nums.detach().cpu().tolist(),
            "sum": int(expert_token_nums.detach().cpu().sum().item()),
        },
    }
    calibration_data_validated = all(check["nonzero_nibbles"] for check in packed_checks.values())
    readback_finite = (
        output_stats["routed_x_scale_active"]["finite"]
        and
        output_stats["gmm1_post_dequant_active"]["finite"]
        and output_stats["gmm2_post_dequant_active"]["finite"]
    )
    readback_nonzero = (
        output_stats["routed_x_int8_active"]["nonzero"]
        and output_stats["routed_x_scale_active"]["nonzero"]
        and
        output_stats["gmm1_post_dequant_active"]["nonzero"]
        and output_stats["gmm2_post_dequant_active"]["nonzero"]
    )
    routed_rows_match = output_stats["expert_token_nums"]["sum"] == active_rows
    calibration_stage_passed = calibration_data_validated and routed_rows_match
    readback_health_passed = readback_finite and readback_nonzero
    hidden_prequant_health = {
        "finite": output_stats["gmm1_hidden_prequant_active"]["finite"],
        "nonzero": output_stats["gmm1_hidden_prequant_active"]["nonzero"],
    }
    hidden_prequant_health_passed = hidden_prequant_health["finite"] and hidden_prequant_health["nonzero"]
    passed = calibration_stage_passed and readback_health_passed and hidden_prequant_health_passed
    return {
        "stage": "official_svdqw4a8_debug_readback_deterministic_packed_int4_calibration",
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_debug_readback -> aclnnSVDQW4A8DebugReadback",
        "official_aic_aiv_launch_completed": True,
        "calibration_data_validated": calibration_data_validated,
        "calibration_stage_passed": calibration_stage_passed,
        "readback_finite": readback_finite,
        "readback_nonzero": readback_nonzero,
        "readback_health_passed": readback_health_passed,
        "hidden_prequant_health": hidden_prequant_health,
        "hidden_prequant_health_passed": hidden_prequant_health_passed,
        "routed_rows_match": routed_rows_match,
        "synthetic_calibration_numerical_gate": False,
        "public_grouped_matmul_used": False,
        "real_checkpoint_validation": False,
        "shape": {
            "num_experts": args.num_experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "max_output_size": args.max_output_size,
            "active_rows": active_rows,
        },
        "calibration_metadata": calibration_metadata,
        "packed_int4_checks": packed_checks,
        "outputs": output_stats,
        "passed": passed,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    symbol_status = _official_debug_symbol_status()
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_debug_readback_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "official_debug_symbol_status": symbol_status,
        "production_svdq_host_tiling_fail_closed": True,
        "next_required_stage": "real-checkpoint GMM1/GMM2 debug readback comparison against an unfused reference",
    }
    if not (env["torch_npu_imported"] and env["npu_available"] and int(env["npu_device_count"]) > args.device_id):
        summary["skipped"] = not args.require_npu
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch_npu import and an available selected NPU are required."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    torch.npu.set_device(args.device_id)
    try:
        registered = _has_registered_debug_op()
    except Exception as exc:
        summary["preflight_failed"] = True
        summary["failure_reason"] = f"failed to enable custom ops: {type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1
    summary["torch_op_registered"] = registered
    if not registered:
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_w4a8_debug_readback is not registered."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1

    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl_group"] = group_info
        stage = _run_debug_readback(args, str(group_info["group"]))
        summary["stage"] = stage
        summary["passed"] = bool(stage["passed"])
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["passed"] = False
    finally:
        _destroy_hccl_if_needed(group_info)

    _write_summary(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
