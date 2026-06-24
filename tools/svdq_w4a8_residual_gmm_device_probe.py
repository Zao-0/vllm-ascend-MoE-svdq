#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Smoke-test and gate official W4A8 residual GMM kernels on NPU.

This probe exercises the real residual GMM dimensions used by the Qwen3.5
W4A8-SVDQ MoE path without enabling the production fused SVDQ op. It uses
synthetic zero packed-W4 weights with valid ModelSlim per-channel scale packing
so the expected residual accumulator is exactly zero.

With ``--real-checkpoint`` it also loads real ModelSlim residual tensors through
the official W4A8 post-load path, launches the same grouped-matmul surface, and
compares against a CPU reference for the official boundary:

``INT4 dot with per-channel weight scale + scale_bias, then per-token scale``.

The real-checkpoint mode is an acceptance gate. It is expected to fail until the
official W4A8 launch surface is correctly isolated for SVDQ validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _ensure_minimal_ascend_config_for_official_postload,
    _group_keys_by_shard,
    _load_residual_checkpoint_tensor,
    _make_official_w4a8_method,
    _make_residual_validation_layer,
    _read_json,
    _residual_checkpoint_keys,
    _weight_map,
)

from vllm_ascend.quantization.svdq_spec import build_svdq_moe_layer_spec  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_f_residual_gmm_device_probe_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--real-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--zero-abs-tol", type=float, default=1e-30)
    parser.add_argument("--real-max-abs-tol", type=float, default=0.5)
    parser.add_argument("--real-mean-abs-tol", type=float, default=0.05)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _npu_environment(device_id: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "has_torch_npu_attr": hasattr(torch, "npu"),
        "torch_npu_imported": False,
        "torch_npu_version": None,
        "npu_available": False,
        "npu_device_count": 0,
        "selected_device": device_id,
        "has_npu_grouped_matmul": False,
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
        info["has_npu_grouped_matmul"] = hasattr(torch_npu, "npu_grouped_matmul")
    except Exception as exc:
        info["torch_npu_import_error"] = f"{type(exc).__name__}: {exc}"

    if hasattr(torch, "npu"):
        try:
            info["npu_available"] = bool(torch.npu.is_available())
            info["npu_device_count"] = int(torch.npu.device_count())
            if info["npu_available"] and info["npu_device_count"] > device_id:
                try:
                    info["runtime_soc_version"] = torch.npu.get_device_name(device_id)
                except Exception as exc:
                    info["runtime_soc_query_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            info["npu_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _target_metadata(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    quant_path = Path(model_path) / "quant_model_description.json"
    hidden_size = 2048
    intermediate_size = 512
    num_experts = 256
    top_k = 8
    config_source = "qwen35_svdq_default"
    if config_path.exists():
        config = _read_json(str(config_path))
        text_config = config.get("text_config", config)
        hidden_size = int(text_config.get("hidden_size", hidden_size))
        intermediate_size = int(
            text_config.get("moe_intermediate_size", text_config.get("intermediate_size", intermediate_size))
        )
        num_experts = int(text_config.get("num_experts", num_experts))
        top_k = int(text_config.get("num_experts_per_tok", top_k))
        config_source = str(config_path)

    quant_description: dict[str, Any] = {}
    if quant_path.exists():
        quant_description = _read_json(str(quant_path))
    return {
        "config_source": config_source,
        "quant_source": str(quant_path) if quant_path.exists() else None,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "quant_version": quant_description.get("version"),
        "group_size": quant_description.get("group_size"),
    }


def _pack_modelslim_per_channel_scale(raw_scale: torch.Tensor) -> torch.Tensor:
    scale = raw_scale.transpose(1, 2).contiguous().cpu().numpy()
    scale.dtype = np.uint32
    return torch.from_numpy(scale.astype(np.int64))


def _make_int8_input(rows: int, width: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-16, 16, (rows, width), dtype=torch.int8, generator=generator)
    return x.to(device=device)


def _stage_error(tensor: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all().item()) if tensor.numel() else True
    abs_tensor = torch.nan_to_num(tensor.detach().float()).abs()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(abs_tensor.max().item()) if abs_tensor.numel() else 0.0,
        "mean_abs": float(abs_tensor.mean().item()) if abs_tensor.numel() else 0.0,
        "nan_count": int(torch.isnan(tensor).sum().item()) if tensor.is_floating_point() else 0,
    }


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    actual_finite = bool(torch.isfinite(actual_cpu).all().item()) if actual_cpu.numel() else True
    expected_finite = bool(torch.isfinite(expected_cpu).all().item()) if expected_cpu.numel() else True
    diff = (actual_cpu - expected_cpu).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
        "nan_count": int(torch.isnan(actual_cpu).sum().item()),
        "numel": int(actual_cpu.numel()),
    }


def _int64_float_bits_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    scale_cpu = scale.detach().contiguous().cpu().numpy().astype(np.uint64).astype(np.uint32)
    scale_fp32 = torch.from_numpy(scale_cpu.view(np.float32).copy()).float()
    if scale_fp32.dim() == 3 and scale_fp32.shape[1] == 1:
        return scale_fp32[:, 0, :]
    return scale_fp32


def _packed_int4_column(weight: torch.Tensor, output_column: int) -> torch.Tensor:
    words = weight[:, output_column // 8].to(torch.int32)
    values = (words >> (4 * (output_column % 8))) & 0xF
    return torch.where(values >= 8, values - 16, values).to(torch.int32)


def _row_expert_ids(group_list: torch.Tensor) -> list[int]:
    expert_ids: list[int] = []
    for expert_id, count in enumerate(group_list.detach().cpu().tolist()):
        expert_ids.extend([expert_id] * int(count))
    return expert_ids


def _cpu_w4a8_reference(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_bias: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    output_columns: int,
) -> torch.Tensor:
    x_cpu = x.detach().cpu().to(torch.int32)
    weight_cpu = weight.detach().cpu().contiguous()
    scale_cpu = _int64_float_bits_to_fp32(weight_scale)
    bias_cpu = scale_bias.detach().cpu().float().contiguous()
    per_token_cpu = per_token_scale.detach().cpu().float()
    expert_ids = _row_expert_ids(group_list)
    if len(expert_ids) != x_cpu.shape[0]:
        raise ValueError(f"group_list expands to {len(expert_ids)} rows, but x has {x_cpu.shape[0]} rows.")

    reference = torch.empty((x_cpu.shape[0], output_columns), dtype=torch.float32)
    for row, expert_id in enumerate(expert_ids):
        for column in range(output_columns):
            int4_column = _packed_int4_column(weight_cpu[expert_id], column)
            accumulator = (x_cpu[row] * int4_column).sum().float()
            reference[row, column] = (
                accumulator * scale_cpu[expert_id, column] + bias_cpu[expert_id, column]
            ) * per_token_cpu[row]
    return reference


def _make_group_list(rows: int, experts: int, *, device: torch.device) -> torch.Tensor:
    group_list = torch.full((experts,), rows // experts, dtype=torch.int64, device=device)
    group_list[-1] += rows - int(group_list.sum().item())
    return group_list


def _run_gmm1(
    *, metadata: dict[str, Any], rows: int, experts: int, seed: int, device: torch.device, zero_abs_tol: float
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    hidden_size = int(metadata["hidden_size"])
    intermediate_size = int(metadata["intermediate_size"])
    x = _make_int8_input(rows, hidden_size, seed=seed, device=device)
    weight = torch.zeros((experts, hidden_size, (2 * intermediate_size) // 8), dtype=torch.int32, device=device)
    raw_scale = torch.ones((experts, 2 * intermediate_size, 1), dtype=torch.float32)
    scale = _pack_modelslim_per_channel_scale(raw_scale).squeeze(1).to(device=device)
    bias = torch.zeros((experts, 2 * intermediate_size), dtype=torch.float32, device=device)
    per_token_scale = torch.ones((rows,), dtype=torch.float32, device=device)
    group_list = _make_group_list(rows, experts, device=device)
    output = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=[scale],
        bias=[bias],
        per_token_scale=[per_token_scale],
        group_list=group_list,
        group_list_type=1,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    error = _stage_error(output)
    return {
        "stage": "w4a8_residual_gmm1",
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "scale_shape": list(scale.shape),
        "bias_shape": list(bias.shape),
        "group_list": group_list.detach().cpu().tolist(),
        "zero_abs_tolerance": zero_abs_tol,
        "output": error,
        "passed": error["finite"] and error["max_abs"] <= zero_abs_tol,
    }


def _run_gmm2(
    *, metadata: dict[str, Any], rows: int, experts: int, seed: int, device: torch.device, zero_abs_tol: float
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    hidden_size = int(metadata["hidden_size"])
    intermediate_size = int(metadata["intermediate_size"])
    x = _make_int8_input(rows, intermediate_size, seed=seed + 1, device=device)
    weight = torch.zeros((experts, intermediate_size, hidden_size // 8), dtype=torch.int32, device=device)
    raw_scale = torch.ones((experts, hidden_size, 1), dtype=torch.float32)
    scale = _pack_modelslim_per_channel_scale(raw_scale).to(device=device)
    bias = torch.zeros((experts, hidden_size), dtype=torch.float32, device=device)
    per_token_scale = torch.ones((rows,), dtype=torch.float32, device=device)
    group_list = _make_group_list(rows, experts, device=device)
    output = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=[scale],
        bias=[bias],
        per_token_scale=[per_token_scale],
        group_list=group_list,
        group_list_type=1,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    error = _stage_error(output)
    return {
        "stage": "w4a8_residual_gmm2",
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "scale_shape": list(scale.shape),
        "bias_shape": list(bias.shape),
        "group_list": group_list.detach().cpu().tolist(),
        "zero_abs_tolerance": zero_abs_tol,
        "output": error,
        "passed": error["finite"] and error["max_abs"] <= zero_abs_tol,
    }


def _load_real_residual_layer(
    *,
    model_path: str,
    layer_index: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.nn.Module, Any, int]:
    quant_description = _read_json(os.path.join(model_path, "quant_model_description.json"))
    weight_map = _weight_map(model_path)
    spec = build_svdq_moe_layer_spec(
        quant_description=quant_description,
        prefix=f"model.language_model.layers.{layer_index}.mlp.experts",
        model_path=model_path,
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
    )
    method = _make_official_w4a8_method(quant_description, tp_size=tp_size)
    layer = _make_residual_validation_layer(method=method, spec=spec, tp_size=tp_size, tp_rank=tp_rank)
    residual_keys = _residual_checkpoint_keys(spec, weight_map)
    for shard, shard_keys in _group_keys_by_shard(residual_keys, weight_map).items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                _load_residual_checkpoint_tensor(layer=layer, spec=spec, key=key, loaded_weight=f.get_tensor(key))
    _ensure_minimal_ascend_config_for_official_postload()
    method.process_weights_after_loading(layer)
    return layer, spec, len(residual_keys)


def _run_real_gmm_stage(
    *,
    stage_name: str,
    x_width: int,
    output_columns: int,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_bias: torch.Tensor,
    rows: int,
    experts: int,
    seed: int,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    x = _make_int8_input(rows, x_width, seed=seed, device=device)
    per_token_scale = torch.linspace(0.0625, 0.125, rows, dtype=torch.float32, device=device)
    group_list = _make_group_list(rows, experts, device=device)
    npu_output = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight[:experts]],
        scale=[weight_scale[:experts]],
        bias=[scale_bias[:experts]],
        per_token_scale=[per_token_scale],
        group_list=group_list,
        group_list_type=1,
        group_type=0,
        split_item=2,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    reference = _cpu_w4a8_reference(
        x=x,
        weight=weight[:experts],
        weight_scale=weight_scale[:experts],
        scale_bias=scale_bias[:experts],
        per_token_scale=per_token_scale,
        group_list=group_list,
        output_columns=output_columns,
    )
    error = _tensor_error(npu_output, reference)
    return {
        "stage": stage_name,
        "input_shape": list(x.shape),
        "weight_shape": list(weight[:experts].shape),
        "scale_shape": list(weight_scale[:experts].shape),
        "bias_shape": list(scale_bias[:experts].shape),
        "group_list": group_list.detach().cpu().tolist(),
        "per_token_scale": per_token_scale.detach().cpu().tolist(),
        "reference_formula": (
            "(sum(int8_activation * signed_int4_weight) * weight_scale + scale_bias) * per_token_scale"
        ),
        "error": error,
        "max_abs_tolerance": max_abs_tol,
        "mean_abs_tolerance": mean_abs_tol,
        "passed": (
            error["actual_finite"]
            and error["expected_finite"]
            and error["diff_finite"]
            and error["max_abs"] <= max_abs_tol
            and error["mean_abs"] <= mean_abs_tol
        ),
    }


def _run_real_checkpoint_layer(
    *,
    model_path: str,
    layer_index: int,
    rows: int,
    experts: int,
    seed: int,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> dict[str, Any]:
    layer, spec, residual_key_count = _load_real_residual_layer(
        model_path=model_path,
        layer_index=layer_index,
        tp_size=1,
        tp_rank=0,
    )
    gmm1 = _run_real_gmm_stage(
        stage_name="real_checkpoint_w4a8_residual_gmm1",
        x_width=spec.hidden_size,
        output_columns=2 * spec.intermediate_size,
        weight=layer.w13_weight,
        weight_scale=layer.w13_weight_scale,
        scale_bias=layer.w13_scale_bias,
        rows=rows,
        experts=experts,
        seed=seed + layer_index * 1000,
        device=device,
        max_abs_tol=max_abs_tol,
        mean_abs_tol=mean_abs_tol,
    )
    gmm2 = _run_real_gmm_stage(
        stage_name="real_checkpoint_w4a8_residual_gmm2",
        x_width=spec.intermediate_size,
        output_columns=spec.hidden_size,
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        scale_bias=layer.w2_scale_bias,
        rows=rows,
        experts=experts,
        seed=seed + layer_index * 1000 + 1,
        device=device,
        max_abs_tol=max_abs_tol,
        mean_abs_tol=mean_abs_tol,
    )
    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "residual_checkpoint_key_count": residual_key_count,
        "postload_tensor_shapes": {
            "w13_weight": list(layer.w13_weight.shape),
            "w2_weight": list(layer.w2_weight.shape),
            "w13_weight_scale": list(layer.w13_weight_scale.shape),
            "w2_weight_scale": list(layer.w2_weight_scale.shape),
            "w13_scale_bias": list(layer.w13_scale_bias.shape),
            "w2_scale_bias": list(layer.w2_scale_bias.shape),
        },
        "stages": [gmm1, gmm2],
        "passed": gmm1["passed"] and gmm2["passed"],
    }


def _write_summary(evidence_dir: str, summary_name: str, summary: dict[str, Any]) -> str:
    os.makedirs(evidence_dir, exist_ok=True)
    path = os.path.join(evidence_dir, summary_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def main() -> int:
    args = _parse_args()
    env = _npu_environment(args.device_id)
    metadata = _target_metadata(args.model_path)
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_residual_gmm_device_probe",
        "model_path": args.model_path,
        "metadata": metadata,
        "environment": env,
        "mode": "real_checkpoint" if args.real_checkpoint else "synthetic_zero",
        "stages": [],
        "passed": False,
        "skipped": False,
    }
    if not (env["torch_npu_imported"] and env["has_npu_grouped_matmul"] and env["npu_available"]):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = "torch_npu.npu_grouped_matmul and an available NPU are required."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    if int(env["npu_device_count"]) <= args.device_id:
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = f"device_id {args.device_id} is outside npu_device_count={env['npu_device_count']}."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    if metadata["quant_version"] != "1.0.0" or metadata["group_size"] != 0:
        raise ValueError(f"expected ModelSlim W4A8 version=1.0.0 group_size=0, got {metadata!r}")
    if args.num_tokens < args.num_experts:
        raise ValueError("--num-tokens must be >= --num-experts so every synthetic expert receives at least one row.")

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    if args.real_checkpoint:
        layers = [
            _run_real_checkpoint_layer(
                model_path=args.model_path,
                layer_index=layer_index,
                rows=args.num_tokens,
                experts=args.num_experts,
                seed=args.seed,
                device=device,
                max_abs_tol=args.real_max_abs_tol,
                mean_abs_tol=args.real_mean_abs_tol,
            )
            for layer_index in args.layers
        ]
        summary["layers"] = layers
        summary["passed"] = all(layer["passed"] for layer in layers)
    else:
        stages = [
            _run_gmm1(
                metadata=metadata,
                rows=args.num_tokens,
                experts=args.num_experts,
                seed=args.seed,
                device=device,
                zero_abs_tol=args.zero_abs_tol,
            ),
            _run_gmm2(
                metadata=metadata,
                rows=args.num_tokens,
                experts=args.num_experts,
                seed=args.seed,
                device=device,
                zero_abs_tol=args.zero_abs_tol,
            ),
        ]
        summary["stages"] = stages
        summary["passed"] = all(stage["passed"] for stage in stages)
    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    print(json.dumps({"summary_path": path, "passed": summary["passed"], "mode": summary["mode"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
