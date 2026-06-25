#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Launch SVDQLowRankDebugReadback and compare real-checkpoint BF16 stages."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_bf16_stage_device_probe import (  # noqa: E402
    _load_validation_layer,
    _make_inputs,
    _npu_environment,
    _stage_error,
)
from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _read_json,
    _weight_map,
)

from vllm_ascend.quantization.methods.svdq_post_load import (  # noqa: E402
    build_svdq_bf16_stage_reference,
)
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

DEBUG_OP_NAME = "SVDQLowRankDebugReadback"
PRODUCTION_OP_NAME = "DispatchFFNCombineW4A8SVDQ"
DEFAULT_SUMMARY_NAME = "phase_j_lowrank_debug_readback_probe_summary.json"
CUSTOM_OP_CONFIG_ROOT = (
    REPO_ROOT
    / "vllm_ascend"
    / "_cann_ops_custom"
    / "vendors"
    / "custom_transformer"
    / "op_impl"
    / "ai_core"
    / "tbe"
    / "kernel"
    / "config"
)
CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
_PRELOADED_CUSTOM_OPAPI_GLOBAL = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 39])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--input-scale", type=float, default=0.03125)
    parser.add_argument("--max-abs-tol", type=float, default=0.5)
    parser.add_argument("--mean-abs-tol", type=float, default=0.02)
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument(
        "--require-accumulator-readback",
        action="store_true",
        help="Require FP32 accumulator buffers to match references. Use with a package built with "
        "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK=ON.",
    )
    parser.add_argument(
        "--skip-copy-only-consumer-gate",
        action="store_true",
        help="Skip the deterministic mixed-AIV BF16 low-rank GM load/cast gate.",
    )
    return parser.parse_args()


def _normalize_soc_name(raw_name: object) -> str | None:
    if raw_name is None:
        return None
    text = str(raw_name).strip().lower().replace("-", "_")
    if not text:
        return None
    if "910b" in text:
        return "ascend910b"
    if "910_93" in text or "910c" in text or "ascend910_939" in text or "ascend910_938" in text:
        return "ascend910_93"
    if "310p" in text:
        return "ascend310p"
    if "950" in text:
        return "ascend950"
    if "kirinx90" in text:
        return "kirinx90"
    if text.startswith("ascend"):
        return text
    return None


def _runtime_soc(device_id: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device_id": device_id,
        "device_name": None,
        "soc_version": None,
        "normalized_soc": None,
    }
    if not hasattr(torch, "npu"):
        return info
    try:
        info["device_name"] = torch.npu.get_device_name(device_id)
    except Exception as exc:
        info["device_name_error"] = f"{type(exc).__name__}: {exc}"
    try:
        info["soc_version"] = torch.npu.get_soc_version()
    except Exception as exc:
        info["soc_version_error"] = f"{type(exc).__name__}: {exc}"
    info["normalized_soc"] = _normalize_soc_name(info["device_name"])
    return info


def _target_dimensions(model_path: str) -> dict[str, int | str]:
    config_path = Path(model_path) / "config.json"
    hidden_size = 2048
    intermediate_size = 512
    source = "qwen35_svdq_default"
    if config_path.exists():
        config = _read_json(str(config_path))
        text_config = config.get("text_config", config)
        hidden_size = int(text_config.get("hidden_size", hidden_size))
        intermediate_size = int(
            text_config.get("moe_intermediate_size", text_config.get("intermediate_size", intermediate_size))
        )
        source = str(config_path)
    return {
        "source": source,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def _deterministic_bf16_pattern(shape: tuple[int, ...], *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(-113, 114, shape, dtype=torch.int16, generator=generator).float()
    return (values * scale).to(torch.bfloat16).contiguous()


def _pack_factor_as_bf16_b_zn(
    factor: torch.Tensor,
    *,
    name: str,
    padded_n_cols: int | None = None,
    padded_k_rows: int | None = None,
) -> torch.Tensor:
    """Pack logical [expert, N, K] BF16 factors as official BF16 B zN [K, N]."""
    if factor.dim() != 3:
        raise ValueError(f"{name} must be rank-3, got shape={tuple(factor.shape)}")
    if factor.dtype != torch.bfloat16:
        factor = factor.to(torch.bfloat16)
    source = factor.detach().cpu().contiguous()
    experts, n_cols, k_rows = (int(dim) for dim in source.shape)
    logical_n_cols = n_cols if padded_n_cols is None else int(padded_n_cols)
    logical_k_rows = k_rows if padded_k_rows is None else int(padded_k_rows)
    if logical_n_cols < n_cols or logical_k_rows < k_rows:
        raise ValueError(
            f"{name} padded logical shape must cover source shape: "
            f"source={(experts, n_cols, k_rows)}, padded={(experts, logical_n_cols, logical_k_rows)}"
        )
    if logical_n_cols != n_cols or logical_k_rows != k_rows:
        padded = torch.zeros((experts, logical_n_cols, logical_k_rows), dtype=source.dtype)
        padded[:, :n_cols, :k_rows] = source
        source = padded
        n_cols = logical_n_cols
        k_rows = logical_k_rows
    c0 = 16
    elems_per_c0 = 16
    k_round = ((k_rows + c0 - 1) // c0) * c0
    n_round = ((n_cols + elems_per_c0 - 1) // elems_per_c0) * elems_per_c0
    capacity = k_round * n_round
    if capacity != k_rows * n_cols:
        raise ValueError(
            f"{name} cannot be represented in-place as zN with the debug op's logical shape: "
            f"logical={(experts, n_cols, k_rows)}, zN_capacity={capacity}"
        )

    k_index = torch.arange(k_rows, dtype=torch.long).view(k_rows, 1)
    n_index = torch.arange(n_cols, dtype=torch.long).view(1, n_cols)
    offsets = (
        (k_index // c0) * (c0 * elems_per_c0)
        + (n_index // elems_per_c0) * (k_round * elems_per_c0)
        + (k_index % c0) * elems_per_c0
        + (n_index % elems_per_c0)
    )
    packed = torch.empty((experts, capacity), dtype=source.dtype)
    packed[:, offsets.reshape(-1)] = source.transpose(1, 2).reshape(experts, -1)
    return packed.view_as(source).contiguous()


def _exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _preload_custom_opapi() -> bool:
    global _PRELOADED_CUSTOM_OPAPI_GLOBAL
    if not CUSTOM_OPAPI_LIB.exists():
        return False
    ctypes.CDLL(str(CUSTOM_OPAPI_LIB), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_CUSTOM_OPAPI_GLOBAL = True
    return True


def _run_copy_only_consumer_gate(
    *,
    model_path: str,
    rows: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    dims = _target_dimensions(model_path)
    hidden_size = int(dims["hidden_size"])
    intermediate_size = int(dims["intermediate_size"])
    gate_up_columns = intermediate_size * 2

    bootstrap_custom_op_env(include_vendor_lib=True)
    _preload_custom_opapi()
    enable_custom_op()
    op = getattr(torch.ops._C_ascend, "svdq_mixed_epilogue_debug_readback", None)
    if op is None:
        raise RuntimeError(
            "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback is not registered. "
            "Rebuild/install vllm-ascend after adding the mixed AIV debug binding."
        )

    gate_up_lowrank = _deterministic_bf16_pattern((rows, gate_up_columns), seed=seed + 17, scale=1.0 / 64.0)
    down_lowrank = _deterministic_bf16_pattern((rows, hidden_size), seed=seed + 23, scale=1.0 / 64.0)
    residual_gate_up = torch.zeros((rows, gate_up_columns), dtype=torch.float32)
    residual_down = torch.zeros((rows, hidden_size), dtype=torch.float32)

    gate_up_total, _hidden_bf16, _hidden_int8, _hidden_scale, down_total, _out_bf16 = op(
        residual_gate_up.to(device=device).contiguous(),
        gate_up_lowrank.to(device=device).contiguous(),
        residual_down.to(device=device).contiguous(),
        down_lowrank.to(device=device).contiguous(),
        0.0,
    )
    torch.npu.synchronize()

    gate_up_error = _stage_error(gate_up_total.detach().float().cpu(), gate_up_lowrank.float())
    down_error = _stage_error(down_total.detach().float().cpu(), down_lowrank.float())
    gate_up_passed = (
        bool(gate_up_error["actual_finite"])
        and bool(gate_up_error["expected_finite"])
        and bool(gate_up_error["diff_finite"])
        and float(gate_up_error["max_abs"]) == 0.0
        and float(gate_up_error["mean_abs"]) == 0.0
    )
    down_passed = (
        bool(down_error["actual_finite"])
        and bool(down_error["expected_finite"])
        and bool(down_error["diff_finite"])
        and float(down_error["max_abs"]) == 0.0
        and float(down_error["mean_abs"]) == 0.0
    )
    return {
        "evaluated": True,
        "passed": gate_up_passed and down_passed,
        "mode": "deterministic_zero_residual_mixed_aiv_lowrank_load_cast",
        "source": (
            "svdq_mixed_epilogue_debug_readback reads BF16 low-rank GM tensors with the same "
            "row-major row * columns + column address formula used by the mixed AIV consumer, "
            "casts them to FP32, and writes the residual-plus-low-rank totals."
        ),
        "dimensions": {
            **dims,
            "rows": rows,
            "gate_up_columns": gate_up_columns,
        },
        "metadata": {
            "gate_up_logical_shape": list(gate_up_lowrank.shape),
            "down_logical_shape": list(down_lowrank.shape),
            "gate_up_row_stride_elements": int(gate_up_lowrank.stride(0)),
            "down_row_stride_elements": int(down_lowrank.stride(0)),
            "gate_up_row_stride_bytes": int(gate_up_lowrank.stride(0) * gate_up_lowrank.element_size()),
            "down_row_stride_bytes": int(down_lowrank.stride(0) * down_lowrank.element_size()),
            "element_size_bytes": int(gate_up_lowrank.element_size()),
        },
        "gate_up_lowrank_bf16_load_cast": gate_up_error,
        "down_lowrank_bf16_load_cast": down_error,
    }


def _custom_package_debug_op_support(config_root: Path = CUSTOM_OP_CONFIG_ROOT) -> dict[str, Any]:
    config_files = sorted(config_root.glob("*/binary_info_config.json"))
    by_soc: dict[str, dict[str, Any]] = {}
    for config_file in config_files:
        soc = config_file.parent.name
        entry: dict[str, Any] = {
            "config_path": str(config_file),
            "has_debug_op": False,
            "has_production_op": False,
            "op_count": 0,
        }
        try:
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["op_count"] = len(payload) if isinstance(payload, dict) else 0
            entry["has_debug_op"] = isinstance(payload, dict) and DEBUG_OP_NAME in payload
            entry["has_production_op"] = isinstance(payload, dict) and PRODUCTION_OP_NAME in payload
        by_soc[soc] = entry
    return {
        "config_root": str(config_root),
        "config_files": [str(path) for path in config_files],
        "supported_socs": sorted(by_soc),
        "debug_op_name": DEBUG_OP_NAME,
        "production_op_name": PRODUCTION_OP_NAME,
        "debug_op_supported_socs": sorted(soc for soc, entry in by_soc.items() if entry["has_debug_op"]),
        "production_op_supported_socs": sorted(soc for soc, entry in by_soc.items() if entry["has_production_op"]),
        "by_soc": by_soc,
    }


def _package_supports_runtime_soc(
    *,
    package_support: dict[str, Any],
    runtime_soc: str | None,
    supported_socs_key: str = "debug_op_supported_socs",
) -> bool:
    if runtime_soc is None:
        return False
    return runtime_soc in set(package_support.get(supported_socs_key, ()))


def _write_preflight_failure(
    *,
    summary_path: str,
    args: argparse.Namespace,
    npu_env: dict[str, Any],
    runtime_soc: dict[str, Any],
    package_support: dict[str, Any],
) -> None:
    normalized_soc = runtime_soc.get("normalized_soc")
    supported_socs = package_support.get("debug_op_supported_socs", [])
    summary = {
        "model_path": args.model_path,
        "evidence_dir": args.evidence_dir,
        "passed": False,
        "skipped": False,
        "preflight_failed": True,
        "failure_reason": (
            f"{DEBUG_OP_NAME} is not advertised for runtime SOC {normalized_soc!r}; "
            f"installed package support={supported_socs}."
        ),
        "npu_environment": npu_env,
        "runtime_soc": runtime_soc,
        "custom_package_debug_op_support": package_support,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def _stage_passed(error: dict[str, float | bool | int], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) <= max_abs_tol
        and float(error["mean_abs"]) <= mean_abs_tol
    )


def _tensor_device_metadata(tensor: torch.Tensor, *, name: str) -> dict[str, Any]:
    element_size = int(tensor.element_size())
    metadata: dict[str, Any] = {
        "name": name,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_shape": list(tensor.shape),
        "physical_shape": list(tensor.shape),
        "stride_elements": list(tensor.stride()),
        "stride_bytes": [int(stride) * element_size for stride in tensor.stride()],
        "element_size_bytes": element_size,
        "storage_offset_elements": int(tensor.storage_offset()),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "storage_nbytes": int(tensor.untyped_storage().nbytes()),
    }
    try:
        metadata["producer_gm_base"] = hex(tensor.data_ptr())
    except Exception as exc:
        metadata["producer_gm_base_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _make_expert_contiguous_inputs(
    *,
    layer: torch.nn.Module,
    spec: Any,
    layer_index: int,
    experts: list[int],
    num_tokens: int,
    seed: int,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, tuple[int, int]], dict[int, dict[str, torch.Tensor]]]:
    intermediate_size = int(layer.gate_svdq_l2.shape[1])
    expert_token_nums = torch.zeros(int(spec.num_experts), dtype=torch.int32)
    x_segments = []
    hidden_segments = []
    references: dict[int, dict[str, torch.Tensor]] = {}
    spans: dict[int, tuple[int, int]] = {}
    offset = 0
    for expert in sorted(experts):
        x, hidden = _make_inputs(
            hidden_size=int(spec.hidden_size),
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            layer_index=layer_index,
            expert=expert,
            input_scale=input_scale,
        )
        reference = build_svdq_bf16_stage_reference(layer, expert, x=x, hidden=hidden, num_tokens=num_tokens)
        x_segments.append(x)
        hidden_segments.append(hidden)
        references[expert] = reference["reference"]
        spans[expert] = (offset, offset + num_tokens)
        expert_token_nums[expert] = num_tokens
        offset += num_tokens
    return (
        torch.cat(x_segments, dim=0),
        torch.cat(hidden_segments, dim=0),
        expert_token_nums,
        spans,
        references,
    )


def _launch_debug_readback(
    *,
    layer: torch.nn.Module,
    routed_x: torch.Tensor,
    hidden: torch.Tensor,
    expert_token_nums: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enable_custom_op()
    op = getattr(torch.ops._C_ascend, "svdq_low_rank_debug_readback", None)
    if op is None:
        raise RuntimeError(
            "torch.ops._C_ascend.svdq_low_rank_debug_readback is not registered. "
            "Rebuild/install vllm-ascend after adding the debug binding."
        )
    padded_down_rank = ((int(layer.svdq_down_rank) + 255) // 256) * 256
    gate_up_svdq_l1 = _pack_factor_as_bf16_b_zn(layer.gate_up_svdq_l1, name="gate_up_svdq_l1")
    gate_svdq_l2 = _pack_factor_as_bf16_b_zn(layer.gate_svdq_l2, name="gate_svdq_l2")
    up_svdq_l2 = _pack_factor_as_bf16_b_zn(layer.up_svdq_l2, name="up_svdq_l2")
    down_svdq_l1 = _pack_factor_as_bf16_b_zn(
        layer.down_svdq_l1,
        name="down_svdq_l1",
        padded_n_cols=padded_down_rank,
    )
    down_svdq_l2 = _pack_factor_as_bf16_b_zn(
        layer.down_svdq_l2,
        name="down_svdq_l2",
        padded_k_rows=padded_down_rank,
    )

    outputs = op(
        routed_x.to(device=device, dtype=torch.bfloat16),
        hidden.to(device=device, dtype=torch.bfloat16),
        gate_up_svdq_l1.to(device=device, dtype=torch.bfloat16),
        gate_svdq_l2.to(device=device, dtype=torch.bfloat16),
        up_svdq_l2.to(device=device, dtype=torch.bfloat16),
        down_svdq_l1.to(device=device, dtype=torch.bfloat16),
        down_svdq_l2.to(device=device, dtype=torch.bfloat16),
        expert_token_nums.to(device=device, dtype=torch.int32),
        int(layer.svdq_gate_rank),
        int(layer.svdq_up_rank),
        int(layer.svdq_down_rank),
        int(layer.svdq_gate_rank_offset),
        int(layer.svdq_up_rank_offset),
    )
    torch.npu.synchronize()
    gate_up_output, down_output, gate_up_accumulator, down_accumulator = outputs
    return {
        "device_metadata": {
            "gate_up_output": _tensor_device_metadata(gate_up_output, name="gate_up_output"),
            "down_output": _tensor_device_metadata(down_output, name="down_output"),
            "gate_up_accumulator": _tensor_device_metadata(gate_up_accumulator, name="gate_up_accumulator"),
            "down_accumulator": _tensor_device_metadata(down_accumulator, name="down_accumulator"),
        },
        "gate_up_output_bf16": gate_up_output.detach().cpu(),
        "down_output_bf16": down_output.detach().cpu(),
        "gate_up_output": gate_up_output.detach().float().cpu(),
        "down_output": down_output.detach().float().cpu(),
        "gate_up_accumulator": gate_up_accumulator.detach().float().cpu(),
        "down_accumulator": down_accumulator.detach().float().cpu(),
    }


def _first_boundary_mismatch(
    *,
    stage: str,
    actual_bf16: torch.Tensor,
    expected_bf16: torch.Tensor,
    accumulator_fp32: torch.Tensor,
    expert: int,
    row_start: int,
    row_stride_elements: int,
    element_size_bytes: int,
) -> dict[str, Any] | None:
    actual_fp32 = actual_bf16.float()
    expected_fp32 = expected_bf16.float()
    mismatch = actual_fp32 != expected_fp32
    nonfinite = ~torch.isfinite(actual_fp32) | ~torch.isfinite(expected_fp32) | ~torch.isfinite(accumulator_fp32)
    bad = mismatch | nonfinite
    bad_indices = bad.nonzero(as_tuple=False)
    if bad_indices.numel() == 0:
        return None
    row, column = (int(value.item()) for value in bad_indices[0])
    physical_element_offset = (row_start + row) * row_stride_elements + column
    return {
        "stage": stage,
        "logical_index": [row, column],
        "physical_element_offset": int(physical_element_offset),
        "physical_byte_offset": int(physical_element_offset * element_size_bytes),
        "expert_id": int(expert),
        "routed_row": int(row_start + row),
        "column": int(column),
        "fp32_accumulator_value": float(accumulator_fp32[row, column].item()),
        "expected_bf16_value": float(expected_fp32[row, column].item()),
        "stored_bf16_value": float(actual_fp32[row, column].item()),
        "aiv_loaded_value": None,
        "aiv_loaded_value_status": "not_evaluated_copy_only_debug_gate_missing",
        "actual_finite": bool(torch.isfinite(actual_fp32[row, column]).item()),
        "expected_finite": bool(torch.isfinite(expected_fp32[row, column]).item()),
        "accumulator_finite": bool(torch.isfinite(accumulator_fp32[row, column]).item()),
    }


def _bf16_output_boundary_check(
    *,
    stage: str,
    actual_bf16: torch.Tensor,
    accumulator_fp32: torch.Tensor,
    expert: int,
    span: tuple[int, int],
    row_stride_elements: int,
) -> dict[str, Any]:
    start, end = span
    actual = actual_bf16[start:end]
    accumulator = accumulator_fp32[start:end]
    expected = accumulator.to(torch.bfloat16)
    error = _stage_error(actual.float(), expected.float())
    active_rows = end - start
    active_actual = actual.float()
    active_expected = expected.float()
    active_accumulator = accumulator.float()
    active_nonzero = bool((active_actual.abs().sum(dim=1) > 0).all().item()) if active_rows > 0 else True
    actual_abs = active_actual.abs()
    expected_abs = active_expected.abs()
    accumulator_abs = active_accumulator.abs()
    metadata = {
        "stage": stage,
        "producer_gm_base": "recorded_in_device_metadata",
        "workspace_region_offset_bytes": 0,
        "expert_offset": int(expert),
        "routed_row_offset": int(start),
        "output_column_offset": 0,
        "logical_shape": list(actual.shape),
        "physical_padded_shape": list(actual.shape),
        "row_stride_elements": int(row_stride_elements),
        "row_stride_bytes": int(row_stride_elements * actual.element_size()),
        "store_byte_count": int(actual.numel() * actual.element_size()),
        "alignment_bytes": int(actual.element_size()),
        "active_tile_dimensions": list(actual.shape),
    }
    first_mismatch = _first_boundary_mismatch(
        stage=stage,
        actual_bf16=actual,
        expected_bf16=expected,
        accumulator_fp32=accumulator,
        expert=expert,
        row_start=start,
        row_stride_elements=row_stride_elements,
        element_size_bytes=int(actual.element_size()),
    )
    return {
        "stage": stage,
        "producer_bf16_to_accumulator_cast_error": error,
        "actual_bf16_finite": bool(torch.isfinite(active_actual).all().item()),
        "actual_bf16_nonzero_active_rows": active_nonzero,
        "actual_bf16_max_abs": float(actual_abs.max().item()) if actual_abs.numel() else 0.0,
        "actual_bf16_mean_abs": float(actual_abs.mean().item()) if actual_abs.numel() else 0.0,
        "expected_bf16_from_accumulator_max_abs": float(expected_abs.max().item()) if expected_abs.numel() else 0.0,
        "expected_bf16_from_accumulator_mean_abs": float(expected_abs.mean().item()) if expected_abs.numel() else 0.0,
        "fp32_accumulator_max_abs": float(accumulator_abs.max().item()) if accumulator_abs.numel() else 0.0,
        "fp32_accumulator_mean_abs": float(accumulator_abs.mean().item()) if accumulator_abs.numel() else 0.0,
        "expected_bf16_from_accumulator_dtype": str(expected.dtype),
        "actual_bf16_dtype": str(actual.dtype),
        "metadata": metadata,
        "copy_only_debug_gate": {
            "evaluated": False,
            "passed": False,
            "reason": "copy-only AIV consumer readback operator is not implemented yet",
        },
        "first_mismatch_or_nonfinite": first_mismatch,
        "passed": (
            bool(error["actual_finite"])
            and bool(error["expected_finite"])
            and bool(error["diff_finite"])
            and float(error["max_abs"]) == 0.0
            and active_nonzero
        ),
    }


def _compare_expert(
    *,
    layer: torch.nn.Module,
    expert: int,
    span: tuple[int, int],
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    max_abs_tol: float,
    mean_abs_tol: float,
    require_accumulator_readback: bool,
) -> dict[str, Any]:
    start, end = span
    expected_down = reference["down_l2_output"].detach().float().cpu()

    comparisons = {
        "down_l2_output": _stage_error(actual["down_output"][start:end], expected_down),
    }
    boundary_checks = {}
    if require_accumulator_readback:
        boundary_checks["down_output_bf16_vs_accumulator_cast"] = _bf16_output_boundary_check(
            stage="down_output",
            actual_bf16=actual["down_output_bf16"],
            accumulator_fp32=actual["down_accumulator"],
            expert=expert,
            span=span,
            row_stride_elements=int(actual["down_output_bf16"].stride(0)),
        )
        comparisons["down_accumulator"] = _stage_error(
            actual["down_accumulator"][start:end],
            expected_down,
        )
    accumulator_diagnostics = {
        "down_accumulator_finite": {
            "actual_finite": bool(torch.isfinite(actual["down_accumulator"][start:end]).all().item()),
            "numel": int(actual["down_accumulator"][start:end].numel()),
            "required": bool(require_accumulator_readback),
        }
    }
    if require_accumulator_readback:
        required_names = ("down_l2_output", "down_accumulator")
    else:
        required_names = ("down_l2_output",)
    passed_by_stage = {
        name: _stage_passed(comparisons[name], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol)
        for name in required_names
    }
    boundary_passed = all(check["passed"] for check in boundary_checks.values())
    boundary_required = bool(require_accumulator_readback)
    return {
        "expert": expert,
        "span": list(span),
        "debug_invocation": "down_only_svdq_l1_l2",
        "stage_errors": comparisons,
        "accumulator_diagnostics": accumulator_diagnostics,
        "bf16_output_boundary_checks": boundary_checks,
        "required_stage_names": list(required_names),
        "stage_passed": passed_by_stage,
        "bf16_output_boundary_required": boundary_required,
        "bf16_output_boundary_passed": boundary_passed if boundary_required else None,
        "passed": all(passed_by_stage.values()) and (not boundary_required or boundary_passed),
    }


def _probe_layer(
    *,
    model_path: str,
    quant_description: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    experts: list[int],
    num_tokens: int,
    tp_size: int,
    tp_rank: int,
    seed: int,
    input_scale: float,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
    require_accumulator_readback: bool,
) -> dict[str, Any]:
    layer, spec, load_count = _load_validation_layer(
        model_path=model_path,
        quant_description=quant_description,
        weight_map=weight_map,
        layer_index=layer_index,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    routed_x, hidden, expert_token_nums, spans, references = _make_expert_contiguous_inputs(
        layer=layer,
        spec=spec,
        layer_index=layer_index,
        experts=experts,
        num_tokens=num_tokens,
        seed=seed,
        input_scale=input_scale,
    )
    actual = _launch_debug_readback(
        layer=layer,
        routed_x=routed_x,
        hidden=hidden,
        expert_token_nums=expert_token_nums,
        device=device,
    )
    expert_results = [
        _compare_expert(
            layer=layer,
            expert=expert,
            span=spans[expert],
            actual=actual,
            reference=references[expert],
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
            require_accumulator_readback=require_accumulator_readback,
        )
        for expert in sorted(experts)
    ]
    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "num_load_records": load_count,
        "num_tokens_per_expert": num_tokens,
        "expert_token_nums_nonzero": {
            str(expert): int(expert_token_nums[expert].item()) for expert in sorted(experts)
        },
        "rank_metadata": {
            "gate_rank": int(layer.svdq_gate_rank),
            "up_rank": int(layer.svdq_up_rank),
            "down_rank": int(layer.svdq_down_rank),
            "gate_rank_offset": int(layer.svdq_gate_rank_offset),
            "up_rank_offset": int(layer.svdq_up_rank_offset),
        },
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in actual.items() if isinstance(tensor, torch.Tensor)
        },
        "device_metadata": actual["device_metadata"],
        "workspace_boundary": {
            "rank_workspace_non_overlap_verified": False,
            "final_projection_workspace_non_overlap_verified": True,
            "reason": (
                "debug ACLNN workspace base for rank workspace is not exposed to Python; "
                "projection output and accumulator output tensors are separate torch allocations"
            ),
        },
        "experts": expert_results,
        "passed": all(result["passed"] for result in expert_results),
    }


def _aggregate_stage_errors(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "layer_count": len(results),
        "expert_count": 0,
        "stage_comparison_count": 0,
        "max_abs_by_stage": {},
        "mean_abs_by_stage": {},
        "all_stage_outputs_finite": True,
    }
    max_abs_by_stage: dict[str, float] = {}
    mean_abs_by_stage: dict[str, float] = {}
    for layer_result in results:
        for expert_result in layer_result["experts"]:
            aggregate["expert_count"] += 1
            for stage_name, stage_error in expert_result["stage_errors"].items():
                if "max_abs" not in stage_error or "mean_abs" not in stage_error:
                    aggregate["all_stage_outputs_finite"] = (
                        aggregate["all_stage_outputs_finite"] and bool(stage_error.get("actual_finite", False))
                    )
                    continue
                aggregate["stage_comparison_count"] += 1
                max_abs_by_stage[stage_name] = max(
                    max_abs_by_stage.get(stage_name, 0.0),
                    float(stage_error["max_abs"]),
                )
                mean_abs_by_stage[stage_name] = max(
                    mean_abs_by_stage.get(stage_name, 0.0),
                    float(stage_error["mean_abs"]),
                )
                aggregate["all_stage_outputs_finite"] = (
                    aggregate["all_stage_outputs_finite"]
                    and bool(stage_error["actual_finite"])
                    and bool(stage_error["expected_finite"])
                    and bool(stage_error["diff_finite"])
                )
    aggregate["max_abs_by_stage"] = max_abs_by_stage
    aggregate["mean_abs_by_stage"] = mean_abs_by_stage
    aggregate["max_abs_overall"] = max(max_abs_by_stage.values(), default=0.0)
    aggregate["mean_abs_overall"] = max(mean_abs_by_stage.values(), default=0.0)
    return aggregate


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    summary_path = os.path.join(args.evidence_dir, args.summary_name)
    npu_env = _npu_environment()
    device_count = int(npu_env.get("npu_device_count") or 0)
    can_run = bool(npu_env.get("npu_available")) and args.device_id < device_count
    if not can_run:
        summary = {
            "model_path": args.model_path,
            "evidence_dir": args.evidence_dir,
            "passed": not args.require_npu,
            "skipped": True,
            "skip_reason": f"NPU device {args.device_id} unavailable; device_count={device_count}.",
            "npu_environment": npu_env,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        if args.require_npu:
            raise SystemExit(1)
        return

    torch.npu.set_device(args.device_id)
    runtime_soc = _runtime_soc(args.device_id)
    package_support = _custom_package_debug_op_support()
    if not _package_supports_runtime_soc(
        package_support=package_support,
        runtime_soc=runtime_soc.get("normalized_soc"),
    ):
        _write_preflight_failure(
            summary_path=summary_path,
            args=args,
            npu_env=npu_env,
            runtime_soc=runtime_soc,
            package_support=package_support,
        )
        raise SystemExit(1)

    device = torch.device(f"npu:{args.device_id}")
    quant_description = _read_json(os.path.join(args.model_path, "quant_model_description.json"))
    weights = _weight_map(args.model_path)
    copy_only_consumer_gate: dict[str, Any]
    if args.skip_copy_only_consumer_gate:
        copy_only_consumer_gate = {
            "evaluated": False,
            "passed": False,
            "reason": "skipped by --skip-copy-only-consumer-gate",
        }
    else:
        try:
            copy_only_consumer_gate = _run_copy_only_consumer_gate(
                model_path=args.model_path,
                rows=max(1, len(args.experts) * args.num_tokens),
                seed=args.seed,
                device=device,
            )
        except Exception as exc:
            copy_only_consumer_gate = {
                "evaluated": True,
                "passed": False,
                "exception": _exception_payload(exc),
            }

    try:
        results = [
            _probe_layer(
                model_path=args.model_path,
                quant_description=quant_description,
                weight_map=weights,
                layer_index=layer_index,
                experts=args.experts,
                num_tokens=args.num_tokens,
                tp_size=args.tp_size,
                tp_rank=args.tp_rank,
                seed=args.seed,
                input_scale=args.input_scale,
                device=device,
                max_abs_tol=args.max_abs_tol,
                mean_abs_tol=args.mean_abs_tol,
                require_accumulator_readback=args.require_accumulator_readback,
            )
            for layer_index in args.layers
        ]
        producer_exception = None
    except Exception as exc:
        results = []
        producer_exception = _exception_payload(exc)

    summary = {
        "model_path": args.model_path,
        "evidence_dir": args.evidence_dir,
        "layers": args.layers,
        "experts": args.experts,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "device": str(device),
        "max_abs_tol": args.max_abs_tol,
        "mean_abs_tol": args.mean_abs_tol,
        "require_accumulator_readback": args.require_accumulator_readback,
        "npu_environment": npu_env,
        "runtime_soc": runtime_soc,
        "custom_package_debug_op_support": package_support,
        "copy_only_consumer_gate": copy_only_consumer_gate,
        "producer_exception": producer_exception,
        "aggregate_stage_errors": _aggregate_stage_errors(results),
        "results": results,
        "passed": (
            producer_exception is None
            and bool(copy_only_consumer_gate["passed"])
            and all(result["passed"] for result in results)
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        sys.stdout.flush()
        sys.stderr.flush()
        if _PRELOADED_CUSTOM_OPAPI_GLOBAL and os.environ.get("SVDQ_LOWRANK_DEBUG_ALLOW_CANN_TEARDOWN") != "1":
            os._exit(1)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
