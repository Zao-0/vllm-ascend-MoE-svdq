#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate official W4A8 GMM2 from SVDQ mixed-epilogue packed hidden.

This Stage 2.2 probe keeps the production SVDQ operator fail-closed. It uses
the already isolated mixed-epilogue debug op to produce canonical SVDQ-modified
hidden, plain INT8 hidden, official high/low packed INT4 hidden, and hidden
scale. It then relaunches the official W4A8 GMM2 consumer path through
``torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback`` with real post-loaded W2
checkpoint weights and compares the official GMM2 post-dequant readback against
the existing unfused reference derived from the official W4A8 contract.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_loader_pre_kernel_validate import DEFAULT_EVIDENCE_DIR, DEFAULT_MODEL_PATH  # noqa: E402
from svdq_mixed_epilogue_device_probe import (  # noqa: E402
    _make_inputs as _make_mixed_epilogue_inputs,
    _npu_environment,
    _run_npu_mixed_epilogue,
)
from svdq_w4a8_debug_readback_probe import (  # noqa: E402
    _destroy_hccl_if_needed,
    _init_single_rank_hccl,
)
from svdq_w4a8_debug_readback_real_checkpoint_probe import (  # noqa: E402
    _float_stats,
    _int64_float_bits_to_fp32,
    _load_real_residual_layer,
    _official_w4a8_scaled_half_c2,
    _unpack_postloaded_w4_columns,
    _unpack_postloaded_w4_columns_zN,
    _official_gmm2_raw_c2_reference,
    _official_gmm2_unfused_reference,
    _postload_metadata,
    _tensor_error,
)
from vllm_ascend.quantization.methods.svdq_post_load import pack_official_hidden_i4_reference  # noqa: E402
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
DEFAULT_SUMMARY_NAME = "phase_stage2_gmm2_from_mixed_hidden_summary.json"
_PRELOADED_CUSTOM_OPAPI_GLOBAL = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=Path(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--route-experts", type=int, nargs="*", default=None)
    parser.add_argument("--local-num-experts", type=int, default=None)
    parser.add_argument("--max-output-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--gmm2-reference-max-rows", type=int, default=64)
    parser.add_argument("--gmm2-reference-max-abs-tol", type=float, default=2e-4)
    parser.add_argument("--gmm2-reference-mean-abs-tol", type=float, default=2e-5)
    parser.add_argument("--gmm2-raw-c2-reference-max-abs-tol", type=float, default=2e-4)
    parser.add_argument("--gmm2-raw-c2-reference-mean-abs-tol", type=float, default=2e-5)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _preload_custom_opapi() -> bool:
    global _PRELOADED_CUSTOM_OPAPI_GLOBAL
    if not CUSTOM_OPAPI_LIB.exists():
        return False
    ctypes.CDLL(str(CUSTOM_OPAPI_LIB), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_CUSTOM_OPAPI_GLOBAL = True
    return True


def _enable_custom_ops() -> None:
    bootstrap_custom_op_env(include_vendor_lib=True)
    _preload_custom_opapi()
    enable_custom_op()


def _has_registered_gmm2_debug_op() -> bool:
    _enable_custom_ops()
    return getattr(torch.ops._C_ascend, "svdq_w4a8_gmm2_debug_readback", None) is not None


def _routed_experts(args: argparse.Namespace) -> list[int]:
    experts = list(range(args.top_k)) if args.route_experts is None else list(args.route_experts)
    if len(experts) != args.top_k:
        raise ValueError("route-experts length must match top-k.")
    if len(set(experts)) != len(experts):
        raise ValueError("route-experts must not contain duplicates.")
    if min(experts) < 0:
        raise ValueError("route-experts must be non-negative.")
    return experts


def _local_num_experts(args: argparse.Namespace, routed_experts: list[int]) -> int:
    local_num_experts = max(routed_experts) + 1 if args.local_num_experts is None else int(args.local_num_experts)
    if local_num_experts <= 0:
        raise ValueError("local-num-experts must be positive.")
    if max(routed_experts) >= local_num_experts:
        raise ValueError("local-num-experts must cover every routed expert id.")
    return local_num_experts


def _make_expert_idx(routed_experts: list[int], num_tokens: int, *, device: torch.device) -> torch.Tensor:
    route = torch.tensor(routed_experts, dtype=torch.int32).reshape(1, len(routed_experts)).expand(num_tokens, -1)
    return route.contiguous().to(device=device)


def _make_expert_token_nums(
    *, routed_experts: list[int], num_tokens: int, local_num_experts: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros((1, local_num_experts), dtype=torch.int32)
    for expert_id in routed_experts:
        counts[0, int(expert_id)] = int(num_tokens)
    return counts.to(device=device)


def _tensor_int_exact(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    diff = (actual_cpu.to(torch.int16) - expected_cpu.to(torch.int16)).abs()
    mismatch = diff != 0
    first_mismatch_coords = mismatch.nonzero()[:16]
    first_mismatches: list[dict[str, Any]] = []
    for coord in first_mismatch_coords:
        index = tuple(int(v) for v in coord.tolist())
        first_mismatches.append(
            {
                "index": list(index),
                "actual": int(actual_cpu[index].item()),
                "expected": int(expected_cpu[index].item()),
                "abs_diff": int(diff[index].item()),
            }
        )
    mismatch_rows = mismatch.any(dim=1) if mismatch.dim() >= 2 else torch.empty((0,), dtype=torch.bool)
    mismatch_cols = mismatch.any(dim=0) if mismatch.dim() >= 2 else torch.empty((0,), dtype=torch.bool)
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual_cpu.dtype),
        "expected_dtype": str(expected_cpu.dtype),
        "exact_match": bool(torch.equal(actual_cpu, expected_cpu)),
        "mismatch_count": int((diff != 0).sum().item()),
        "max_abs_diff": int(diff.max().item()) if diff.numel() else 0,
        "mismatch_row_count": int(mismatch_rows.sum().item()) if mismatch.dim() >= 2 else None,
        "mismatch_col_count": int(mismatch_cols.sum().item()) if mismatch.dim() >= 2 else None,
        "mismatch_rows_first32": mismatch_rows.nonzero().flatten()[:32].tolist() if mismatch.dim() >= 2 else [],
        "mismatch_cols_first32": mismatch_cols.nonzero().flatten()[:32].tolist() if mismatch.dim() >= 2 else [],
        "first_mismatches": first_mismatches,
        "actual_sample": actual_cpu.flatten()[:16].tolist(),
        "expected_sample": expected_cpu.flatten()[:16].tolist(),
    }


def _tensor_float_exact_locations(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    mismatch = diff != 0
    first_mismatch_coords = mismatch.nonzero()[:16]
    first_mismatches: list[dict[str, Any]] = []
    for coord in first_mismatch_coords:
        index = tuple(int(v) for v in coord.tolist())
        first_mismatches.append(
            {
                "index": list(index),
                "actual": float(actual_cpu[index].item()),
                "expected": float(expected_cpu[index].item()),
                "abs_diff": float(diff[index].item()),
            }
        )
    max_abs_index: list[int] = []
    if diff.numel():
        flat_index = int(diff.flatten().argmax().item())
        max_abs_index = list(torch.unravel_index(torch.tensor(flat_index), diff.shape))
        max_abs_index = [int(v) for v in max_abs_index]
    return {
        "exact_mismatch_count": int(mismatch.sum().item()),
        "exact_mismatch_indices_first32": mismatch.nonzero().flatten()[:32].tolist()
        if mismatch.dim() == 1
        else mismatch.nonzero()[:32].tolist(),
        "max_abs_index": max_abs_index,
        "first_exact_mismatches": first_mismatches,
    }


def _fp16_bit_distance_diagnostics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_fp16 = actual.detach().cpu().float().numpy().astype(np.float16, copy=False)
    expected_fp16 = expected.detach().cpu().float().numpy().astype(np.float16, copy=False)
    actual_bits = actual_fp16.view(np.uint16)
    expected_bits = expected_fp16.view(np.uint16)
    equal = actual_bits == expected_bits

    def _ordered(bits: np.ndarray) -> np.ndarray:
        bits_u32 = bits.astype(np.uint32, copy=False)
        sign = bits_u32 & np.uint32(0x8000)
        return np.where(sign != 0, np.uint32(0x8000) - bits_u32, bits_u32 + np.uint32(0x8000)).astype(
            np.int32,
            copy=False,
        )

    ulp = np.abs(_ordered(actual_bits).astype(np.int32) - _ordered(expected_bits).astype(np.int32))
    finite = np.isfinite(actual_fp16.astype(np.float32)) & np.isfinite(expected_fp16.astype(np.float32))
    mismatch_indices = np.argwhere(~equal)
    first_mismatches: list[dict[str, Any]] = []
    for coord in mismatch_indices[:16]:
        index = tuple(int(v) for v in coord.tolist())
        first_mismatches.append(
            {
                "index": list(index),
                "actual_fp16_bits": int(actual_bits[index]),
                "expected_fp16_bits": int(expected_bits[index]),
                "actual": float(actual_fp16[index]),
                "expected": float(expected_fp16[index]),
                "ulp_abs": int(ulp[index]),
            }
        )
    return {
        "fp16_bit_exact_match_count": int(equal.sum()),
        "fp16_bit_mismatch_count": int((~equal).sum()),
        "fp16_bit_total_count": int(equal.size),
        "fp16_bit_exact_match_ratio": float(equal.mean()) if equal.size else 1.0,
        "fp16_ulp_max_abs": int(ulp.max()) if ulp.size else 0,
        "fp16_ulp_mean_abs": float(ulp.mean()) if ulp.size else 0.0,
        "fp16_ulp_gt_1_count": int((ulp > 1).sum()),
        "fp16_ulp_gt_2_count": int((ulp > 2).sum()),
        "fp16_finite_pair_count": int(finite.sum()),
        "fp16_bit_mismatch_first16": first_mismatches,
    }


def _threshold_error_counts(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_tol: float,
) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    finite_diff = torch.isfinite(diff)
    denominator = expected_cpu.abs().clamp_min(1.0e-12)
    relative = torch.where(finite_diff, diff / denominator, torch.full_like(diff, float("inf")))
    failed = diff > max_abs_tol
    return {
        "failed_element_count_abs_gt_tolerance": int(failed.sum().item()) if failed.numel() else 0,
        "failed_rows_first32": failed.any(dim=1).nonzero().flatten()[:32].tolist() if failed.ndim == 2 else [],
        "failed_cols_first32": failed.any(dim=0).nonzero().flatten()[:32].tolist() if failed.ndim == 2 else [],
        "actual_nan_count": int(torch.isnan(actual_cpu).sum().item()),
        "actual_inf_count": int(torch.isinf(actual_cpu).sum().item()),
        "expected_nan_count": int(torch.isnan(expected_cpu).sum().item()),
        "expected_inf_count": int(torch.isinf(expected_cpu).sum().item()),
        "diff_nan_count": int(torch.isnan(diff).sum().item()),
        "diff_inf_count": int(torch.isinf(diff).sum().item()),
        "max_relative_error": float(relative.max().item()) if relative.numel() else 0.0,
        "mean_relative_error": float(relative.mean().item()) if relative.numel() else 0.0,
    }


def _residual_distribution_diagnostics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    abs_thresholds: tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2),
    top_k: int = 16,
) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    signed = actual_cpu - expected_cpu
    abs_diff = signed.abs()
    finite = bool(torch.isfinite(abs_diff).all().item()) if abs_diff.numel() else True
    flat_abs = abs_diff.flatten()
    flat_signed = signed.flatten()
    if not flat_abs.numel() or not finite:
        return {
            "enabled": True,
            "finite": finite,
            "numel": int(flat_abs.numel()),
            "reason": "empty or non-finite residual tensor",
        }

    quantiles = torch.tensor(
        [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0],
        dtype=torch.float32,
    )
    quantile_values = torch.quantile(flat_abs, quantiles)
    threshold_counts = {
        f"abs_le_{threshold:g}": int((flat_abs <= threshold).sum().item()) for threshold in abs_thresholds
    }
    threshold_counts.update(
        {f"abs_gt_{threshold:g}": int((flat_abs > threshold).sum().item()) for threshold in abs_thresholds}
    )

    top_count = min(int(top_k), int(flat_abs.numel()))
    top_values, top_indices = torch.topk(flat_abs, k=top_count)
    top_entries: list[dict[str, Any]] = []
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        index_tuple = torch.unravel_index(torch.tensor(int(flat_index)), abs_diff.shape)
        index = [int(v) for v in index_tuple]
        idx = tuple(index)
        top_entries.append(
            {
                "index": index,
                "abs_diff": float(value),
                "signed_diff": float(signed[idx].item()),
                "actual": float(actual_cpu[idx].item()),
                "expected": float(expected_cpu[idx].item()),
            }
        )

    row_max = abs_diff.max(dim=1).values if abs_diff.ndim == 2 and abs_diff.shape[0] else torch.empty(0)
    col_max = abs_diff.max(dim=0).values if abs_diff.ndim == 2 and abs_diff.shape[1] else torch.empty(0)
    row_top_values, row_top_indices = (
        torch.topk(row_max, k=min(16, int(row_max.numel()))) if row_max.numel() else (torch.empty(0), torch.empty(0, dtype=torch.int64))
    )
    col_top_values, col_top_indices = (
        torch.topk(col_max, k=min(16, int(col_max.numel()))) if col_max.numel() else (torch.empty(0), torch.empty(0, dtype=torch.int64))
    )

    positive_count = int((flat_signed > 0).sum().item())
    negative_count = int((flat_signed < 0).sum().item())
    zero_count = int((flat_signed == 0).sum().item())
    return {
        "enabled": True,
        "finite": finite,
        "numel": int(flat_abs.numel()),
        "signed": {
            "mean": float(flat_signed.mean().item()),
            "min": float(flat_signed.min().item()),
            "max": float(flat_signed.max().item()),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
        },
        "abs": {
            "mean": float(flat_abs.mean().item()),
            "max": float(flat_abs.max().item()),
            "quantiles": {f"q{float(q):g}": float(v) for q, v in zip(quantiles.tolist(), quantile_values.tolist(), strict=True)},
            "threshold_counts": threshold_counts,
        },
        "top_abs_entries": top_entries,
        "row_max_abs_top16": [
            {"row": int(idx), "max_abs": float(value)}
            for value, idx in zip(row_top_values.tolist(), row_top_indices.tolist(), strict=True)
        ],
        "col_max_abs_top16": [
            {"col": int(idx), "max_abs": float(value)}
            for value, idx in zip(col_top_values.tolist(), col_top_indices.tolist(), strict=True)
        ],
        "diagnostic_only": (
            "Residual distribution for narrowing the official FP16 D2/Fixpipe mismatch. "
            "It does not relax tolerances and is not a substitute reference implementation."
        ),
    }


def _rowwise_abs_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    row_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    if row_mask is not None:
        mask_cpu = row_mask.detach().cpu().bool()
        actual_cpu = actual_cpu[mask_cpu]
        expected_cpu = expected_cpu[mask_cpu]
    diff = (actual_cpu - expected_cpu).abs()
    finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    row_mean = diff.mean(dim=1) if diff.ndim == 2 and diff.shape[0] else torch.empty((0,), dtype=torch.float32)
    row_max = diff.max(dim=1).values if diff.ndim == 2 and diff.shape[0] else torch.empty((0,), dtype=torch.float32)
    return {
        "row_count": int(actual_cpu.shape[0]) if actual_cpu.ndim >= 1 else 0,
        "finite": finite,
        "mean_abs": float(diff.mean().item()) if diff.numel() and finite else float("inf"),
        "max_abs": float(diff.max().item()) if diff.numel() and finite else float("inf"),
        "row_mean_abs_first32": row_mean[:32].tolist(),
        "row_max_abs_first32": row_max[:32].tolist(),
    }


def _row_alignment_summary(actual: torch.Tensor, expected: torch.Tensor, *, max_rows: int = 64) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()[:max_rows]
    expected_cpu = expected.detach().cpu().float()[:max_rows]
    row_count = min(int(actual_cpu.shape[0]), int(expected_cpu.shape[0]))
    if row_count == 0:
        return {"enabled": True, "row_count": 0}
    actual_cpu = actual_cpu[:row_count]
    expected_cpu = expected_cpu[:row_count]
    pairwise_mean_abs = torch.cdist(actual_cpu, expected_cpu, p=1) / max(1, int(actual_cpu.shape[1]))
    best_error, best_expected_row = pairwise_mean_abs.min(dim=1)
    diagonal_error = pairwise_mean_abs.diag()
    nonidentity = best_expected_row != torch.arange(row_count, dtype=best_expected_row.dtype)
    return {
        "enabled": True,
        "row_count": row_count,
        "diagonal_mean_abs_first32": diagonal_error[:32].tolist(),
        "best_expected_row_for_actual_first32": best_expected_row[:32].tolist(),
        "best_row_mean_abs_first32": best_error[:32].tolist(),
        "nonidentity_best_row_count": int(nonidentity.sum().item()),
        "nonidentity_actual_rows_first32": nonidentity.nonzero().flatten()[:32].tolist(),
        "mean_best_row_abs": float(best_error.mean().item()),
        "mean_diagonal_abs": float(diagonal_error.mean().item()),
    }


def _row_pair_pattern_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    even_rows = torch.arange(diff.shape[0]) % 2 == 0 if diff.ndim == 2 else torch.empty((0,), dtype=torch.bool)
    odd_rows = ~even_rows if even_rows.numel() else even_rows
    adjacent_actual = (
        (actual_cpu[0::2] - actual_cpu[1::2]).abs().mean(dim=1)
        if actual_cpu.ndim == 2 and actual_cpu.shape[0] > 1
        else torch.empty((0,), dtype=torch.float32)
    )
    adjacent_expected = (
        (expected_cpu[0::2] - expected_cpu[1::2]).abs().mean(dim=1)
        if expected_cpu.ndim == 2 and expected_cpu.shape[0] > 1
        else torch.empty((0,), dtype=torch.float32)
    )
    adjacent_cross = (
        (actual_cpu[0::2] - expected_cpu[1::2]).abs().mean(dim=1)
        if actual_cpu.ndim == 2 and actual_cpu.shape[0] > 1
        else torch.empty((0,), dtype=torch.float32)
    )

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        if not mask.numel() or not bool(mask.any().item()):
            return {"row_count": 0, "mean_abs": 0.0, "max_abs": 0.0}
        selected = diff[mask]
        return {
            "row_count": int(mask.sum().item()),
            "mean_abs": float(selected.mean().item()),
            "max_abs": float(selected.max().item()),
        }

    return {
        "even_rows": summarize(even_rows),
        "odd_rows": summarize(odd_rows),
        "actual_adjacent_even_odd_mean_abs_first32": adjacent_actual[:32].tolist(),
        "expected_adjacent_even_odd_mean_abs_first32": adjacent_expected[:32].tolist(),
        "actual_even_to_expected_odd_mean_abs_first32": adjacent_cross[:32].tolist(),
        "actual_row_norm_first32": actual_cpu.norm(dim=1)[:32].tolist() if actual_cpu.ndim == 2 else [],
        "expected_row_norm_first32": expected_cpu.norm(dim=1)[:32].tolist() if expected_cpu.ndim == 2 else [],
    }


def _clip_group_counts_for_limit(counts: torch.Tensor, row_limit: int) -> list[int]:
    remaining = int(row_limit)
    clipped: list[int] = []
    for count in counts.detach().cpu().to(torch.int64).flatten().tolist():
        take = min(int(count), remaining)
        clipped.append(take)
        remaining -= take
        if remaining <= 0:
            clipped.extend([0] * (int(counts.numel()) - len(clipped)))
            break
    return clipped


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.int16).numpy().tobytes()
    return cpu.numpy().tobytes()


def _tensor_byte_manifest(tensor: torch.Tensor, *, max_sample_bytes: int = 64) -> dict[str, Any]:
    raw = _tensor_raw_bytes(tensor)
    try:
        storage_nbytes = int(tensor.untyped_storage().nbytes())
    except Exception:
        storage_nbytes = None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "logical_nbytes": int(tensor.numel() * tensor.element_size()),
        "storage_nbytes": storage_nbytes,
        "is_contiguous": bool(tensor.is_contiguous()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sample_bytes": list(raw[:max_sample_bytes]),
    }


def _tensor_metadata_manifest(tensor: torch.Tensor, *, max_sample_elements: int = 64) -> dict[str, Any]:
    try:
        storage_nbytes = int(tensor.untyped_storage().nbytes())
    except Exception:
        storage_nbytes = None
    sample = tensor.detach().flatten()[:max_sample_elements].cpu().contiguous()
    sample_bytes = _tensor_raw_bytes(sample)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "logical_nbytes": int(tensor.numel() * tensor.element_size()),
        "storage_nbytes": storage_nbytes,
        "is_contiguous": bool(tensor.is_contiguous()),
        "sample_element_count": int(sample.numel()),
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "sample_bytes": list(sample_bytes[:128]),
    }


def _row_sha256_first(tensor: torch.Tensor, *, row_count: int = 16) -> list[str]:
    cpu = tensor.detach().cpu()
    if cpu.ndim < 2:
        return []
    rows = []
    for row_idx in range(min(int(cpu.shape[0]), int(row_count))):
        rows.append(hashlib.sha256(_tensor_raw_bytes(cpu[row_idx])).hexdigest())
    return rows


def _routing_identity_manifest(
    *,
    routed_experts: list[int],
    num_tokens: int,
    top_k: int,
    local_num_experts: int,
    max_output_size: int,
    expert_token_nums: torch.Tensor,
    hidden_x_int4_packed: torch.Tensor,
    hidden_x_scale: torch.Tensor,
    hidden_x_readback: torch.Tensor | None,
    hidden_scale_readback: torch.Tensor | None,
    reference_group_counts: list[int] | None = None,
) -> dict[str, Any]:
    counts = [int(v) for v in expert_token_nums.detach().cpu().to(torch.int64).flatten().tolist()]
    prefixes: list[int] = []
    running = 0
    for count in counts:
        prefixes.append(running)
        running += int(count)
    active_rows = int(num_tokens * top_k)
    active_experts = [idx for idx, count in enumerate(counts) if count > 0]
    route_slot_by_expert = {int(expert): slot for slot, expert in enumerate(routed_experts)}

    row_map: list[dict[str, Any]] = []
    token_major_matches = True
    for expert_id, count in enumerate(counts):
        for expert_local_offset in range(int(count)):
            routed_row = prefixes[expert_id] + expert_local_offset
            topk_slot = route_slot_by_expert.get(expert_id)
            token_major_row = None
            if topk_slot is not None:
                token_major_row = expert_local_offset * top_k + topk_slot
                token_major_matches = token_major_matches and routed_row == token_major_row
            if len(row_map) < 64:
                row_map.append(
                    {
                        "routed_row": routed_row,
                        "source_token_id": expert_local_offset,
                        "top_k_slot": topk_slot,
                        "selected_expert_id": expert_id,
                        "local_expert_id": expert_id,
                        "expert_local_row_offset": expert_local_offset,
                        "expert_row_start": prefixes[expert_id],
                        "token_major_row_if_applicable": token_major_row,
                    }
                )

    padded_hidden = hidden_x_int4_packed[active_rows:max_output_size].detach().cpu()
    padded_scale = hidden_x_scale[active_rows:max_output_size].detach().cpu()
    padded_row_count = max(0, int(max_output_size) - active_rows)
    padded_hidden_zero = bool((padded_hidden == 0).all().item()) if padded_hidden.numel() else True
    padded_scale_zero = bool((padded_scale == 0).all().item()) if padded_scale.numel() else True

    readback_manifest: dict[str, Any] = {
        "enabled": hidden_x_readback is not None and hidden_scale_readback is not None,
    }
    if hidden_x_readback is not None:
        readback_manifest["hidden_int4_packed_active"] = _tensor_byte_manifest(hidden_x_readback[:active_rows])
        readback_manifest["hidden_int4_row_sha256_first16"] = _row_sha256_first(hidden_x_readback[:active_rows])
    if hidden_scale_readback is not None:
        readback_manifest["hidden_scale_active"] = _tensor_byte_manifest(hidden_scale_readback[:active_rows])

    source_counts_match_reference = None
    if reference_group_counts is not None:
        source_counts_match_reference = list(reference_group_counts) == counts[: len(reference_group_counts)]

    return {
        "manifest_scope": {
            "validates": (
                "The external hidden tensor supplied to the official GMM2 boundary is ordered "
                "expert-contiguously according to expert_token_nums, and the same bytes/scales are "
                "read back from the post-override boundary."
            ),
            "does_not_validate": (
                "For top_k > 1, this synthetic probe does not prove that multiple routed rows sharing "
                "the same source_token_id carry identical pre-routing token payloads. It validates the "
                "GMM2 input boundary row identity, not the full upstream router's token duplication."
            ),
            "same_source_token_payload_across_topk_slots_proven": bool(top_k == 1),
        },
        "expert_token_nums_shape": list(expert_token_nums.shape),
        "expert_token_nums": [counts],
        "expert_token_total": int(sum(counts)),
        "expert_token_total_matches_active_rows": int(sum(counts)) == active_rows,
        "active_expert_ids": active_experts,
        "routed_experts_argument": [int(v) for v in routed_experts],
        "route_slot_to_expert": [
            {"top_k_slot": slot, "expert_id": int(expert)} for slot, expert in enumerate(routed_experts)
        ],
        "expert_prefix_sums": prefixes,
        "expert_local_row_starts": {str(expert): prefixes[expert] for expert in active_experts},
        "expert_local_row_offsets": {
            str(expert): list(range(min(int(counts[expert]), 32))) for expert in active_experts
        },
        "source_token_set": {
            "count": int(num_tokens),
            "first32": list(range(min(int(num_tokens), 32))),
        },
        "top_k_expansion": {
            "top_k": int(top_k),
            "expanded_row_count": active_rows,
            "token_major_order_matches_expert_contiguous_order": bool(token_major_matches),
        },
        "expert_contiguous_rows": True,
        "row_source": (
            "External hidden rows are supplied to the official GMM2 boundary in expert-contiguous order; "
            "for each active expert, source_token_id is the expert-local row offset."
        ),
        "routed_row_map_first64": row_map,
        "reference_group_counts": reference_group_counts,
        "reference_group_counts_match_expert_token_nums": source_counts_match_reference,
        "tp_ep_mapping": {
            "tp_size": 1,
            "tp_rank": 0,
            "ep_size": 1,
            "ep_rank": 0,
            "local_expert_id_equals_global_expert_id": True,
        },
        "active_boundary": {
            "hidden_int4_packed_active": _tensor_byte_manifest(hidden_x_int4_packed[:active_rows]),
            "hidden_scale_active": _tensor_byte_manifest(hidden_x_scale[:active_rows]),
            "hidden_int4_row_sha256_first16": _row_sha256_first(hidden_x_int4_packed[:active_rows]),
            "hidden_scale_values_first32": hidden_x_scale[:active_rows].detach().cpu().float()[:32].tolist(),
        },
        "post_override_readback_boundary": readback_manifest,
        "padded_row_interpretation": {
            "max_output_size": int(max_output_size),
            "active_rows": active_rows,
            "padded_row_count": padded_row_count,
            "padded_rows_start": active_rows,
            "hidden_int4_padded_rows_zero": padded_hidden_zero,
            "hidden_scale_padded_rows_zero": padded_scale_zero,
            "hidden_int4_padded_nonzero_count": int((padded_hidden != 0).sum().item()) if padded_hidden.numel() else 0,
            "hidden_scale_padded_nonzero_count": int((padded_scale != 0).sum().item()) if padded_scale.numel() else 0,
        },
    }


def _gate_a_input_boundary_manifest(
    *,
    mixed: dict[str, torch.Tensor],
    hidden_x_int4_packed: torch.Tensor,
    hidden_x_scale: torch.Tensor,
    external_expert_token_nums: torch.Tensor,
    layer: Any,
    routing_identity: dict[str, Any],
    active_rows: int,
    max_output_size: int,
) -> dict[str, Any]:
    return {
        "gate": "Gate A - GMM2 Input Boundary",
        "status": "diagnostic_manifest_only",
        "scope": (
            "Records the exact tensors and row identity supplied to the official modified-hidden "
            "W4A8 GMM2 debug boundary. This does not claim Gate B or Gate C numerical success."
        ),
        "canonical_hidden_bf16": _tensor_byte_manifest(mixed["hidden_bf16"][:active_rows]),
        "hidden_int8": _tensor_byte_manifest(mixed["hidden_q"][:active_rows]),
        "hidden_int4_packed_active": _tensor_byte_manifest(hidden_x_int4_packed[:active_rows]),
        "hidden_int4_packed_full_padded": _tensor_byte_manifest(hidden_x_int4_packed),
        "hidden_scale_active": _tensor_byte_manifest(hidden_x_scale[:active_rows]),
        "hidden_scale_full_padded": _tensor_byte_manifest(hidden_x_scale),
        "expert_token_nums": _tensor_byte_manifest(external_expert_token_nums),
        "routing_identity": {
            "active_expert_ids": routing_identity["active_expert_ids"],
            "expert_token_nums": routing_identity["expert_token_nums"],
            "expert_token_total": routing_identity["expert_token_total"],
            "expert_token_total_matches_active_rows": routing_identity["expert_token_total_matches_active_rows"],
            "expert_prefix_sums": routing_identity["expert_prefix_sums"],
            "expert_local_row_starts": routing_identity["expert_local_row_starts"],
            "expert_local_row_offsets": routing_identity["expert_local_row_offsets"],
            "routed_row_map_first64": routing_identity["routed_row_map_first64"],
            "reference_group_counts": routing_identity["reference_group_counts"],
            "reference_group_counts_match_expert_token_nums": routing_identity[
                "reference_group_counts_match_expert_token_nums"
            ],
            "top_k_expansion": routing_identity["top_k_expansion"],
            "tp_ep_mapping": routing_identity["tp_ep_mapping"],
        },
        "active_row_count": int(active_rows),
        "max_output_size": int(max_output_size),
        "padded_row_interpretation": routing_identity["padded_row_interpretation"],
        "w2_packed_weight_metadata": _tensor_metadata_manifest(layer.w2_weight),
        "w2_scale_metadata": _tensor_metadata_manifest(layer.w2_weight_scale),
        "w2_scale_bias_metadata": _tensor_metadata_manifest(layer.w2_scale_bias),
        "official_source_contract": {
            "packed_hidden_workspace": (
                "dispatch_ffn_combine_w4_a8_kernel.hpp binds gmA2I4/gmA2I4_I8 to "
                "workspaceInfo.ptrA2Int4 and GMM2 consumes gmA2I4 through the official BlockMmad path."
            ),
            "hidden_scale_workspace": (
                "dispatch_ffn_combine_w4_a8_kernel.hpp binds gmPerTokenScale2 to "
                "workspaceInfo.ptrPerTokenScale2 and BlockEpilogue2 consumes that scale."
            ),
            "w2_access": (
                "GMM2 selects W2 and W2 scale through GetTensorAddr on params.ptrB2 and params.ptrScale2, "
                "preserving the official postloaded packed-W4 zN layout."
            ),
        },
    }


def _unpack_i4_bytes_variant(
    bytes_tensor: torch.Tensor,
    *,
    unpacked_columns: int,
    nibble_order: str,
) -> torch.Tensor:
    unsigned = bytes_tensor.detach().cpu().contiguous().to(torch.int16) & 0xFF
    low_nibble = unsigned & 0x0F
    high_nibble = (unsigned >> 4) & 0x0F
    if nibble_order == "low_high":
        ordered = torch.stack((low_nibble, high_nibble), dim=-1)
    elif nibble_order == "high_low":
        ordered = torch.stack((high_nibble, low_nibble), dim=-1)
    else:
        raise ValueError(f"unsupported nibble_order: {nibble_order}")
    unpacked = ordered.reshape(bytes_tensor.shape[0], unpacked_columns)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.int32)


def _packed_i4_hidden_to_parts_variant(
    hidden_x_int4_packed: torch.Tensor,
    *,
    half_order: str,
    nibble_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = hidden_x_int4_packed.detach().cpu().contiguous()
    if packed.ndim != 2 or packed.shape[1] % 2 != 0:
        raise ValueError("hidden packed INT4 debug tensor must have shape [rows, even_intermediate_size].")
    packed_columns = int(packed.shape[1])
    half_columns = packed_columns // 2
    first = _unpack_i4_bytes_variant(
        packed[:, :half_columns],
        unpacked_columns=packed_columns,
        nibble_order=nibble_order,
    )
    second = _unpack_i4_bytes_variant(
        packed[:, half_columns:packed_columns],
        unpacked_columns=packed_columns,
        nibble_order=nibble_order,
    )
    if half_order == "high_low":
        return first, second
    if half_order == "low_high":
        return second, first
    raise ValueError(f"unsupported half_order: {half_order}")


def _raw_c2_reference_from_parts(
    *,
    x_high: torch.Tensor,
    x_low: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
) -> torch.Tensor:
    row_count = min(int(x_high.shape[0]), int(x_low.shape[0]), int(max_rows))
    counts = _clip_group_counts_for_limit(expert_token_nums, row_count)
    weight_scale_fp32 = _int64_float_bits_to_fp32(weight_scale)
    unpacked_weight = _unpack_postloaded_w4_columns(weight, output_columns)

    outputs: list[torch.Tensor] = []
    row_start = 0
    for expert_id, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        row_end = row_start + count
        weight_e = unpacked_weight[expert_id]
        high_acc = x_high[row_start:row_end].matmul(weight_e)
        low_acc = x_low[row_start:row_end].matmul(weight_e)
        outputs.append(_official_w4a8_scaled_half_c2(high_acc, low_acc, weight_scale_fp32[expert_id]))
        row_start = row_end
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty((0, output_columns), dtype=torch.float32)


def _official_gmm2_d2_half_reference(
    *,
    hidden_x_int4_packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
    half: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_high, x_low = _packed_i4_hidden_to_parts_variant(
        hidden_x_int4_packed[:max_rows],
        half_order="high_low",
        nibble_order="low_high",
    )
    row_count = min(int(x_high.shape[0]), int(x_low.shape[0]), int(max_rows))
    counts = _clip_group_counts_for_limit(expert_token_nums, row_count)
    weight_scale_fp32 = _int64_float_bits_to_fp32(weight_scale)
    unpacked_weight = _unpack_postloaded_w4_columns_zN(weight, output_columns)
    if half not in {"high", "low"}:
        raise ValueError(f"unsupported D2 half reference: {half}")

    outputs: list[torch.Tensor] = []
    row_start = 0
    for expert_id, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        row_end = row_start + count
        weight_e = unpacked_weight[expert_id]
        if half == "high":
            acc = x_high[row_start:row_end].matmul(weight_e)
        else:
            acc = x_low[row_start:row_end].matmul(weight_e)
        outputs.append((acc * weight_scale_fp32[expert_id]).to(torch.float16).float())
        row_start = row_end
    reference = torch.cat(outputs, dim=0) if outputs else torch.empty((0, output_columns), dtype=torch.float32)
    return reference, {
        "enabled": True,
        "half": half,
        "source_boundary": (
            "BlockEpilogue2 reads FP16 gmC2 high/low rows, casts the selected half to FP32, "
            "and the debug tap copies it before high*16+low, aux bias, hidden scale, BF16 cast, "
            "or peer-output routing."
        ),
        "weight_layout": "official postloaded W2 Catlass layout::zN::MakeLayout<int4b_t>",
        "d2_storage": "float16_t per-channel Fixpipe output, compared after FP16 storage rounding and FP32 cast",
        "group_counts": counts,
        "diagnostic_only": (
            "Per-half D2 boundary comparator for isolating the official Fixpipe/FP16 rounding contract; "
            "not a substitute GMM2 implementation."
        ),
    }


def _scale_bits_to_fp32(scale: torch.Tensor, *, word: str) -> torch.Tensor:
    scale_u64 = scale.detach().contiguous().cpu().numpy().astype(np.uint64, copy=False)
    if word == "low32":
        scale_u32 = (scale_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32, copy=False)
    elif word == "high32":
        scale_u32 = (scale_u64 >> np.uint64(32)).astype(np.uint32, copy=False)
    else:
        raise ValueError(f"unsupported scale word: {word}")
    scale_fp32 = torch.from_numpy(scale_u32.view(np.float32).copy()).float()
    if scale_fp32.dim() == 3 and scale_fp32.shape[1] == 1:
        return scale_fp32[:, 0, :]
    return scale_fp32


def _fp32_to_fp16_mode(values: torch.Tensor, *, mode: str) -> torch.Tensor:
    values_cpu = values.detach().cpu().float()
    if mode == "none":
        return values_cpu
    if mode == "nearest_even":
        return values_cpu.to(torch.float16).float()

    values_np = values_cpu.numpy().astype(np.float32, copy=False)
    rounded = values_np.astype(np.float16)
    rounded_fp32 = rounded.astype(np.float32)
    adjusted = rounded.copy()
    if mode == "toward_zero":
        mask = np.abs(rounded_fp32) > np.abs(values_np)
        if np.any(mask):
            adjusted[mask] = np.nextafter(rounded[mask], np.float16(0.0)).astype(np.float16)
    elif mode == "floor":
        mask = rounded_fp32 > values_np
        if np.any(mask):
            adjusted[mask] = np.nextafter(rounded[mask], np.float16(-np.inf)).astype(np.float16)
    elif mode == "ceil":
        mask = rounded_fp32 < values_np
        if np.any(mask):
            adjusted[mask] = np.nextafter(rounded[mask], np.float16(np.inf)).astype(np.float16)
    else:
        raise ValueError(f"unsupported fp16 rounding mode: {mode}")
    return torch.from_numpy(adjusted.astype(np.float32, copy=True)).float()


def _official_gmm2_d2_half_variant_diagnostics(
    *,
    actual: torch.Tensor,
    hidden_x_int4_packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
    half: str,
    max_abs_tol: float,
) -> dict[str, Any]:
    if half not in {"high", "low"}:
        return {"enabled": False, "reason": f"unsupported D2 half mode: {half}"}
    x_high, x_low = _packed_i4_hidden_to_parts_variant(
        hidden_x_int4_packed[:max_rows],
        half_order="high_low",
        nibble_order="low_high",
    )
    row_count = min(int(x_high.shape[0]), int(x_low.shape[0]), int(max_rows))
    counts = _clip_group_counts_for_limit(expert_token_nums, row_count)
    unpacked_weight = _unpack_postloaded_w4_columns_zN(weight, output_columns)
    scale_variants = {
        "low32": _scale_bits_to_fp32(weight_scale, word="low32"),
        "high32": _scale_bits_to_fp32(weight_scale, word="high32"),
    }
    rounding_modes = ("nearest_even", "toward_zero", "floor", "ceil", "none")

    variant_reports: dict[str, Any] = {}
    best_name: str | None = None
    best_key: tuple[float, float, int] | None = None
    for scale_name, scale_fp32 in scale_variants.items():
        for rounding_mode in rounding_modes:
            outputs: list[torch.Tensor] = []
            row_start = 0
            for expert_id, count in enumerate(counts):
                count = int(count)
                if count <= 0:
                    continue
                row_end = row_start + count
                weight_e = unpacked_weight[expert_id]
                if half == "high":
                    acc = x_high[row_start:row_end].matmul(weight_e)
                else:
                    acc = x_low[row_start:row_end].matmul(weight_e)
                product = acc.float() * scale_fp32[expert_id].reshape(1, -1)
                outputs.append(_fp32_to_fp16_mode(product, mode=rounding_mode))
                row_start = row_end
            reference = torch.cat(outputs, dim=0) if outputs else torch.empty((0, output_columns), dtype=torch.float32)
            actual_slice = actual[: reference.shape[0], : reference.shape[1]]
            error = _tensor_error(actual_slice, reference)
            error.update(_threshold_error_counts(actual_slice, reference, max_abs_tol=max_abs_tol))
            error.update(_fp16_bit_distance_diagnostics(actual_slice, reference))
            name = f"scale_{scale_name}_fp16_{rounding_mode}"
            variant_reports[name] = {
                "scale_word": scale_name,
                "fp16_rounding_mode": rounding_mode,
                "error": error,
            }
            key = (
                float(error["max_abs"]),
                float(error["mean_abs"]),
                int(error["failed_element_count_abs_gt_tolerance"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_name = name

    return {
        "enabled": True,
        "diagnostic_only": (
            "Compares the official D2 half readback against narrow Fixpipe scale-word and FP16 rounding "
            "candidates. This does not alter the strict gate or replace the official kernel path."
        ),
        "half": half,
        "group_counts": counts,
        "best_variant": best_name,
        "best_key": list(best_key) if best_key is not None else None,
        "variants": variant_reports,
    }


def _round_up(value: int, align: int) -> int:
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _unpack_postloaded_w4_columns_zn_diagnostic(
    weight: torch.Tensor,
    output_columns: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Diagnostic-only CATLASS nZ interpretation kept for historical comparison."""
    words = weight.detach().cpu().contiguous().to(torch.int32)
    if words.ndim != 3:
        raise ValueError("postloaded W4 weight diagnostic expects [experts, k, packed_n] int32 words.")
    experts, k_rows, packed_columns = (int(v) for v in words.shape)
    if packed_columns * 8 != int(output_columns):
        raise ValueError(
            f"packed W4 columns {packed_columns} do not match output columns {output_columns}."
        )

    flat_words = words.reshape(experts, -1)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).reshape(1, 1, 8)
    flat_i4 = ((flat_words.unsqueeze(-1) >> shifts) & 0xF).reshape(experts, -1)
    flat_i4 = torch.where(flat_i4 >= 8, flat_i4 - 16, flat_i4).to(torch.int32)

    byte_per_c0 = 32
    c0_num_per_fractal = 16
    byte_per_fractal = byte_per_c0 * c0_num_per_fractal
    int4_bits = 4
    ele_num_per_c0 = (byte_per_c0 * 8) // int4_bits
    ele_num_per_fractal = (byte_per_fractal * 8) // int4_bits
    rows_round = _round_up(k_rows, ele_num_per_c0)
    cols_round = _round_up(output_columns, c0_num_per_fractal)

    row_ids = torch.arange(k_rows, dtype=torch.int64).reshape(k_rows, 1)
    col_ids = torch.arange(output_columns, dtype=torch.int64).reshape(1, output_columns)
    offsets = (
        (row_ids // ele_num_per_c0) * (cols_round * ele_num_per_c0)
        + (col_ids // c0_num_per_fractal) * ele_num_per_fractal
        + (row_ids % ele_num_per_c0)
        + (col_ids % c0_num_per_fractal) * ele_num_per_c0
    )
    if int(offsets.max().item()) >= int(flat_i4.shape[1]):
        raise ValueError(
            "computed nZ int4 offset exceeds flattened W4 storage; "
            f"max_offset={int(offsets.max().item())}, flat_i4={int(flat_i4.shape[1])}"
        )
    unpacked = flat_i4[:, offsets.reshape(-1)].reshape(experts, k_rows, output_columns)
    contract = {
        "diagnostic_only": True,
        "layout": "Catlass::layout::nZ / zN MakeLayout<Element=int4b_t>",
        "source_locations": {
            "kernel_weight_layout": (
                "dispatch_ffn_combine_w4_a8.h uses LayoutB = layout::zN when weightNz=true "
                "and creates layoutB2 = LayoutBInitializer<LayoutB, int4b_t>::create(k2, n2)"
            ),
            "op_api_attr": (
                "op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.cpp hard-codes "
                "transB=false and weightNz=true"
            ),
            "catlass_layout": (
                "third_party/catlass/include/catlass/layout/matrix.hpp nZ::MakeLayout and GetOffset"
            ),
            "postload": (
                "vllm_ascend/quantization/methods/w4a8.py process_weights_after_loading_modelslim "
                "applies maybe_trans_nz before the debug op consumes W2"
            ),
        },
        "constants": {
            "BYTE_PER_C0": byte_per_c0,
            "C0_NUM_PER_FRACTAL": c0_num_per_fractal,
            "BYTE_PER_FRACTAL": byte_per_fractal,
            "int4_bits": int4_bits,
            "ELE_NUM_PER_C0": ele_num_per_c0,
            "ELE_NUM_PER_FRACTAL": ele_num_per_fractal,
        },
        "shape": {
            "experts": experts,
            "k_rows": k_rows,
            "packed_int32_columns": packed_columns,
            "output_columns": int(output_columns),
            "flattened_int4_per_expert": int(flat_i4.shape[1]),
            "rows_round": rows_round,
            "cols_round": cols_round,
            "max_offset": int(offsets.max().item()),
        },
        "formula": (
            "offset = row/64*(colsRound*64) + col/16*1024 + row%64 + col%16*64 "
            "for int4b_t with BYTE_PER_C0=32 and C0_NUM_PER_FRACTAL=16"
        ),
    }
    return unpacked, contract


def _unpack_postloaded_w4_columns_zN_diagnostic(
    weight: torch.Tensor,
    output_columns: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Diagnostic-only CATLASS zN interpretation used by official W4A8 GMM2 B."""
    words = weight.detach().cpu().contiguous().to(torch.int32)
    if words.ndim != 3:
        raise ValueError("postloaded W4 weight diagnostic expects [experts, k, packed_n] int32 words.")
    experts, k_rows, packed_columns = (int(v) for v in words.shape)
    if packed_columns * 8 != int(output_columns):
        raise ValueError(
            f"packed W4 columns {packed_columns} do not match output columns {output_columns}."
        )

    flat_words = words.reshape(experts, -1)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).reshape(1, 1, 8)
    flat_i4 = ((flat_words.unsqueeze(-1) >> shifts) & 0xF).reshape(experts, -1)
    flat_i4 = torch.where(flat_i4 >= 8, flat_i4 - 16, flat_i4).to(torch.int32)

    byte_per_c0 = 32
    c0_num_per_fractal = 16
    byte_per_fractal = byte_per_c0 * c0_num_per_fractal
    int4_bits = 4
    ele_num_per_c0 = (byte_per_c0 * 8) // int4_bits
    ele_num_per_fractal = (byte_per_fractal * 8) // int4_bits
    rows_round = _round_up(k_rows, c0_num_per_fractal)
    cols_round = _round_up(output_columns, ele_num_per_c0)

    row_ids = torch.arange(k_rows, dtype=torch.int64).reshape(k_rows, 1)
    col_ids = torch.arange(output_columns, dtype=torch.int64).reshape(1, output_columns)
    offsets = (
        (row_ids // c0_num_per_fractal) * ele_num_per_fractal
        + (col_ids // ele_num_per_c0) * (rows_round * ele_num_per_c0)
        + (row_ids % c0_num_per_fractal) * ele_num_per_c0
        + (col_ids % ele_num_per_c0)
    )
    if int(offsets.max().item()) >= int(flat_i4.shape[1]):
        raise ValueError(
            "computed zN int4 offset exceeds flattened W4 storage; "
            f"max_offset={int(offsets.max().item())}, flat_i4={int(flat_i4.shape[1])}"
        )
    unpacked = flat_i4[:, offsets.reshape(-1)].reshape(experts, k_rows, output_columns)
    contract = {
        "diagnostic_only": True,
        "layout": "Catlass::layout::zN::MakeLayout<Element=int4b_t>",
        "source_locations": {
            "kernel_weight_layout": (
                "dispatch_ffn_combine_w4_a8.h uses LayoutB = layout::zN when weightNz=true "
                "and creates layoutB2 = LayoutBInitializer<LayoutB, int4b_t>::create(k2, n2)"
            ),
            "layout_initializer": (
                "utils/select_helper.hpp specializes LayoutBInitializer<layout::zN, int4b_t> "
                "to call layout::zN::MakeLayout<int4b_t>(k, n)"
            ),
            "catlass_layout": "third_party/catlass/include/catlass/layout/matrix.hpp zN::MakeLayout and GetOffset",
            "copy_path": (
                "CopyGmToL1<layout::zN> keeps zN in L1; CopyL1ToL0B<layout::zN> uses "
                "LoadDataWithTranspose for B"
            ),
            "postload": (
                "vllm_ascend/quantization/methods/w4a8.py process_weights_after_loading_modelslim "
                "applies maybe_trans_nz before the debug op consumes W2"
            ),
        },
        "constants": {
            "BYTE_PER_C0": byte_per_c0,
            "C0_NUM_PER_FRACTAL": c0_num_per_fractal,
            "BYTE_PER_FRACTAL": byte_per_fractal,
            "int4_bits": int4_bits,
            "ELE_NUM_PER_C0": ele_num_per_c0,
            "ELE_NUM_PER_FRACTAL": ele_num_per_fractal,
        },
        "shape": {
            "experts": experts,
            "k_rows": k_rows,
            "packed_int32_columns": packed_columns,
            "output_columns": int(output_columns),
            "flattened_int4_per_expert": int(flat_i4.shape[1]),
            "rows_round": rows_round,
            "cols_round": cols_round,
            "max_offset": int(offsets.max().item()),
        },
        "formula": (
            "offset = row/16*1024 + col/64*(rowsRound*64) + row%16*64 + col%64 "
            "for int4b_t with C0_NUM_PER_FRACTAL=16 and ELE_NUM_PER_C0=64"
        ),
    }
    return unpacked, contract


def _raw_c2_reference_with_unpacked_weight(
    *,
    x_high: torch.Tensor,
    x_low: torch.Tensor,
    unpacked_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    max_rows: int,
) -> torch.Tensor:
    row_count = min(int(x_high.shape[0]), int(x_low.shape[0]), int(max_rows))
    counts = _clip_group_counts_for_limit(expert_token_nums, row_count)
    weight_scale_fp32 = _int64_float_bits_to_fp32(weight_scale)

    outputs: list[torch.Tensor] = []
    row_start = 0
    for expert_id, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        row_end = row_start + count
        weight_e = unpacked_weight[expert_id]
        high_acc = x_high[row_start:row_end].matmul(weight_e)
        low_acc = x_low[row_start:row_end].matmul(weight_e)
        outputs.append(_official_w4a8_scaled_half_c2(high_acc, low_acc, weight_scale_fp32[expert_id]))
        row_start = row_end
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty((0, int(unpacked_weight.shape[-1])), dtype=torch.float32)


def _raw_c2_layout_variant_diagnostics(
    *,
    actual: torch.Tensor,
    hidden_x_int4_packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
    max_abs_tol: float,
) -> dict[str, Any]:
    variants = [
        ("official_high_low_low_high_nibbles", "high_low", "low_high"),
        ("high_low_high_low_nibbles", "high_low", "high_low"),
        ("low_high_low_high_nibbles", "low_high", "low_high"),
        ("low_high_high_low_nibbles", "low_high", "high_low"),
    ]
    reports: dict[str, Any] = {}
    best_name = None
    best_mean_abs = None
    for name, half_order, nibble_order in variants:
        x_high, x_low = _packed_i4_hidden_to_parts_variant(
            hidden_x_int4_packed[:max_rows],
            half_order=half_order,
            nibble_order=nibble_order,
        )
        reference = _raw_c2_reference_from_parts(
            x_high=x_high,
            x_low=x_low,
            weight=weight,
            weight_scale=weight_scale,
            expert_token_nums=expert_token_nums,
            output_columns=output_columns,
            max_rows=max_rows,
        )
        variant_actual = actual[: reference.shape[0], : reference.shape[1]]
        error = _tensor_error(variant_actual, reference)
        error.update(
            _threshold_error_counts(
                variant_actual,
                reference,
                max_abs_tol=max_abs_tol,
            )
        )
        reports[name] = {
            "half_order": half_order,
            "nibble_order": nibble_order,
            "error": error,
            "row_alignment": _row_alignment_summary(variant_actual, reference),
            "diagnostic_only": (
                "Reference variant for isolating the official A2 physical-layout boundary; "
                "not a proposed repack, scale formula, or substitute GMM2 implementation."
            ),
        }
        mean_abs = float(error["mean_abs"])
        if best_mean_abs is None or mean_abs < best_mean_abs:
            best_name = name
            best_mean_abs = mean_abs
    return {
        "enabled": True,
        "source": (
            "Official producer writes high-half bytes then low-half bytes; official GMM2 consumes "
            "the same GM region as doubled-M int4 A2. Variants only test whether the host Gate B "
            "reference is using the same physical interpretation as the official consumer."
        ),
        "official_expected_variant": "official_high_low_low_high_nibbles",
        "best_by_mean_abs": best_name,
        "variants": reports,
    }


def _raw_c2_weight_layout_variant_diagnostics(
    *,
    actual: torch.Tensor,
    hidden_x_int4_packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
    max_abs_tol: float,
) -> dict[str, Any]:
    x_high, x_low = _packed_i4_hidden_to_parts_variant(
        hidden_x_int4_packed[:max_rows],
        half_order="high_low",
        nibble_order="low_high",
    )
    row_major_weight = _unpack_postloaded_w4_columns(weight, output_columns)
    zn_weight, zn_contract = _unpack_postloaded_w4_columns_zn_diagnostic(weight, output_columns)
    zN_weight, zN_contract = _unpack_postloaded_w4_columns_zN_diagnostic(weight, output_columns)
    variants = {
        "current_row_major_host_reference": {
            "unpacked_weight": row_major_weight,
            "contract": {
                "diagnostic_only": True,
                "layout": "contiguous int32 row-major host unpack used by the existing Gate B comparator",
            },
        },
        "catlass_weight_nz_host_interpretation": {
            "unpacked_weight": zn_weight,
            "contract": zn_contract,
        },
        "catlass_weight_zN_official_b_layout": {
            "unpacked_weight": zN_weight,
            "contract": zN_contract,
        },
    }
    reports: dict[str, Any] = {}
    best_name = None
    best_mean_abs = None
    for name, variant in variants.items():
        reference = _raw_c2_reference_with_unpacked_weight(
            x_high=x_high,
            x_low=x_low,
            unpacked_weight=variant["unpacked_weight"],
            weight_scale=weight_scale,
            expert_token_nums=expert_token_nums,
            max_rows=max_rows,
        )
        variant_actual = actual[: reference.shape[0], : reference.shape[1]]
        error = _tensor_error(variant_actual, reference)
        error.update(
            _threshold_error_counts(
                variant_actual,
                reference,
                max_abs_tol=max_abs_tol,
            )
        )
        reports[name] = {
            "contract": variant["contract"],
            "error": error,
            "row_alignment": _row_alignment_summary(variant_actual, reference),
            "diagnostic_only": (
                "Host-side Gate B comparator diagnostic for the official weightNz=true W2 physical layout; "
                "not a substitute GMM2 implementation and not a proposed weight repack."
            ),
        }
        mean_abs = float(error["mean_abs"])
        if best_mean_abs is None or mean_abs < best_mean_abs:
            best_name = name
            best_mean_abs = mean_abs
    return {
        "enabled": True,
        "official_expected_weight_path": "transB=false, weightNz=true, LayoutB=layout::zN, ElementB=int4b_t",
        "best_by_mean_abs": best_name,
        "variants": reports,
    }


def _official_gmm2_layout_contract(
    *,
    active_rows: int,
    max_output_size: int,
    packed_row_bytes: int,
    output_columns: int,
) -> dict[str, Any]:
    return {
        "source_locations": {
            "producer_row_offset": (
                "block_epilogue_w4a8post_pertoken_swiglu.hpp:210-214 and :234 "
                "use ChunkTileLen=blockN/2 and gmTileD=gmD[loopIdx * ChunkTileLen]"
            ),
            "producer_high_low_writes": (
                "block_epilogue_w4a8post_pertoken_swiglu.hpp:388 and :412 write "
                "high-half bytes then low-half bytes"
            ),
            "layout_a2_d1": "dispatch_ffn_combine_w4_a8.h:309-314 creates layoutA2{m,k2} and layoutD1{maxOutputSize,k2}",
            "gmm2_doubled_m": "dispatch_ffn_combine_w4_a8_kernel.hpp:690-724 sets n2=k, k2=n/2, then doubles currentM for int4",
            "gmm2_a2_offset": "dispatch_ffn_combine_w4_a8_kernel.hpp:752-767 consumes gmA2I4 and advances by M*K int4 elements",
            "c2_high_low_epilogue": "block_epilogue_w4a8post_pertoken_v2.hpp:154-199 reads adjacent high/low C2 rows and computes high*16+low",
        },
        "computed_layout": {
            "active_rows": int(active_rows),
            "max_output_size": int(max_output_size),
            "packed_row_bytes": int(packed_row_bytes),
            "producer_high_half_bytes": int(packed_row_bytes // 2),
            "producer_low_half_bytes": int(packed_row_bytes // 2),
            "gmm2_a2_logical_rows_after_int4_m_double": int(active_rows * 2),
            "gmm2_a2_k2_int4_columns": int(packed_row_bytes),
            "gmm2_a2_group_int4_elements": int(active_rows * 2 * packed_row_bytes),
            "gmm2_a2_group_physical_bytes": int(active_rows * packed_row_bytes),
            "producer_d1_group_physical_bytes": int(active_rows * packed_row_bytes),
            "a2_physical_bytes_match_producer_d1_bytes": True,
            "gmm2_c2_n2_output_columns": int(output_columns),
            "gmm2_c2_high_low_row_offset": int(output_columns),
        },
    }


def _official_gmm2_fixpipe_contract() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "gate_progress": False,
        "source_locations": {
            "accumulator_type": "catlass/gemm/helper.hpp:137-140 maps int4b_t x int4b_t to int32_t accumulator",
            "quant_tile_copy": "catlass/gemm/tile/tile_copy.hpp:218-230 selects PER_CHANNEL CopyL0CToGm and uint64 scale copy paths",
            "vdeqf16_mode": "catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:144-150 maps int32_t -> half PER_CHANNEL to QuantMode_t::VDEQF16",
            "fixpipe_call": "catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:344-364 sets FixpipeParamsV220, SetFixPipeConfig<uint64_t,false>, then Fixpipe<half,int32_t,CFG_ROW_MAJOR>",
            "cann_vector_prequant": "/usr/local/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a5/common.hpp maps int32_t -> half vector pre-quant to VDEQF16",
        },
        "contract": {
            "arch": "Catlass::Arch::AtlasA2",
            "a_type": "GemmType<AscendC::int4b_t, layout::RowMajor>",
            "b_type": "GemmType<AscendC::int4b_t, layout::zN>",
            "c_type": "GemmType<half, layout::RowMajor>",
            "element_accumulator": "int32_t",
            "scale_granularity": "ScaleGranularity::PER_CHANNEL",
            "copy_l0c_to_gm_quant_pre": "QuantMode_t::VDEQF16",
            "fixpipe_params_type": "AscendC::FixpipeParamsV220",
            "fixpipe_template": "AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>",
            "scale_tensor_type": "AscendC::LocalTensor<uint64_t>",
            "scale_path": "GM uint64 vector -> A1 uint64 vector -> C2PIPE2GM Fixpipe scale buffer",
            "set_fixpipe_config": "AscendC::SetFixPipeConfig<uint64_t, false>(scale, false)",
            "pipe_barrier": "AscendC::PipeBarrier<PIPE_FIX>()",
            "relu_enable": False,
            "unit_flag_source": "forwarded from copyL0CToGm caller; official DispatchFFNCombineW4A8 uses enableUnitFlag=false",
            "is_channel_split_explicitly_set": False,
        },
        "layout_params": {
            "nSize": "dstLayout.shape(1)",
            "mSize": "dstLayout.shape(0)",
            "srcStride": "srcLayout.stride(3) / srcLayout.stride(0)",
            "dstStride": "dstLayout.stride(0)",
        },
        "interpretation": (
            "The next valid Stage 2.2 comparator boundary is the official hardware "
            "VDEQF16 Fixpipe dequantization from int32 L0C to FP16 row-major GM. "
            "Host-side scale-bit and rounding variants remain diagnostic only."
        ),
    }


def _pad_hidden_boundary(
    *,
    hidden_x_int4_packed: torch.Tensor,
    hidden_x_scale: torch.Tensor,
    max_output_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_rows = int(hidden_x_int4_packed.shape[0])
    if active_rows > max_output_size:
        raise ValueError("active routed rows exceed max-output-size.")
    packed = torch.zeros((max_output_size, intermediate_size), dtype=torch.int8, device=device)
    scale = torch.zeros((max_output_size,), dtype=torch.float32, device=device)
    packed[:active_rows].copy_(hidden_x_int4_packed.to(device=device, dtype=torch.int8))
    scale[:active_rows].copy_(hidden_x_scale.to(device=device, dtype=torch.float32))
    return packed, scale


def _is_gmm2_loop_stats_debug(swiglu_limit: float) -> bool:
    return 430000.0 < float(swiglu_limit) < 440000.0


def _is_gmm2_raw_c2_debug(swiglu_limit: float) -> bool:
    return 450000.0 < float(swiglu_limit) < 460000.0


def _gmm2_raw_debug_mode(swiglu_limit: float) -> str | None:
    value = float(swiglu_limit)
    if 450000.0 < value < 452000.0:
        return "d2_high_half"
    if 452000.0 < value < 454000.0:
        return "d2_low_half"
    if 454000.0 < value < 460000.0:
        return "combined_high16_plus_low"
    return None


def _parse_gmm2_loop_stats(tensor: torch.Tensor, *, expert_per_rank: int) -> dict[str, Any]:
    values = tensor.detach().cpu().flatten()[:512].tolist()
    group_count = min(int(values[1]) if len(values) > 1 else 0, int(expert_per_rank), 48)
    groups = []
    for group_idx in range(group_count):
        base = 16 + group_idx * 5
        groups.append(
            {
                "group_idx": group_idx,
                "raw_current_m": int(values[base + 0]),
                "clipped_current_m": int(values[base + 1]),
                "doubled_current_m": int(values[base + 2]),
                "core_loops": int(values[base + 3]),
                "pre_current_m_sum": int(values[base + 4]),
            }
        )
    state_count = min(int(expert_per_rank), 32)
    state_probe = {
        "magic": float(values[256]) if len(values) > 256 else 0.0,
        "valid_magic": bool(len(values) > 256 and abs(float(values[256]) - 434344.0) < 0.5),
        "ep": int(values[257]) if len(values) > 257 else 0,
        "rank": int(values[258]) if len(values) > 258 else 0,
        "cumsum_base": int(values[259]) if len(values) > 259 else 0,
        "layout_base": int(values[260]) if len(values) > 260 else 0,
        "layout_base_equals_cumsum_base": bool(len(values) > 261 and int(values[261]) != 0),
        "external_expert_token_nums_first32": [int(values[272 + idx]) for idx in range(state_count)],
        "token_per_expert_cumsum_base_first32": [int(values[304 + idx]) for idx in range(state_count)],
        "token_per_expert_layout_base_first32": [int(values[336 + idx]) for idx in range(state_count)],
        "cumsum_mm_last_rank_first32": [int(values[368 + idx]) for idx in range(state_count)],
    }
    return {
        "enabled": True,
        "magic": float(values[0]) if values else 0.0,
        "expert_per_rank": int(values[1]) if len(values) > 1 else 0,
        "max_output_size": int(values[2]) if len(values) > 2 else 0,
        "total_active_rows": int(values[3]) if len(values) > 3 else 0,
        "total_doubled_rows": int(values[4]) if len(values) > 4 else 0,
        "total_core_loops": int(values[5]) if len(values) > 5 else 0,
        "groups_with_work": int(values[6]) if len(values) > 6 else 0,
        "final_pre_current_m_sum": int(values[7]) if len(values) > 7 else 0,
        "n2": int(values[8]) if len(values) > 8 else 0,
        "k2": int(values[9]) if len(values) > 9 else 0,
        "core_num": int(values[10]) if len(values) > 10 else 0,
        "groups": groups,
        "state_probe": state_probe,
        "valid_magic": bool(values and abs(float(values[0]) - 434343.0) < 0.5),
        "active_tile_count_nonzero": bool(len(values) > 5 and int(values[5]) > 0),
    }


def _run_stage(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    routed_experts = _routed_experts(args)
    local_num_experts = _local_num_experts(args, routed_experts)
    active_rows = int(args.num_tokens * args.top_k)
    if args.max_output_size < active_rows:
        raise ValueError("max-output-size must cover num_tokens * top_k routed rows.")

    layer, spec, residual_key_count = _load_real_residual_layer(
        model_path=args.model_path,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
        routed_experts=set(routed_experts),
        local_num_experts=local_num_experts,
    )
    if max(routed_experts) >= spec.num_experts:
        raise ValueError(f"route-experts={routed_experts} must be within [0, {spec.num_experts}).")

    mixed_inputs = _make_mixed_epilogue_inputs(
        num_tokens=active_rows,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        seed=args.seed,
    )
    mixed = _run_npu_mixed_epilogue(inputs=mixed_inputs, device=device, swiglu_limit=args.swiglu_limit)
    expected_packed = pack_official_hidden_i4_reference(mixed["hidden_q"])
    packed_exact = _tensor_int_exact(mixed["hidden_q_packed"], expected_packed)

    hidden_x_int4_packed, hidden_x_scale = _pad_hidden_boundary(
        hidden_x_int4_packed=mixed["hidden_q_packed"],
        hidden_x_scale=mixed["hidden_scale"],
        max_output_size=args.max_output_size,
        intermediate_size=spec.intermediate_size,
        device=device,
    )
    external_expert_token_nums = _make_expert_token_nums(
        routed_experts=routed_experts,
        num_tokens=args.num_tokens,
        local_num_experts=local_num_experts,
        device=device,
    )

    x = torch.zeros((args.num_tokens, spec.hidden_size), dtype=torch.bfloat16, device=device)
    expert_idx = _make_expert_idx(routed_experts, args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)

    _enable_custom_ops()
    op = getattr(torch.ops._C_ascend, "svdq_w4a8_gmm2_debug_readback", None)
    if op is None:
        raise RuntimeError("torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered.")

    debug_outputs = op(
        x,
        [layer.w13_weight],
        [layer.w2_weight],
        expert_idx,
        [layer.w13_weight_scale],
        [layer.w2_weight_scale],
        [layer.w13_scale_bias],
        [layer.w2_scale_bias],
        probs,
        hidden_x_int4_packed,
        hidden_x_scale,
        external_expert_token_nums,
        group,
        args.max_output_size,
        x_active_mask,
        args.swiglu_limit,
    )
    if isinstance(debug_outputs, tuple):
        gmm2_post_dequant, hidden_x_readback, hidden_scale_readback = debug_outputs
    else:
        # Backward-compatible fallback for stale installs; the ABI test requires
        # the tuple-returning debug op after this diagnostic patch is installed.
        gmm2_post_dequant = debug_outputs
        hidden_x_readback = None
        hidden_scale_readback = None
    torch.npu.synchronize()
    hidden_x_readback_exact = (
        _tensor_int_exact(hidden_x_readback[:active_rows], hidden_x_int4_packed[:active_rows])
        if hidden_x_readback is not None
        else None
    )
    hidden_scale_readback_error = (
        _tensor_error(hidden_scale_readback[:active_rows], hidden_x_scale[:active_rows])
        if hidden_scale_readback is not None
        else None
    )
    if hidden_scale_readback_error is not None:
        hidden_scale_readback_error.update(
            _tensor_float_exact_locations(hidden_scale_readback[:active_rows], hidden_x_scale[:active_rows])
        )
    hidden_scale_readback_exact = (
        hidden_scale_readback_error is not None
        and hidden_scale_readback_error["actual_finite"]
        and hidden_scale_readback_error["expected_finite"]
        and hidden_scale_readback_error["diff_finite"]
        and hidden_scale_readback_error["max_abs"] == 0.0
    )

    loop_stats_debug = _is_gmm2_loop_stats_debug(args.swiglu_limit)
    raw_c2_debug = _is_gmm2_raw_c2_debug(args.swiglu_limit)
    reference, reference_contract = _official_gmm2_unfused_reference(
        hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
        hidden_x_scale=hidden_x_scale[:active_rows],
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        scale_bias=layer.w2_scale_bias,
        expert_token_nums=external_expert_token_nums,
        output_columns=spec.hidden_size,
        max_rows=args.gmm2_reference_max_rows,
    )
    routing_identity = _routing_identity_manifest(
        routed_experts=routed_experts,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
        local_num_experts=local_num_experts,
        max_output_size=args.max_output_size,
        expert_token_nums=external_expert_token_nums,
        hidden_x_int4_packed=hidden_x_int4_packed,
        hidden_x_scale=hidden_x_scale,
        hidden_x_readback=hidden_x_readback,
        hidden_scale_readback=hidden_scale_readback,
        reference_group_counts=reference_contract.get("group_counts"),
    )
    gate_a_input_boundary = _gate_a_input_boundary_manifest(
        mixed=mixed,
        hidden_x_int4_packed=hidden_x_int4_packed,
        hidden_x_scale=hidden_x_scale,
        external_expert_token_nums=external_expert_token_nums,
        layer=layer,
        routing_identity=routing_identity,
        active_rows=active_rows,
        max_output_size=args.max_output_size,
    )
    if loop_stats_debug:
        hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
        loop_stats = _parse_gmm2_loop_stats(gmm2_post_dequant, expert_per_rank=local_num_experts)
        return {
            "stage": "stage2_modified_hidden_official_w4a8_gmm2_loop_stats",
            "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback -> aclnnSVDQW4A8GMM2DebugReadback",
            "mixed_hidden_source_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
            "official_source_of_truth": "dispatch_ffn_combine_w4_a8 GMM2 AIC scheduling path",
            "public_grouped_matmul_used": False,
            "real_checkpoint_validation": True,
            "production_svdq_host_tiling_fail_closed": True,
            "diagnostic_mode": "gmm2_loop_stats_only",
            "diagnostic_swiglu_limit": args.swiglu_limit,
            "layer_index": args.layer,
            "layer_name": spec.prefix,
            "residual_checkpoint_key_count": residual_key_count,
            "routed_experts": routed_experts,
            "shape": {
                "num_experts": spec.num_experts,
                "local_num_experts": local_num_experts,
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "num_tokens": args.num_tokens,
                "top_k": args.top_k,
                "max_output_size": args.max_output_size,
                "active_rows": active_rows,
            },
            "routing_identity": routing_identity,
            "gate_a_input_boundary": gate_a_input_boundary,
            "official_postload": {
                "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
                "metadata": _postload_metadata(layer),
            },
            "hidden_boundary": {
                "canonical_hidden_bf16": _float_stats(mixed["hidden_bf16"]),
                "hidden_int8": _float_stats(mixed["hidden_q"]),
                "hidden_int4_packed": _float_stats(hidden_x_int4_packed[:active_rows]),
                "hidden_scale": _float_stats(hidden_scale_active),
                "hidden_q_packed_exact_reference": packed_exact,
                "hidden_q_post_override_readback_exact_reference": hidden_x_readback_exact,
                "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
                "hidden_scale_post_override_exact_match": hidden_scale_readback_exact,
                "hidden_q_exact_match_from_stage2_1_probe": None,
                "hidden_q_packed_exact_match": packed_exact["exact_match"],
                "hidden_q_packed_mismatch_count": packed_exact["mismatch_count"],
            },
            "gmm2": {
                "loop_stats": loop_stats,
                "unfused_reference": {
                    "enabled": True,
                    "contract": reference_contract,
                    "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
                    "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
                },
            },
            "checks": {
                "official_gmm2_entry_reached": bool(loop_stats["valid_magic"]),
                "official_gmm2_loop_count": int(loop_stats["total_core_loops"]),
                "official_gmm2_active_tile_count": int(loop_stats["total_core_loops"]),
                "official_gmm2_active_tile_count_nonzero": bool(loop_stats["active_tile_count_nonzero"]),
                "hidden_packed_exact": packed_exact["exact_match"],
                "hidden_packed_mismatch_count_zero": packed_exact["mismatch_count"] == 0,
                "hidden_scale_finite": bool(torch.isfinite(hidden_scale_active).all().item()),
                "hidden_scale_nonzero": bool(torch.any(hidden_scale_active.abs() > 0).item()),
                "canonical_hidden_finite": bool(torch.isfinite(mixed["hidden_bf16"].float()).all().item()),
                "canonical_hidden_nonzero": bool(torch.any(mixed["hidden_bf16"].float().abs() > 0).item()),
                "official_gmm2_numerical_gate_passed": False,
            },
            "passed": False,
        }

    if raw_c2_debug:
        raw_debug_mode = _gmm2_raw_debug_mode(args.swiglu_limit)
        hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
        raw_c2_active = gmm2_post_dequant[:active_rows].detach().cpu()
        raw_c2_stats = _float_stats(raw_c2_active)
        raw_c2_finite = bool(torch.isfinite(raw_c2_active).all().item())
        raw_c2_nonzero = bool(torch.any(raw_c2_active.abs() > 0).item())
        if raw_debug_mode == "d2_high_half":
            raw_reference, raw_reference_contract = _official_gmm2_d2_half_reference(
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
                half="high",
            )
        elif raw_debug_mode == "d2_low_half":
            raw_reference, raw_reference_contract = _official_gmm2_d2_half_reference(
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
                half="low",
            )
        else:
            raw_reference, raw_reference_contract = _official_gmm2_raw_c2_reference(
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
            )
        raw_actual = raw_c2_active[: raw_reference.shape[0], : raw_reference.shape[1]]
        raw_c2_error = _tensor_error(raw_actual, raw_reference)
        raw_c2_error.update(
            _threshold_error_counts(
                raw_actual,
                raw_reference,
                max_abs_tol=args.gmm2_raw_c2_reference_max_abs_tol,
            )
        )
        raw_c2_error.update(_fp16_bit_distance_diagnostics(raw_actual, raw_reference))
        raw_c2_reference_passed = (
            raw_c2_error["actual_finite"]
            and raw_c2_error["expected_finite"]
            and raw_c2_error["diff_finite"]
            and raw_c2_error["max_abs"] <= args.gmm2_raw_c2_reference_max_abs_tol
            and raw_c2_error["mean_abs"] <= args.gmm2_raw_c2_reference_mean_abs_tol
        )
        if raw_debug_mode in {"d2_high_half", "d2_low_half"}:
            raw_c2_d2_half_variant_diagnostics = _official_gmm2_d2_half_variant_diagnostics(
                actual=raw_actual,
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
                half="high" if raw_debug_mode == "d2_high_half" else "low",
                max_abs_tol=args.gmm2_raw_c2_reference_max_abs_tol,
            )
        else:
            raw_c2_d2_half_variant_diagnostics = {
                "enabled": False,
                "reason": "D2 half variant diagnostics require raw debug mode d2_high_half or d2_low_half",
            }
        hidden_row_exact_mask = None
        hidden_readback_raw_reference = None
        hidden_readback_raw_reference_passed = None
        hidden_readback_raw_reference_report: dict[str, Any] = {
            "enabled": False,
            "reason": "debug op did not return hidden_x_readback",
        }
        if hidden_x_readback is not None:
            hidden_readback_active = hidden_x_readback[:active_rows].detach().cpu()
            source_hidden_active = hidden_x_int4_packed[:active_rows].detach().cpu()
            hidden_row_exact_mask = (hidden_readback_active == source_hidden_active).all(dim=1)
            if raw_debug_mode == "d2_high_half":
                hidden_readback_raw_reference, hidden_readback_raw_contract = _official_gmm2_d2_half_reference(
                    hidden_x_int4_packed=hidden_x_readback[:active_rows],
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale,
                    expert_token_nums=external_expert_token_nums,
                    output_columns=spec.hidden_size,
                    max_rows=args.gmm2_reference_max_rows,
                    half="high",
                )
            elif raw_debug_mode == "d2_low_half":
                hidden_readback_raw_reference, hidden_readback_raw_contract = _official_gmm2_d2_half_reference(
                    hidden_x_int4_packed=hidden_x_readback[:active_rows],
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale,
                    expert_token_nums=external_expert_token_nums,
                    output_columns=spec.hidden_size,
                    max_rows=args.gmm2_reference_max_rows,
                    half="low",
                )
            else:
                hidden_readback_raw_reference, hidden_readback_raw_contract = _official_gmm2_raw_c2_reference(
                    hidden_x_int4_packed=hidden_x_readback[:active_rows],
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale,
                    expert_token_nums=external_expert_token_nums,
                    output_columns=spec.hidden_size,
                    max_rows=args.gmm2_reference_max_rows,
                )
            hidden_readback_raw_actual = raw_c2_active[
                : hidden_readback_raw_reference.shape[0], : hidden_readback_raw_reference.shape[1]
            ]
            hidden_readback_raw_error = _tensor_error(hidden_readback_raw_actual, hidden_readback_raw_reference)
            hidden_readback_raw_error.update(
                _threshold_error_counts(
                    hidden_readback_raw_actual,
                    hidden_readback_raw_reference,
                    max_abs_tol=args.gmm2_raw_c2_reference_max_abs_tol,
                )
            )
            hidden_readback_raw_error.update(
                _fp16_bit_distance_diagnostics(hidden_readback_raw_actual, hidden_readback_raw_reference)
            )
            hidden_readback_raw_reference_passed = (
                hidden_readback_raw_error["actual_finite"]
                and hidden_readback_raw_error["expected_finite"]
                and hidden_readback_raw_error["diff_finite"]
                and hidden_readback_raw_error["max_abs"] <= args.gmm2_raw_c2_reference_max_abs_tol
                and hidden_readback_raw_error["mean_abs"] <= args.gmm2_raw_c2_reference_mean_abs_tol
            )
            hidden_readback_raw_reference_report = {
                "enabled": True,
                "passed": bool(hidden_readback_raw_reference_passed),
                "contract": hidden_readback_raw_contract,
                "error": hidden_readback_raw_error,
                "max_abs_tolerance": args.gmm2_raw_c2_reference_max_abs_tol,
                "mean_abs_tolerance": args.gmm2_raw_c2_reference_mean_abs_tol,
            }
        raw_c2_row_diagnostics = {
            "source_hidden_row_error": _rowwise_abs_error_summary(raw_actual, raw_reference),
            "source_hidden_row_alignment": _row_alignment_summary(raw_actual, raw_reference),
            "source_hidden_even_odd_pattern": _row_pair_pattern_summary(raw_actual, raw_reference),
        }
        if raw_debug_mode == "combined_high16_plus_low":
            raw_c2_layout_variants = _raw_c2_layout_variant_diagnostics(
                actual=raw_actual,
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
                max_abs_tol=args.gmm2_raw_c2_reference_max_abs_tol,
            )
            raw_c2_weight_layout_variants = _raw_c2_weight_layout_variant_diagnostics(
                actual=raw_actual,
                hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale,
                expert_token_nums=external_expert_token_nums,
                output_columns=spec.hidden_size,
                max_rows=args.gmm2_reference_max_rows,
                max_abs_tol=args.gmm2_raw_c2_reference_max_abs_tol,
            )
        else:
            raw_c2_layout_variants = {
                "enabled": False,
                "reason": "layout variants compare the combined high*16+low boundary, not a single D2 half",
            }
            raw_c2_weight_layout_variants = {
                "enabled": False,
                "reason": "weight-layout variants compare the combined high*16+low boundary, not a single D2 half",
            }
        if hidden_row_exact_mask is not None:
            exact_count = int(hidden_row_exact_mask.sum().item())
            mismatch_count = int((~hidden_row_exact_mask).sum().item())
            raw_c2_row_diagnostics.update(
                {
                    "hidden_readback_exact_row_count": exact_count,
                    "hidden_readback_mismatched_row_count": mismatch_count,
                    "source_reference_error_on_hidden_exact_rows": _rowwise_abs_error_summary(
                        raw_actual,
                        raw_reference,
                        row_mask=hidden_row_exact_mask[: raw_actual.shape[0]],
                    ),
                    "source_reference_error_on_hidden_mismatched_rows": _rowwise_abs_error_summary(
                        raw_actual,
                        raw_reference,
                        row_mask=(~hidden_row_exact_mask[: raw_actual.shape[0]]),
                    ),
                }
            )
            if hidden_readback_raw_reference is not None:
                raw_c2_row_diagnostics.update(
                    {
                        "readback_hidden_row_error": _rowwise_abs_error_summary(
                            raw_actual,
                            hidden_readback_raw_reference[: raw_actual.shape[0], : raw_actual.shape[1]],
                        ),
                        "readback_hidden_row_alignment": _row_alignment_summary(
                            raw_actual,
                            hidden_readback_raw_reference[: raw_actual.shape[0], : raw_actual.shape[1]],
                        ),
                        "readback_hidden_even_odd_pattern": _row_pair_pattern_summary(
                            raw_actual,
                            hidden_readback_raw_reference[: raw_actual.shape[0], : raw_actual.shape[1]],
                        ),
                    }
                )
        return {
            "stage": "stage2_modified_hidden_official_w4a8_gmm2_raw_c2",
            "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback -> aclnnSVDQW4A8GMM2DebugReadback",
            "mixed_hidden_source_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
            "official_source_of_truth": (
                "dispatch_ffn_combine_w4_a8 full lifecycle GMM2 AIC path plus "
                "BlockEpilogue2 rawDebugOnly C2 high/low decode"
            ),
            "public_grouped_matmul_used": False,
            "real_checkpoint_validation": True,
            "production_svdq_host_tiling_fail_closed": True,
            "diagnostic_mode": f"full_lifecycle_gmm2_{raw_debug_mode}_readback",
            "diagnostic_swiglu_limit": args.swiglu_limit,
            "layer_index": args.layer,
            "layer_name": spec.prefix,
            "residual_checkpoint_key_count": residual_key_count,
            "routed_experts": routed_experts,
            "shape": {
                "num_experts": spec.num_experts,
                "local_num_experts": local_num_experts,
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "num_tokens": args.num_tokens,
                "top_k": args.top_k,
                "max_output_size": args.max_output_size,
                "active_rows": active_rows,
            },
            "routing_identity": routing_identity,
            "gate_a_input_boundary": gate_a_input_boundary,
            "official_postload": {
                "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
                "metadata": _postload_metadata(layer),
            },
            "hidden_boundary": {
                "canonical_hidden_bf16": _float_stats(mixed["hidden_bf16"]),
                "hidden_int8": _float_stats(mixed["hidden_q"]),
                "hidden_int4_packed": _float_stats(hidden_x_int4_packed[:active_rows]),
                "hidden_scale": _float_stats(hidden_scale_active),
                "hidden_q_packed_exact_reference": packed_exact,
                "hidden_q_post_override_readback_exact_reference": hidden_x_readback_exact,
                "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
                "hidden_scale_post_override_exact_match": hidden_scale_readback_exact,
                "hidden_q_exact_match_from_stage2_1_probe": None,
                "hidden_q_packed_exact_match": packed_exact["exact_match"],
                "hidden_q_packed_mismatch_count": packed_exact["mismatch_count"],
            },
            "gmm2": {
                "raw_c2_high_low_decoded_active": raw_c2_stats,
                "official_a2_c2_physical_layout_contract": _official_gmm2_layout_contract(
                    active_rows=active_rows,
                    max_output_size=args.max_output_size,
                    packed_row_bytes=int(hidden_x_int4_packed.shape[1]),
                    output_columns=spec.hidden_size,
                ),
                "official_fixpipe_contract": _official_gmm2_fixpipe_contract(),
                "raw_c2_contract": {
                    "source_boundary": (
                        "BlockEpilogue2 reads official gmC2 after GMM2/C2V and writes the "
                        "existing W4A8_DEBUG gmGMM2 tap before aux bias, hidden scale, "
                        "BF16 cast, or peer-output routing. Mode 1 writes high*16+low; "
                        "mode 2 writes the high FP16 D2 half after FP32 cast; mode 3 writes "
                        "the low FP16 D2 half after FP32 cast."
                    ),
                    "source_file": (
                        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
                        "block_epilogue_w4a8post_pertoken_v2.hpp"
                    ),
                    "enabled_by_swiglu_limit_range": [450000.0, 460000.0],
                    "debug_mode": raw_debug_mode,
                },
                "raw_c2_unfused_reference": {
                    "enabled": True,
                    "passed": bool(raw_c2_reference_passed),
                    "contract": raw_reference_contract,
                    "error": raw_c2_error,
                    "residual_distribution": _residual_distribution_diagnostics(raw_actual, raw_reference),
                    "d2_half_variant_diagnostics": raw_c2_d2_half_variant_diagnostics,
                    "max_abs_tolerance": args.gmm2_raw_c2_reference_max_abs_tol,
                    "mean_abs_tolerance": args.gmm2_raw_c2_reference_mean_abs_tol,
                    "diagnostic_only": (
                        "This host-side official-contract reference is a Gate B comparator, "
                        "not an alternative implementation or a substitute for the official kernel path."
                    ),
                },
                "raw_c2_readback_hidden_reference": hidden_readback_raw_reference_report,
                "raw_c2_row_diagnostics": raw_c2_row_diagnostics,
                "raw_c2_layout_variant_diagnostics": raw_c2_layout_variants,
                "raw_c2_weight_layout_variant_diagnostics": raw_c2_weight_layout_variants,
                "unfused_reference": {
                    "enabled": True,
                    "contract": reference_contract,
                    "post_dequant_comparison_run": False,
                    "reason": "raw C2 diagnostic stops before aux/scale/final post-dequant semantics",
                    "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
                    "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
                },
            },
            "checks": {
                "official_gmm2_entry_reached": True,
                "official_gmm2_loop_count": None,
                "official_gmm2_active_tile_count": None,
                "official_gmm2_aic_raw_output_finite": raw_c2_finite,
                "official_gmm2_aic_raw_output_nonzero": raw_c2_nonzero,
                "official_gmm2_aic_reference_passed": bool(raw_c2_reference_passed),
                "official_gmm2_c2v_handoff_verified": bool(raw_c2_finite and raw_c2_nonzero),
                "official_gmm2_post_dequant_finite": False,
                "official_gmm2_post_dequant_nonzero": False,
                "official_gmm2_post_dequant_reference_passed": False,
                "official_gmm2_numerical_gate_passed": False,
                "hidden_packed_exact": packed_exact["exact_match"],
                "hidden_packed_mismatch_count_zero": packed_exact["mismatch_count"] == 0,
                "hidden_post_override_readback_exact": (
                    hidden_x_readback_exact is not None and hidden_x_readback_exact["exact_match"]
                ),
                "hidden_scale_post_override_readback_exact": bool(hidden_scale_readback_exact),
                "hidden_scale_finite": bool(torch.isfinite(hidden_scale_active).all().item()),
                "hidden_scale_nonzero": bool(torch.any(hidden_scale_active.abs() > 0).item()),
                "canonical_hidden_finite": bool(torch.isfinite(mixed["hidden_bf16"].float()).all().item()),
                "canonical_hidden_nonzero": bool(torch.any(mixed["hidden_bf16"].float().abs() > 0).item()),
            },
            "passed": False,
        }

    actual = gmm2_post_dequant[: reference.shape[0], : reference.shape[1]]
    error = _tensor_error(actual, reference)
    gmm2_reference_passed = (
        error["actual_finite"]
        and error["expected_finite"]
        and error["diff_finite"]
        and error["max_abs"] <= args.gmm2_reference_max_abs_tol
        and error["mean_abs"] <= args.gmm2_reference_mean_abs_tol
    )

    hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
    gmm2_active = gmm2_post_dequant[:active_rows].detach().cpu()
    return {
        "stage": "stage2_modified_hidden_official_w4a8_gmm2",
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback -> aclnnSVDQW4A8GMM2DebugReadback",
        "mixed_hidden_source_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "official_source_of_truth": "dispatch_ffn_combine_w4_a8 GMM2 AIC path and W4A8_DEBUG AIV post-dequant tap",
        "public_grouped_matmul_used": False,
        "real_checkpoint_validation": True,
        "production_svdq_host_tiling_fail_closed": True,
        "layer_index": args.layer,
        "layer_name": spec.prefix,
        "residual_checkpoint_key_count": residual_key_count,
        "routed_experts": routed_experts,
        "shape": {
            "num_experts": spec.num_experts,
            "local_num_experts": local_num_experts,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "max_output_size": args.max_output_size,
            "active_rows": active_rows,
        },
        "routing_identity": routing_identity,
        "gate_a_input_boundary": gate_a_input_boundary,
        "official_postload": {
            "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
            "metadata": _postload_metadata(layer),
        },
        "hidden_boundary": {
            "canonical_hidden_bf16": _float_stats(mixed["hidden_bf16"]),
            "hidden_int8": _float_stats(mixed["hidden_q"]),
            "hidden_int4_packed": _float_stats(hidden_x_int4_packed[:active_rows]),
            "hidden_scale": _float_stats(hidden_scale_active),
            "hidden_q_packed_exact_reference": packed_exact,
            "hidden_q_post_override_readback_exact_reference": hidden_x_readback_exact,
            "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
            "hidden_scale_post_override_exact_match": hidden_scale_readback_exact,
            "hidden_q_exact_match_from_stage2_1_probe": None,
            "hidden_q_packed_exact_match": packed_exact["exact_match"],
            "hidden_q_packed_mismatch_count": packed_exact["mismatch_count"],
        },
        "gmm2": {
            "post_dequant_active": _float_stats(gmm2_active),
            "unfused_reference": {
                "enabled": True,
                "passed": bool(gmm2_reference_passed),
                "contract": reference_contract,
                "error": error,
                "residual_distribution": _residual_distribution_diagnostics(actual, reference),
                "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
                "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
            },
        },
        "checks": {
            "official_gmm2_kernel_launched": True,
            "hidden_packed_exact": packed_exact["exact_match"],
            "hidden_packed_mismatch_count_zero": packed_exact["mismatch_count"] == 0,
            "hidden_post_override_readback_exact": (
                hidden_x_readback_exact is not None and hidden_x_readback_exact["exact_match"]
            ),
            "hidden_scale_post_override_readback_exact": bool(hidden_scale_readback_exact),
            "hidden_scale_finite": bool(torch.isfinite(hidden_scale_active).all().item()),
            "hidden_scale_nonzero": bool(torch.any(hidden_scale_active.abs() > 0).item()),
            "canonical_hidden_finite": bool(torch.isfinite(mixed["hidden_bf16"].float()).all().item()),
            "canonical_hidden_nonzero": bool(torch.any(mixed["hidden_bf16"].float().abs() > 0).item()),
            "gmm2_reference_passed": bool(gmm2_reference_passed),
        },
        "passed": bool(
            packed_exact["exact_match"]
            and packed_exact["mismatch_count"] == 0
            and hidden_x_readback_exact is not None
            and hidden_x_readback_exact["exact_match"]
            and hidden_scale_readback_exact
            and torch.isfinite(hidden_scale_active).all().item()
            and torch.any(hidden_scale_active.abs() > 0).item()
            and torch.isfinite(mixed["hidden_bf16"].float()).all().item()
            and torch.any(mixed["hidden_bf16"].float().abs() > 0).item()
            and gmm2_reference_passed
        ),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _finish_probe(payload: dict[str, Any], exit_code: int) -> int:
    print(json.dumps(payload, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if _PRELOADED_CUSTOM_OPAPI_GLOBAL and os.environ.get("SVDQ_GMM2_DEBUG_ALLOW_CANN_TEARDOWN") != "1":
        os._exit(exit_code)
    return exit_code


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_gmm2_from_mixed_hidden_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "production_svdq_host_tiling_fail_closed": True,
    }
    if not (env["torch_npu_imported"] and env["npu_available"] and int(env["npu_device_count"]) > args.device_id):
        summary["skipped"] = not args.require_npu
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch_npu import and an available selected NPU are required."
        _write_summary(summary_path, summary)
        return _finish_probe(
            {"summary_path": str(summary_path), "passed": False, "skipped": summary["skipped"]},
            2 if args.require_npu else 0,
        )

    torch.npu.set_device(args.device_id)
    try:
        registered = _has_registered_gmm2_debug_op()
    except Exception as exc:
        summary["preflight_failed"] = True
        summary["failure_reason"] = f"failed to enable custom ops: {type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        return _finish_probe({"summary_path": str(summary_path), "passed": False, "skipped": False}, 1)
    summary["torch_op_registered"] = registered
    if not registered:
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered."
        _write_summary(summary_path, summary)
        return _finish_probe({"summary_path": str(summary_path), "passed": False, "skipped": False}, 1)

    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl_group"] = group_info
        stage = _run_stage(args, str(group_info["group"]))
        summary["stage"] = stage
        summary["passed"] = bool(stage["passed"])
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["passed"] = False
    finally:
        _destroy_hccl_if_needed(group_info)

    _write_summary(summary_path, summary)
    return _finish_probe(
        {"summary_path": str(summary_path), "passed": summary["passed"], "skipped": False},
        0 if summary["passed"] else 1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
