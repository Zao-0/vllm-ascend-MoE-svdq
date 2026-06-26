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
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sample_bytes": list(raw[:max_sample_bytes]),
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
        hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
        raw_c2_active = gmm2_post_dequant[:active_rows].detach().cpu()
        raw_c2_stats = _float_stats(raw_c2_active)
        raw_c2_finite = bool(torch.isfinite(raw_c2_active).all().item())
        raw_c2_nonzero = bool(torch.any(raw_c2_active.abs() > 0).item())
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
        raw_c2_reference_passed = (
            raw_c2_error["actual_finite"]
            and raw_c2_error["expected_finite"]
            and raw_c2_error["diff_finite"]
            and raw_c2_error["max_abs"] <= args.gmm2_raw_c2_reference_max_abs_tol
            and raw_c2_error["mean_abs"] <= args.gmm2_raw_c2_reference_mean_abs_tol
        )
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
            "diagnostic_mode": "full_lifecycle_gmm2_raw_c2_readback",
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
                "raw_c2_contract": {
                    "source_boundary": (
                        "BlockEpilogue2 reads official gmC2 after GMM2/C2V, casts high/low "
                        "FP16 halves to FP32, computes high * 16 + low, and writes the "
                        "existing W4A8_DEBUG gmGMM2 tap before aux bias, hidden scale, "
                        "BF16 cast, or peer-output routing."
                    ),
                    "source_file": (
                        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
                        "block_epilogue_w4a8post_pertoken_v2.hpp"
                    ),
                    "enabled_by_swiglu_limit_range": [450000.0, 460000.0],
                },
                "raw_c2_unfused_reference": {
                    "enabled": True,
                    "passed": bool(raw_c2_reference_passed),
                    "contract": raw_reference_contract,
                    "error": raw_c2_error,
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
