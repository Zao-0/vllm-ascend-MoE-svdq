#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate SVDQ mixed epilogues using official W4A8 debug taps.

This probe is intentionally isolated from production
``DispatchFFNCombineW4A8SVDQ``. It uses the official
``dispatch_ffn_combine_w4_a8`` debug readback as the W4A8 source of truth,
loads real checkpoint SVDQ factors, computes BF16 low-rank outputs through the
SVDQ low-rank debug op, and feeds official residual taps plus BF16 low-rank
outputs into the mixed-epilogue debug op.
"""

from __future__ import annotations

import argparse
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

from svdq_bf16_stage_device_probe import _load_validation_layer  # noqa: E402
from svdq_loader_pre_kernel_validate import DEFAULT_EVIDENCE_DIR, DEFAULT_MODEL_PATH, _read_json, _weight_map  # noqa: E402
from svdq_lowrank_debug_readback_probe import _launch_debug_readback as _launch_lowrank_debug  # noqa: E402
from svdq_mixed_epilogue_device_probe import _run_npu_mixed_epilogue  # noqa: E402
from svdq_w4a8_gmm2_from_mixed_hidden_probe import (  # noqa: E402
    _gate_a_input_boundary_manifest,
    _has_registered_gmm2_debug_op,
    _official_lifecycle_debug_contract,
    _pad_hidden_boundary,
    _routing_identity_manifest,
    _tensor_int_exact,
    _unpack_gmm2_debug_outputs,
)
from svdq_w4a8_debug_readback_probe import _destroy_hccl_if_needed, _init_single_rank_hccl  # noqa: E402
from svdq_w4a8_debug_readback_real_checkpoint_probe import (  # noqa: E402
    _has_registered_debug_op,
    _load_real_residual_layer,
    _local_num_experts,
    _make_expert_idx,
    _make_input,
    _npu_environment,
    _official_debug_symbol_status,
    _official_gmm1_unfused_reference,
    _official_gmm2_unfused_reference,
    _routed_experts,
)

from vllm_ascend.quantization.methods.svdq_post_load import (  # noqa: E402
    build_svdq_final_combine_reference,
    build_svdq_mixed_epilogue_reference,
    pack_official_hidden_i4_reference,
)

DEFAULT_SUMMARY_NAME = "phase_w4a8_tap_mixed_epilogue_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=Path(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--route-experts", type=int, nargs="*", default=None)
    parser.add_argument("--local-num-experts", type=int, default=None)
    parser.add_argument("--max-output-size", type=int, default=64)
    parser.add_argument("--input-mode", choices=("linear",), default="linear")
    parser.add_argument("--linear-input-scale", type=float, default=0.01)
    parser.add_argument("--mixed-max-abs-tol", type=float, default=0.02)
    parser.add_argument("--mixed-mean-abs-tol", type=float, default=0.002)
    parser.add_argument("--scale-tol", type=float, default=1e-7)
    parser.add_argument("--hidden-q-mismatch-count-tol", type=int, default=2)
    parser.add_argument("--hidden-q-max-abs-diff-tol", type=int, default=1)
    parser.add_argument("--w4a8-reference-max-rows", type=int, default=64)
    parser.add_argument("--gmm1-reference-max-abs-tol", type=float, default=2e-2)
    parser.add_argument("--gmm1-reference-mean-abs-tol", type=float, default=2e-3)
    parser.add_argument("--gmm2-reference-max-abs-tol", type=float, default=2e-4)
    parser.add_argument("--gmm2-reference-mean-abs-tol", type=float, default=2e-5)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual_cpu.dtype),
        "expected_dtype": str(expected_cpu.dtype),
        "actual_finite": bool(torch.isfinite(actual_cpu.float()).all().item()) if actual_cpu.numel() else True,
        "expected_finite": bool(torch.isfinite(expected_cpu.float()).all().item()) if expected_cpu.numel() else True,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
        "numel": int(diff.numel()),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(cpu).all().item()) if cpu.numel() else True
    abs_cpu = torch.nan_to_num(cpu).abs()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(abs_cpu.max().item()) if abs_cpu.numel() else 0.0,
        "mean_abs": float(abs_cpu.mean().item()) if abs_cpu.numel() else 0.0,
        "nan_count": int(torch.isnan(cpu).sum().item()),
        "inf_count": int(torch.isinf(cpu).sum().item()),
    }


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.int16).numpy().tobytes()
    return cpu.numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_raw_bytes(tensor)).hexdigest()


def _row_sha256_first(tensor: torch.Tensor, *, row_count: int = 16) -> list[str]:
    cpu = tensor.detach().cpu()
    if cpu.ndim < 2:
        return []
    rows = []
    for row_idx in range(min(int(cpu.shape[0]), int(row_count))):
        rows.append(hashlib.sha256(_tensor_raw_bytes(cpu[row_idx])).hexdigest())
    return rows


def _nonzero_finite(stats: dict[str, Any]) -> bool:
    return bool(stats["finite"] and float(stats["max_abs"]) > 0.0)


def _stage_passed(error: dict[str, Any], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) <= max_abs_tol
        and float(error["mean_abs"]) <= mean_abs_tol
    )


def _not_evaluated_stage_passed() -> dict[str, bool]:
    return {
        "gate_mixed": False,
        "up_mixed": False,
        "hidden_bf16": False,
        "hidden_scale": False,
        "hidden_q": False,
        "down_mixed": False,
        "out_bf16": False,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _active_expert_counts(expert_token_nums: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    counts = expert_token_nums.detach().cpu().reshape(-1).to(torch.int64)
    active_experts = [int(index) for index, count in enumerate(counts.tolist()) if int(count) > 0]
    return active_experts, counts


def _group_routed_x_by_expert(x: torch.Tensor, counts: torch.Tensor, active_experts: list[int]) -> torch.Tensor:
    rows = []
    x_cpu = x.detach().cpu().to(torch.bfloat16)
    for expert in active_experts:
        count = int(counts[expert].item())
        if count > int(x_cpu.shape[0]):
            raise ValueError(f"expert {expert} requested {count} rows, but input has {x_cpu.shape[0]}.")
        rows.append(x_cpu[:count])
    if not rows:
        return torch.empty((0, x_cpu.shape[1]), dtype=torch.bfloat16)
    return torch.cat(rows, dim=0).contiguous()


def _expand_counts_for_svdq(counts: torch.Tensor, num_experts: int) -> torch.Tensor:
    expanded = torch.zeros(num_experts, dtype=torch.int32)
    limit = min(num_experts, int(counts.numel()))
    expanded[:limit] = counts[:limit].to(torch.int32)
    return expanded


def _external_expert_token_nums_for_gmm2(
    counts: torch.Tensor,
    *,
    local_num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    external = torch.zeros((1, local_num_experts), dtype=torch.int32)
    limit = min(local_num_experts, int(counts.numel()))
    external[0, :limit] = counts[:limit].to(torch.int32)
    return external.to(device=device)


def _run_official_gmm2_from_mixed_hidden(
    *,
    args: argparse.Namespace,
    group: str,
    residual_layer: torch.nn.Module,
    spec: Any,
    taps: dict[str, torch.Tensor],
    hidden_bf16: torch.Tensor,
    hidden_q: torch.Tensor,
    hidden_q_packed: torch.Tensor,
    hidden_scale: torch.Tensor,
    counts: torch.Tensor,
    local_num_experts: int,
    device: torch.device,
) -> dict[str, Any]:
    active_rows = int(args.num_tokens * args.top_k)
    hidden_x_int4_packed, hidden_x_scale = _pad_hidden_boundary(
        hidden_x_int4_packed=hidden_q_packed[:active_rows],
        hidden_x_scale=hidden_scale[:active_rows],
        max_output_size=args.max_output_size,
        intermediate_size=spec.intermediate_size,
        device=device,
    )
    external_expert_token_nums = _external_expert_token_nums_for_gmm2(
        counts,
        local_num_experts=local_num_experts,
        device=device,
    )
    x = taps["x"].to(device=device, dtype=torch.bfloat16).contiguous()
    expert_idx = _make_expert_idx(_routed_experts(args), args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)

    op = getattr(torch.ops._C_ascend, "svdq_w4a8_gmm2_debug_readback", None)
    if op is None:
        raise RuntimeError("torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered.")

    gmm2_post_dequant, hidden_x_readback, hidden_scale_readback, gmm2_accumulator_int32 = _unpack_gmm2_debug_outputs(
        op(
            x,
            [residual_layer.w13_weight],
            [residual_layer.w2_weight],
            expert_idx,
            [residual_layer.w13_weight_scale],
            [residual_layer.w2_weight_scale],
            [residual_layer.w13_scale_bias],
            [residual_layer.w2_scale_bias],
            probs,
            hidden_x_int4_packed,
            hidden_x_scale,
            external_expert_token_nums,
            group,
            args.max_output_size,
            x_active_mask,
            float(args.swiglu_limit),
        )
    )
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
    hidden_scale_readback_exact = (
        hidden_scale_readback_error is not None
        and bool(hidden_scale_readback_error["actual_finite"])
        and bool(hidden_scale_readback_error["expected_finite"])
        and bool(hidden_scale_readback_error["diff_finite"])
        and float(hidden_scale_readback_error["max_abs"]) == 0.0
    )
    reference, reference_contract = _official_gmm2_unfused_reference(
        hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
        hidden_x_scale=hidden_x_scale[:active_rows],
        weight=residual_layer.w2_weight,
        weight_scale=residual_layer.w2_weight_scale,
        scale_bias=residual_layer.w2_scale_bias,
        expert_token_nums=external_expert_token_nums,
        output_columns=spec.hidden_size,
        max_rows=args.w4a8_reference_max_rows,
    )
    actual = gmm2_post_dequant.detach().cpu()[: reference.shape[0], : reference.shape[1]]
    error = _tensor_error(actual, reference)
    reference_passed = _stage_passed(
        error,
        max_abs_tol=args.gmm2_reference_max_abs_tol,
        mean_abs_tol=args.gmm2_reference_mean_abs_tol,
    )
    post_dequant_active = gmm2_post_dequant.detach().cpu()[:active_rows]
    post_dequant_nonzero = bool(torch.any(post_dequant_active.float().abs() > 0).item())
    routing_identity = _routing_identity_manifest(
        routed_experts=_routed_experts(args),
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
        mixed={
            "hidden_bf16": hidden_bf16[:active_rows],
            "hidden_q": hidden_q[:active_rows],
        },
        hidden_x_int4_packed=hidden_x_int4_packed,
        hidden_x_scale=hidden_x_scale,
        external_expert_token_nums=external_expert_token_nums,
        layer=residual_layer,
        routing_identity=routing_identity,
        active_rows=active_rows,
        max_output_size=args.max_output_size,
    )
    passed = bool(
        routing_identity["expert_token_total_matches_active_rows"]
        and routing_identity["reference_group_counts_match_expert_token_nums"]
        and hidden_x_readback_exact is not None
        and hidden_x_readback_exact["exact_match"]
        and hidden_scale_readback_exact
        and bool(error["actual_finite"])
        and post_dequant_nonzero
        and reference_passed
    )
    return {
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback",
        "official_lifecycle_debug_contract": _official_lifecycle_debug_contract(),
        "routing_identity": routing_identity,
        "gate_a_input_boundary": gate_a_input_boundary,
        "hidden_post_override_readback_exact": hidden_x_readback_exact,
        "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
        "hidden_scale_post_override_exact_match": bool(hidden_scale_readback_exact),
        "post_dequant_active": _tensor_stats(post_dequant_active),
        "unfused_reference": {
            "enabled": True,
            "passed": bool(reference_passed),
            "contract": reference_contract,
            "error": error,
            "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
        },
        "checks": {
            "official_gmm2_entry_reached": True,
            "gate_a_input_boundary_passed": bool(
                routing_identity["expert_token_total_matches_active_rows"]
                and routing_identity["reference_group_counts_match_expert_token_nums"]
                and hidden_x_readback_exact is not None
                and hidden_x_readback_exact["exact_match"]
                and hidden_scale_readback_exact
            ),
            "official_gmm2_post_dequant_finite": bool(error["actual_finite"]),
            "official_gmm2_post_dequant_nonzero": post_dequant_nonzero,
            "official_gmm2_post_dequant_reference_passed": bool(reference_passed),
            "official_gmm2_numerical_gate_passed": passed,
        },
        "post_dequant": post_dequant_active,
        "passed": passed,
    }


def _stage2_3_same_routing_manifest(
    *,
    routed_x: torch.Tensor,
    residual_gate_up: torch.Tensor,
    gate_up_lowrank: torch.Tensor,
    first_mixed: dict[str, torch.Tensor],
    official_gmm2: dict[str, Any],
    down_lowrank: torch.Tensor,
    final_mixed: dict[str, torch.Tensor],
    final_combine: dict[str, Any],
    expert_token_nums: torch.Tensor,
    routed_experts: list[int],
    num_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    active_rows = int(num_tokens * top_k)
    counts = [int(v) for v in expert_token_nums.detach().cpu().to(torch.int64).flatten().tolist()]
    prefixes = []
    running = 0
    for count in counts:
        prefixes.append(running)
        running += count
    active_experts = [idx for idx, count in enumerate(counts) if count > 0]
    route_slot_by_expert = {int(expert): slot for slot, expert in enumerate(routed_experts)}
    routed_x_cpu = routed_x.detach().cpu()
    same_source_token_payload = True
    for source_token_id in range(int(num_tokens)):
        reference_row = None
        for expert_id in routed_experts:
            expert_id = int(expert_id)
            if expert_id >= len(prefixes) or source_token_id >= counts[expert_id]:
                same_source_token_payload = False
                continue
            routed_row = prefixes[expert_id] + source_token_id
            row = routed_x_cpu[routed_row]
            if reference_row is None:
                reference_row = row
            else:
                same_source_token_payload = same_source_token_payload and bool(torch.equal(row, reference_row))
    row_map = []
    for expert_id, count in enumerate(counts):
        for expert_local_offset in range(count):
            if len(row_map) >= 64:
                break
            topk_slot = route_slot_by_expert.get(expert_id)
            row_map.append(
                {
                    "routed_row": prefixes[expert_id] + expert_local_offset,
                    "source_token_id": expert_local_offset,
                    "top_k_slot": topk_slot,
                    "selected_expert_id": expert_id,
                    "local_expert_id": expert_id,
                    "expert_row_start": prefixes[expert_id],
                    "expert_local_row_offset": expert_local_offset,
                }
            )
    boundary_hashes = {
        "routed_x_all": _tensor_sha256(routed_x[:active_rows]),
        "routed_x_row_sha256_first16": _row_sha256_first(routed_x[:active_rows]),
        "w4a8_gmm1_residual_gate_up_all": _tensor_sha256(residual_gate_up[:active_rows]),
        "svdq_gate_up_lowrank_all": _tensor_sha256(gate_up_lowrank[:active_rows]),
        "canonical_hidden_all": _tensor_sha256(first_mixed["hidden_bf16"][:active_rows]),
        "canonical_hidden_row_sha256_first16": _row_sha256_first(first_mixed["hidden_bf16"][:active_rows]),
        "svdq_down_hidden_input": _tensor_sha256(first_mixed["hidden_bf16"][:active_rows]),
        "w4a8_gmm2_hidden_quant_source_hidden": _tensor_sha256(first_mixed["hidden_bf16"][:active_rows]),
        "hidden_q_all": _tensor_sha256(first_mixed["hidden_q"][:active_rows]),
        "hidden_q_packed_all": _tensor_sha256(first_mixed["hidden_q_packed"][:active_rows]),
        "hidden_scale_all": _tensor_sha256(first_mixed["hidden_scale"][:active_rows]),
        "official_gmm2_residual_down_all": _tensor_sha256(official_gmm2["post_dequant"][:active_rows]),
        "svdq_down_lowrank_all": _tensor_sha256(down_lowrank[:active_rows]),
        "final_mixed_down_all": _tensor_sha256(final_mixed["down_mixed"][:active_rows]),
        "final_out_bf16_all": _tensor_sha256(final_mixed["out_bf16"][:active_rows]),
        "final_combine_input": _tensor_sha256(final_mixed["down_mixed"][:active_rows]),
        "final_combine_expanded_row_idx": _tensor_sha256(final_combine["expanded_row_idx"]),
        "final_combine_topk_weights": _tensor_sha256(final_combine["topk_weights"]),
        "final_combine_output_all": _tensor_sha256(final_combine["output"]),
    }
    checks = {
        "expert_token_total_matches_active_rows": sum(counts) == active_rows,
        "same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant": (
            boundary_hashes["svdq_down_hidden_input"]
            == boundary_hashes["w4a8_gmm2_hidden_quant_source_hidden"]
        ),
        "official_gmm2_output_feeds_final_mixed_residual_down": (
            boundary_hashes["official_gmm2_residual_down_all"]
            == _tensor_sha256(official_gmm2["post_dequant"][:active_rows])
        ),
        "final_mixed_output_is_final_combine_input": (
            boundary_hashes["final_mixed_down_all"] == _tensor_sha256(final_mixed["down_mixed"][:active_rows])
        ),
        "final_combine_consumes_mixed_down_peer_output": (
            boundary_hashes["final_mixed_down_all"] == boundary_hashes["final_combine_input"]
        ),
        "final_combine_output_validated": bool(final_combine["passed"]),
        "same_source_token_payload_across_topk_slots_proven": bool(same_source_token_payload),
    }
    return {
        "manifest_scope": (
            "Stage 2.3 real-checkpoint same-routing composition evidence for a single-device route. "
            "Official W4A8 GMM1 residual rows and real SVDQ gate/up rows produce one canonical hidden; "
            "the same canonical hidden feeds real SVDQ down and the official W4A8 GMM2 hidden-quant boundary; "
            "official GMM2 residual down plus real SVDQ down then feed the mixed output epilogue."
        ),
        "limitations": (
            "This is a top-1 single-device real-checkpoint probe unless run with broader routing. It does not "
            "enable production DispatchFFNCombineW4A8SVDQ host tiling."
        ),
        "topology": {
            "tp_size": 1,
            "tp_rank": 0,
            "ep_size": 1,
            "ep_rank": 0,
            "num_tokens": int(num_tokens),
            "top_k": int(top_k),
            "active_rows": active_rows,
            "routed_experts": [int(v) for v in routed_experts],
            "active_experts": active_experts,
        },
        "expert_token_nums": [counts],
        "expert_prefix_sums": prefixes,
        "routed_row_map_first64": row_map,
        "boundary_hashes": boundary_hashes,
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
    }


def _final_combine_inputs(
    *,
    counts: torch.Tensor,
    routed_experts: list[int],
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts_list = [int(v) for v in counts.detach().cpu().to(torch.int64).flatten().tolist()]
    if len(routed_experts) != int(top_k):
        raise ValueError(f"final-combine routed expert count {len(routed_experts)} != top_k {top_k}.")
    prefixes: list[int] = []
    running = 0
    for count in counts_list:
        prefixes.append(running)
        running += count
    active_rows = int(num_tokens) * int(top_k)
    if running != active_rows:
        raise ValueError(f"final-combine active row count {running} != expected {active_rows}.")
    for expert_id, count in enumerate(counts_list):
        expected = int(num_tokens) if expert_id in set(int(expert) for expert in routed_experts) else 0
        if count != expected:
            raise ValueError(f"final-combine expert {expert_id} count {count} != expected {expected}.")
    sorted_indices = torch.empty((active_rows,), dtype=torch.int32)
    cursor = 0
    for source_token in range(int(num_tokens)):
        for topk_slot, expert_id in enumerate(routed_experts):
            sorted_indices[cursor] = int(prefixes[int(expert_id)] + source_token)
            cursor += 1
    topk_weights = torch.full((int(num_tokens), int(top_k)), 1.0 / float(top_k), dtype=torch.float32)
    return sorted_indices.contiguous(), topk_weights.contiguous()


def _run_real_final_combine(
    *,
    peer_output: torch.Tensor,
    counts: torch.Tensor,
    routed_experts: list[int],
    num_tokens: int,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    expanded_row_idx, topk_weights = _final_combine_inputs(
        counts=counts,
        routed_experts=routed_experts,
        num_tokens=num_tokens,
        top_k=top_k,
    )
    reference = build_svdq_final_combine_reference(
        routed_output=peer_output.to(torch.bfloat16).contiguous(),
        topk_weights=topk_weights,
        expanded_row_idx=expanded_row_idx,
    )
    expected = reference["stages"]["combined_output"].to(torch.bfloat16)
    actual = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=peer_output.to(device=device, dtype=torch.bfloat16).contiguous(),
        sorted_indices=expanded_row_idx.to(device=device),
        probs=topk_weights.to(device=device),
    )
    torch.npu.synchronize()
    actual_cpu = actual.detach().cpu()
    error = _tensor_error(actual_cpu, expected)
    passed = (
        actual_cpu.shape == expected.shape
        and bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) == 0.0
        and float(error["mean_abs"]) == 0.0
    )
    return {
        "stage": "real_checkpoint_final_combine_token_unpermute",
        "official_surface": "torch_npu.npu_moe_token_unpermute",
        "reference": "build_svdq_final_combine_reference",
        "index_semantics": "official_token_major_output_slots_to_permuted_input_rows",
        "input_shape": list(peer_output.shape),
        "input_dtype": str(peer_output.dtype),
        "expanded_row_idx": expanded_row_idx,
        "topk_weights": topk_weights,
        "expanded_row_idx_shape": list(expanded_row_idx.shape),
        "topk_weights_shape": list(topk_weights.shape),
        "topk_weights_uniform": True,
        "oracle_stage_shapes": reference["stage_shapes"],
        "output": actual_cpu,
        "expected": expected,
        "output_error": error,
        "max_abs_tolerance": 0.0,
        "passed": passed,
    }


def _run_w4a8_debug(
    *,
    args: argparse.Namespace,
    group: str,
    residual_layer: torch.nn.Module,
    spec: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = _make_input(args, spec.hidden_size, device=device)
    expert_idx = _make_expert_idx(_routed_experts(args), args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)
    op = torch.ops._C_ascend.svdq_w4a8_debug_readback
    (
        out,
        expert_token_nums,
        routed_x_int8,
        routed_x_scale,
        gmm1_post_dequant,
        gmm1_hidden_prequant,
        hidden_x_int4_packed,
        hidden_x_scale,
        gmm2_post_dequant,
    ) = op(
        x,
        [residual_layer.w13_weight],
        [residual_layer.w2_weight],
        expert_idx,
        [residual_layer.w13_weight_scale],
        [residual_layer.w2_weight_scale],
        [residual_layer.w13_scale_bias],
        [residual_layer.w2_scale_bias],
        probs,
        group,
        args.max_output_size,
        x_active_mask,
    )
    torch.npu.synchronize()
    return {
        "x": x.detach().cpu(),
        "out": out.detach().cpu(),
        "expert_token_nums": expert_token_nums.detach().cpu(),
        "routed_x_int8": routed_x_int8.detach().cpu(),
        "routed_x_scale": routed_x_scale.detach().cpu(),
        "gmm1_post_dequant": gmm1_post_dequant.detach().cpu(),
        "gmm1_hidden_prequant": gmm1_hidden_prequant.detach().cpu(),
        "hidden_x_int4_packed": hidden_x_int4_packed.detach().cpu(),
        "hidden_x_scale": hidden_x_scale.detach().cpu(),
        "gmm2_post_dequant": gmm2_post_dequant.detach().cpu(),
    }


def _w4a8_reference_checks(
    *,
    args: argparse.Namespace,
    residual_layer: torch.nn.Module,
    spec: Any,
    taps: dict[str, torch.Tensor],
    active_rows: int,
) -> dict[str, Any]:
    reference1, contract1 = _official_gmm1_unfused_reference(
        routed_x_int8=taps["routed_x_int8"][:active_rows],
        routed_x_scale=taps["routed_x_scale"][:active_rows],
        weight=residual_layer.w13_weight,
        weight_scale=residual_layer.w13_weight_scale,
        scale_bias=residual_layer.w13_scale_bias,
        expert_token_nums=taps["expert_token_nums"],
        output_columns=2 * spec.intermediate_size,
        max_rows=args.w4a8_reference_max_rows,
    )
    error1 = _tensor_error(taps["gmm1_post_dequant"][: reference1.shape[0], : reference1.shape[1]], reference1)
    reference2, contract2 = _official_gmm2_unfused_reference(
        hidden_x_int4_packed=taps["hidden_x_int4_packed"][:active_rows],
        hidden_x_scale=taps["hidden_x_scale"][:active_rows],
        weight=residual_layer.w2_weight,
        weight_scale=residual_layer.w2_weight_scale,
        scale_bias=residual_layer.w2_scale_bias,
        expert_token_nums=taps["expert_token_nums"],
        output_columns=spec.hidden_size,
        max_rows=args.w4a8_reference_max_rows,
    )
    error2 = _tensor_error(taps["gmm2_post_dequant"][: reference2.shape[0], : reference2.shape[1]], reference2)
    gmm1_passed = _stage_passed(
        error1,
        max_abs_tol=args.gmm1_reference_max_abs_tol,
        mean_abs_tol=args.gmm1_reference_mean_abs_tol,
    )
    gmm2_passed = _stage_passed(
        error2,
        max_abs_tol=args.gmm2_reference_max_abs_tol,
        mean_abs_tol=args.gmm2_reference_mean_abs_tol,
    )
    return {
        "gmm1": {
            "passed": gmm1_passed,
            "contract": contract1,
            "error": error1,
            "max_abs_tolerance": args.gmm1_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm1_reference_mean_abs_tol,
        },
        "gmm2": {
            "passed": gmm2_passed,
            "contract": contract2,
            "error": error2,
            "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
        },
        "passed": bool(gmm1_passed and gmm2_passed),
    }


def _run_combined_probe(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    routed_experts = _routed_experts(args)
    local_num_experts = _local_num_experts(args, routed_experts)
    quant_description = _read_json(os.path.join(args.model_path, "quant_model_description.json"))
    weights = _weight_map(args.model_path)
    residual_layer, residual_spec, residual_key_count = _load_real_residual_layer(
        model_path=args.model_path,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
        routed_experts=set(routed_experts),
        local_num_experts=local_num_experts,
    )
    svdq_layer, svdq_spec, factor_load_count = _load_validation_layer(
        model_path=args.model_path,
        quant_description=quant_description,
        weight_map=weights,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
    )
    if residual_spec.hidden_size != svdq_spec.hidden_size or residual_spec.intermediate_size != svdq_spec.intermediate_size:
        raise ValueError("residual and SVDQ specs disagree on hidden/intermediate size.")

    taps = _run_w4a8_debug(
        args=args,
        group=group,
        residual_layer=residual_layer,
        spec=residual_spec,
        device=device,
    )
    active_rows = args.num_tokens * args.top_k
    active_experts, counts = _active_expert_counts(taps["expert_token_nums"])
    routed_x_grouped = _group_routed_x_by_expert(taps["x"], counts, active_experts)
    expert_token_nums_svdq = _expand_counts_for_svdq(counts, int(svdq_spec.num_experts))
    w4a8_reference = _w4a8_reference_checks(
        args=args,
        residual_layer=residual_layer,
        spec=residual_spec,
        taps=taps,
        active_rows=active_rows,
    )
    base_result: dict[str, Any] = {
        "stage": "stage2_3_real_checkpoint_same_route_w4a8_svdq_composition",
        "official_w4a8_debug_op": "torch.ops._C_ascend.svdq_w4a8_debug_readback",
        "official_w4a8_gmm2_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback",
        "svdq_lowrank_debug_op": "torch.ops._C_ascend.svdq_low_rank_debug_readback",
        "svdq_mixed_epilogue_debug_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "public_grouped_matmul_used": False,
        "production_svdq_host_tiling_fail_closed": True,
        "superseded_limitation": (
            "GMM2 residual tap comes from the official W4A8 debug path for its internally quantized hidden. "
            "Earlier versions of this probe validated mixed residual-plus-SVDQ epilogues with official W4A8 taps "
            "but did not yet relaunch official GMM2 from the SVDQ-modified hidden activation."
        ),
        "stage2_3_scope": (
            "This probe now relaunches the official W4A8 GMM2 path from the SVDQ-modified hidden produced by "
            "the mixed epilogue, then adds that official residual down output to the actual SVDQ down output."
        ),
        "layer_index": args.layer,
        "routed_experts": routed_experts,
        "active_experts_from_w4a8": active_experts,
        "residual_checkpoint_key_count": residual_key_count,
        "factor_load_count": factor_load_count,
        "shape": {
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "active_rows": active_rows,
            "hidden_size": int(svdq_spec.hidden_size),
            "intermediate_size": int(svdq_spec.intermediate_size),
        },
        "rank_metadata": {
            "gate_rank": int(svdq_layer.svdq_gate_rank),
            "up_rank": int(svdq_layer.svdq_up_rank),
            "down_rank": int(svdq_layer.svdq_down_rank),
            "gate_rank_offset": int(svdq_layer.svdq_gate_rank_offset),
            "up_rank_offset": int(svdq_layer.svdq_up_rank_offset),
        },
        "w4a8_reference": w4a8_reference,
    }
    hidden_placeholder = torch.zeros((active_rows, svdq_spec.intermediate_size), dtype=torch.bfloat16)
    lowrank_gate = _launch_lowrank_debug(
        layer=svdq_layer,
        routed_x=routed_x_grouped,
        hidden=hidden_placeholder,
        expert_token_nums=expert_token_nums_svdq,
        device=device,
    )
    gate_up_output = lowrank_gate["gate_up_output"][:active_rows]
    gate_up_accumulator = lowrank_gate["gate_up_accumulator"][:active_rows]
    gate_lowrank = gate_up_output[:, : svdq_spec.intermediate_size]
    up_lowrank = gate_up_output[:, svdq_spec.intermediate_size :]
    lowrank_stats = {
        "source_for_mixed_epilogue": "bf16_lowrank_output_readback",
        "bf16_output_used_for_mixed_epilogue": True,
        "gate_up_output": _tensor_stats(gate_up_output),
        "gate_up_accumulator": _tensor_stats(gate_up_accumulator),
        "down_output": None,
        "down_accumulator": None,
    }
    if not _nonzero_finite(lowrank_stats["gate_up_output"]):
        return {
            **base_result,
            "lowrank_readback": lowrank_stats,
            "lowrank_output_health_passed": False,
            "mixed_epilogue_evaluated": False,
            "mixed_epilogue_skip_reason": (
                "BF16 gate/up low-rank readback is not finite and nonzero; mixed AIV epilogue not evaluated."
            ),
            "stage_errors": {},
            "stage_passed": _not_evaluated_stage_passed(),
            "hidden_q_exact_match": False,
            "hidden_q_mismatch_count": None,
            "hidden_q_max_abs_diff": None,
            "hidden_q_mismatch_count_tolerance": args.hidden_q_mismatch_count_tol,
            "hidden_q_max_abs_diff_tolerance": args.hidden_q_max_abs_diff_tol,
            "passed": False,
        }
    zero_residual_down = torch.zeros((active_rows, svdq_spec.hidden_size), dtype=torch.float32)
    zero_down_lowrank = torch.zeros((active_rows, svdq_spec.hidden_size), dtype=torch.bfloat16)
    first_actual = _run_npu_mixed_epilogue(
        inputs={
            "residual_gate_up": taps["gmm1_post_dequant"][:active_rows].float(),
            "gate_lowrank": gate_lowrank.to(torch.bfloat16),
            "up_lowrank": up_lowrank.to(torch.bfloat16),
            "residual_down": zero_residual_down,
            "down_lowrank": zero_down_lowrank,
        },
        device=device,
        swiglu_limit=args.swiglu_limit,
    )
    first_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=taps["gmm1_post_dequant"][:active_rows].float(),
        gate_lowrank=gate_lowrank.to(torch.bfloat16),
        up_lowrank=up_lowrank.to(torch.bfloat16),
        residual_down=zero_residual_down,
        down_lowrank=zero_down_lowrank,
        swiglu_limit=args.swiglu_limit,
    )
    first_stage_errors = {
        "gate_mixed": _tensor_error(first_actual["gate_mixed"], first_reference["stages"]["gate_mixed"]),
        "up_mixed": _tensor_error(first_actual["up_mixed"], first_reference["stages"]["up_mixed"]),
        "hidden_bf16": _tensor_error(first_actual["hidden_bf16"], first_reference["stages"]["hidden_bf16"]),
        "hidden_scale": _tensor_error(first_actual["hidden_scale"], first_reference["stages"]["hidden_scale"]),
    }
    first_q_diff = (
        first_actual["hidden_q"].to(torch.int16) - first_reference["stages"]["hidden_q"].to(torch.int16)
    ).abs()
    first_hidden_q_mismatch_count = int((first_q_diff != 0).sum().item())
    first_hidden_q_max_abs_diff = int(first_q_diff.max().item()) if first_q_diff.numel() else 0
    first_hidden_q_passed = (
        first_hidden_q_mismatch_count <= args.hidden_q_mismatch_count_tol
        and first_hidden_q_max_abs_diff <= args.hidden_q_max_abs_diff_tol
    )
    expected_packed = pack_official_hidden_i4_reference(first_actual["hidden_q"])
    first_hidden_q_packed_exact = _tensor_int_exact(first_actual["hidden_q_packed"], expected_packed)
    first_stage_passed = {
        "gate_mixed": _stage_passed(
            first_stage_errors["gate_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "up_mixed": _stage_passed(
            first_stage_errors["up_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_bf16": _stage_passed(
            first_stage_errors["hidden_bf16"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_scale": (
            bool(first_stage_errors["hidden_scale"]["actual_finite"])
            and bool(first_stage_errors["hidden_scale"]["expected_finite"])
            and bool(first_stage_errors["hidden_scale"]["diff_finite"])
            and float(first_stage_errors["hidden_scale"]["max_abs"]) <= args.scale_tol
        ),
        "hidden_q": first_hidden_q_passed,
        "hidden_q_packed": bool(first_hidden_q_packed_exact["exact_match"]),
    }
    hidden_bf16 = first_actual["hidden_bf16"]
    official_gmm2 = _run_official_gmm2_from_mixed_hidden(
        args=args,
        group=group,
        residual_layer=residual_layer,
        spec=residual_spec,
        taps=taps,
        hidden_bf16=first_actual["hidden_bf16"],
        hidden_q=first_actual["hidden_q"],
        hidden_q_packed=first_actual["hidden_q_packed"],
        hidden_scale=first_actual["hidden_scale"],
        counts=counts,
        local_num_experts=local_num_experts,
        device=device,
    )
    lowrank_down = _launch_lowrank_debug(
        layer=svdq_layer,
        routed_x=routed_x_grouped,
        hidden=hidden_bf16,
        expert_token_nums=expert_token_nums_svdq,
        device=device,
    )
    down_output = lowrank_down["down_output"][:active_rows]
    down_accumulator = lowrank_down["down_accumulator"][:active_rows]
    down_lowrank = down_output
    lowrank_stats["down_output"] = _tensor_stats(down_output)
    lowrank_stats["down_accumulator"] = _tensor_stats(down_accumulator)
    lowrank_output_health_passed = bool(
        _nonzero_finite(lowrank_stats["gate_up_output"]) and _nonzero_finite(lowrank_stats["down_output"])
    )
    if not lowrank_output_health_passed:
        return {
            **base_result,
            "lowrank_readback": lowrank_stats,
            "lowrank_output_health_passed": lowrank_output_health_passed,
            "mixed_epilogue_evaluated": False,
            "mixed_epilogue_skip_reason": (
                "BF16 down low-rank readback is not finite and nonzero; mixed AIV epilogue not evaluated."
            ),
            "stage_errors": {},
            "stage_passed": _not_evaluated_stage_passed(),
            "hidden_q_exact_match": False,
            "hidden_q_mismatch_count": None,
            "hidden_q_max_abs_diff": None,
            "hidden_q_mismatch_count_tolerance": args.hidden_q_mismatch_count_tol,
            "hidden_q_max_abs_diff_tolerance": args.hidden_q_max_abs_diff_tol,
            "passed": False,
        }
    mixed_actual = _run_npu_mixed_epilogue(
        inputs={
            "residual_gate_up": taps["gmm1_post_dequant"][:active_rows].float(),
            "gate_lowrank": gate_lowrank.to(torch.bfloat16),
            "up_lowrank": up_lowrank.to(torch.bfloat16),
            "residual_down": official_gmm2["post_dequant"][:active_rows].float(),
            "down_lowrank": down_lowrank.to(torch.bfloat16),
        },
        device=device,
        swiglu_limit=args.swiglu_limit,
    )
    mixed_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=taps["gmm1_post_dequant"][:active_rows].float(),
        gate_lowrank=gate_lowrank.to(torch.bfloat16),
        up_lowrank=up_lowrank.to(torch.bfloat16),
        residual_down=official_gmm2["post_dequant"][:active_rows].float(),
        down_lowrank=down_lowrank.to(torch.bfloat16),
        swiglu_limit=args.swiglu_limit,
    )
    final_combine = _run_real_final_combine(
        peer_output=mixed_actual["down_mixed"][:active_rows].to(torch.bfloat16),
        counts=counts,
        routed_experts=routed_experts,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
        device=device,
    )
    stage_errors = {
        "gate_mixed": _tensor_error(mixed_actual["gate_mixed"], mixed_reference["stages"]["gate_mixed"]),
        "up_mixed": _tensor_error(mixed_actual["up_mixed"], mixed_reference["stages"]["up_mixed"]),
        "hidden_bf16": _tensor_error(mixed_actual["hidden_bf16"], mixed_reference["stages"]["hidden_bf16"]),
        "hidden_scale": _tensor_error(mixed_actual["hidden_scale"], mixed_reference["stages"]["hidden_scale"]),
        "down_mixed": _tensor_error(mixed_actual["down_mixed"], mixed_reference["stages"]["down_mixed"]),
        "out_bf16": _tensor_error(mixed_actual["out_bf16"], mixed_reference["stages"]["down_mixed"].to(torch.bfloat16)),
        "final_combine_output": final_combine["output_error"],
    }
    q_diff = (mixed_actual["hidden_q"].to(torch.int16) - mixed_reference["stages"]["hidden_q"].to(torch.int16)).abs()
    hidden_q_mismatch_count = int((q_diff != 0).sum().item())
    hidden_q_max_abs_diff = int(q_diff.max().item()) if q_diff.numel() else 0
    hidden_q_passed = (
        hidden_q_mismatch_count <= args.hidden_q_mismatch_count_tol
        and hidden_q_max_abs_diff <= args.hidden_q_max_abs_diff_tol
    )
    stage_passed = {
        "first_mixed_epilogue": all(first_stage_passed.values()),
        "official_gmm2_from_svdq_hidden": bool(official_gmm2["passed"]),
        "gate_mixed": _stage_passed(
            stage_errors["gate_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "up_mixed": _stage_passed(
            stage_errors["up_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_bf16": _stage_passed(
            stage_errors["hidden_bf16"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_scale": (
            bool(stage_errors["hidden_scale"]["actual_finite"])
            and bool(stage_errors["hidden_scale"]["expected_finite"])
            and bool(stage_errors["hidden_scale"]["diff_finite"])
            and float(stage_errors["hidden_scale"]["max_abs"]) <= args.scale_tol
        ),
        "hidden_q": hidden_q_passed,
        "down_mixed": _stage_passed(
            stage_errors["down_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "out_bf16": _stage_passed(
            stage_errors["out_bf16"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "final_combine_output": bool(final_combine["passed"]),
    }
    same_routing_manifest = _stage2_3_same_routing_manifest(
        routed_x=routed_x_grouped,
        residual_gate_up=taps["gmm1_post_dequant"][:active_rows],
        gate_up_lowrank=gate_up_output,
        first_mixed=first_actual,
        official_gmm2=official_gmm2,
        down_lowrank=down_lowrank,
        final_mixed=mixed_actual,
        final_combine=final_combine,
        expert_token_nums=taps["expert_token_nums"],
        routed_experts=routed_experts,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
    )
    stage_passed["same_routing_identity"] = bool(same_routing_manifest["passed"])
    return {
        **base_result,
        "lowrank_readback": lowrank_stats,
        "lowrank_output_health_passed": lowrank_output_health_passed,
        "first_mixed_epilogue": {
            "stage_errors": first_stage_errors,
            "stage_passed": first_stage_passed,
            "hidden_q_exact_match": bool(torch.equal(first_actual["hidden_q"], first_reference["stages"]["hidden_q"])),
            "hidden_q_mismatch_count": first_hidden_q_mismatch_count,
            "hidden_q_max_abs_diff": first_hidden_q_max_abs_diff,
            "hidden_q_packed_exact_reference": first_hidden_q_packed_exact,
        },
        "official_gmm2_from_svdq_hidden": {
            key: value for key, value in official_gmm2.items() if key != "post_dequant"
        },
        "stage2_3_same_routing_manifest": same_routing_manifest,
        "real_final_combine": {
            key: value
            for key, value in final_combine.items()
            if key not in {"output", "expected", "expanded_row_idx", "topk_weights"}
        },
        "mixed_epilogue_evaluated": True,
        "mixed_epilogue_skip_reason": None,
        "stage_errors": stage_errors,
        "stage_passed": stage_passed,
        "hidden_q_exact_match": bool(torch.equal(mixed_actual["hidden_q"], mixed_reference["stages"]["hidden_q"])),
        "hidden_q_mismatch_count": hidden_q_mismatch_count,
        "hidden_q_max_abs_diff": hidden_q_max_abs_diff,
        "hidden_q_mismatch_count_tolerance": args.hidden_q_mismatch_count_tol,
        "hidden_q_max_abs_diff_tolerance": args.hidden_q_max_abs_diff_tol,
        "passed": bool(
            w4a8_reference["gmm1"]["passed"]
            and lowrank_output_health_passed
            and all(stage_passed.values())
        ),
    }


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    symbol_status = _official_debug_symbol_status()
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_tap_mixed_epilogue_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "official_debug_symbol_status": symbol_status,
    }
    if not (env["torch_npu_imported"] and env["npu_available"] and int(env["npu_device_count"]) > args.device_id):
        summary["skipped"] = not args.require_npu
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch_npu import and an available selected NPU are required."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0
    if args.max_output_size < args.num_tokens * args.top_k:
        summary["failure_reason"] = "max-output-size must cover num_tokens * top_k."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1

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
    try:
        gmm2_registered = _has_registered_gmm2_debug_op()
    except Exception as exc:
        summary["preflight_failed"] = True
        summary["failure_reason"] = f"failed to enable GMM2 custom op: {type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1
    summary["gmm2_torch_op_registered"] = gmm2_registered
    if not gmm2_registered:
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1

    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl_group"] = group_info
        stage = _run_combined_probe(args, str(group_info["group"]))
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
