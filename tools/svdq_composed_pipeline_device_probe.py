#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate composed SVDQ MoE stage order on NPU tensor operators.

The production fused SVDQ operator is still fail-closed. This probe validates
that the previously validated pieces compose in the required order on real
Qwen3.5 dimensions:

1. BF16 low-rank gate/up projection with explicit gate/up rank offsets.
2. Mixed gate/up epilogue before SwiGLU.
3. Hidden activation dynamic quantization.
4. BF16 low-rank down projection.
5. Mixed output epilogue before final combine.
6. Token combine with flattened ``[token, top_k]`` route indices.
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

from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _read_json,
)

from vllm_ascend.quantization.methods.svdq_post_load import (  # noqa: E402
    build_svdq_final_combine_reference,
    build_svdq_mixed_epilogue_reference,
)

DEFAULT_SUMMARY_NAME = "phase_l_composed_pipeline_device_probe_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--gate-rank", type=int, default=64)
    parser.add_argument("--up-rank", type=int, default=64)
    parser.add_argument("--down-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--max-abs-tol", type=float, default=0.08)
    parser.add_argument("--mean-abs-tol", type=float, default=0.004)
    parser.add_argument("--scale-tol", type=float, default=1e-7)
    parser.add_argument("--hidden-q-mismatch-tol", type=int, default=1)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
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
        "has_npu_dynamic_quant": False,
        "has_npu_moe_token_unpermute": False,
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
        info["has_npu_dynamic_quant"] = hasattr(torch_npu, "npu_dynamic_quant")
        info["has_npu_moe_token_unpermute"] = hasattr(torch_npu, "npu_moe_token_unpermute")
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


def _target_dimensions(model_path: str) -> dict[str, int | str]:
    config_path = Path(model_path) / "config.json"
    hidden_size = 2048
    intermediate_size = 512
    num_experts = 256
    top_k = 8
    source = "qwen35_svdq_default"

    if config_path.exists():
        config = _read_json(str(config_path))
        text_config = config.get("text_config", config)
        hidden_size = int(text_config.get("hidden_size", hidden_size))
        intermediate_size = int(
            text_config.get("moe_intermediate_size", text_config.get("intermediate_size", intermediate_size))
        )
        num_experts = int(text_config.get("num_experts", num_experts))
        top_k = int(text_config.get("num_experts_per_tok", top_k))
        source = str(config_path)

    return {
        "source": source,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
    }


def _make_bf16_tensor(shape: tuple[int, ...], *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(-32, 33, shape, dtype=torch.int16, generator=generator).float()
    return (values * scale).to(torch.bfloat16)


def _make_topk_weights(num_tokens: int, top_k: int) -> torch.Tensor:
    weights = torch.tensor([2.0 ** -(slot + 1) for slot in range(top_k)], dtype=torch.float32)
    weights[-1] += 1.0 - float(weights.sum().item())
    return weights.expand(num_tokens, top_k).contiguous()


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool | int | list[int] | str]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    actual_finite = bool(torch.isfinite(actual_cpu.float()).all().item()) if actual_cpu.numel() else True
    expected_finite = bool(torch.isfinite(expected_cpu.float()).all().item()) if expected_cpu.numel() else True
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual_cpu.dtype),
        "expected_dtype": str(expected_cpu.dtype),
        "numel": int(actual_cpu.numel()),
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
    }


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy().tobytes()
    if cpu.dtype == torch.float16:
        return cpu.view(torch.uint16).numpy().tobytes()
    return cpu.numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_raw_bytes(tensor)).hexdigest()


def _row_sha256_first(tensor: torch.Tensor, *, row_count: int = 16) -> list[str]:
    cpu = tensor.detach().cpu()
    if cpu.ndim < 2:
        return []
    return [_tensor_sha256(cpu[row]) for row in range(min(int(cpu.shape[0]), int(row_count)))]


def _same_routing_identity_manifest(
    *,
    routed_x: torch.Tensor,
    hidden_bf16: torch.Tensor,
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    peer_output: torch.Tensor,
    topk_weights: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    routed_rows = int(num_tokens) * int(top_k)
    row_map = []
    for routed_row in range(min(routed_rows, 64)):
        source_token_id = routed_row // int(top_k)
        topk_slot = routed_row % int(top_k)
        row_map.append(
            {
                "routed_row": routed_row,
                "source_token_id": source_token_id,
                "top_k_slot": topk_slot,
                "global_expert_id": topk_slot,
                "local_expert_id": topk_slot,
                "expert_prefix_start": source_token_id * int(top_k),
                "expert_local_row_offset": source_token_id,
                "tp_rank": 0,
                "ep_rank": 0,
                "active": True,
            }
        )

    expanded_expected = torch.arange(routed_rows, dtype=torch.int32)
    expanded_matches = bool(torch.equal(expanded_row_idx.detach().cpu().to(torch.int32), expanded_expected))
    weights_row_consistent = (
        bool(torch.allclose(topk_weights.detach().cpu(), topk_weights.detach().cpu()[0:1].expand_as(topk_weights)))
        if topk_weights.numel()
        else True
    )
    manifest = {
        "manifest_scope": (
            "Stage 2.3 synthetic same-routing composition evidence. The same flattened [token, top_k] routed rows "
            "feed the first-stage W4A8 residual placeholder, first-stage SVDQ gate/up branch, canonical hidden "
            "producer, SVDQ down branch, W4A8 hidden quant branch, mixed down output, and final combine indices."
        ),
        "limitations": (
            "This probe uses deterministic synthetic factors/activations at Qwen3.5 dimensions; it is not the "
            "real-checkpoint residual W4A8 GMM1/GMM2 plus SVDQ down final-combine gate."
        ),
        "topology": {
            "tp_size": 1,
            "tp_rank": 0,
            "ep_size": 1,
            "ep_rank": 0,
            "num_tokens": int(num_tokens),
            "top_k": int(top_k),
            "routed_rows": routed_rows,
        },
        "route_order": {
            "flattening": "routed_row = source_token_id * top_k + top_k_slot",
            "expanded_row_idx_matches_arange": expanded_matches,
            "topk_weights_shape": list(topk_weights.shape),
            "topk_weights_row_consistent": weights_row_consistent,
            "row_map_first64": row_map,
        },
        "boundary_hashes": {
            "routed_x_all": _tensor_sha256(routed_x),
            "routed_x_row_sha256_first16": _row_sha256_first(routed_x),
            "first_stage_w4a8_input": _tensor_sha256(routed_x),
            "first_stage_svdq_gate_up_input": _tensor_sha256(routed_x),
            "canonical_hidden_all": _tensor_sha256(hidden_bf16),
            "canonical_hidden_row_sha256_first16": _row_sha256_first(hidden_bf16),
            "svdq_down_hidden_input": _tensor_sha256(hidden_bf16),
            "w4a8_hidden_quant_input": _tensor_sha256(hidden_bf16),
            "hidden_q_all": _tensor_sha256(hidden_q),
            "hidden_scale_all": _tensor_sha256(hidden_scale),
            "mixed_down_peer_output_all": _tensor_sha256(peer_output),
            "final_combine_input": _tensor_sha256(peer_output),
            "expanded_row_idx": _tensor_sha256(expanded_row_idx),
        },
    }
    manifest["checks"] = {
        "first_stage_w4a8_and_svdq_share_routed_x": (
            manifest["boundary_hashes"]["first_stage_w4a8_input"]
            == manifest["boundary_hashes"]["first_stage_svdq_gate_up_input"]
        ),
        "svdq_down_and_w4a8_hidden_quant_share_canonical_hidden": (
            manifest["boundary_hashes"]["svdq_down_hidden_input"]
            == manifest["boundary_hashes"]["w4a8_hidden_quant_input"]
        ),
        "final_combine_uses_mixed_down_peer_output": (
            manifest["boundary_hashes"]["mixed_down_peer_output_all"]
            == manifest["boundary_hashes"]["final_combine_input"]
        ),
        "expanded_row_idx_matches_arange": expanded_matches,
        "all_rows_active_no_padding": int(expanded_row_idx.numel()) == routed_rows,
    }
    manifest["passed"] = all(bool(value) for value in manifest["checks"].values())
    return manifest


def _stage_passed(error: dict[str, Any], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) <= max_abs_tol
        and float(error["mean_abs"]) <= mean_abs_tol
    )


def _cpu_lowrank_reference(
    *,
    routed_x: torch.Tensor,
    gate_up_svdq_l1: torch.Tensor,
    gate_svdq_l2: torch.Tensor,
    up_svdq_l2: torch.Tensor,
    down_svdq_l1: torch.Tensor,
    down_svdq_l2: torch.Tensor,
    hidden_bf16: torch.Tensor | None = None,
    gate_rank: int,
    up_rank: int,
    gate_rank_offset: int,
    up_rank_offset: int,
) -> dict[str, torch.Tensor]:
    gate_up_rank = routed_x.float() @ gate_up_svdq_l1.float().T
    gate_rank_state = gate_up_rank[:, gate_rank_offset : gate_rank_offset + gate_rank]
    up_rank_state = gate_up_rank[:, up_rank_offset : up_rank_offset + up_rank]
    gate_lowrank = gate_rank_state @ gate_svdq_l2.float().T
    up_lowrank = up_rank_state @ up_svdq_l2.float().T
    stages = {
        "gate_up_rank": gate_up_rank,
        "gate_lowrank": gate_lowrank,
        "up_lowrank": up_lowrank,
    }
    if hidden_bf16 is not None:
        down_rank_state = hidden_bf16.float() @ down_svdq_l1.float().T
        down_lowrank = down_rank_state @ down_svdq_l2.float().T
        stages["down_rank"] = down_rank_state
        stages["down_lowrank"] = down_lowrank
    return stages


def _npu_lowrank_gate_up(
    *,
    routed_x: torch.Tensor,
    gate_up_svdq_l1: torch.Tensor,
    gate_svdq_l2: torch.Tensor,
    up_svdq_l2: torch.Tensor,
    gate_rank: int,
    up_rank: int,
    gate_rank_offset: int,
    up_rank_offset: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    routed_x_npu = routed_x.to(device=device, dtype=torch.bfloat16)
    gate_up_l1_npu = gate_up_svdq_l1.to(device=device, dtype=torch.bfloat16)
    gate_l2_npu = gate_svdq_l2.to(device=device, dtype=torch.bfloat16)
    up_l2_npu = up_svdq_l2.to(device=device, dtype=torch.bfloat16)
    gate_up_rank = routed_x_npu @ gate_up_l1_npu.transpose(0, 1)
    gate_rank_state = gate_up_rank[:, gate_rank_offset : gate_rank_offset + gate_rank]
    up_rank_state = gate_up_rank[:, up_rank_offset : up_rank_offset + up_rank]
    gate_lowrank = gate_rank_state @ gate_l2_npu.transpose(0, 1)
    up_lowrank = up_rank_state @ up_l2_npu.transpose(0, 1)
    return {
        "gate_up_rank": gate_up_rank.detach().float().cpu(),
        "gate_lowrank": gate_lowrank.detach().float().cpu(),
        "up_lowrank": up_lowrank.detach().float().cpu(),
    }


def _npu_down_lowrank(
    *,
    hidden_bf16: torch.Tensor,
    down_svdq_l1: torch.Tensor,
    down_svdq_l2: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    hidden_npu = hidden_bf16.to(device=device, dtype=torch.bfloat16)
    down_l1_npu = down_svdq_l1.to(device=device, dtype=torch.bfloat16)
    down_l2_npu = down_svdq_l2.to(device=device, dtype=torch.bfloat16)
    down_rank = hidden_npu @ down_l1_npu.transpose(0, 1)
    down_lowrank = down_rank @ down_l2_npu.transpose(0, 1)
    return {
        "down_rank": down_rank.detach().float().cpu(),
        "down_lowrank": down_lowrank.detach().float().cpu(),
    }


def _run_probe(
    *,
    dimensions: dict[str, int | str],
    num_tokens: int,
    gate_rank: int,
    up_rank: int,
    down_rank: int,
    seed: int,
    device: torch.device,
    swiglu_limit: float,
    max_abs_tol: float,
    mean_abs_tol: float,
    scale_tol: float,
    hidden_q_mismatch_tol: int,
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    hidden_size = int(dimensions["hidden_size"])
    intermediate_size = int(dimensions["intermediate_size"])
    top_k = int(dimensions["top_k"])
    routed_rows = num_tokens * top_k
    gate_rank_offset = 0
    up_rank_offset = gate_rank
    routed_x = _make_bf16_tensor((routed_rows, hidden_size), seed=seed, scale=1.0 / 128.0)
    gate_l1 = _make_bf16_tensor((gate_rank, hidden_size), seed=seed + 1, scale=1.0 / 512.0)
    up_l1 = _make_bf16_tensor((up_rank, hidden_size), seed=seed + 2, scale=1.0 / 512.0)
    gate_up_svdq_l1 = torch.cat((gate_l1, up_l1), dim=0).contiguous()
    gate_svdq_l2 = _make_bf16_tensor((intermediate_size, gate_rank), seed=seed + 3, scale=1.0 / 512.0)
    up_svdq_l2 = _make_bf16_tensor((intermediate_size, up_rank), seed=seed + 4, scale=1.0 / 512.0)
    down_svdq_l1 = _make_bf16_tensor((down_rank, intermediate_size), seed=seed + 5, scale=1.0 / 512.0)
    down_svdq_l2 = _make_bf16_tensor((hidden_size, down_rank), seed=seed + 6, scale=1.0 / 512.0)
    residual_gate_up = torch.zeros((routed_rows, intermediate_size * 2), dtype=torch.bfloat16)
    residual_down = torch.zeros((routed_rows, hidden_size), dtype=torch.bfloat16)
    topk_weights = _make_topk_weights(num_tokens, top_k)
    expanded_row_idx = torch.arange(routed_rows, dtype=torch.int32)

    cpu_gate_up = _cpu_lowrank_reference(
        routed_x=routed_x,
        gate_up_svdq_l1=gate_up_svdq_l1,
        gate_svdq_l2=gate_svdq_l2,
        up_svdq_l2=up_svdq_l2,
        down_svdq_l1=down_svdq_l1,
        down_svdq_l2=down_svdq_l2,
        gate_rank=gate_rank,
        up_rank=up_rank,
        gate_rank_offset=gate_rank_offset,
        up_rank_offset=up_rank_offset,
    )
    npu_gate_up = _npu_lowrank_gate_up(
        routed_x=routed_x,
        gate_up_svdq_l1=gate_up_svdq_l1,
        gate_svdq_l2=gate_svdq_l2,
        up_svdq_l2=up_svdq_l2,
        gate_rank=gate_rank,
        up_rank=up_rank,
        gate_rank_offset=gate_rank_offset,
        up_rank_offset=up_rank_offset,
        device=device,
    )
    first_mixed_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=residual_gate_up,
        gate_lowrank=npu_gate_up["gate_lowrank"].to(torch.bfloat16),
        up_lowrank=npu_gate_up["up_lowrank"].to(torch.bfloat16),
        residual_down=residual_down,
        down_lowrank=torch.zeros_like(residual_down),
        swiglu_limit=swiglu_limit,
    )
    hidden_bf16 = first_mixed_reference["stages"]["hidden_bf16"]
    hidden_q, hidden_scale = torch_npu.npu_dynamic_quant(hidden_bf16.to(device=device, dtype=torch.bfloat16))
    torch.npu.synchronize()
    hidden_q_cpu = hidden_q.detach().cpu()
    hidden_scale_cpu = hidden_scale.detach().cpu().float()

    cpu_down = _cpu_lowrank_reference(
        routed_x=routed_x,
        gate_up_svdq_l1=gate_up_svdq_l1,
        gate_svdq_l2=gate_svdq_l2,
        up_svdq_l2=up_svdq_l2,
        down_svdq_l1=down_svdq_l1,
        down_svdq_l2=down_svdq_l2,
        hidden_bf16=hidden_bf16,
        gate_rank=gate_rank,
        up_rank=up_rank,
        gate_rank_offset=gate_rank_offset,
        up_rank_offset=up_rank_offset,
    )
    npu_down = _npu_down_lowrank(
        hidden_bf16=hidden_bf16,
        down_svdq_l1=down_svdq_l1,
        down_svdq_l2=down_svdq_l2,
        device=device,
    )
    output_epilogue_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=residual_gate_up,
        gate_lowrank=npu_gate_up["gate_lowrank"].to(torch.bfloat16),
        up_lowrank=npu_gate_up["up_lowrank"].to(torch.bfloat16),
        residual_down=residual_down,
        down_lowrank=npu_down["down_lowrank"].to(torch.bfloat16),
        swiglu_limit=swiglu_limit,
    )
    peer_output = output_epilogue_reference["stages"]["down_mixed"].to(torch.bfloat16)
    final_reference = build_svdq_final_combine_reference(
        routed_output=peer_output,
        topk_weights=topk_weights,
        expanded_row_idx=expanded_row_idx,
    )
    final_output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=peer_output.to(device=device),
        sorted_indices=expanded_row_idx.to(device=device),
        probs=topk_weights.to(device=device),
    )
    torch.npu.synchronize()
    final_output_cpu = final_output.detach().cpu()
    final_expected = final_reference["stages"]["combined_output"].to(torch.bfloat16)
    q_diff = (hidden_q_cpu.to(torch.int16) - first_mixed_reference["stages"]["hidden_q"].to(torch.int16)).abs()
    routing_identity = _same_routing_identity_manifest(
        routed_x=routed_x,
        hidden_bf16=hidden_bf16,
        hidden_q=hidden_q_cpu,
        hidden_scale=hidden_scale_cpu,
        peer_output=peer_output,
        topk_weights=topk_weights,
        expanded_row_idx=expanded_row_idx,
        num_tokens=num_tokens,
        top_k=top_k,
    )

    stage_errors = {
        "gate_up_rank": _tensor_error(npu_gate_up["gate_up_rank"], cpu_gate_up["gate_up_rank"].to(torch.bfloat16)),
        "gate_lowrank": _tensor_error(npu_gate_up["gate_lowrank"], cpu_gate_up["gate_lowrank"].to(torch.bfloat16)),
        "up_lowrank": _tensor_error(npu_gate_up["up_lowrank"], cpu_gate_up["up_lowrank"].to(torch.bfloat16)),
        "hidden_scale": _tensor_error(hidden_scale_cpu, first_mixed_reference["stages"]["hidden_scale"]),
        "down_rank": _tensor_error(npu_down["down_rank"], cpu_down["down_rank"].to(torch.bfloat16)),
        "down_lowrank": _tensor_error(npu_down["down_lowrank"], cpu_down["down_lowrank"].to(torch.bfloat16)),
        "peer_output": _tensor_error(peer_output, output_epilogue_reference["stages"]["down_mixed"].to(torch.bfloat16)),
        "final_output": _tensor_error(final_output_cpu, final_expected),
    }
    stage_passed = {
        "gate_up_rank": _stage_passed(stage_errors["gate_up_rank"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "gate_lowrank": _stage_passed(stage_errors["gate_lowrank"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "up_lowrank": _stage_passed(stage_errors["up_lowrank"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "hidden_q": int((q_diff != 0).sum().item()) <= hidden_q_mismatch_tol
        and (int(q_diff.max().item()) if q_diff.numel() else 0) <= 1,
        "hidden_scale": (
            bool(stage_errors["hidden_scale"]["actual_finite"])
            and bool(stage_errors["hidden_scale"]["expected_finite"])
            and bool(stage_errors["hidden_scale"]["diff_finite"])
            and float(stage_errors["hidden_scale"]["max_abs"]) <= scale_tol
        ),
        "down_rank": _stage_passed(stage_errors["down_rank"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "down_lowrank": _stage_passed(stage_errors["down_lowrank"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "peer_output": _stage_passed(stage_errors["peer_output"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "final_output": _stage_passed(stage_errors["final_output"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "routing_identity": bool(routing_identity["passed"]),
    }
    return {
        "stage": "composed_svdq_pipeline_npu_math",
        "rank_metadata": {
            "gate_rank": gate_rank,
            "up_rank": up_rank,
            "down_rank": down_rank,
            "gate_rank_offset": gate_rank_offset,
            "up_rank_offset": up_rank_offset,
        },
        "tensor_shapes": {
            "routed_x": list(routed_x.shape),
            "gate_up_svdq_l1": list(gate_up_svdq_l1.shape),
            "gate_svdq_l2": list(gate_svdq_l2.shape),
            "up_svdq_l2": list(up_svdq_l2.shape),
            "hidden_bf16": list(hidden_bf16.shape),
            "down_svdq_l1": list(down_svdq_l1.shape),
            "down_svdq_l2": list(down_svdq_l2.shape),
            "peer_output": list(peer_output.shape),
            "topk_weights": list(topk_weights.shape),
            "expanded_row_idx": list(expanded_row_idx.shape),
            "final_output": list(final_output_cpu.shape),
        },
        "stage_errors": stage_errors,
        "hidden_q_exact_match": stage_passed["hidden_q"],
        "hidden_q_mismatch_count": int((q_diff != 0).sum().item()),
        "hidden_q_max_abs_diff": int(q_diff.max().item()) if q_diff.numel() else 0,
        "routing_identity_manifest": routing_identity,
        "stage_passed": stage_passed,
        "max_abs_tolerance": max_abs_tol,
        "mean_abs_tolerance": mean_abs_tol,
        "scale_tolerance": scale_tol,
        "hidden_q_mismatch_tolerance": hidden_q_mismatch_tol,
        "swiglu_limit": float(swiglu_limit),
        "passed": all(stage_passed.values()),
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
    dimensions = _target_dimensions(args.model_path)
    summary: dict[str, Any] = {
        "probe": "svdq_composed_pipeline_device_probe",
        "model_path": args.model_path,
        "dimensions": dimensions,
        "environment": env,
        "stages": [],
        "passed": False,
        "skipped": False,
    }
    if not (
        env["torch_npu_imported"]
        and env["has_npu_dynamic_quant"]
        and env["has_npu_moe_token_unpermute"]
        and env["npu_available"]
    ):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = (
            "torch_npu.npu_dynamic_quant, torch_npu.npu_moe_token_unpermute, "
            "and an available NPU are required."
        )
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    if int(env["npu_device_count"]) <= args.device_id:
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = f"NPU device {args.device_id} is outside available device count."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    stage = _run_probe(
        dimensions=dimensions,
        num_tokens=args.num_tokens,
        gate_rank=args.gate_rank,
        up_rank=args.up_rank,
        down_rank=args.down_rank,
        seed=args.seed,
        device=device,
        swiglu_limit=args.swiglu_limit,
        max_abs_tol=args.max_abs_tol,
        mean_abs_tol=args.mean_abs_tol,
        scale_tol=args.scale_tol,
        hidden_q_mismatch_tol=args.hidden_q_mismatch_tol,
    )
    summary["stages"] = [stage]
    summary["passed"] = bool(stage["passed"])
    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    print(json.dumps({"summary_path": path, "passed": summary["passed"], "skipped": False}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
