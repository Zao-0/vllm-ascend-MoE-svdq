#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Launch official W4A8 debug readback with real SVDQ checkpoint weights.

This is an isolated Appendix-1 probe for the official
``dispatch_ffn_combine_w4_a8`` path. It loads real residual W4A8 tensors through
the official W4A8 post-load implementation, launches only
``torch.ops._C_ascend.svdq_w4a8_debug_readback``, and verifies debug readback
health on real post-loaded weights with zero or deterministic nonzero inputs.

The health gate proves the real-checkpoint launch/readback path, finite debug
buffers, and routed token counts. It is not the final nonzero real-checkpoint
GMM numerical gate because it does not yet compare against an unfused numerical
reference.
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
from svdq_w4a8_debug_readback_probe import (  # noqa: E402
    _destroy_hccl_if_needed,
    _has_registered_debug_op,
    _init_single_rank_hccl,
    _npu_environment,
    _official_debug_symbol_status,
)

from vllm_ascend.quantization.svdq_spec import build_svdq_moe_layer_spec  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_bb_w4a8_debug_readback_real_checkpoint_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=Path(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--route-experts", type=int, nargs="*", default=None)
    parser.add_argument("--local-num-experts", type=int, default=None)
    parser.add_argument("--max-output-size", type=int, default=512)
    parser.add_argument("--input-mode", choices=("zero", "linear"), default="zero")
    parser.add_argument("--linear-input-scale", type=float, default=0.01)
    parser.add_argument("--out-zero-abs-tol", type=float, default=1e-6)
    parser.add_argument("--compare-gmm1-reference", action="store_true")
    parser.add_argument("--gmm1-reference-max-rows", type=int, default=64)
    parser.add_argument("--gmm1-reference-max-abs-tol", type=float, default=2e-2)
    parser.add_argument("--gmm1-reference-mean-abs-tol", type=float, default=2e-3)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _load_real_residual_layer(
    *,
    model_path: str,
    layer_index: int,
    tp_size: int,
    tp_rank: int,
    routed_experts: set[int],
    local_num_experts: int,
) -> tuple[torch.nn.Module, Any, int]:
    quant_description = _read_json(os.path.join(model_path, "quant_model_description.json"))
    weight_map = _weight_map(model_path)
    spec = build_svdq_moe_layer_spec(
        quant_description=quant_description,
        prefix=f"model.language_model.layers.{layer_index}.mlp.experts",
        model_path=model_path,
        num_experts=local_num_experts,
        hidden_size=2048,
        intermediate_size=512,
    )
    method = _make_official_w4a8_method(quant_description, tp_size=tp_size)
    layer = _make_residual_validation_layer(method=method, spec=spec, tp_size=tp_size, tp_rank=tp_rank)
    for param in layer.parameters():
        param.data.zero_()
    residual_keys = [
        key
        for key in _residual_checkpoint_keys(spec, weight_map)
        if int(key.removeprefix(f"{spec.checkpoint_prefix}.").split(".", 1)[0]) in routed_experts
    ]
    for shard, shard_keys in _group_keys_by_shard(residual_keys, weight_map).items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                _load_residual_checkpoint_tensor(layer=layer, spec=spec, key=key, loaded_weight=f.get_tensor(key))
    _ensure_minimal_ascend_config_for_official_postload()
    method.process_weights_after_loading(layer)
    return layer, spec, len(residual_keys)


def _index_runs(mask: torch.Tensor, *, limit: int = 16) -> list[list[int]]:
    indices = mask.nonzero().flatten().tolist()
    if not indices:
        return []
    runs = []
    start = prev = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index == prev + 1:
            prev = index
            continue
        runs.append([start, prev])
        if len(runs) >= limit:
            return runs
        start = prev = index
    runs.append([start, prev])
    return runs[:limit]


def _float_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(cpu).all().item()) if cpu.numel() else True
    abs_cpu = torch.nan_to_num(cpu).abs()
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(abs_cpu.max().item()) if abs_cpu.numel() else 0.0,
        "mean_abs": float(abs_cpu.mean().item()) if abs_cpu.numel() else 0.0,
        "nonzero": bool(torch.any(abs_cpu > 0).item()) if abs_cpu.numel() else False,
        "nan_count": int(torch.isnan(cpu).sum().item()),
        "inf_count": int(torch.isinf(cpu).sum().item()),
        "sample": cpu.flatten()[:8].tolist(),
    }
    if cpu.ndim == 2 and cpu.numel():
        nan_mask = torch.isnan(cpu)
        inf_mask = torch.isinf(cpu)
        nonfinite_mask = ~torch.isfinite(cpu)
        stats.update(
            {
                "nan_row_count": int(nan_mask.any(dim=1).sum().item()),
                "nan_col_count": int(nan_mask.any(dim=0).sum().item()),
                "inf_row_count": int(inf_mask.any(dim=1).sum().item()),
                "inf_col_count": int(inf_mask.any(dim=0).sum().item()),
                "nonfinite_row_count": int(nonfinite_mask.any(dim=1).sum().item()),
                "nonfinite_col_count": int(nonfinite_mask.any(dim=0).sum().item()),
                "nan_rows_first32": nan_mask.any(dim=1).nonzero().flatten()[:32].tolist(),
                "nan_cols_first32": nan_mask.any(dim=0).nonzero().flatten()[:32].tolist(),
                "nan_row_ranges_first16": _index_runs(nan_mask.any(dim=1)),
                "nan_col_ranges_first16": _index_runs(nan_mask.any(dim=0)),
                "nonfinite_rows_first32": nonfinite_mask.any(dim=1).nonzero().flatten()[:32].tolist(),
                "nonfinite_cols_first32": nonfinite_mask.any(dim=0).nonzero().flatten()[:32].tolist(),
                "nonfinite_row_ranges_first16": _index_runs(nonfinite_mask.any(dim=1)),
                "nonfinite_col_ranges_first16": _index_runs(nonfinite_mask.any(dim=0)),
            }
        )
    return stats


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_finite": bool(torch.isfinite(actual_cpu).all().item()) if actual_cpu.numel() else True,
        "expected_finite": bool(torch.isfinite(expected_cpu).all().item()) if expected_cpu.numel() else True,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
        "numel": int(diff.numel()),
        "actual_sample": actual_cpu.flatten()[:8].tolist(),
        "expected_sample": expected_cpu.flatten()[:8].tolist(),
        "diff_sample": diff.flatten()[:8].tolist(),
    }


def _int64_float_bits_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    scale_cpu = scale.detach().contiguous().cpu().numpy().astype(np.uint64).astype(np.uint32)
    scale_fp32 = torch.from_numpy(scale_cpu.view(np.float32).copy()).float()
    if scale_fp32.dim() == 3 and scale_fp32.shape[1] == 1:
        return scale_fp32[:, 0, :]
    return scale_fp32


def _unpack_postloaded_w4_columns(weight: torch.Tensor, output_columns: int) -> torch.Tensor:
    words = weight.detach().cpu().contiguous().to(torch.int32)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).reshape(1, 1, 1, 8)
    unpacked = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(words.shape[0], words.shape[1], output_columns)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.int32)


def _official_int8_to_int4_parts(x_int8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_cpu = x_int8.detach().cpu().to(torch.int16)
    # Mirrors FetchAndPreprocessInt8ToInt4: high=floor(x/16), low=(x & 0x0f)-8.
    x_high = torch.floor(x_cpu.float() / 16.0).to(torch.int32)
    x_low = ((x_cpu & 0x0F) - 8).to(torch.int32)
    return x_high, x_low


def _clip_group_counts(counts: torch.Tensor, row_limit: int) -> torch.Tensor:
    remaining = int(row_limit)
    clipped = []
    flat_counts = counts.detach().cpu().to(torch.int64).flatten().tolist()
    for count in flat_counts:
        take = min(int(count), remaining)
        clipped.append(take)
        remaining -= take
        if remaining <= 0:
            clipped.extend([0] * (len(flat_counts) - len(clipped)))
            break
    return torch.tensor(clipped, dtype=torch.int64)


def _official_gmm1_unfused_reference(
    *,
    routed_x_int8: torch.Tensor,
    routed_x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_bias: torch.Tensor,
    expert_token_nums: torch.Tensor,
    output_columns: int,
    max_rows: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    row_count = min(int(routed_x_int8.shape[0]), int(max_rows))
    counts = _clip_group_counts(expert_token_nums, row_count)
    x_high, x_low = _official_int8_to_int4_parts(routed_x_int8[:row_count])
    per_token_scale = routed_x_scale[:row_count].detach().cpu().float()
    weight_scale_fp32 = _int64_float_bits_to_fp32(weight_scale)
    bias = scale_bias.detach().cpu().float().contiguous()
    unpacked_weight = _unpack_postloaded_w4_columns(weight, output_columns)

    outputs: list[torch.Tensor] = []
    row_start = 0
    for expert_id, count in enumerate(counts.tolist()):
        count = int(count)
        if count <= 0:
            continue
        row_end = row_start + count
        weight_e = unpacked_weight[expert_id]
        high_acc = x_high[row_start:row_end].matmul(weight_e)
        low_acc = x_low[row_start:row_end].matmul(weight_e)
        combined = (high_acc * 16 + low_acc).float()
        dequant = combined * weight_scale_fp32[expert_id].reshape(1, -1) + bias[expert_id].reshape(1, -1)
        outputs.append(dequant * per_token_scale[row_start:row_end].reshape(-1, 1))
        row_start = row_end
    if outputs:
        reference = torch.cat(outputs, dim=0)
    else:
        reference = torch.empty((0, output_columns), dtype=torch.float32)
    return reference, {
        "source": "official dispatch_ffn_combine_w4_a8 AIC/AIV contract",
        "activation_split": "FetchAndPreprocessInt8ToInt4 high=floor(x/16), low=(x&0x0f)-8",
        "epilogue_formula": "(high_acc * 16 + low_acc) * postloaded_weight_scale + scale_bias, then * routed_x_scale",
        "compared_rows": int(reference.shape[0]),
        "group_counts": counts.tolist(),
    }


def _postload_metadata(layer: torch.nn.Module) -> dict[str, Any]:
    names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_scale_bias",
        "w2_scale_bias",
    )
    metadata = {}
    for name in names:
        tensor = getattr(layer, name)
        metadata[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "device": str(tensor.device),
            "health": _float_stats(tensor),
        }
    return metadata


def _routed_experts(args: argparse.Namespace) -> list[int]:
    experts = list(range(args.top_k)) if args.route_experts is None else list(args.route_experts)
    if len(experts) != args.top_k:
        raise ValueError("route-experts length must match top-k.")
    if len(set(experts)) != len(experts):
        raise ValueError("route-experts must not contain duplicates.")
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


def _make_input(args: argparse.Namespace, hidden_size: int, *, device: torch.device) -> torch.Tensor:
    shape = (args.num_tokens, hidden_size)
    if args.input_mode == "zero":
        return torch.zeros(shape, dtype=torch.bfloat16, device=device)
    values = torch.linspace(
        -args.linear_input_scale,
        args.linear_input_scale,
        steps=args.num_tokens * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(shape)
    return values.to(torch.bfloat16).contiguous()


def _run_real_checkpoint(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    routed_experts = _routed_experts(args)
    local_num_experts = _local_num_experts(args, routed_experts)
    layer, spec, residual_key_count = _load_real_residual_layer(
        model_path=args.model_path,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
        routed_experts=set(routed_experts),
        local_num_experts=local_num_experts,
    )
    if args.max_output_size < args.num_tokens * args.top_k:
        raise ValueError("max_output_size must cover all routed token rows.")
    if max(routed_experts) >= spec.num_experts or min(routed_experts) < 0:
        raise ValueError(f"route-experts={routed_experts} must be within [0, {spec.num_experts}).")

    x = _make_input(args, spec.hidden_size, device=device)
    expert_idx = _make_expert_idx(routed_experts, args.num_tokens, device=device)
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
        gmm2_post_dequant,
    ) = op(
        x,
        [layer.w13_weight],
        [layer.w2_weight],
        expert_idx,
        [layer.w13_weight_scale],
        [layer.w2_weight_scale],
        [layer.w13_scale_bias],
        [layer.w2_scale_bias],
        probs,
        group,
        args.max_output_size,
        x_active_mask,
    )
    torch.npu.synchronize()

    active_rows = args.num_tokens * args.top_k
    output_stats = {
        "input": _float_stats(x),
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
    health_reference = {
        "type": f"{args.input_mode}_input_health_oracle",
        "expected_gmm1_post_dequant": (
            "finite official debug tap; not assumed zero because the official epilogue adds weight auxiliary/"
            "scale-bias before per-token scaling"
        ),
        "expected_routed_x_int8": "finite official routed INT8 activation fed to GMM1",
        "expected_routed_x_scale": "finite official routed per-row activation scale fed to GMM1 epilogue",
        "expected_gmm1_hidden_prequant": "finite official hidden prequant debug tap after SwiGLU",
        "expected_gmm2_post_dequant": "finite official debug tap",
        "expected_out": "finite output",
    }
    out_zero_match = output_stats["out"]["max_abs"] <= args.out_zero_abs_tol
    output_health = output_stats["out"]["finite"] and (args.input_mode != "zero" or out_zero_match)
    routed_quant_health = (
        output_stats["routed_x_scale_active"]["finite"]
        and (
            args.input_mode == "zero"
            or (
                output_stats["routed_x_scale_active"]["nonzero"]
                and output_stats["routed_x_int8_active"]["nonzero"]
            )
        )
    )
    readback_finite = (
        routed_quant_health
        and output_stats["gmm1_post_dequant_active"]["finite"]
        and output_stats["gmm2_post_dequant_active"]["finite"]
        and output_stats["out"]["finite"]
    )
    routed_rows_match = output_stats["expert_token_nums"]["sum"] == active_rows
    input_health_gate_passed = readback_finite and routed_rows_match and output_health
    hidden_prequant_health = {
        "finite": output_stats["gmm1_hidden_prequant_active"]["finite"],
        "nonzero": output_stats["gmm1_hidden_prequant_active"]["nonzero"],
    }
    if output_stats["gmm1_post_dequant_active"]["finite"] and hidden_prequant_health["finite"]:
        debug_boundary_classification = "gmm1_post_dequant_and_hidden_prequant_finite"
    elif not output_stats["gmm1_post_dequant_active"]["finite"] and hidden_prequant_health["finite"]:
        debug_boundary_classification = "gmm1_post_dequant_nonfinite_hidden_prequant_finite"
    elif output_stats["gmm1_post_dequant_active"]["finite"] and not hidden_prequant_health["finite"]:
        debug_boundary_classification = "gmm1_post_dequant_finite_hidden_prequant_nonfinite"
    else:
        debug_boundary_classification = "gmm1_post_dequant_and_hidden_prequant_nonfinite"
    gmm1_reference: dict[str, Any] | None = None
    gmm1_reference_passed = None
    if args.compare_gmm1_reference:
        reference, reference_contract = _official_gmm1_unfused_reference(
            routed_x_int8=routed_x_int8[:active_rows],
            routed_x_scale=routed_x_scale[:active_rows],
            weight=layer.w13_weight,
            weight_scale=layer.w13_weight_scale,
            scale_bias=layer.w13_scale_bias,
            expert_token_nums=expert_token_nums,
            output_columns=2 * spec.intermediate_size,
            max_rows=args.gmm1_reference_max_rows,
        )
        actual = gmm1_post_dequant[: reference.shape[0], : reference.shape[1]]
        error = _tensor_error(actual, reference)
        gmm1_reference_passed = (
            error["actual_finite"]
            and error["expected_finite"]
            and error["diff_finite"]
            and error["max_abs"] <= args.gmm1_reference_max_abs_tol
            and error["mean_abs"] <= args.gmm1_reference_mean_abs_tol
        )
        gmm1_reference = {
            "enabled": True,
            "passed": gmm1_reference_passed,
            "contract": reference_contract,
            "error": error,
            "max_abs_tolerance": args.gmm1_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm1_reference_mean_abs_tol,
        }
        input_health_gate_passed = input_health_gate_passed and gmm1_reference_passed
    else:
        gmm1_reference = {
            "enabled": False,
            "reason": "pass --compare-gmm1-reference to run the official-source unfused GMM1 comparator",
        }
    return {
        "stage": f"real_checkpoint_w4a8_debug_readback_{args.input_mode}_input_health",
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_debug_readback -> aclnnSVDQW4A8DebugReadback",
        "official_source_of_truth": "dispatch_ffn_combine_w4_a8 AIC producer and AIV dequant debug taps",
        "public_grouped_matmul_used": False,
        "real_checkpoint_validation": True,
        "nonzero_real_checkpoint_numerical_gate": False,
        "gmm1_real_checkpoint_numerical_gate": (
            bool(gmm1_reference_passed) if gmm1_reference_passed is not None else False
        ),
        "gmm2_real_checkpoint_numerical_gate": False,
        "production_svdq_host_tiling_fail_closed": True,
        "layer_index": args.layer,
        "layer_name": spec.prefix,
        "residual_checkpoint_key_count": residual_key_count,
        "routed_experts": routed_experts,
        "shape": {
            "num_experts": spec.num_experts,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "max_output_size": args.max_output_size,
            "active_rows": active_rows,
            "input_mode": args.input_mode,
        },
        "official_postload": {
            "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
            "metadata": _postload_metadata(layer),
        },
        "unfused_reference": health_reference,
        "gmm1_unfused_reference": gmm1_reference,
        "gmm2_unfused_reference": {
            "enabled": False,
            "reason": (
                "The current debug ABI does not expose official hidden INT8 and hidden per-token scale, "
                "which are required to compare GMM2 without re-deriving hidden quantization."
            ),
        },
        "out_zero_abs_tolerance": args.out_zero_abs_tol,
        "checks": {
            "input_health_gate_passed": input_health_gate_passed,
            "routed_quant_health": routed_quant_health,
            "readback_finite": readback_finite,
            "gmm1_reference_passed": gmm1_reference_passed,
            "gmm2_reference_passed": None,
            "hidden_prequant_health": hidden_prequant_health,
            "debug_boundary_classification": debug_boundary_classification,
            "routed_rows_match": routed_rows_match,
            "output_health": output_health,
            "out_zero_match": out_zero_match,
        },
        "outputs": output_stats,
        "passed": input_health_gate_passed,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_debug_readback_real_checkpoint_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "official_debug_symbol_status": _official_debug_symbol_status(),
        "production_svdq_host_tiling_fail_closed": True,
        "next_required_stage": "nonzero real-checkpoint GMM1/GMM2 comparison against an unfused reference",
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
        stage = _run_real_checkpoint(args, str(group_info["group"]))
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
