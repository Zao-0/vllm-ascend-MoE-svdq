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
``torch.ops._C_ascend.svdq_w4a8_debug_readback``, and compares zero-input GMM1
and GMM2 debug readbacks with an unfused zero oracle.

The zero oracle proves the real-checkpoint launch/readback path and catches
non-finite or unexpected bias leakage. It is not the final nonzero
real-checkpoint GMM numerical gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--zero-abs-tol", type=float, default=1e-6)
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
        "sample": cpu.flatten()[:8].tolist(),
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


def _run_zero_input_real_checkpoint(args: argparse.Namespace, group: str) -> dict[str, Any]:
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

    x = torch.zeros((args.num_tokens, spec.hidden_size), dtype=torch.bfloat16, device=device)
    expert_idx = _make_expert_idx(routed_experts, args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)

    op = torch.ops._C_ascend.svdq_w4a8_debug_readback
    out, expert_token_nums, gmm1_post_dequant, gmm2_post_dequant = op(
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
        "out": _float_stats(out),
        "gmm1_post_dequant_active": _float_stats(gmm1_post_dequant[:active_rows]),
        "gmm2_post_dequant_active": _float_stats(gmm2_post_dequant[:active_rows]),
        "expert_token_nums": {
            "shape": list(expert_token_nums.shape),
            "dtype": str(expert_token_nums.dtype),
            "values": expert_token_nums.detach().cpu().tolist(),
            "sum": int(expert_token_nums.detach().cpu().sum().item()),
        },
    }
    zero_reference = {
        "type": "unfused_zero_input_oracle",
        "expected_gmm1_post_dequant": "all zeros",
        "expected_gmm2_post_dequant": "all zeros",
        "expected_out": "all zeros",
    }
    gmm1_zero_match = output_stats["gmm1_post_dequant_active"]["max_abs"] <= args.zero_abs_tol
    gmm2_zero_match = output_stats["gmm2_post_dequant_active"]["max_abs"] <= args.zero_abs_tol
    out_zero_match = output_stats["out"]["max_abs"] <= args.zero_abs_tol
    readback_finite = (
        output_stats["gmm1_post_dequant_active"]["finite"]
        and output_stats["gmm2_post_dequant_active"]["finite"]
        and output_stats["out"]["finite"]
    )
    routed_rows_match = output_stats["expert_token_nums"]["sum"] == active_rows
    passed = readback_finite and routed_rows_match and gmm1_zero_match and gmm2_zero_match and out_zero_match
    return {
        "stage": "real_checkpoint_w4a8_debug_readback_zero_input",
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_debug_readback -> aclnnSVDQW4A8DebugReadback",
        "official_source_of_truth": "dispatch_ffn_combine_w4_a8 AIC producer and AIV dequant debug taps",
        "public_grouped_matmul_used": False,
        "real_checkpoint_validation": True,
        "nonzero_real_checkpoint_numerical_gate": False,
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
        },
        "official_postload": {
            "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
            "metadata": _postload_metadata(layer),
        },
        "unfused_reference": zero_reference,
        "zero_abs_tolerance": args.zero_abs_tol,
        "checks": {
            "readback_finite": readback_finite,
            "routed_rows_match": routed_rows_match,
            "gmm1_zero_match": gmm1_zero_match,
            "gmm2_zero_match": gmm2_zero_match,
            "out_zero_match": out_zero_match,
        },
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
        stage = _run_zero_input_real_checkpoint(args, str(group_info["group"]))
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
