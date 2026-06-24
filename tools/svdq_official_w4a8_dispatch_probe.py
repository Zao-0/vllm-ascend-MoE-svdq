#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Launch the official W4A8 dispatch_ffn_combine path with SVDQ residual tensors.

Appendix-1 requires isolated validation of the official W4A8 AIC/AIV path
before the production SVDQ operator can be enabled. This probe does not exercise
the SVDQ production op and does not use a scalar production substitute. It loads
real SVDQ residual tensors through the official W4A8 post-load code and calls
``torch.ops._C_ascend.dispatch_ffn_combine``. The existing torch adapter routes
INT32 packed-W4 weights to ``aclnnDispatchFFNCombineW4A8``.

The current numerical gate is deliberately narrow: zero BF16 input routed to
real checkpoint weights should produce finite zero BF16 output. This is an
official full-op smoke gate, not completion of the required nonzero staged
GMM/dequant/mixed-epilogue validation.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
)
from svdq_w4a8_residual_gmm_device_probe import (  # noqa: E402
    _load_real_residual_layer,
    _npu_environment,
)

DEFAULT_SUMMARY_NAME = "phase_as_official_w4a8_dispatch_probe_summary.json"
CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
OFFICIAL_W4A8_OPAPI_SYMBOLS = (
    "aclnnDispatchFFNCombineW4A8GetWorkspaceSize",
    "aclnnDispatchFFNCombineW4A8",
    "aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize",
    "aclnnInnerDispatchFFNCombineW4A8",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--max-output-size", type=int, default=512)
    parser.add_argument("--zero-abs-tol", type=float, default=1e-30)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


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
    dist.init_process_group(
        backend="hccl",
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    return {
        "initialized_here": True,
        "backend": dist.get_backend(),
        "rank": 0,
        "world_size": 1,
        "init_method": f"tcp://127.0.0.1:{port}",
        "group": _get_hccl_group_name(0),
        "device_id": device_id,
    }


def _destroy_hccl_if_needed(group_info: dict[str, Any]) -> None:
    if group_info.get("initialized_here") and dist.is_initialized():
        dist.destroy_process_group()


def _has_official_dispatch_op() -> bool:
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

        bootstrap_custom_op_env(include_vendor_lib=True)
        enable_custom_op()
    except Exception:
        return False
    return hasattr(getattr(torch.ops, "_C_ascend", object()), "dispatch_ffn_combine")


def _official_w4a8_symbol_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "lib_path": str(CUSTOM_OPAPI_LIB),
        "exists": CUSTOM_OPAPI_LIB.exists(),
        "loaded": False,
        "symbols": {name: False for name in OFFICIAL_W4A8_OPAPI_SYMBOLS},
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
    status["symbols"] = {name: hasattr(handle, name) for name in OFFICIAL_W4A8_OPAPI_SYMBOLS}
    return status


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu()
    abs_cpu = torch.nan_to_num(cpu.float()).abs()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "finite": bool(torch.isfinite(cpu.float()).all().item()) if cpu.numel() else True,
        "max_abs": float(abs_cpu.max().item()) if abs_cpu.numel() else 0.0,
        "mean_abs": float(abs_cpu.mean().item()) if abs_cpu.numel() else 0.0,
        "nan_count": int(torch.isnan(cpu.float()).sum().item()),
    }


def _run_official_zero_input_stage(
    *,
    model_path: str,
    layer_index: int,
    num_tokens: int,
    top_k: int,
    expert_id: int,
    max_output_size: int,
    zero_abs_tol: float,
    device: torch.device,
    group: str,
) -> dict[str, Any]:
    layer, spec, residual_key_count = _load_real_residual_layer(
        model_path=model_path,
        layer_index=layer_index,
        tp_size=1,
        tp_rank=0,
    )
    if expert_id < 0 or expert_id >= spec.num_experts:
        raise ValueError(f"expert {expert_id} is outside [0, {spec.num_experts}).")

    x = torch.zeros((num_tokens, spec.hidden_size), dtype=torch.bfloat16, device=device)
    expert_idx = torch.full((num_tokens, top_k), expert_id, dtype=torch.int32, device=device)
    probs = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((num_tokens,), dtype=torch.bool, device=device)
    out = torch.empty_like(x)
    expert_token_nums = torch.empty((1, spec.num_experts), dtype=torch.int32, device=device)

    torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore[attr-defined]
        x=x,
        weight1=[layer.w13_weight],
        weight2=[layer.w2_weight],
        expert_idx=expert_idx,
        scale1=[layer.w13_weight_scale],
        scale2=[layer.w2_weight_scale],
        bias1=[layer.w13_scale_bias],
        bias2=[layer.w2_scale_bias],
        probs=probs,
        group=group,
        max_output_size=max_output_size,
        out=out,
        expert_token_nums=expert_token_nums,
        x_active_mask=x_active_mask,
    )
    torch.npu.synchronize()
    output_stats = _tensor_stats(out)
    expert_token_stats = _tensor_stats(expert_token_nums)
    passed = output_stats["finite"] and output_stats["max_abs"] <= zero_abs_tol
    return {
        "stage": "official_dispatch_ffn_combine_w4a8_zero_input_real_checkpoint",
        "official_api": "torch.ops._C_ascend.dispatch_ffn_combine -> aclnnDispatchFFNCombineW4A8",
        "official_aic_aiv_validation": True,
        "numerical_scope": (
            "zero-input full official W4A8 op smoke; nonzero staged GMM/dequant validation still required"
        ),
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "residual_checkpoint_key_count": residual_key_count,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "expert_idx_shape": list(expert_idx.shape),
        "top_k": top_k,
        "expert_id": expert_id,
        "postload_tensor_shapes": {
            "w13_weight": list(layer.w13_weight.shape),
            "w2_weight": list(layer.w2_weight.shape),
            "w13_weight_scale": list(layer.w13_weight_scale.shape),
            "w2_weight_scale": list(layer.w2_weight_scale.shape),
            "w13_scale_bias": list(layer.w13_scale_bias.shape),
            "w2_scale_bias": list(layer.w2_scale_bias.shape),
        },
        "postload_tensor_dtypes": {
            "w13_weight": str(layer.w13_weight.dtype),
            "w2_weight": str(layer.w2_weight.dtype),
            "w13_weight_scale": str(layer.w13_weight_scale.dtype),
            "w2_weight_scale": str(layer.w2_weight_scale.dtype),
            "w13_scale_bias": str(layer.w13_scale_bias.dtype),
            "w2_scale_bias": str(layer.w2_scale_bias.dtype),
        },
        "output": output_stats,
        "expert_token_nums": expert_token_stats,
        "zero_abs_tolerance": zero_abs_tol,
        "passed": passed,
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
    env["has_torch_ops_c_ascend_dispatch_ffn_combine"] = _has_official_dispatch_op()
    official_w4a8_symbols = _official_w4a8_symbol_status()
    summary: dict[str, Any] = {
        "probe": "svdq_official_w4a8_dispatch_probe",
        "model_path": args.model_path,
        "environment": env,
        "official_w4a8_opapi_symbols": official_w4a8_symbols,
        "stages": [],
        "passed": False,
        "skipped": False,
        "production_svdq_host_tiling_expected": "fail_closed",
    }

    if not (
        env["torch_npu_imported"]
        and env["npu_available"]
        and env["has_torch_ops_c_ascend_dispatch_ffn_combine"]
    ):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = (
            "torch_npu, an available NPU, and torch.ops._C_ascend.dispatch_ffn_combine are required."
        )
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    if int(env["npu_device_count"]) <= args.device_id:
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = f"device_id {args.device_id} is outside npu_device_count={env['npu_device_count']}."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl"] = group_info
        stage = _run_official_zero_input_stage(
            model_path=args.model_path,
            layer_index=args.layer,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            expert_id=args.expert,
            max_output_size=args.max_output_size,
            zero_abs_tol=args.zero_abs_tol,
            device=device,
            group=str(group_info["group"]),
        )
        summary["stages"] = [stage]
        summary["passed"] = stage["passed"]
    except Exception as exc:
        summary["exception"] = f"{type(exc).__name__}: {exc}"
        summary["passed"] = False
    finally:
        _destroy_hccl_if_needed(group_info)

    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    print(json.dumps({"summary_path": path, "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
