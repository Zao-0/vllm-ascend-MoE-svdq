#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate final SVDQ MoE combine semantics on the NPU token-unpermute op.

The production fused SVDQ operator is still fail-closed. This probe validates
the final combine boundary it must eventually reuse: routed expert rows,
flattened ``[token, top_k]`` row indices, and per-route top-k probabilities are
fed to ``torch_npu.npu_moe_token_unpermute`` and compared with the pure PyTorch
SVDQ final-combine oracle.
"""

from __future__ import annotations

import argparse
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

from vllm_ascend.quantization.methods.svdq_post_load import build_svdq_final_combine_reference  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_h_final_combine_device_probe_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--max-abs-tol", type=float, default=0.0)
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
        "has_npu_moe_token_unpermute": False,
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
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


def _make_topk_weights(num_tokens: int, top_k: int) -> torch.Tensor:
    weights = torch.tensor([2.0 ** -(slot + 1) for slot in range(top_k)], dtype=torch.float32)
    weights[-1] += 1.0 - float(weights.sum().item())
    return weights.expand(num_tokens, top_k).contiguous()


def _make_routed_output(num_rows: int, hidden_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(-64, 65, (num_rows, hidden_size), dtype=torch.int16, generator=generator).float()
    row_offsets = torch.arange(num_rows, dtype=torch.float32).unsqueeze(1) / 16.0
    return ((values / 32.0) + row_offsets).to(torch.bfloat16)


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


def _run_device_combine(
    *, dimensions: dict[str, int | str], num_tokens: int, seed: int, device: torch.device, max_abs_tol: float
) -> dict[str, Any]:
    import torch_npu  # type: ignore[import-untyped]

    hidden_size = int(dimensions["hidden_size"])
    top_k = int(dimensions["top_k"])
    num_rows = num_tokens * top_k
    routed_output = _make_routed_output(num_rows, hidden_size, seed)
    topk_weights = _make_topk_weights(num_tokens, top_k)
    expanded_row_idx = torch.arange(num_rows, dtype=torch.int32)

    reference = build_svdq_final_combine_reference(
        routed_output=routed_output,
        topk_weights=topk_weights,
        expanded_row_idx=expanded_row_idx,
    )
    expected = reference["stages"]["combined_output"].to(torch.bfloat16)
    actual = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=routed_output.to(device=device),
        sorted_indices=expanded_row_idx.to(device=device),
        probs=topk_weights.to(device=device),
    )
    torch.npu.synchronize()
    actual_cpu = actual.detach().cpu()
    error = _tensor_error(actual_cpu, expected)
    return {
        "stage": "final_combine_token_unpermute",
        "input_shape": list(routed_output.shape),
        "input_dtype": str(routed_output.dtype),
        "topk_weights_shape": list(topk_weights.shape),
        "topk_weights_dtype": str(topk_weights.dtype),
        "expanded_row_idx_shape": list(expanded_row_idx.shape),
        "expanded_row_idx_dtype": str(expanded_row_idx.dtype),
        "oracle_stage_shapes": reference["stage_shapes"],
        "output_error": error,
        "max_abs_tolerance": max_abs_tol,
        "passed": (
            actual_cpu.shape == expected.shape
            and error["actual_finite"]
            and error["expected_finite"]
            and error["diff_finite"]
            and float(error["max_abs"]) <= max_abs_tol
        ),
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
        "probe": "svdq_final_combine_device_probe",
        "model_path": args.model_path,
        "dimensions": dimensions,
        "environment": env,
        "stages": [],
        "passed": False,
        "skipped": False,
    }
    if not (env["torch_npu_imported"] and env["has_npu_moe_token_unpermute"] and env["npu_available"]):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = "torch_npu.npu_moe_token_unpermute and an available NPU are required."
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
    stage = _run_device_combine(
        dimensions=dimensions,
        num_tokens=args.num_tokens,
        seed=args.seed,
        device=device,
        max_abs_tol=args.max_abs_tol,
    )
    summary["stages"] = [stage]
    summary["passed"] = bool(stage["passed"])
    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    print(json.dumps({"summary_path": path, "passed": summary["passed"], "skipped": False}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
