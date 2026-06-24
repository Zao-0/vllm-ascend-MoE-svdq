#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate W4A8-SVDQ activation quantization against torch_npu semantics.

This probe covers the two residual W4A8 quantization inputs in the SVDQ MoE
dataflow:

* routed BF16 expert input, before the gate/up W4A8 GMM
* BF16 SwiGLU hidden input, before the down W4A8 GMM

It intentionally does not call the production fused SVDQ operator. The output is
an evidence JSON file that records exact int8 agreement and scale/reconstruction
errors for torch_npu.npu_dynamic_quant against the CPU reference formula.
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

DEFAULT_SUMMARY_NAME = "phase_f_activation_quant_probe_summary.json"
STAGES = ("routed_input", "hidden_input")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--scale-tol", type=float, default=1e-7)
    parser.add_argument("--dequant-max-abs-tol", type=float, default=0.02)
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
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
        info["has_npu_dynamic_quant"] = hasattr(torch_npu, "npu_dynamic_quant")
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


def _make_stage_input(*, num_tokens: int, width: int, seed: int, stage_index: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + stage_index * 1009)
    q_values = torch.randint(-120, 121, (num_tokens, width), generator=generator, dtype=torch.int16).float()
    scale = float(stage_index + 1) / 64.0
    values = q_values * scale

    if num_tokens:
        values[0].zero_()
    for row in range(1, num_tokens):
        values[row, 0] = 127.0 * scale
    if num_tokens > 1 and width >= 8:
        pattern = torch.tensor([0.0, 32.0, -32.0, 64.0, -64.0, 16.0, -16.0, 8.0], dtype=torch.float32)
        values[1, :8] = pattern * scale
        values[1, 8] = 127.0 * scale
        values[1, 9] = -126.0 * scale
    return values.to(torch.bfloat16)


def _cpu_dynamic_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.detach().cpu().float()
    max_abs = x_fp32.abs().amax(dim=1)
    scale = (max_abs / 127.0).float()
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x_fp32 / safe_scale[:, None]).clamp(-127, 127).to(torch.int8)
    q = torch.where((scale == 0)[:, None], torch.zeros_like(q), q)
    return q, scale


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool | int]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    actual_finite = bool(torch.isfinite(actual_cpu.float()).all().item()) if actual_cpu.numel() else True
    expected_finite = bool(torch.isfinite(expected_cpu.float()).all().item()) if expected_cpu.numel() else True
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "numel": int(actual_cpu.numel()),
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
    }


def _quantize_on_npu(x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu  # type: ignore[import-untyped]

    q, scale = torch_npu.npu_dynamic_quant(x.to(device=device, dtype=torch.bfloat16))
    torch.npu.synchronize()
    return q.detach().cpu(), scale.detach().cpu().float()


def _stage_metrics(
    *,
    name: str,
    x: torch.Tensor,
    device: torch.device,
    scale_tol: float,
    dequant_max_abs_tol: float,
) -> dict[str, Any]:
    q_expected, scale_expected = _cpu_dynamic_quant_reference(x)
    q_actual, scale_actual = _quantize_on_npu(x, device)
    q_diff = (q_actual.to(torch.int16) - q_expected.to(torch.int16)).abs()
    actual_dequant = q_actual.float() * scale_actual[:, None]
    expected_dequant = q_expected.float() * scale_expected[:, None]
    input_reconstruction = q_actual.float() * scale_actual[:, None]

    scale_error = _tensor_error(scale_actual, scale_expected)
    dequant_error = _tensor_error(actual_dequant, expected_dequant)
    input_error = _tensor_error(input_reconstruction, x.float())
    q_exact_match = bool(torch.equal(q_actual, q_expected))
    scale_within_tol = bool(scale_error["max_abs"] <= scale_tol)
    dequant_within_tol = bool(input_error["max_abs"] <= dequant_max_abs_tol)
    passed = (
        q_exact_match
        and scale_within_tol
        and bool(scale_error["actual_finite"])
        and bool(scale_error["expected_finite"])
        and bool(dequant_error["actual_finite"])
        and bool(dequant_error["expected_finite"])
        and dequant_within_tol
    )

    return {
        "stage": name,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "q_dtype": str(q_actual.dtype),
        "scale_dtype": str(scale_actual.dtype),
        "zero_row_scale": float(scale_actual[0].item()) if scale_actual.numel() else None,
        "q_exact_match": q_exact_match,
        "q_mismatch_count": int((q_diff != 0).sum().item()),
        "q_max_abs_diff": int(q_diff.max().item()) if q_diff.numel() else 0,
        "scale_error": scale_error,
        "npu_vs_reference_dequant_error": dequant_error,
        "npu_dequant_vs_input_error": input_error,
        "scale_tolerance": scale_tol,
        "dequant_max_abs_tolerance": dequant_max_abs_tol,
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
    dimensions = _target_dimensions(args.model_path)
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_activation_quant_probe",
        "model_path": args.model_path,
        "dimensions": dimensions,
        "stages": [],
        "environment": env,
        "passed": False,
        "skipped": False,
    }

    if not (env["torch_npu_imported"] and env["has_npu_dynamic_quant"] and env["npu_available"]):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = "torch_npu.npu_dynamic_quant or an available NPU is required."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    if int(env["npu_device_count"]) <= args.device_id:
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = f"device_id {args.device_id} is outside npu_device_count={env['npu_device_count']}."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        print(json.dumps({"summary_path": path, "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    stage_widths = {
        "routed_input": int(dimensions["hidden_size"]),
        "hidden_input": int(dimensions["intermediate_size"]),
    }
    stage_results = []
    for stage_index, stage_name in enumerate(STAGES):
        x = _make_stage_input(
            num_tokens=args.num_tokens,
            width=stage_widths[stage_name],
            seed=args.seed,
            stage_index=stage_index,
        )
        stage_results.append(
            _stage_metrics(
                name=stage_name,
                x=x,
                device=device,
                scale_tol=args.scale_tol,
                dequant_max_abs_tol=args.dequant_max_abs_tol,
            )
        )

    summary["stages"] = stage_results
    summary["passed"] = all(stage["passed"] for stage in stage_results)
    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    print(json.dumps({"summary_path": path, "passed": summary["passed"], "stages": stage_results}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
