#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate the isolated SVDQ mixed epilogue debug op on NPU.

The production fused SVDQ operator is still fail-closed. This probe validates
the mixed epilogue boundary it must implement: residual and BF16 low-rank
branches are added before SwiGLU, hidden is dynamically quantized, and residual
plus low-rank down outputs are added before final combine.
"""

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

from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _read_json,
)

from vllm_ascend.quantization.methods.svdq_post_load import build_svdq_mixed_epilogue_reference  # noqa: E402
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_k_mixed_epilogue_device_probe_summary.json"
DEBUG_OP_NAME = "SVDQMixedEpilogueDebugReadback"
CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
_PRELOADED_CUSTOM_OPAPI_GLOBAL = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--max-abs-tol", type=float, default=0.02)
    parser.add_argument("--mean-abs-tol", type=float, default=0.002)
    parser.add_argument("--scale-tol", type=float, default=1e-7)
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
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
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
    values = torch.randint(-96, 97, shape, dtype=torch.int16, generator=generator).float()
    return (values * scale).to(torch.bfloat16)


def _make_fp32_tensor(shape: tuple[int, ...], *, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(-96, 97, shape, dtype=torch.int16, generator=generator).float()
    return (values * scale).float()


def _make_inputs(*, num_tokens: int, hidden_size: int, intermediate_size: int, seed: int) -> dict[str, torch.Tensor]:
    return {
        "residual_gate_up": _make_fp32_tensor((num_tokens, intermediate_size * 2), seed=seed, scale=1.0 / 64.0),
        "gate_lowrank": _make_bf16_tensor((num_tokens, intermediate_size), seed=seed + 1, scale=1.0 / 128.0),
        "up_lowrank": _make_bf16_tensor((num_tokens, intermediate_size), seed=seed + 2, scale=1.0 / 128.0),
        "residual_down": _make_fp32_tensor((num_tokens, hidden_size), seed=seed + 3, scale=1.0 / 64.0),
        "down_lowrank": _make_bf16_tensor((num_tokens, hidden_size), seed=seed + 4, scale=1.0 / 128.0),
    }


def _appendix3_case_inputs(base: dict[str, torch.Tensor], case: str) -> dict[str, torch.Tensor]:
    inputs = {name: tensor.clone() for name, tensor in base.items()}
    if case == "residual_only":
        inputs["gate_lowrank"].zero_()
        inputs["up_lowrank"].zero_()
        inputs["down_lowrank"].zero_()
    elif case == "svdq_only":
        inputs["residual_gate_up"].zero_()
        inputs["residual_down"].zero_()
    elif case != "two_branch_nonzero":
        raise ValueError(f"unsupported Appendix 3 mixed-epilogue case: {case}")
    return inputs


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


def _preload_custom_opapi() -> bool:
    global _PRELOADED_CUSTOM_OPAPI_GLOBAL
    if not CUSTOM_OPAPI_LIB.exists():
        return False
    ctypes.CDLL(str(CUSTOM_OPAPI_LIB), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_CUSTOM_OPAPI_GLOBAL = True
    return True


def _finish_probe(payload: dict[str, Any], exit_code: int) -> int:
    print(json.dumps(payload, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if _PRELOADED_CUSTOM_OPAPI_GLOBAL and os.environ.get("SVDQ_MIXED_EPILOGUE_ALLOW_CANN_TEARDOWN") != "1":
        os._exit(exit_code)
    return exit_code


def _has_registered_mixed_debug_op() -> bool:
    bootstrap_custom_op_env(include_vendor_lib=True)
    _preload_custom_opapi()
    enable_custom_op()
    return getattr(torch.ops._C_ascend, "svdq_mixed_epilogue_debug_readback", None) is not None


def _run_npu_mixed_epilogue(
    *,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    swiglu_limit: float,
) -> dict[str, torch.Tensor]:
    bootstrap_custom_op_env(include_vendor_lib=True)
    _preload_custom_opapi()
    enable_custom_op()
    op = getattr(torch.ops._C_ascend, "svdq_mixed_epilogue_debug_readback", None)
    if op is None:
        raise RuntimeError(
            "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback is not registered. "
            "Rebuild/install vllm-ascend after adding the debug binding."
        )

    gate_up_low_rank = torch.cat((inputs["gate_lowrank"], inputs["up_lowrank"]), dim=1).contiguous()
    (
        gate_up_total,
        hidden_bf16,
        hidden_q,
        hidden_scale,
        down_total,
        out_bf16,
    ) = op(
        inputs["residual_gate_up"].to(device=device, dtype=torch.float32).contiguous(),
        gate_up_low_rank.to(device=device, dtype=torch.bfloat16).contiguous(),
        inputs["residual_down"].to(device=device, dtype=torch.float32).contiguous(),
        inputs["down_lowrank"].to(device=device, dtype=torch.bfloat16).contiguous(),
        float(swiglu_limit),
    )
    torch.npu.synchronize()
    gate_mixed, up_mixed = gate_up_total.detach().float().cpu().chunk(2, dim=1)
    return {
        "gate_up_total": gate_up_total.detach().float().cpu(),
        "gate_mixed": gate_mixed,
        "up_mixed": up_mixed,
        "hidden_bf16": hidden_bf16.detach().cpu(),
        "hidden_q": hidden_q.detach().cpu(),
        "hidden_scale": hidden_scale.detach().cpu().float(),
        "down_mixed": down_total.detach().float().cpu(),
        "out_bf16": out_bf16.detach().cpu(),
    }


def _stage_passed(stage_error: dict[str, Any], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(stage_error["actual_finite"])
        and bool(stage_error["expected_finite"])
        and bool(stage_error["diff_finite"])
        and float(stage_error["max_abs"]) <= max_abs_tol
        and float(stage_error["mean_abs"]) <= mean_abs_tol
    )


def _exact_passed(stage_error: dict[str, Any]) -> bool:
    return (
        bool(stage_error["actual_finite"])
        and bool(stage_error["expected_finite"])
        and bool(stage_error["diff_finite"])
        and float(stage_error["max_abs"]) == 0.0
        and float(stage_error["mean_abs"]) == 0.0
    )


def _appendix3_boundary_checks(
    *,
    case: str,
    inputs: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if case == "residual_only":
        checks["w4a8_fp32_gate_up_add_zero_exact"] = _tensor_error(
            actual["gate_up_total"], inputs["residual_gate_up"]
        )
        checks["w4a8_fp32_down_add_zero_exact"] = _tensor_error(actual["down_mixed"], inputs["residual_down"])
    elif case == "svdq_only":
        gate_up_low_rank = torch.cat((inputs["gate_lowrank"], inputs["up_lowrank"]), dim=1).float()
        checks["svdq_bf16_gate_up_cast_to_fp32_exact"] = _tensor_error(actual["gate_up_total"], gate_up_low_rank)
        checks["svdq_bf16_down_cast_to_fp32_exact"] = _tensor_error(
            actual["down_mixed"], inputs["down_lowrank"].float()
        )
    return checks


def _run_probe_case(
    *,
    case: str,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    swiglu_limit: float,
    max_abs_tol: float,
    mean_abs_tol: float,
    scale_tol: float,
) -> dict[str, Any]:
    reference = build_svdq_mixed_epilogue_reference(**inputs, swiglu_limit=swiglu_limit)
    actual = _run_npu_mixed_epilogue(inputs=inputs, device=device, swiglu_limit=swiglu_limit)
    stage_errors = {
        "gate_mixed": _tensor_error(actual["gate_mixed"], reference["stages"]["gate_mixed"]),
        "up_mixed": _tensor_error(actual["up_mixed"], reference["stages"]["up_mixed"]),
        "hidden_bf16": _tensor_error(actual["hidden_bf16"], reference["stages"]["hidden_bf16"]),
        "hidden_scale": _tensor_error(actual["hidden_scale"], reference["stages"]["hidden_scale"]),
        "down_mixed": _tensor_error(actual["down_mixed"], reference["stages"]["down_mixed"]),
        "out_bf16": _tensor_error(actual["out_bf16"], reference["stages"]["down_mixed"].to(torch.bfloat16)),
    }
    q_diff = (actual["hidden_q"].to(torch.int16) - reference["stages"]["hidden_q"].to(torch.int16)).abs()
    stage_passed = {
        "gate_mixed": _stage_passed(stage_errors["gate_mixed"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "up_mixed": _stage_passed(stage_errors["up_mixed"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "hidden_bf16": _stage_passed(stage_errors["hidden_bf16"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "hidden_scale": (
            bool(stage_errors["hidden_scale"]["actual_finite"])
            and bool(stage_errors["hidden_scale"]["expected_finite"])
            and bool(stage_errors["hidden_scale"]["diff_finite"])
            and float(stage_errors["hidden_scale"]["max_abs"]) <= scale_tol
        ),
        "hidden_q": bool(torch.equal(actual["hidden_q"], reference["stages"]["hidden_q"])),
        "down_mixed": _stage_passed(stage_errors["down_mixed"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
        "out_bf16": _stage_passed(stage_errors["out_bf16"], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol),
    }
    boundary_checks = _appendix3_boundary_checks(case=case, inputs=inputs, actual=actual)
    boundary_passed = {name: _exact_passed(error) for name, error in boundary_checks.items()}
    return {
        "stage": "mixed_epilogue_debug_readback",
        "appendix3_case": case,
        "zero_branch_isolation_only": case in ("residual_only", "svdq_only"),
        "debug_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "debug_op_name": DEBUG_OP_NAME,
        "input_shapes": {name: list(tensor.shape) for name, tensor in inputs.items()},
        "oracle_stage_shapes": reference["stage_shapes"],
        "actual_stage_shapes": {name: list(tensor.shape) for name, tensor in actual.items()},
        "boundary_checks": boundary_checks,
        "boundary_passed": boundary_passed,
        "stage_errors": stage_errors,
        "hidden_q_exact_match": stage_passed["hidden_q"],
        "hidden_q_mismatch_count": int((q_diff != 0).sum().item()),
        "hidden_q_max_abs_diff": int(q_diff.max().item()) if q_diff.numel() else 0,
        "max_abs_tolerance": max_abs_tol,
        "mean_abs_tolerance": mean_abs_tol,
        "scale_tolerance": scale_tol,
        "swiglu_limit": float(swiglu_limit),
        "stage_passed": stage_passed,
        "passed": all(stage_passed.values()) and all(boundary_passed.values()),
    }


def _run_probe(
    *,
    dimensions: dict[str, int | str],
    num_tokens: int,
    seed: int,
    device: torch.device,
    swiglu_limit: float,
    max_abs_tol: float,
    mean_abs_tol: float,
    scale_tol: float,
) -> dict[str, Any]:
    base_inputs = _make_inputs(
        num_tokens=num_tokens,
        hidden_size=int(dimensions["hidden_size"]),
        intermediate_size=int(dimensions["intermediate_size"]),
        seed=seed,
    )
    appendix3_gate_order = ("residual_only", "svdq_only", "two_branch_nonzero")
    cases = {
        case: _run_probe_case(
            case=case,
            inputs=_appendix3_case_inputs(base_inputs, case),
            device=device,
            swiglu_limit=swiglu_limit,
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
            scale_tol=scale_tol,
        )
        for case in appendix3_gate_order
    }
    two_branch = cases["two_branch_nonzero"]
    return {
        "stage": "mixed_epilogue_debug_readback_appendix3_sequence",
        "debug_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "debug_op_name": DEBUG_OP_NAME,
        "appendix3_gate_order": list(appendix3_gate_order),
        "appendix3_gates": cases,
        "appendix3_zero_branch_tests_are_isolation_only": True,
        "ordered_gate_passed": {case: bool(cases[case]["passed"]) for case in appendix3_gate_order},
        "input_shapes": two_branch["input_shapes"],
        "oracle_stage_shapes": two_branch["oracle_stage_shapes"],
        "actual_stage_shapes": two_branch["actual_stage_shapes"],
        "stage_errors": two_branch["stage_errors"],
        "hidden_q_exact_match": two_branch["hidden_q_exact_match"],
        "hidden_q_mismatch_count": two_branch["hidden_q_mismatch_count"],
        "hidden_q_max_abs_diff": two_branch["hidden_q_max_abs_diff"],
        "max_abs_tolerance": max_abs_tol,
        "mean_abs_tolerance": mean_abs_tol,
        "scale_tolerance": scale_tol,
        "swiglu_limit": float(swiglu_limit),
        "stage_passed": two_branch["stage_passed"],
        "passed": all(bool(cases[case]["passed"]) for case in appendix3_gate_order),
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
        "probe": "svdq_mixed_epilogue_device_probe",
        "model_path": args.model_path,
        "dimensions": dimensions,
        "environment": env,
        "debug_op_name": DEBUG_OP_NAME,
        "stages": [],
        "passed": False,
        "skipped": False,
    }
    if not (env["torch_npu_imported"] and env["npu_available"]):
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = "torch_npu import and an available NPU are required."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        return _finish_probe(
            {"summary_path": path, "passed": False, "skipped": summary["skipped"]},
            2 if args.require_npu else 0,
        )

    if int(env["npu_device_count"]) <= args.device_id:
        summary["skipped"] = not args.require_npu
        summary["skip_reason"] = f"NPU device {args.device_id} is outside available device count."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        return _finish_probe(
            {"summary_path": path, "passed": False, "skipped": summary["skipped"]},
            2 if args.require_npu else 0,
        )

    torch.npu.set_device(args.device_id)
    try:
        registered = _has_registered_mixed_debug_op()
    except Exception as exc:
        summary["failure_reason"] = f"failed to enable custom ops: {type(exc).__name__}: {exc}"
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        return _finish_probe({"summary_path": path, "passed": False, "skipped": False}, 1)
    summary["torch_op_registered"] = registered
    if not registered:
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback is not registered."
        path = _write_summary(args.evidence_dir, args.summary_name, summary)
        return _finish_probe({"summary_path": path, "passed": False, "skipped": False}, 1)

    device = torch.device(f"npu:{args.device_id}")
    stage = _run_probe(
        dimensions=dimensions,
        num_tokens=args.num_tokens,
        seed=args.seed,
        device=device,
        swiglu_limit=args.swiglu_limit,
        max_abs_tol=args.max_abs_tol,
        mean_abs_tol=args.mean_abs_tol,
        scale_tol=args.scale_tol,
    )
    summary["stages"] = [stage]
    summary["passed"] = bool(stage["passed"])
    path = _write_summary(args.evidence_dir, args.summary_name, summary)
    return _finish_probe(
        {"summary_path": path, "passed": summary["passed"], "skipped": False},
        0 if summary["passed"] else 1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
