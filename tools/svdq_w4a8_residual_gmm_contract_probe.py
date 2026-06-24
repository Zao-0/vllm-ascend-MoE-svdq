#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Record the W4A8 residual GMM contract used by SVDQ fused MoE.

This probe intentionally stays at the contract boundary. It cross-checks the
production SVDQ tiling payload against the official W4A8 grouped-matmul call
surface that residual stages must reuse:

* GMM1 consumes routed int8 activations, per-token x scale, W4 packed w1,
  packed weight scale, and scale-bias assist matrix.
* GMM2 consumes hidden int8 activations, per-token hidden scale, W4 packed w2,
  packed weight scale, and scale-bias assist matrix.

It does not enable the production fused SVDQ op. The host tiler must remain
fail-closed until the actual residual kernels and mixed epilogues are
implemented and numerically validated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = Path("/root/workspace/lza/svdq_clean_evidence")
DEFAULT_SUMMARY_NAME = "phase_f_residual_gmm_contract_probe_summary.json"

MOE_MLP = Path("vllm_ascend/ops/fused_moe/moe_mlp.py")
DEVICE_OP = Path("vllm_ascend/device/device_op.py")
HOST_TILING = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp")
KERNEL_TILING = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    return parser.parse_args()


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _contains_all(source: str, required: tuple[str, ...]) -> dict[str, bool]:
    return {item: item in source for item in required}


def _official_w4a8_gmm1_contract(moe_mlp: str) -> dict[str, Any]:
    required = (
        "hidden_states = torch_npu.npu_grouped_matmul(",
        "x=[hidden_states]",
        "weight=w1",
        "scale=w1_scale",
        "bias=bias1",
        "per_token_scale=[pertoken_scale]",
        "split_item=2",
        "group_list_type=group_list_type",
        "group_type=0",
        "group_list=group_list",
        "output_dtype=_output_dtype",
    )
    checks = _contains_all(moe_mlp, required)
    return {
        "stage": "official_w4a8_gmm1",
        "source": str(MOE_MLP),
        "input_activation": "routed int8 activations from npu_dynamic_quant",
        "activation_scale": "pertoken_scale",
        "weight": "w1 / w13_weight packed W4 operator tensor",
        "weight_scale": "w1_scale / w13_weight_scale",
        "scale_bias": "bias1 / w13_scale_bias",
        "output": "gate_up residual projection before SwiGLU or mixed epilogue",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _official_w4a8_gmm2_contract(device_op: str) -> dict[str, Any]:
    required = (
        "def npu_grouped_matmul_gmm2",
        "return torch_npu.npu_grouped_matmul(",
        "x=[hidden_states]",
        "weight=weight",
        "scale=weight_scale",
        "bias=bias",
        "per_token_scale=[per_token_scale]",
        "split_item=2",
        "group_list_type=group_list_type",
        "group_type=0",
        "group_list=group_list",
        "output_dtype=fallback_output_dtype",
    )
    checks = _contains_all(device_op, required)
    return {
        "stage": "official_w4a8_gmm2",
        "source": str(DEVICE_OP),
        "input_activation": "hidden int8 activations from post-SwiGLU npu_dynamic_quant",
        "activation_scale": "per_token_scale / hidden_scale",
        "weight": "w2 / w2_weight packed W4 operator tensor",
        "weight_scale": "w2_scale / w2_weight_scale",
        "scale_bias": "bias / w2_scale_bias",
        "output": "down residual projection before output mixed epilogue",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _svdq_tiling_contract(host_tiling: str, kernel_tiling: str) -> dict[str, Any]:
    required = (
        "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
        "SVDQ_REGION_X_Q",
        "SVDQ_REGION_X_SCALE",
        "SVDQ_REGION_ACCUMULATOR_1",
        "SVDQ_REGION_HIDDEN_Q",
        "SVDQ_REGION_HIDDEN_SCALE",
        "SVDQ_REGION_ACCUMULATOR_2",
        "scale1Slot",
        "scale2Slot",
        "BuildResidualStageShapeTable",
        "GRAPH_FAILED",
        "AscendC kernel is not implemented yet",
    )
    combined = f"{host_tiling}\n{kernel_tiling}"
    checks = _contains_all(combined, required)
    return {
        "stage": "svdq_residual_gmm_tiling",
        "sources": [str(HOST_TILING), str(KERNEL_TILING)],
        "gmm1_binding": {
            "input": "SVDQ_REGION_X_Q",
            "activation_scale": "SVDQ_REGION_X_SCALE",
            "weight_slot": "w1",
            "weight_scale_slot": "scale1Slot",
            "output": "SVDQ_REGION_ACCUMULATOR_1",
        },
        "gmm2_binding": {
            "input": "SVDQ_REGION_HIDDEN_Q",
            "activation_scale": "SVDQ_REGION_HIDDEN_SCALE",
            "weight_slot": "w2",
            "weight_scale_slot": "scale2Slot",
            "output": "SVDQ_REGION_ACCUMULATOR_2",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_summary() -> dict[str, Any]:
    moe_mlp = _read(MOE_MLP)
    device_op = _read(DEVICE_OP)
    host_tiling = _read(HOST_TILING)
    kernel_tiling = _read(KERNEL_TILING)
    stages = [
        _official_w4a8_gmm1_contract(moe_mlp),
        _official_w4a8_gmm2_contract(device_op),
        _svdq_tiling_contract(host_tiling, kernel_tiling),
    ]
    return {
        "probe": "svdq_w4a8_residual_gmm_contract_probe",
        "passed": all(stage["passed"] for stage in stages),
        "production_fail_closed": (
            "GRAPH_FAILED" in host_tiling and "AscendC kernel is not implemented yet" in host_tiling
        ),
        "stages": stages,
    }


def main() -> int:
    args = _parse_args()
    summary = build_summary()
    os.makedirs(args.evidence_dir, exist_ok=True)
    output_path = args.evidence_dir / args.summary_name
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(output_path), "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
