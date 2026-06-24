#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Record the official W4A8 AIC/AIV contract that SVDQ must reuse.

This probe is intentionally source/evidence only. It does not claim numerical
acceptance for the production SVDQ operator. Appendix-1 requires production
host tiling to remain fail-closed until isolated W4A8 and mixed-epilogue NPU
validation passes, so this probe treats fail-closed production tiling as the
expected state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = Path("/root/workspace/lza/svdq_clean_evidence")
DEFAULT_SUMMARY_NAME = "phase_an_official_w4a8_aic_aiv_contract_summary.json"

OFFICIAL_MAIN_CPP = Path("csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.cpp")
OFFICIAL_MAIN_H = Path("csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h")
OFFICIAL_KERNEL_HPP = Path("csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp")
OFFICIAL_BLOCK_MMAD = Path("csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp")
OFFICIAL_EPILOGUE_SWIGLU = Path(
    "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
)
OFFICIAL_EPILOGUE_V2 = Path(
    "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp"
)
OFFICIAL_HOST_TILING = Path("csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp")
SVDQ_KERNEL_H = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h")
SVDQ_HOST_TILING = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    return parser.parse_args()


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _contains_all(source: str, required: tuple[str, ...]) -> dict[str, bool]:
    return {item: item in source for item in required}


def _contract(name: str, sources: list[Path], checks: dict[str, bool], **metadata: Any) -> dict[str, Any]:
    return {
        "name": name,
        "sources": [str(path) for path in sources],
        "checks": checks,
        "passed": all(checks.values()),
        **metadata,
    }


def _official_kernel_boundary(main_cpp: str, main_h: str, kernel_hpp: str) -> dict[str, Any]:
    combined = "\n".join((main_cpp, main_h, kernel_hpp))
    checks = _contains_all(
        combined,
        (
            "KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2)",
            "CATLASS_DEVICE void operator()<AscendC::AIC>",
            "GMM1(params);",
            "GMM2(params);",
            "CATLASS_DEVICE void operator()<AscendC::AIV>",
            "DispatchAndCombine(params);",
            "BlockMmad",
            "BlockEpilogue1",
            "BlockEpilogue2",
            "ptrC = params.ptrWorkspace",
            "ptrC2 = params.ptrWorkspace",
            "ptrPerTokenScale = params.ptrWorkspace",
            "ptrPerTokenScale2 = params.ptrWorkspace",
            "ptrA1Int4 = params.ptrWorkspace",
            "ptrA2Int4 = params.ptrWorkspace",
        ),
    )
    return _contract(
        "official_mixed_aic_aiv_kernel_boundary",
        [OFFICIAL_MAIN_CPP, OFFICIAL_MAIN_H, OFFICIAL_KERNEL_HPP],
        checks,
        aic_role="GMM1 and GMM2 grouped W4A8 MMAD producers",
        aiv_role="routing, W4A8 dequant epilogues, SwiGLU/hidden quant, GMM2 epilogue, and final combine",
        task_type="KERNEL_TYPE_MIX_AIC_1_2",
    )


def _official_aic_gmm_contract(kernel_hpp: str, block_mmad: str) -> dict[str, Any]:
    combined = "\n".join((kernel_hpp, block_mmad))
    checks = _contains_all(
        combined,
        (
            "void GMM1(Params const &params)",
            "void GMM2(Params const &params)",
            "gmB1.SetGlobalBuffer",
            "params.ptrB1",
            "gmS.SetGlobalBuffer",
            "params.ptrScale1",
            "gmB2.SetGlobalBuffer",
            "params.ptrB2",
            "gmS2.SetGlobalBuffer",
            "params.ptrScale2",
            "GetTensorAddr<AscendC::int4b_t>",
            "GetTensorAddr<int64_t>",
            "copyGmToL1Scale",
            "ElementAccumulator",
            "TileMmad",
            "CopyL0CToGm",
            "callbackBeforeFixpipe",
            "callbackAfterFixpipe",
            "blockMmad.Finalize(",
        ),
    )
    return _contract(
        "official_aic_w4a8_gmm_contract",
        [OFFICIAL_KERNEL_HPP, OFFICIAL_BLOCK_MMAD],
        checks,
        accumulator_boundary=(
            "AIC BlockMmad accumulates W4A8 MMAD tiles and writes half workspace tiles; "
            "official dequantization is completed by AIV epilogues."
        ),
        gmm1_binding={
            "activation": "workspaceInfo.ptrA1Int4 / gmA1I4",
            "weight": "params.ptrB1 / gmB1",
            "scale": "params.ptrScale1 / gmS",
            "workspace_output": "workspaceInfo.ptrC / gmC",
        },
        gmm2_binding={
            "activation": "workspaceInfo.ptrA2Int4 / gmA2I4",
            "weight": "params.ptrB2 / gmB2",
            "scale": "params.ptrScale2 / gmS2",
            "workspace_output": "workspaceInfo.ptrC2 / gmC2",
        },
    )


def _official_gmm1_epilogue_contract(epilogue: str) -> dict[str, Any]:
    checks = _contains_all(
        epilogue,
        (
            "EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant",
            "AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE",
            "constexpr float DEFAULT_MUL_SCALE = 16.0f",
            "AscendC::Muls(ubCFp32, ubCFp32, DEFAULT_MUL_SCALE",
            "AscendC::Add(ubCFp32, ubCFp32, ubCFp32[blockN]",
            "AscendC::Add(ubCFp32, ubCFp32, ubweighAux",
            "ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx)",
            "AscendC::Muls(ubCFp32, ubCFp32, perTokenScale",
            "AscendC::ClampMax",
            "AscendC::Exp",
            "AscendC::Div",
            "AscendC::Mul(ubCFp32ChunkN",
            "AscendC::ReduceMax<float>",
            "GMubDequantScale / 127.f",
            "AscendC::Muls(ubOutputTmp, ubCFp32ChunkN, 127.f / GMubDequantScale",
            "AscendC::Cast(ubQuantS32",
            "Cast(xHighI4Tensor",
            "Cast(xLowI4Tensor",
            "copyUbToGmDequantScale",
        ),
    )
    return _contract(
        "official_aiv_gmm1_dequant_swiglu_hidden_quant_contract",
        [OFFICIAL_EPILOGUE_SWIGLU],
        checks,
        dequant_formula="float(hi_half) * 16 + float(lo_half) + scale_bias, then multiply per-token scale",
        hidden_quant="AIV computes SiLU(gate) * up, derives per-row max/127 scale, and packs hidden int4.",
        debug_boundary="W4A8_DEBUG optionally stores post-dequant pre-SwiGLU GMM1 FP32 to gmGMM1.",
    )


def _official_gmm2_epilogue_contract(epilogue: str) -> dict[str, Any]:
    checks = _contains_all(
        epilogue,
        (
            "EpilogueAtlasA2W4A8PostPerTokenDequantV2",
            "Cast<float, ElementC, false>(ubFp32, ubCH",
            "Cast<float, ElementC, false>(ubFp32L, ubCL",
            "constexpr float DEFAULT_MUL_SCALE = 16.0f",
            "AscendC::Muls(ubFp32, ubFp32, DEFAULT_MUL_SCALE",
            "AscendC::Add(ubFp32, ubFp32, ubFp32L",
            "AscendC::Add(ubFp32[i * n0], ubFp32[i * n0], ubweighAux",
            "float scale = gmPerTokenScale(gmScaleOffset + row)",
            "Muls<float, false>(ubFp32[n0 * row], ubFp32[n0 * row], scale",
            "copyUbToGmGMM2",
            "AscendC::Cast<ElementD, float, false>(ubD, ubFp32, AscendC::RoundMode::CAST_RINT",
            "copyUbToGmD",
            "params.shmem(params.offsetD, dstEpIdx)",
            "tokenPerExpert(tokenPerExpertLayout",
        ),
    )
    return _contract(
        "official_aiv_gmm2_dequant_final_output_contract",
        [OFFICIAL_EPILOGUE_V2],
        checks,
        dequant_formula="float(hi_half) * 16 + float(lo_half) + scale_bias, then multiply hidden per-token scale",
        output_boundary="AIV casts dequantized FP32 output to ElementD and scatters to peer output memory.",
        debug_boundary="W4A8_DEBUG optionally stores post-dequant GMM2 FP32 to gmGMM2.",
    )


def _svdq_fail_closed_contract(host_tiling: str, kernel_h: str) -> dict[str, Any]:
    checks = _contains_all(
        host_tiling,
        (
            "DispatchFFNCombineW4A8SVDQ production tiling is fail-closed",
            "official W4A8 AIC/AIV",
            "return ge::GRAPH_FAILED;",
            "IMPL_OP_OPTILING(DispatchFFNCombineW4A8SVDQ)",
            "IMPL_OP_OPTILING(SVDQLowRankDebugReadback)",
        ),
    )
    checks.update(
        _contains_all(
            kernel_h,
            (
                "__aicore__ inline bool DispatchRoutingReady() const",
                "__aicore__ inline bool RunDispatchRoutingStage() const",
                "return false;",
                "RunResidualScalarDynamicQuantStage",
                "RunResidualPackedW4A8ScalarGmmStage",
                "RunMixedEpilogueStage",
                "RunFinalCombine",
            ),
        )
    )
    fail_closed_idx = host_tiling.index("DispatchFFNCombineW4A8SVDQ production tiling is fail-closed")
    failed_return_idx = host_tiling.index("return ge::GRAPH_FAILED;", fail_closed_idx)
    checks["fail_closed_before_production_validation"] = failed_return_idx < host_tiling.index(
        "DispatchFFNCombineW4A8SVDQCheckAttrAndSetTiling",
        failed_return_idx,
    )
    return _contract(
        "svdq_production_fail_closed_and_scalar_placeholders_disabled",
        [SVDQ_HOST_TILING, SVDQ_KERNEL_H],
        checks,
        production_host_tiling_fail_closed=True,
        debug_lowrank_tiling_enabled=True,
        scalar_placeholder_status=(
            "Scalar residual/mixed/final helper names remain in source but production tiling and routing "
            "are fail-closed. They are not acceptance evidence and must be removed or replaced by official "
            "AIC/AIV ports before production can open."
        ),
    )


def _official_host_tiling_contract(host_tiling: str) -> dict[str, Any]:
    checks = _contains_all(
        host_tiling,
        (
            "MoeInitRoutingQuantV2TilingBase",
            "workspaceSize_",
            "SetBlockDim",
            "SetTilingKey",
            "CalcTschBlockDim",
            "GetGroupRankSize",
            "GetWorkspaceSizes",
        ),
    )
    return _contract(
        "official_host_tiling_contract",
        [OFFICIAL_HOST_TILING],
        checks,
        scope="Official W4A8 tiling, routing workspace, HCCL/group metadata, block dim, and tiling key source.",
    )


def build_summary() -> dict[str, Any]:
    main_cpp = _read(OFFICIAL_MAIN_CPP)
    main_h = _read(OFFICIAL_MAIN_H)
    kernel_hpp = _read(OFFICIAL_KERNEL_HPP)
    block_mmad = _read(OFFICIAL_BLOCK_MMAD)
    epilogue_swiglu = _read(OFFICIAL_EPILOGUE_SWIGLU)
    epilogue_v2 = _read(OFFICIAL_EPILOGUE_V2)
    official_host_tiling = _read(OFFICIAL_HOST_TILING)
    svdq_kernel_h = _read(SVDQ_KERNEL_H)
    svdq_host_tiling = _read(SVDQ_HOST_TILING)

    contracts = [
        _official_host_tiling_contract(official_host_tiling),
        _official_kernel_boundary(main_cpp, main_h, kernel_hpp),
        _official_aic_gmm_contract(kernel_hpp, block_mmad),
        _official_gmm1_epilogue_contract(epilogue_swiglu),
        _official_gmm2_epilogue_contract(epilogue_v2),
        _svdq_fail_closed_contract(svdq_host_tiling, svdq_kernel_h),
    ]
    return {
        "probe": "svdq_w4a8_residual_gmm_contract_probe",
        "passed": all(contract["passed"] for contract in contracts),
        "production_host_tiling_expected": "fail_closed",
        "numerical_acceptance_claimed": False,
        "official_aic_aiv_boundary": {
            "aic": "official GMM1/GMM2 BlockMmad W4A8 MMAD producers",
            "aiv": (
                "official routing, dequant, scale-bias add, activation-scale multiply, "
                "SwiGLU, hidden quant, GMM2 output"
            ),
            "svdq_next_required_port": (
                "reuse or port official AIC GMM producers and replace scalar placeholders with AIV mixed epilogues"
            ),
        },
        "contracts": contracts,
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
