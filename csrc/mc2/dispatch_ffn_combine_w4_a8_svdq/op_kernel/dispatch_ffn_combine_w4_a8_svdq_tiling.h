/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_DISPATCH_FFN_COMBINE_W4A8_SVDQ_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_W4A8_SVDQ_TILING_H

#include <cstdint>

#include "lowrank/svdq_fused_down_up_tiling.h"

constexpr uint32_t SVDQ_WORKSPACE_REGION_COUNT = 14;
constexpr uint32_t SVDQ_SYNC_FLAG_COUNT = 14;
constexpr uint32_t SVDQ_BF16_STAGE_COUNT = 7;
constexpr uint32_t SVDQ_RESIDUAL_STAGE_COUNT = 4;
constexpr uint32_t SVDQ_INVALID_ID = 0xffffffffU;

enum SVDQWorkspaceRegionId : uint32_t {
    SVDQ_REGION_EXPANDED_ROW_IDX = 0,
    SVDQ_REGION_ROUTED_X = 1,
    SVDQ_REGION_X_Q = 2,
    SVDQ_REGION_X_SCALE = 3,
    SVDQ_REGION_PROJECTION_1 = 4,
    SVDQ_REGION_ACCUMULATOR_1 = 5,
    SVDQ_REGION_HIDDEN = 6,
    SVDQ_REGION_HIDDEN_Q = 7,
    SVDQ_REGION_HIDDEN_SCALE = 8,
    SVDQ_REGION_PROJECTION_2 = 9,
    SVDQ_REGION_ACCUMULATOR_2 = 10,
    SVDQ_REGION_LOWRANK_ACCUMULATOR_1 = 11,
    SVDQ_REGION_LOWRANK_ACCUMULATOR_2 = 12,
    SVDQ_REGION_PEER_OUTPUT = 13,
};

enum SVDQWorkspaceDType : uint32_t {
    SVDQ_DTYPE_INT8 = 1,
    SVDQ_DTYPE_INT32 = 2,
    SVDQ_DTYPE_BF16 = 3,
    SVDQ_DTYPE_FP32 = 4,
};

enum SVDQWorkspaceStage : uint32_t {
    SVDQ_STAGE_BF16_DISPATCH = 1,
    SVDQ_STAGE_QUANT_1 = 2,
    SVDQ_STAGE_LOWRANK_1 = 3,
    SVDQ_STAGE_W4A8_GEMM_1 = 4,
    SVDQ_STAGE_MIXED_EPILOGUE_1 = 5,
    SVDQ_STAGE_QUANT_2 = 6,
    SVDQ_STAGE_LOWRANK_2 = 7,
    SVDQ_STAGE_W4A8_GEMM_2 = 8,
    SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE = 9,
    SVDQ_STAGE_UNPERMUTE_COMBINE = 10,
};

enum SVDQFactorId : uint32_t {
    SVDQ_FACTOR_GATE_UP_L1 = 0,
    SVDQ_FACTOR_GATE_L2 = 1,
    SVDQ_FACTOR_UP_L2 = 2,
    SVDQ_FACTOR_DOWN_L1 = 3,
    SVDQ_FACTOR_DOWN_L2 = 4,
};

enum SVDQBF16LowRankStageId : uint32_t {
    SVDQ_BF16_STAGE_ROUTING = 0,
    SVDQ_BF16_STAGE_GATE_UP_L1_GEMM = 1,
    SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT = 2,
    SVDQ_BF16_STAGE_GATE_L2_GEMM = 3,
    SVDQ_BF16_STAGE_UP_L2_GEMM = 4,
    SVDQ_BF16_STAGE_DOWN_L1_GEMM = 5,
    SVDQ_BF16_STAGE_DOWN_L2_GEMM = 6,
};

enum SVDQResidualStageId : uint32_t {
    SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT = 0,
    SVDQ_RESIDUAL_STAGE_W4A8_GMM1 = 1,
    SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN = 2,
    SVDQ_RESIDUAL_STAGE_W4A8_GMM2 = 3,
};

struct SVDQWorkspaceRegion {
    uint64_t offset;
    uint64_t size;
    uint32_t dtype;
    uint32_t producerStage;
    uint32_t consumerStage;
    uint32_t lifetimeId;
};

enum SVDQSyncFlagId : uint32_t {
    SVDQ_SYNC_DISPATCH_TO_QUANT_1 = 0,
    SVDQ_SYNC_DISPATCH_TO_LOWRANK_1 = 1,
    SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1 = 2,
    SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1 = 3,
    SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1 = 4,
    SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1 = 5,
    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2 = 6,
    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2 = 7,
    SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2 = 8,
    SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE = 9,
    SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE = 10,
    SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE = 11,
    SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE = 12,
    SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE = 13,
};

struct SVDQSyncFlag {
    uint32_t flagId;
    uint32_t producerStage;
    uint32_t consumerStage;
    uint32_t workspaceRegionId;
    uint32_t producerSignalIndex;
    uint32_t consumerWaitIndex;
};

struct SVDQBF16StageShape {
    uint32_t stageId;
    uint32_t factorId;
    uint32_t inputRegionId;
    uint32_t outputRegionId;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t inputColumnOffset;
    uint32_t outputColumnOffset;
    uint32_t factorColumnOffset;
};

struct SVDQResidualStageShape {
    uint32_t stageId;
    uint32_t inputRegionId;
    uint32_t scaleRegionId;
    uint32_t outputRegionId;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t residualWeightSlot;
    uint32_t residualScaleSlot;
    bool residualOnly;
};

struct DispatchFFNCombineW4A8SVDQInfo {
    uint32_t m;
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t expertPerRank;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t maxOutputSize;
    uint32_t gateRank;
    uint32_t upRank;
    uint32_t downRank;
    uint32_t gateRankOffset;
    uint32_t upRankOffset;
    uint32_t lowRankCoreCount;
    uint64_t workspaceBytes;
    uint32_t syncFlagCount;
    float swigluLimit;
};

struct DispatchFFNCombineW4A8SVDQTilingData {
    DispatchFFNCombineW4A8SVDQInfo info;
    SVDQWorkspaceRegion workspaceRegions[SVDQ_WORKSPACE_REGION_COUNT];
    SVDQSyncFlag syncFlags[SVDQ_SYNC_FLAG_COUNT];
    SVDQBF16StageShape bf16StageShapes[SVDQ_BF16_STAGE_COUNT];
    SVDQResidualStageShape residualStageShapes[SVDQ_RESIDUAL_STAGE_COUNT];
    DispatchFFNCombineW4A8SVDQImpl::SVDQFusedDownUpTiling
        lowRankInvocations[DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_COUNT];
};

#endif  // ASCENDC_DISPATCH_FFN_COMBINE_W4A8_SVDQ_TILING_H
