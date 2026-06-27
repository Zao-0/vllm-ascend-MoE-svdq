/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_W4A8_SVDQ_H
#define DISPATCH_FFN_COMBINE_W4A8_SVDQ_H

#include "kernel_operator.h"
#include "dispatch_ffn_combine_w4_a8_svdq_tiling.h"
#include "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
#include "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_gather_out.h"
#include "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_v2_init_routing_fullload.h"
#include "../../dispatch_ffn_combine_w4_a8/op_kernel/unpermute/moe_token_unpermute.h"
#include "lowrank/svdq_fused_down_up.hpp"

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_FACTOR_COUNT = 5;
constexpr uint32_t SVDQ_BF16_STAGE_COUNT = 7;
constexpr uint32_t SVDQ_MIXED_EPILOGUE_COUNT = 2;
constexpr uint32_t SVDQ_INVALID_ID = 0xffffffffU;
constexpr uint32_t SVDQ_OFFICIAL_W4A8_KERNEL_DISPATCH_FFN_COMBINE = 1;
constexpr uint32_t SVDQ_OFFICIAL_W4A8_AIC_GMM = 2;
constexpr uint32_t SVDQ_OFFICIAL_W4A8_AIV_DEQUANT = 3;
constexpr uint32_t SVDQ_RESIDUAL_WEIGHT1_SLOT = 1;
constexpr uint32_t SVDQ_RESIDUAL_WEIGHT2_SLOT = 2;
constexpr uint32_t SVDQ_RESIDUAL_SCALE1_SLOT = 4;
constexpr uint32_t SVDQ_RESIDUAL_SCALE2_SLOT = 5;
constexpr uint32_t SVDQ_RESIDUAL_BIAS1_SLOT = 6;
constexpr uint32_t SVDQ_RESIDUAL_BIAS2_SLOT = 7;
constexpr uint32_t SVDQ_MIXED_EPILOGUE_VECTOR_TILE = 64;
constexpr uint32_t SVDQ_MIXED_EPILOGUE_UB_BYTES = 196352;

template <class DTYPE_X = bfloat16_t>
__aicore__ inline void svdq_moe_init_routing_v2(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
    const optiling::InnerMoeInitRoutingV2TilingData* tilingData, uint64_t tilingKey)
{
    if (g_coreType == AIC || workspace == nullptr) {
        return;
    }

    if (tilingKey == 20000) {
        TPipe sortPipe;
        MoeInitRoutingQuantV2::MoeV2FullLoad<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    if (tilingKey == 10001 || tilingKey == 10011) {
        TPipe sortPipe;
        MoeInitRoutingQuantV2::MoeV2SortOneCore op;
        op.Init<optiling::InnerMoeInitRoutingV2TilingData>(
            expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (tilingKey == 10002 || tilingKey == 10012) {
        TPipe sortPipe;
        MoeInitRoutingQuantV2::MoeV2SortMultiCore op;
        op.Init<optiling::InnerMoeInitRoutingV2TilingData>(
            expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (tilingKey == 10001 || tilingKey == 10002) {
        if (tilingData->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeInitRoutingQuantV2::MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<optiling::InnerMoeInitRoutingV2TilingData>(
                expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, workspace, tilingData,
                &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
        TPipe srcToDstPipe;
        MoeInitRoutingQuantV2::MoeV2SrcToDstOp srcToDstOp;
        srcToDstOp.Init<optiling::InnerMoeInitRoutingV2TilingData>(
            expandedRowIdx, workspace, tilingData, &srcToDstPipe);
        srcToDstOp.Process();
        srcToDstPipe.Destroy();
    } else if (tilingKey == 10011 || tilingKey == 10012) {
        TPipe expertTokenOutPipe;
        MoeInitRoutingQuantV2::MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<optiling::InnerMoeInitRoutingV2TilingData>(
            expertTokensCountOrCumsum, expertTokensBeforeCapacity, expandedRowIdx, workspace, tilingData,
            &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe srcToDstPipe;
        MoeInitRoutingQuantV2::MoeV2SrcToDstWithCapacity<DTYPE_X, optiling::InnerMoeInitRoutingV2TilingData>
            srcToDstWithCapacityOp;
        srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, workspace, tilingData, &srcToDstPipe);
        srcToDstWithCapacityOp.Process();
        srcToDstPipe.Destroy();
    }

    TPipe gatherPipe;
    MoeInitRoutingQuantV2::MoeV2GatherOut<DTYPE_X> gatherOp;
    gatherOp.Init(x, expandedRowIdx, expandedX, workspace, tilingData, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
}

enum SVDQBF16LowRankStageId : uint32_t {
    SVDQ_BF16_STAGE_ROUTING = 0,
    SVDQ_BF16_STAGE_GATE_UP_L1_GEMM = 1,
    SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT = 2,
    SVDQ_BF16_STAGE_GATE_L2_GEMM = 3,
    SVDQ_BF16_STAGE_UP_L2_GEMM = 4,
    SVDQ_BF16_STAGE_DOWN_L1_GEMM = 5,
    SVDQ_BF16_STAGE_DOWN_L2_GEMM = 6,
};

struct SVDQFactorGM {
    GM_ADDR gateUpSvdqL1;
    GM_ADDR gateSvdqL2;
    GM_ADDR upSvdqL2;
    GM_ADDR downSvdqL1;
    GM_ADDR downSvdqL2;
};

struct SVDQResidualGM {
    GM_ADDR w1;
    GM_ADDR w2;
    GM_ADDR scale1;
    GM_ADDR scale2;
    GM_ADDR bias1;
    GM_ADDR bias2;
};

struct SVDQRuntimeGM {
    GM_ADDR x;
    GM_ADDR expertId;
    GM_ADDR probs;
    GM_ADDR xActiveMask;
    GM_ADDR out;
    GM_ADDR expertTokenNums;
    GM_ADDR workspace;
    SVDQResidualGM residual;
    SVDQFactorGM factors;
};

struct SVDQWorkspaceGM {
    GM_ADDR expandedRowIdx;
    GM_ADDR routedX;
    GM_ADDR xQ;
    GM_ADDR xScale;
    GM_ADDR projection1;
    GM_ADDR accumulator1;
    GM_ADDR hidden;
    GM_ADDR hiddenQ;
    GM_ADDR hiddenScale;
    GM_ADDR projection2;
    GM_ADDR accumulator2;
    GM_ADDR lowRankAccumulator1;
    GM_ADDR lowRankAccumulator2;
    GM_ADDR peerOutput;
    GM_ADDR lowRankRank1;
    GM_ADDR lowRankRank2;
};

struct SVDQBF16StageContract {
    uint32_t stageId;
    uint32_t tilingStageId;
    uint32_t inputRegionId;
    uint32_t outputRegionId;
    uint32_t waitFlagId;
    uint32_t signalFlagId;
};

struct SVDQDispatchRoutingContract {
    uint32_t stageId;
    uint32_t inputRegionId;
    uint32_t routedOutputRegionId;
    uint32_t routeIndexRegionId;
    uint32_t signalQuantFlagId;
    uint32_t signalLowRankFlagId;
    uint32_t signalFinalCombineFlagId;
};

struct SVDQResidualStageContract {
    uint32_t stageId;
    uint32_t tilingStageId;
    uint32_t inputRegionId;
    uint32_t scaleRegionId;
    uint32_t outputRegionId;
    uint32_t residualWeightSlot;
    uint32_t residualScaleSlot;
    uint32_t waitFlagId;
    uint32_t secondWaitFlagId;
    uint32_t signalFlagId;
    uint32_t secondSignalFlagId;
    bool residualOnly;
};

enum SVDQResidualOpKind : uint32_t {
    SVDQ_RESIDUAL_OP_DYNAMIC_QUANT = 1,
    SVDQ_RESIDUAL_OP_W4A8_GMM = 2,
};

struct SVDQResidualExecutionPlan {
    uint32_t stageId;
    uint32_t opKind;
    uint32_t inputRegionId;
    uint32_t activationScaleRegionId;
    uint32_t outputRegionId;
    uint32_t residualWeightSlot;
    uint32_t residualScaleSlot;
    uint32_t residualBiasSlot;
    bool residualOnly;
};

struct SVDQResidualQuantLaunch {
    uint32_t stageId;
    GM_ADDR input;
    GM_ADDR activationScale;
    GM_ADDR output;
    GM_ADDR routeIndex;
    GM_ADDR expertTokenNums;
    GM_ADDR workspace;
    uint32_t m;
    uint32_t k;
    uint32_t scaleElements;
    bool usesRouting;
    bool residualOnly;
};

struct SVDQResidualGmmLaunch {
    uint32_t stageId;
    GM_ADDR input;
    GM_ADDR activationScale;
    GM_ADDR output;
    GM_ADDR weight;
    GM_ADDR weightScale;
    GM_ADDR bias;
    GM_ADDR expertTokenNums;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t listLen;
    uint32_t groupListType;
    uint32_t groupType;
    uint32_t splitItem;
    bool transB;
    bool weightNz;
    bool residualOnly;
};

struct SVDQResidualGmmOfficialBridgeContract {
    uint32_t stageId;
    uint32_t officialKernelId;
    uint32_t officialAicProducerId;
    uint32_t officialAivConsumerId;
    uint32_t inputRegionId;
    uint32_t activationScaleRegionId;
    uint32_t outputRegionId;
    bool requiresPackedW4Weights;
    bool requiresOfficialAicAccumulator;
    bool requiresOfficialC2VHandoff;
    bool requiresOfficialAivDequant;
    bool producesBF16Residual;
};

struct SVDQMixedEpilogueContract {
    uint32_t stageId;
    uint32_t residualRegionId;
    uint32_t lowRankRegionId;
    uint32_t scaleRegionId;
    uint32_t outputRegionId;
    uint32_t waitLowRankFlagId;
    uint32_t waitResidualFlagId;
    uint32_t waitScaleFlagId;
    uint32_t signalPrimaryFlagId;
    uint32_t signalSecondaryFlagId;
    bool appliesSwiGLU;
};

struct SVDQMixedEpilogueLaunch {
    uint32_t stageId;
    GM_ADDR residualAccumulator;
    GM_ADDR lowRankOutput;
    GM_ADDR activationScale;
    GM_ADDR output;
    uint32_t m;
    uint32_t residualColumns;
    uint32_t lowRankColumns;
    uint32_t outputColumns;
    uint32_t gateColumnOffset;
    uint32_t upColumnOffset;
    float swigluLimit;
    bool appliesSwiGLU;
};

struct SVDQFinalCombineContract {
    uint32_t stageId;
    uint32_t inputRegionId;
    uint32_t routeRegionId;
    uint32_t waitOutputFlagId;
    uint32_t waitRouteFlagId;
};

struct SVDQFinalCombineLaunch {
    uint32_t stageId;
    GM_ADDR input;
    GM_ADDR routeIndex;
    GM_ADDR expertId;
    GM_ADDR probs;
    GM_ADDR output;
    uint32_t m;
    uint32_t routedRows;
    uint32_t hiddenSize;
    uint32_t topK;
    uint32_t activeSlots;
};

class DispatchFFNCombineW4A8SVDQ {
public:
    __aicore__ inline DispatchFFNCombineW4A8SVDQ() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR bias1,
        GM_ADDR bias2, GM_ADDR probs, GM_ADDR gateUpSvdqL1, GM_ADDR gateSvdqL2, GM_ADDR upSvdqL2,
        GM_ADDR downSvdqL1, GM_ADDR downSvdqL2, GM_ADDR xActiveMask, GM_ADDR out, GM_ADDR expertTokenNums,
        GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(DispatchFFNCombineW4A8SVDQTilingData);
        GET_TILING_DATA(tilingData, tilingGM);

        runtime_.x = x;
        runtime_.expertId = expertId;
        runtime_.probs = probs;
        runtime_.xActiveMask = xActiveMask;
        runtime_.out = out;
        runtime_.expertTokenNums = expertTokenNums;
        runtime_.workspace = workspaceGM;
        runtime_.residual.w1 = w1;
        runtime_.residual.w2 = w2;
        runtime_.residual.scale1 = scale1;
        runtime_.residual.scale2 = scale2;
        runtime_.residual.bias1 = bias1;
        runtime_.residual.bias2 = bias2;
        runtime_.factors.gateUpSvdqL1 = gateUpSvdqL1;
        runtime_.factors.gateSvdqL2 = gateSvdqL2;
        runtime_.factors.upSvdqL2 = upSvdqL2;
        runtime_.factors.downSvdqL1 = downSvdqL1;
        runtime_.factors.downSvdqL2 = downSvdqL2;
        tilingData_ = tilingData;

        ResolveWorkspaceAddresses();
        pipe_.InitBuffer(mixedEpilogueUb_, SVDQ_MIXED_EPILOGUE_UB_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (!HasCompleteTilingContract()) {
            return;
        }
        if (!RunDispatchRoutingStage()) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_STAGE_BF16_DISPATCH,
            SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_ROUTED_X)) {
            return;
        }
        if (!ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_LOWRANK_1,
            SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_PROJECTION_1)) {
            return;
        }
        if (!RunResidualStage(SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_STAGE_QUANT_1,
            SVDQ_STAGE_W4A8_GEMM_1, SVDQ_REGION_X_Q)) {
            return;
        }
        if (!RunResidualStage(SVDQ_RESIDUAL_STAGE_W4A8_GMM1)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_W4A8_GEMM_1,
            SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_ACCUMULATOR_1)) {
            return;
        }
        if (!RunMixedEpilogueStage(0)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_STAGE_MIXED_EPILOGUE_1,
            SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_HIDDEN)) {
            return;
        }
        if (!ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_DOWN)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_LOWRANK_2,
            SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_PROJECTION_2)) {
            return;
        }
        if (!RunResidualStage(SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_STAGE_QUANT_2,
            SVDQ_STAGE_W4A8_GEMM_2, SVDQ_REGION_HIDDEN_Q)) {
            return;
        }
        if (!RunResidualStage(SVDQ_RESIDUAL_STAGE_W4A8_GMM2)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_W4A8_GEMM_2,
            SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_ACCUMULATOR_2)) {
            return;
        }
        if (!RunMixedEpilogueStage(1)) {
            return;
        }
        if (!SynchronizeStageBoundary(SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE,
            SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_PEER_OUTPUT)) {
            return;
        }
        if (!RunFinalCombine()) {
            return;
        }
    }

    __aicore__ inline bool HasCompleteTilingContract() const
    {
        return tilingData_.info.workspaceBytes > 0 && tilingData_.info.syncFlagCount == SVDQ_SYNC_FLAG_COUNT &&
               tilingData_.info.gateRankOffset == 0 && tilingData_.info.upRankOffset == tilingData_.info.gateRank &&
               HasCompleteSyncFlagTable();
    }

    __aicore__ inline GM_ADDR WorkspaceAddress(uint32_t regionId) const
    {
        return runtime_.workspace + tilingData_.workspaceRegions[regionId].offset;
    }

    __aicore__ inline SVDQWorkspaceRegion WorkspaceRegion(uint32_t regionId) const
    {
        return tilingData_.workspaceRegions[regionId];
    }

    __aicore__ inline SVDQSyncFlag SyncFlag(uint32_t flagId) const
    {
        return tilingData_.syncFlags[flagId];
    }

    __aicore__ inline bool SynchronizeStageBoundary(
        uint32_t flagId, uint32_t producerStage, uint32_t consumerStage, uint32_t workspaceRegionId) const
    {
        if (flagId >= SVDQ_SYNC_FLAG_COUNT) {
            return false;
        }
        SVDQSyncFlag flag = SyncFlag(flagId);
        if (flag.flagId != flagId || flag.producerStage != producerStage ||
            flag.consumerStage != consumerStage || flag.workspaceRegionId != workspaceRegionId ||
            flag.producerSignalIndex != flag.consumerWaitIndex) {
            return false;
        }
        AscendC::SyncAll();
        return true;
    }

    __aicore__ inline bool ValidateSyncFlag(
        uint32_t flagId, uint32_t producerStage, uint32_t consumerStage, uint32_t workspaceRegionId) const
    {
        if (flagId >= SVDQ_SYNC_FLAG_COUNT) {
            return false;
        }
        SVDQSyncFlag flag = SyncFlag(flagId);
        return flag.flagId == flagId && flag.producerStage == producerStage &&
               flag.consumerStage == consumerStage && flag.workspaceRegionId == workspaceRegionId &&
               flag.producerSignalIndex == flagId && flag.consumerWaitIndex == flagId;
    }

    __aicore__ inline bool HasCompleteSyncFlagTable() const
    {
        return ValidateSyncFlag(SVDQ_SYNC_DISPATCH_TO_QUANT_1, SVDQ_STAGE_BF16_DISPATCH,
                   SVDQ_STAGE_QUANT_1, SVDQ_REGION_ROUTED_X) &&
               ValidateSyncFlag(SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_STAGE_BF16_DISPATCH,
                   SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_ROUTED_X) &&
               ValidateSyncFlag(SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_STAGE_QUANT_1,
                   SVDQ_STAGE_W4A8_GEMM_1, SVDQ_REGION_X_Q) &&
               ValidateSyncFlag(SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_QUANT_1,
                   SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_X_SCALE) &&
               ValidateSyncFlag(SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_LOWRANK_1,
                   SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_PROJECTION_1) &&
               ValidateSyncFlag(SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_W4A8_GEMM_1,
                   SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_ACCUMULATOR_1) &&
               ValidateSyncFlag(SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2, SVDQ_STAGE_MIXED_EPILOGUE_1,
                   SVDQ_STAGE_QUANT_2, SVDQ_REGION_HIDDEN) &&
               ValidateSyncFlag(SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_STAGE_MIXED_EPILOGUE_1,
                   SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_HIDDEN) &&
               ValidateSyncFlag(SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_STAGE_QUANT_2,
                   SVDQ_STAGE_W4A8_GEMM_2, SVDQ_REGION_HIDDEN_Q) &&
               ValidateSyncFlag(SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_QUANT_2,
                   SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_HIDDEN_SCALE) &&
               ValidateSyncFlag(SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_LOWRANK_2,
                   SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_PROJECTION_2) &&
               ValidateSyncFlag(SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_W4A8_GEMM_2,
                   SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_ACCUMULATOR_2) &&
               ValidateSyncFlag(SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE,
                   SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_PEER_OUTPUT) &&
               ValidateSyncFlag(SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE, SVDQ_STAGE_BF16_DISPATCH,
                   SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_EXPANDED_ROW_IDX);
    }

    __aicore__ inline SVDQBF16StageShape BF16StageShape(uint32_t stageId) const
    {
        return tilingData_.bf16StageShapes[stageId];
    }

    __aicore__ inline SVDQResidualStageShape ResidualStageShape(uint32_t stageId) const
    {
        return tilingData_.residualStageShapes[stageId];
    }

    __aicore__ inline SVDQResidualQuantShape ResidualQuantShape(uint32_t quantId) const
    {
        return tilingData_.residualQuantShapes[quantId];
    }

    __aicore__ inline SVDQResidualGmmShape ResidualGmmShape(uint32_t gmmId) const
    {
        return tilingData_.residualGmmShapes[gmmId];
    }

    __aicore__ inline SVDQMixedEpilogueShape MixedEpilogueShape(uint32_t epilogueId) const
    {
        return tilingData_.mixedEpilogueShapes[epilogueId];
    }

    __aicore__ inline SVDQFinalCombineShape FinalCombineShape() const
    {
        return tilingData_.finalCombineShape;
    }

    __aicore__ inline SVDQDispatchRoutingTiling DispatchRoutingTiling() const
    {
        return tilingData_.dispatchRouting;
    }

    __aicore__ inline SVDQResidualW4A8BridgeTiling ResidualW4A8BridgeTiling() const
    {
        return tilingData_.residualW4A8Bridge;
    }

    __aicore__ inline GM_ADDR DispatchRoutingTempWorkspace() const
    {
        return runtime_.workspace + tilingData_.info.workspaceBytes;
    }

    __aicore__ inline GM_ADDR DispatchQuantRoutingTempWorkspace() const
    {
        return runtime_.workspace + tilingData_.info.workspaceBytes +
               tilingData_.dispatchRouting.bf16RoutingWorkspaceBytes;
    }

    __aicore__ inline GM_ADDR FactorAddress(uint32_t factorId) const
    {
        switch (factorId) {
            case SVDQ_FACTOR_GATE_UP_L1:
                return runtime_.factors.gateUpSvdqL1;
            case SVDQ_FACTOR_GATE_L2:
                return runtime_.factors.gateSvdqL2;
            case SVDQ_FACTOR_UP_L2:
                return runtime_.factors.upSvdqL2;
            case SVDQ_FACTOR_DOWN_L1:
                return runtime_.factors.downSvdqL1;
            case SVDQ_FACTOR_DOWN_L2:
                return runtime_.factors.downSvdqL2;
            default:
                return nullptr;
        }
    }

    __aicore__ inline GM_ADDR ResidualWeightAddress(uint32_t residualWeightSlot) const
    {
        switch (residualWeightSlot) {
            case SVDQ_RESIDUAL_WEIGHT1_SLOT:
                return runtime_.residual.w1;
            case SVDQ_RESIDUAL_WEIGHT2_SLOT:
                return runtime_.residual.w2;
            default:
                return nullptr;
        }
    }

    __aicore__ inline GM_ADDR ResidualScaleAddress(uint32_t residualScaleSlot) const
    {
        switch (residualScaleSlot) {
            case SVDQ_RESIDUAL_SCALE1_SLOT:
                return runtime_.residual.scale1;
            case SVDQ_RESIDUAL_SCALE2_SLOT:
                return runtime_.residual.scale2;
            default:
                return nullptr;
        }
    }

    __aicore__ inline GM_ADDR ResidualBiasAddress(uint32_t residualBiasSlot) const
    {
        switch (residualBiasSlot) {
            case SVDQ_RESIDUAL_BIAS1_SLOT:
                return runtime_.residual.bias1;
            case SVDQ_RESIDUAL_BIAS2_SLOT:
                return runtime_.residual.bias2;
            default:
                return nullptr;
        }
    }

    __aicore__ inline SVDQFusedDownUpTiling LowRankInvocation(uint32_t invocationId) const
    {
        return tilingData_.lowRankInvocations[invocationId];
    }

    __aicore__ inline SVDQFusedDownUpArgs BuildLowRankArgs(uint32_t invocationId) const
    {
        SVDQFusedDownUpTiling invocation = LowRankInvocation(invocationId);
        return {
            WorkspaceAddress(invocation.inputRegionId),
            FactorAddress(invocation.downFactorId),
            FactorAddress(invocation.upFactorId),
            FactorAddress(invocation.secondUpFactorId),
            WorkspaceAddress(invocation.rankRegionId),
            WorkspaceAddress(invocation.outputRegionId),
            WorkspaceAddress(invocation.accumulatorRegionId),
            runtime_.expertTokenNums,
            tilingData_.info.expertPerRank,
            invocation,
        };
    }

    __aicore__ inline bool LowRankInvocationReady(uint32_t invocationId) const
    {
        SVDQFusedDownUp lowRankOp;
        lowRankOp.Init(BuildLowRankArgs(invocationId));
        return lowRankOp.HasCompleteContract() && lowRankOp.IsImplemented();
    }

    __aicore__ inline bool ExecuteLowRankInvocation(uint32_t invocationId) const
    {
        SVDQFusedDownUp lowRankOp;
        lowRankOp.Init(BuildLowRankArgs(invocationId));
        if (!lowRankOp.HasCompleteContract() || !lowRankOp.IsImplemented()) {
            return false;
        }
        lowRankOp.Process();
        return true;
    }

    __aicore__ inline bool RunBF16LowRankStages() const
    {
        return ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP) &&
               ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_DOWN);
    }

    __aicore__ inline SVDQBF16StageContract BF16StageContract(uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_BF16_STAGE_ROUTING:
                return {stageId, SVDQ_STAGE_BF16_DISPATCH, SVDQ_INVALID_ID, SVDQ_REGION_ROUTED_X,
                    SVDQ_INVALID_ID, SVDQ_SYNC_DISPATCH_TO_LOWRANK_1};
            case SVDQ_BF16_STAGE_GATE_UP_L1_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_ROUTED_X, SVDQ_REGION_LOWRANK_RANK_1,
                    SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_LOWRANK_RANK_1, SVDQ_REGION_LOWRANK_RANK_1,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_GATE_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_LOWRANK_RANK_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_UP_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_LOWRANK_RANK_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_INVALID_ID, SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1};
            case SVDQ_BF16_STAGE_DOWN_L1_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_HIDDEN, SVDQ_REGION_LOWRANK_RANK_2,
                    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_DOWN_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_LOWRANK_RANK_2, SVDQ_REGION_PROJECTION_2,
                    SVDQ_INVALID_ID, SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
        }
    }

    __aicore__ inline SVDQDispatchRoutingContract DispatchRoutingContract() const
    {
        return {SVDQ_STAGE_BF16_DISPATCH, SVDQ_INVALID_ID, SVDQ_REGION_ROUTED_X,
            SVDQ_REGION_EXPANDED_ROW_IDX, SVDQ_SYNC_DISPATCH_TO_QUANT_1,
            SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE};
    }

    __aicore__ inline bool DispatchRoutingReady() const
    {
        SVDQDispatchRoutingContract contract = DispatchRoutingContract();
        SVDQDispatchRoutingTiling routingTiling = DispatchRoutingTiling();
        return runtime_.x != nullptr && runtime_.expertId != nullptr && runtime_.expertTokenNums != nullptr &&
               runtime_.workspace != nullptr && contract.stageId == SVDQ_STAGE_BF16_DISPATCH &&
               contract.routedOutputRegionId == SVDQ_REGION_ROUTED_X &&
               contract.routeIndexRegionId == SVDQ_REGION_EXPANDED_ROW_IDX &&
               contract.signalQuantFlagId == SVDQ_SYNC_DISPATCH_TO_QUANT_1 &&
               contract.signalLowRankFlagId == SVDQ_SYNC_DISPATCH_TO_LOWRANK_1 &&
               contract.signalFinalCombineFlagId == SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE &&
               WorkspaceRegion(contract.routedOutputRegionId).size > 0 &&
               WorkspaceRegion(contract.routeIndexRegionId).size > 0 &&
               routingTiling.bf16RoutingTilingKey != 0 &&
               routingTiling.bf16RoutingWorkspaceBytes > 0 && routingTiling.aivNum > 0;
    }

    __aicore__ inline bool RunDispatchRoutingStage() const
    {
        if (!DispatchRoutingReady()) {
            return false;
        }
        SVDQDispatchRoutingContract contract = DispatchRoutingContract();
        SVDQDispatchRoutingTiling routingTiling = DispatchRoutingTiling();
        svdq_moe_init_routing_v2<bfloat16_t>(runtime_.x, runtime_.expertId,
            WorkspaceAddress(contract.routedOutputRegionId), WorkspaceAddress(contract.routeIndexRegionId),
            runtime_.expertTokenNums, nullptr, DispatchRoutingTempWorkspace(),
            &routingTiling.moeInitRoutingV2TilingData, routingTiling.bf16RoutingTilingKey);
        return true;
    }

    __aicore__ inline SVDQResidualStageContract ResidualStageContract(uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT:
                return {stageId, SVDQ_STAGE_QUANT_1, SVDQ_REGION_ROUTED_X, SVDQ_REGION_X_SCALE, SVDQ_REGION_X_Q,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_SYNC_DISPATCH_TO_QUANT_1, SVDQ_INVALID_ID,
                    SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1, true};
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM1:
                return {stageId, SVDQ_STAGE_W4A8_GEMM_1, SVDQ_REGION_X_Q, SVDQ_REGION_X_SCALE,
                    SVDQ_REGION_ACCUMULATOR_1, SVDQ_RESIDUAL_WEIGHT1_SLOT, SVDQ_RESIDUAL_SCALE1_SLOT,
                    SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_INVALID_ID,
                    SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_INVALID_ID, true};
            case SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN:
                return {stageId, SVDQ_STAGE_QUANT_2, SVDQ_REGION_HIDDEN, SVDQ_REGION_HIDDEN_SCALE,
                    SVDQ_REGION_HIDDEN_Q, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2, SVDQ_INVALID_ID,
                    SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE, true};
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM2:
                return {stageId, SVDQ_STAGE_W4A8_GEMM_2, SVDQ_REGION_HIDDEN_Q, SVDQ_REGION_HIDDEN_SCALE,
                    SVDQ_REGION_ACCUMULATOR_2, SVDQ_RESIDUAL_WEIGHT2_SLOT, SVDQ_RESIDUAL_SCALE2_SLOT,
                    SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_INVALID_ID,
                    SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_INVALID_ID, true};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, false};
        }
    }

    __aicore__ inline SVDQResidualExecutionPlan ResidualExecutionPlan(uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT:
                return {stageId, SVDQ_RESIDUAL_OP_DYNAMIC_QUANT, SVDQ_REGION_ROUTED_X, SVDQ_REGION_X_SCALE,
                    SVDQ_REGION_X_Q, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, true};
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM1:
                return {stageId, SVDQ_RESIDUAL_OP_W4A8_GMM, SVDQ_REGION_X_Q, SVDQ_REGION_X_SCALE,
                    SVDQ_REGION_ACCUMULATOR_1, SVDQ_RESIDUAL_WEIGHT1_SLOT, SVDQ_RESIDUAL_SCALE1_SLOT,
                    SVDQ_RESIDUAL_BIAS1_SLOT, true};
            case SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN:
                return {stageId, SVDQ_RESIDUAL_OP_DYNAMIC_QUANT, SVDQ_REGION_HIDDEN, SVDQ_REGION_HIDDEN_SCALE,
                    SVDQ_REGION_HIDDEN_Q, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, true};
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM2:
                return {stageId, SVDQ_RESIDUAL_OP_W4A8_GMM, SVDQ_REGION_HIDDEN_Q, SVDQ_REGION_HIDDEN_SCALE,
                    SVDQ_REGION_ACCUMULATOR_2, SVDQ_RESIDUAL_WEIGHT2_SLOT, SVDQ_RESIDUAL_SCALE2_SLOT,
                    SVDQ_RESIDUAL_BIAS2_SLOT, true};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, false};
        }
    }

    __aicore__ inline uint32_t ResidualQuantIdForStage(uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT:
                return 0;
            case SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN:
                return 1;
            default:
                return SVDQ_INVALID_ID;
        }
    }

    __aicore__ inline SVDQResidualQuantLaunch BuildResidualQuantLaunch(uint32_t stageId) const
    {
        uint32_t quantId = ResidualQuantIdForStage(stageId);
        if (quantId == SVDQ_INVALID_ID) {
            return {SVDQ_INVALID_ID, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                0, 0, 0, false, false};
        }
        SVDQResidualQuantShape shape = ResidualQuantShape(quantId);
        SVDQDispatchRoutingContract contract = DispatchRoutingContract();
        GM_ADDR routeIndex = shape.usesRouting ? WorkspaceAddress(contract.routeIndexRegionId) : nullptr;
        GM_ADDR workspace = shape.usesRouting ? DispatchQuantRoutingTempWorkspace() : nullptr;
        return {shape.stageId, WorkspaceAddress(shape.inputRegionId),
            WorkspaceAddress(shape.activationScaleRegionId), WorkspaceAddress(shape.outputRegionId),
            routeIndex, runtime_.expertTokenNums, workspace, shape.m, shape.k, shape.scaleElements,
            shape.usesRouting, shape.residualOnly};
    }

    __aicore__ inline uint32_t ResidualGmmIdForStage(uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM1:
                return 0;
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM2:
                return 1;
            default:
                return SVDQ_INVALID_ID;
        }
    }

    __aicore__ inline SVDQResidualGmmLaunch BuildResidualGmmLaunch(uint32_t stageId) const
    {
        uint32_t gmmId = ResidualGmmIdForStage(stageId);
        if (gmmId == SVDQ_INVALID_ID) {
            return {SVDQ_INVALID_ID, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                0, 0, 0, 0, 0, 0, 0, false, false, false};
        }
        SVDQResidualGmmShape shape = ResidualGmmShape(gmmId);
        return {shape.stageId, WorkspaceAddress(shape.inputRegionId), WorkspaceAddress(shape.activationScaleRegionId),
            WorkspaceAddress(shape.outputRegionId), ResidualWeightAddress(shape.residualWeightSlot),
            ResidualScaleAddress(shape.residualScaleSlot), ResidualBiasAddress(shape.residualBiasSlot),
            runtime_.expertTokenNums, shape.m, shape.k, shape.n, shape.listLen, shape.groupListType, shape.groupType,
            shape.splitItem, shape.transB, shape.weightNz, shape.residualOnly};
    }

    __aicore__ inline bool ResidualStageReady(uint32_t stageId) const
    {
        SVDQResidualStageShape shape = ResidualStageShape(stageId);
        SVDQResidualStageContract contract = ResidualStageContract(stageId);
        if (contract.stageId == SVDQ_INVALID_ID || shape.stageId != contract.stageId ||
            shape.inputRegionId != contract.inputRegionId || shape.scaleRegionId != contract.scaleRegionId ||
            shape.outputRegionId != contract.outputRegionId || shape.residualWeightSlot != contract.residualWeightSlot ||
            shape.residualScaleSlot != contract.residualScaleSlot || shape.residualOnly != contract.residualOnly ||
            !shape.residualOnly) {
            return false;
        }

        if (WorkspaceAddress(contract.inputRegionId) == nullptr ||
            WorkspaceAddress(contract.scaleRegionId) == nullptr ||
            WorkspaceAddress(contract.outputRegionId) == nullptr) {
            return false;
        }
        if (contract.residualWeightSlot != SVDQ_INVALID_ID &&
            ResidualWeightAddress(contract.residualWeightSlot) == nullptr) {
            return false;
        }
        if (contract.residualScaleSlot != SVDQ_INVALID_ID &&
            ResidualScaleAddress(contract.residualScaleSlot) == nullptr) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool ResidualExecutionPlanReady(uint32_t stageId) const
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        if (!ResidualStageReady(stageId) || plan.stageId != stageId || !plan.residualOnly ||
            WorkspaceAddress(plan.inputRegionId) == nullptr ||
            WorkspaceAddress(plan.activationScaleRegionId) == nullptr ||
            WorkspaceAddress(plan.outputRegionId) == nullptr) {
            return false;
        }
        if (plan.opKind == SVDQ_RESIDUAL_OP_DYNAMIC_QUANT) {
            return plan.residualWeightSlot == SVDQ_INVALID_ID && plan.residualScaleSlot == SVDQ_INVALID_ID &&
                   plan.residualBiasSlot == SVDQ_INVALID_ID;
        }
        if (plan.opKind == SVDQ_RESIDUAL_OP_W4A8_GMM) {
            return ResidualWeightAddress(plan.residualWeightSlot) != nullptr &&
                   ResidualScaleAddress(plan.residualScaleSlot) != nullptr &&
                   ResidualBiasAddress(plan.residualBiasSlot) != nullptr;
        }
        return false;
    }

    __aicore__ inline bool ResidualQuantLaunchReady(uint32_t stageId) const
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        uint32_t quantId = ResidualQuantIdForStage(stageId);
        if (plan.opKind != SVDQ_RESIDUAL_OP_DYNAMIC_QUANT || quantId == SVDQ_INVALID_ID ||
            !ResidualExecutionPlanReady(stageId)) {
            return false;
        }
        SVDQResidualQuantShape shape = ResidualQuantShape(quantId);
        SVDQResidualQuantLaunch launch = BuildResidualQuantLaunch(stageId);
        if (shape.stageId != stageId || shape.inputRegionId != plan.inputRegionId ||
            shape.activationScaleRegionId != plan.activationScaleRegionId ||
            shape.outputRegionId != plan.outputRegionId || shape.m == 0 || shape.k == 0 ||
            shape.scaleElements != shape.m || !shape.residualOnly || launch.input == nullptr ||
            launch.activationScale == nullptr || launch.output == nullptr ||
            launch.expertTokenNums == nullptr) {
            return false;
        }
        if (shape.usesRouting) {
            SVDQDispatchRoutingTiling routingTiling = DispatchRoutingTiling();
            return launch.routeIndex != nullptr && launch.workspace != nullptr &&
                   routingTiling.initRoutingQuantTilingKey != 0 &&
                   routingTiling.routingWorkspaceBytes > 0;
        }
        return launch.routeIndex == nullptr && launch.workspace == nullptr;
    }

    __aicore__ inline bool ResidualGmmLaunchReady(uint32_t stageId) const
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        uint32_t gmmId = ResidualGmmIdForStage(stageId);
        if (plan.opKind != SVDQ_RESIDUAL_OP_W4A8_GMM || gmmId == SVDQ_INVALID_ID ||
            !ResidualExecutionPlanReady(stageId)) {
            return false;
        }
        SVDQResidualGmmShape shape = ResidualGmmShape(gmmId);
        SVDQResidualGmmLaunch launch = BuildResidualGmmLaunch(stageId);
        return shape.stageId == stageId && shape.inputRegionId == plan.inputRegionId &&
               shape.activationScaleRegionId == plan.activationScaleRegionId &&
               shape.outputRegionId == plan.outputRegionId &&
               shape.residualWeightSlot == plan.residualWeightSlot &&
               shape.residualScaleSlot == plan.residualScaleSlot &&
               shape.residualBiasSlot == plan.residualBiasSlot && shape.m > 0 && shape.k > 0 && shape.n > 0 &&
               shape.listLen == tilingData_.info.expertPerRank && shape.groupListType == 1 &&
               shape.groupType == 0 && shape.splitItem == 2 && !shape.transB && shape.weightNz &&
               shape.residualOnly && launch.input != nullptr && launch.activationScale != nullptr &&
               launch.output != nullptr && launch.weight != nullptr && launch.weightScale != nullptr &&
               launch.bias != nullptr && launch.expertTokenNums != nullptr;
    }

    __aicore__ inline SVDQResidualGmmOfficialBridgeContract ResidualGmmOfficialBridgeContract(
        uint32_t stageId) const
    {
        switch (stageId) {
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM1:
                return {stageId, SVDQ_OFFICIAL_W4A8_KERNEL_DISPATCH_FFN_COMBINE, SVDQ_OFFICIAL_W4A8_AIC_GMM,
                    SVDQ_OFFICIAL_W4A8_AIV_DEQUANT, SVDQ_REGION_X_Q, SVDQ_REGION_X_SCALE,
                    SVDQ_REGION_ACCUMULATOR_1, true, true, true, true, true};
            case SVDQ_RESIDUAL_STAGE_W4A8_GMM2:
                return {stageId, SVDQ_OFFICIAL_W4A8_KERNEL_DISPATCH_FFN_COMBINE, SVDQ_OFFICIAL_W4A8_AIC_GMM,
                    SVDQ_OFFICIAL_W4A8_AIV_DEQUANT, SVDQ_REGION_HIDDEN_Q, SVDQ_REGION_HIDDEN_SCALE,
                    SVDQ_REGION_ACCUMULATOR_2, true, true, true, true, true};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, false, false, false, false, false};
        }
    }

    __aicore__ inline bool ResidualGmmOfficialBridgeReady(uint32_t stageId) const
    {
        if (!ResidualGmmLaunchReady(stageId)) {
            return false;
        }
        SVDQResidualGmmLaunch launch = BuildResidualGmmLaunch(stageId);
        SVDQResidualGmmOfficialBridgeContract bridge = ResidualGmmOfficialBridgeContract(stageId);
        return bridge.stageId == stageId &&
               bridge.officialKernelId == SVDQ_OFFICIAL_W4A8_KERNEL_DISPATCH_FFN_COMBINE &&
               bridge.officialAicProducerId == SVDQ_OFFICIAL_W4A8_AIC_GMM &&
               bridge.officialAivConsumerId == SVDQ_OFFICIAL_W4A8_AIV_DEQUANT &&
               bridge.inputRegionId == ResidualExecutionPlan(stageId).inputRegionId &&
               bridge.activationScaleRegionId == ResidualExecutionPlan(stageId).activationScaleRegionId &&
               bridge.outputRegionId == ResidualExecutionPlan(stageId).outputRegionId &&
               bridge.requiresPackedW4Weights && bridge.requiresOfficialAicAccumulator &&
               bridge.requiresOfficialC2VHandoff && bridge.requiresOfficialAivDequant &&
               bridge.producesBF16Residual && launch.weightNz && launch.residualOnly;
    }

    __aicore__ inline bool ResidualGmmOfficialTilingBridgeReady(uint32_t stageId) const
    {
        if (!ResidualGmmOfficialBridgeReady(stageId)) {
            return false;
        }
        SVDQResidualGmmLaunch launch = BuildResidualGmmLaunch(stageId);
        SVDQResidualW4A8BridgeTiling bridge = ResidualW4A8BridgeTiling();
        DispatchFFNCombineW4A8Info officialInfo = bridge.officialTiling.dispatchFFNCombineW4A8Info;
        auto officialCoc = bridge.officialTiling.cocTiling;
        const bool fullOfficialShapeReady = bridge.officialM == tilingData_.info.m &&
            bridge.officialK == tilingData_.info.hiddenSize &&
            bridge.officialN == tilingData_.info.intermediateSize * 2 &&
            bridge.officialListLen == tilingData_.info.expertPerRank &&
            bridge.officialWorkspaceBytes > 0 && bridge.hostExecutionFailClosed;
        const bool officialInfoReady = officialInfo.M == bridge.officialM &&
            officialInfo.K == bridge.officialK && officialInfo.N == bridge.officialN &&
            officialInfo.expertPerRank == tilingData_.info.expertPerRank &&
            officialInfo.maxOutputSize == tilingData_.info.maxOutputSize &&
            !officialInfo.isTransposeB && officialInfo.isWeightNz &&
            officialInfo.topK == tilingData_.info.topK &&
            officialInfo.worldSize == tilingData_.info.worldSize &&
            officialInfo.listLen == bridge.officialListLen &&
            officialInfo.swigluLimit == tilingData_.info.swigluLimit;
        const bool officialCocReady = officialCoc.m0 == 128 && officialCoc.k0 == 256 &&
            officialCoc.n0 == 256 && officialCoc.swizzleDirect == 1 &&
            officialCoc.swizzleOffset == 7 && officialCoc.ubMoveNum == 16 * 1024 &&
            officialCoc.pValue == 1 && officialCoc.commNpuSplit == tilingData_.info.worldSize &&
            officialCoc.commDataSplit == 1 &&
            officialCoc.lenPerLoop == officialCoc.m0 * officialCoc.n0 / 2 &&
            officialCoc.initRoutingQuantTilingKey == tilingData_.dispatchRouting.initRoutingQuantTilingKey;
        const bool stageShapeMatchesOfficial =
            (stageId == SVDQ_RESIDUAL_STAGE_W4A8_GMM1 &&
                launch.k == bridge.officialK && launch.n == bridge.officialN) ||
            (stageId == SVDQ_RESIDUAL_STAGE_W4A8_GMM2 &&
                launch.k == bridge.officialN / 2 && launch.n == bridge.officialK);
        return fullOfficialShapeReady && officialInfoReady && officialCocReady &&
            stageShapeMatchesOfficial && launch.listLen == bridge.officialListLen;
    }

    __aicore__ inline void CopyInResidualQuantBf16(LocalTensor<bfloat16_t> dst,
        const GlobalTensor<bfloat16_t>& src, uint32_t offset, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst, src[offset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyOutResidualQuantScale(GlobalTensor<float>& dst, uint32_t offset,
        LocalTensor<float> src, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline void CopyOutResidualQuantI8(GlobalTensor<int8_t>& dst, uint32_t offset,
        LocalTensor<int8_t> src, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline void ResidualQuantSyncMte2ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncVToMte3() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncMte3ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncMte3ToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncVToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncVToS() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void ResidualQuantSyncSToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void PackResidualHiddenOfficialI4AIV(const SVDQResidualQuantLaunch& launch,
        GlobalTensor<int8_t>& outputGm, uint32_t row, uint32_t column, LocalTensor<int8_t> hiddenI8,
        LocalTensor<half> quantHalf, LocalTensor<int4b_t> hiddenHighI4, LocalTensor<int4b_t> hiddenLowI4,
        LocalTensor<half> lowHalf, LocalTensor<half> lowHalf2, LocalTensor<int16_t> lowMask)
    {
        constexpr half ONE_SIXTEENTH = static_cast<half>(0.0625f);
        constexpr half MINUS_EIGHT = static_cast<half>(-8.0f);
        const uint32_t vectorTile = SVDQ_MIXED_EPILOGUE_VECTOR_TILE;
        const uint32_t rowOffset = row * launch.k;
        const uint32_t packedOffset = column / 2;
        const uint32_t packedCount = vectorTile / 2;

        Cast(quantHalf, hiddenI8, RoundMode::CAST_NONE, vectorTile);
        PipeBarrier<PIPE_V>();
        Muls(quantHalf, quantHalf, ONE_SIXTEENTH, vectorTile);
        PipeBarrier<PIPE_V>();
        Cast(hiddenHighI4, quantHalf, RoundMode::CAST_FLOOR, vectorTile);
        PipeBarrier<PIPE_V>();
        ResidualQuantSyncVToMte3();
        CopyOutResidualQuantI8(outputGm, rowOffset + packedOffset,
            hiddenHighI4.template ReinterpretCast<int8_t>(), packedCount);
        ResidualQuantSyncMte3ToV();
        ResidualQuantSyncMte3ToMte2();

        And(lowHalf.template ReinterpretCast<int16_t>(), hiddenI8.template ReinterpretCast<int16_t>(), lowMask,
            packedCount, 1, {1, 1, 1, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Cast(lowHalf2.template ReinterpretCast<half>(), lowHalf.template ReinterpretCast<int8_t>(),
            RoundMode::CAST_NONE, vectorTile);
        PipeBarrier<PIPE_V>();
        Adds(quantHalf, lowHalf2, MINUS_EIGHT, vectorTile);
        PipeBarrier<PIPE_V>();
        Cast(hiddenLowI4, quantHalf, RoundMode::CAST_NONE, vectorTile);
        PipeBarrier<PIPE_V>();
        ResidualQuantSyncVToMte3();
        CopyOutResidualQuantI8(outputGm, rowOffset + launch.k / 2 + packedOffset,
            hiddenLowI4.template ReinterpretCast<int8_t>(), packedCount);
        ResidualQuantSyncMte3ToV();
        ResidualQuantSyncMte3ToMte2();
    }

    __aicore__ inline void QuantizeResidualHiddenRowAIV(const SVDQResidualQuantLaunch& launch,
        const GlobalTensor<bfloat16_t>& inputGm, GlobalTensor<int8_t>& outputGm, GlobalTensor<float>& scaleGm,
        uint32_t row)
    {
        LocalTensor<float> ub = mixedEpilogueUb_.Get<float>();
        LocalTensor<float> hiddenFp32 = ub;
        LocalTensor<float> absHidden = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE];
        LocalTensor<float> reduceTmp = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 2];
        LocalTensor<float> scaleLocal = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 3];
        LocalTensor<bfloat16_t> hiddenBf16 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 4].template ReinterpretCast<bfloat16_t>();
        LocalTensor<int8_t> hiddenI8 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 5].template ReinterpretCast<int8_t>();
        LocalTensor<int32_t> quantS32 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 6].template ReinterpretCast<int32_t>();
        LocalTensor<half> quantHalf =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 7].template ReinterpretCast<half>();
        LocalTensor<int4b_t> hiddenHighI4 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 8].template ReinterpretCast<int4b_t>();
        LocalTensor<int4b_t> hiddenLowI4 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 9].template ReinterpretCast<int4b_t>();
        LocalTensor<half> lowHalf =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 10].template ReinterpretCast<half>();
        LocalTensor<half> lowHalf2 =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 11].template ReinterpretCast<half>();
        LocalTensor<int16_t> lowMask =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 12].template ReinterpretCast<int16_t>();

        float maxAbs = 0.0f;
        for (uint32_t column = 0; column < launch.k; column += SVDQ_MIXED_EPILOGUE_VECTOR_TILE) {
            CopyInResidualQuantBf16(hiddenBf16, inputGm, row * launch.k + column,
                SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            ResidualQuantSyncMte2ToV();
            Cast(hiddenFp32, hiddenBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            PipeBarrier<PIPE_V>();
            ResidualQuantSyncVToMte2();
            Abs(absHidden, hiddenFp32, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            PipeBarrier<PIPE_V>();
            ReduceMax(reduceTmp, absHidden, scaleLocal, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            ResidualQuantSyncVToS();
            const float chunkMax = reduceTmp.GetValue(0);
            if (chunkMax > maxAbs) {
                maxAbs = chunkMax;
            }
            ResidualQuantSyncSToV();
        }

        const float scale = maxAbs / 127.0f;
        scaleLocal.SetValue(0, scale);
        ResidualQuantSyncVToMte3();
        CopyOutResidualQuantScale(scaleGm, row, scaleLocal, 1);
        ResidualQuantSyncMte3ToV();
        Duplicate(lowMask, static_cast<int16_t>(0x0F0F), 128);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < launch.k; column += SVDQ_MIXED_EPILOGUE_VECTOR_TILE) {
            if (scale == 0.0f) {
                Duplicate<float>(hiddenFp32, 0.0f, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            } else {
                CopyInResidualQuantBf16(hiddenBf16, inputGm, row * launch.k + column,
                    SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                ResidualQuantSyncMte2ToV();
                Cast(hiddenFp32, hiddenBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                ResidualQuantSyncVToMte2();
                Muls(hiddenFp32, hiddenFp32, 1.0f / scale, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Maxs(hiddenFp32, hiddenFp32, -127.0f, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Mins(hiddenFp32, hiddenFp32, 127.0f, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            }
            PipeBarrier<PIPE_V>();
            Cast(quantS32, hiddenFp32, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0f));
            Cast(quantHalf, quantS32, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            PipeBarrier<PIPE_V>();
            Cast(hiddenI8, quantHalf, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
            PipeBarrier<PIPE_V>();
            PackResidualHiddenOfficialI4AIV(launch, outputGm, row, column, hiddenI8, quantHalf, hiddenHighI4,
                hiddenLowI4, lowHalf, lowHalf2, lowMask);
        }
    }

    __aicore__ inline bool RunResidualHiddenQuantAIV(const SVDQResidualQuantLaunch& launch)
    {
        if (g_coreType == AIC) {
            return true;
        }
        if (launch.usesRouting || launch.scaleElements != launch.m || launch.k == 0 ||
            launch.k % SVDQ_MIXED_EPILOGUE_VECTOR_TILE != 0) {
            return false;
        }

        GlobalTensor<bfloat16_t> inputGm;
        GlobalTensor<int8_t> outputGm;
        GlobalTensor<float> scaleGm;
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.input);
        outputGm.SetGlobalBuffer((__gm__ int8_t*)launch.output);
        scaleGm.SetGlobalBuffer((__gm__ float*)launch.activationScale);

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t row = blockIdx; row < launch.m; row += blockNum) {
            QuantizeResidualHiddenRowAIV(launch, inputGm, outputGm, scaleGm, row);
        }
        return true;
    }

    __aicore__ inline bool RunResidualDynamicQuantStage(uint32_t stageId)
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        if (plan.opKind != SVDQ_RESIDUAL_OP_DYNAMIC_QUANT || !ResidualQuantLaunchReady(stageId)) {
            return false;
        }
        SVDQResidualQuantLaunch launch = BuildResidualQuantLaunch(stageId);
        if (launch.usesRouting) {
            SVDQDispatchRoutingTiling routingTiling = DispatchRoutingTiling();
            moe_init_routing_quant_v2<bfloat16_t>(runtime_.x, runtime_.expertId, nullptr, nullptr,
                launch.output, launch.routeIndex, launch.expertTokenNums, nullptr, launch.activationScale,
                launch.workspace, &routingTiling.moeInitRoutingQuantV2TilingData,
                routingTiling.initRoutingQuantTilingKey);
            return true;
        }
        return RunResidualHiddenQuantAIV(launch);
    }

    __aicore__ inline bool RunResidualGmmStage(uint32_t stageId) const
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        if (plan.opKind != SVDQ_RESIDUAL_OP_W4A8_GMM ||
            !ResidualGmmOfficialTilingBridgeReady(stageId)) {
            return false;
        }
        // Execution remains fail-closed until this bridge directly reuses the official W4A8 AIC/AIV lifecycle.
        return false;
    }

    __aicore__ inline bool RunResidualStage(uint32_t stageId)
    {
        SVDQResidualExecutionPlan plan = ResidualExecutionPlan(stageId);
        if (plan.opKind == SVDQ_RESIDUAL_OP_DYNAMIC_QUANT) {
            return RunResidualDynamicQuantStage(stageId);
        }
        if (plan.opKind == SVDQ_RESIDUAL_OP_W4A8_GMM) {
            return RunResidualGmmStage(stageId);
        }
        return false;
    }

    __aicore__ inline bool RunW4A8ResidualStages()
    {
        for (uint32_t stageId = 0; stageId < SVDQ_RESIDUAL_STAGE_COUNT; ++stageId) {
            if (!RunResidualStage(stageId)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline SVDQMixedEpilogueContract MixedEpilogueContract(uint32_t epilogueId) const
    {
        switch (epilogueId) {
            case 0:
                return {SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_ACCUMULATOR_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_REGION_X_SCALE, SVDQ_REGION_HIDDEN, SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1,
                    SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1,
                    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2, SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, true};
            case 1:
                return {SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_ACCUMULATOR_2, SVDQ_REGION_PROJECTION_2,
                    SVDQ_REGION_HIDDEN_SCALE, SVDQ_REGION_PEER_OUTPUT,
                    SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE,
                    SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE,
                    SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE,
                    SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE, SVDQ_INVALID_ID, false};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, false};
        }
    }

    __aicore__ inline SVDQMixedEpilogueLaunch BuildMixedEpilogueLaunch(uint32_t epilogueId) const
    {
        SVDQMixedEpilogueContract contract = MixedEpilogueContract(epilogueId);
        if (contract.stageId == SVDQ_INVALID_ID || epilogueId >= SVDQ_MIXED_EPILOGUE_COUNT) {
            return {SVDQ_INVALID_ID, nullptr, nullptr, nullptr, nullptr,
                0, 0, 0, 0, SVDQ_INVALID_ID, SVDQ_INVALID_ID, 0.0f, false};
        }
        SVDQMixedEpilogueShape shape = MixedEpilogueShape(epilogueId);
        return {shape.stageId, WorkspaceAddress(shape.residualRegionId),
            WorkspaceAddress(shape.lowRankRegionId), WorkspaceAddress(shape.scaleRegionId),
            WorkspaceAddress(shape.outputRegionId), shape.m, shape.residualColumns, shape.lowRankColumns,
            shape.outputColumns, shape.gateColumnOffset, shape.upColumnOffset, shape.swigluLimit,
            shape.appliesSwiGLU};
    }

    __aicore__ inline bool MixedEpilogueReady(uint32_t epilogueId) const
    {
        SVDQMixedEpilogueContract contract = MixedEpilogueContract(epilogueId);
        if (contract.stageId == SVDQ_INVALID_ID || epilogueId >= SVDQ_MIXED_EPILOGUE_COUNT) {
            return false;
        }
        SVDQMixedEpilogueShape shape = MixedEpilogueShape(epilogueId);
        SVDQMixedEpilogueLaunch launch = BuildMixedEpilogueLaunch(epilogueId);
        if (shape.stageId != contract.stageId || shape.residualRegionId != contract.residualRegionId ||
            shape.lowRankRegionId != contract.lowRankRegionId || shape.scaleRegionId != contract.scaleRegionId ||
            shape.outputRegionId != contract.outputRegionId || shape.appliesSwiGLU != contract.appliesSwiGLU ||
            shape.m == 0 || shape.residualColumns == 0 || shape.lowRankColumns == 0 ||
            shape.outputColumns == 0 || launch.residualAccumulator == nullptr ||
            launch.lowRankOutput == nullptr || launch.activationScale == nullptr || launch.output == nullptr) {
            return false;
        }
        if (shape.appliesSwiGLU) {
            return shape.residualColumns == tilingData_.info.intermediateSize * 2 &&
                   shape.lowRankColumns == tilingData_.info.intermediateSize * 2 &&
                   shape.outputColumns == tilingData_.info.intermediateSize &&
                   shape.gateColumnOffset == 0 &&
                   shape.upColumnOffset == tilingData_.info.intermediateSize;
        }
        return shape.residualColumns == tilingData_.info.hiddenSize &&
               shape.lowRankColumns == tilingData_.info.hiddenSize &&
               shape.outputColumns == tilingData_.info.hiddenSize &&
               shape.gateColumnOffset == SVDQ_INVALID_ID &&
               shape.upColumnOffset == SVDQ_INVALID_ID;
    }

    __aicore__ inline void MixedEpilogueSyncMte2ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void MixedEpilogueSyncVToMte3() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void MixedEpilogueSyncMte3ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void MixedEpilogueSyncMte3ToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopyInMixedEpilogueBf16(LocalTensor<bfloat16_t> dst,
        const GlobalTensor<bfloat16_t>& src, uint32_t offset, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst, src[offset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyOutMixedEpilogueBf16(GlobalTensor<bfloat16_t>& dst, uint32_t offset,
        LocalTensor<bfloat16_t> src, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline bool RunMixedOutputEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)
    {
        if (g_coreType == AIC) {
            return true;
        }
        if (launch.appliesSwiGLU || launch.residualColumns != launch.outputColumns ||
            launch.lowRankColumns != launch.outputColumns || launch.outputColumns == 0 ||
            launch.outputColumns % SVDQ_MIXED_EPILOGUE_VECTOR_TILE != 0) {
            return false;
        }

        GlobalTensor<bfloat16_t> residualGm;
        GlobalTensor<bfloat16_t> lowRankGm;
        GlobalTensor<bfloat16_t> outputGm;
        residualGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.residualAccumulator);
        lowRankGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.lowRankOutput);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.output);

        LocalTensor<float> ub = mixedEpilogueUb_.Get<float>();
        LocalTensor<float> residual = ub;
        LocalTensor<float> lowRank = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE];
        LocalTensor<bfloat16_t> bf16Ub =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 2].template ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> residualBf16 = bf16Ub;
        LocalTensor<bfloat16_t> lowRankBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE];
        LocalTensor<bfloat16_t> outputBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 2];

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t row = blockIdx; row < launch.m; row += blockNum) {
            for (uint32_t column = 0; column < launch.outputColumns;
                 column += SVDQ_MIXED_EPILOGUE_VECTOR_TILE) {
                const uint32_t offset = row * launch.outputColumns + column;
                CopyInMixedEpilogueBf16(residualBf16, residualGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                CopyInMixedEpilogueBf16(lowRankBf16, lowRankGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                Cast(residual, residualBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Cast(lowRank, lowRankBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Add(residual, residual, lowRank, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Cast(outputBf16, residual, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                MixedEpilogueSyncVToMte3();
                CopyOutMixedEpilogueBf16(outputGm, offset, outputBf16, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte3ToV();
                MixedEpilogueSyncMte3ToMte2();
            }
        }
        return true;
    }

    __aicore__ inline bool RunMixedSwiGLUEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)
    {
        if (g_coreType == AIC) {
            return true;
        }
        if (!launch.appliesSwiGLU || launch.residualColumns != launch.outputColumns * 2 ||
            launch.lowRankColumns != launch.outputColumns * 2 || launch.gateColumnOffset != 0 ||
            launch.upColumnOffset != launch.outputColumns ||
            launch.outputColumns % SVDQ_MIXED_EPILOGUE_VECTOR_TILE != 0) {
            return false;
        }

        GlobalTensor<bfloat16_t> residualGm;
        GlobalTensor<bfloat16_t> lowRankGm;
        GlobalTensor<bfloat16_t> outputGm;
        residualGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.residualAccumulator);
        lowRankGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.lowRankOutput);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)launch.output);

        LocalTensor<float> ub = mixedEpilogueUb_.Get<float>();
        LocalTensor<float> gate = ub;
        LocalTensor<float> up = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE];
        LocalTensor<float> tmp = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 2];
        LocalTensor<float> hidden = ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 3];
        LocalTensor<bfloat16_t> bf16Ub =
            ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 4].template ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> residualGateBf16 = bf16Ub;
        LocalTensor<bfloat16_t> lowRankGateBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE];
        LocalTensor<bfloat16_t> residualUpBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 2];
        LocalTensor<bfloat16_t> lowRankUpBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 3];
        LocalTensor<bfloat16_t> hiddenBf16 = bf16Ub[SVDQ_MIXED_EPILOGUE_VECTOR_TILE * 4];

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t row = blockIdx; row < launch.m; row += blockNum) {
            for (uint32_t column = 0; column < launch.outputColumns;
                 column += SVDQ_MIXED_EPILOGUE_VECTOR_TILE) {
                const uint32_t gateOffset = row * launch.residualColumns + launch.gateColumnOffset + column;
                const uint32_t upOffset = row * launch.residualColumns + launch.upColumnOffset + column;
                const uint32_t outputOffset = row * launch.outputColumns + column;

                CopyInMixedEpilogueBf16(residualGateBf16, residualGm, gateOffset,
                    SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                CopyInMixedEpilogueBf16(lowRankGateBf16, lowRankGm, gateOffset,
                    SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                Cast(gate, residualGateBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Cast(tmp, lowRankGateBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Add(gate, gate, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();

                CopyInMixedEpilogueBf16(residualUpBf16, residualGm, upOffset,
                    SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                CopyInMixedEpilogueBf16(lowRankUpBf16, lowRankGm, upOffset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte2ToV();
                Cast(up, residualUpBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Cast(tmp, lowRankUpBf16, RoundMode::CAST_NONE, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Add(up, up, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();

                if (launch.swigluLimit > 0.0f) {
                    Maxs(gate, gate, -launch.swigluLimit, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                    PipeBarrier<PIPE_V>();
                    Mins(gate, gate, launch.swigluLimit, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                    PipeBarrier<PIPE_V>();
                    Maxs(up, up, -launch.swigluLimit, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                    PipeBarrier<PIPE_V>();
                    Mins(up, up, launch.swigluLimit, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                    PipeBarrier<PIPE_V>();
                }

                Muls(tmp, gate, -1.0f, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Exp(tmp, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Adds(tmp, tmp, 1.0f, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Div(hidden, gate, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Mul(hidden, hidden, up, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                Cast(hiddenBf16, hidden, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                PipeBarrier<PIPE_V>();
                MixedEpilogueSyncVToMte3();
                CopyOutMixedEpilogueBf16(outputGm, outputOffset, hiddenBf16,
                    SVDQ_MIXED_EPILOGUE_VECTOR_TILE);
                MixedEpilogueSyncMte3ToV();
                MixedEpilogueSyncMte3ToMte2();
            }
        }
        return true;
    }

    __aicore__ inline bool RunMixedEpilogueStage(uint32_t epilogueId)
    {
        if (!MixedEpilogueReady(epilogueId)) {
            return false;
        }
        SVDQMixedEpilogueLaunch launch = BuildMixedEpilogueLaunch(epilogueId);
        if (launch.appliesSwiGLU) {
            return RunMixedSwiGLUEpilogueAIV(launch);
        }
        return RunMixedOutputEpilogueAIV(launch);
    }

    __aicore__ inline bool RunMixedEpilogueStages()
    {
        for (uint32_t epilogueId = 0; epilogueId < SVDQ_MIXED_EPILOGUE_COUNT; ++epilogueId) {
            if (!RunMixedEpilogueStage(epilogueId)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline SVDQFinalCombineContract FinalCombineContract() const
    {
        return {SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_PEER_OUTPUT, SVDQ_REGION_EXPANDED_ROW_IDX,
            SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE, SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE};
    }

    __aicore__ inline SVDQFinalCombineLaunch BuildFinalCombineLaunch() const
    {
        SVDQFinalCombineShape shape = FinalCombineShape();
        return {shape.stageId, WorkspaceAddress(shape.inputRegionId), WorkspaceAddress(shape.routeRegionId),
            runtime_.expertId, runtime_.probs, runtime_.out, shape.m, shape.routedRows, shape.hiddenSize,
            shape.topK, shape.activeSlots};
    }

    __aicore__ inline bool FinalCombineReady() const
    {
        SVDQFinalCombineContract contract = FinalCombineContract();
        SVDQFinalCombineShape shape = FinalCombineShape();
        SVDQFinalCombineLaunch launch = BuildFinalCombineLaunch();
        SVDQFinalCombineTiling finalCombineTiling = tilingData_.finalCombine;
        return shape.stageId == contract.stageId && shape.inputRegionId == contract.inputRegionId &&
               shape.routeRegionId == contract.routeRegionId && shape.m == tilingData_.info.m &&
               shape.routedRows == tilingData_.info.maxOutputSize && shape.hiddenSize == tilingData_.info.hiddenSize &&
               shape.topK == tilingData_.info.topK && shape.activeSlots == tilingData_.info.m * tilingData_.info.topK &&
               shape.m > 0 && shape.routedRows >= shape.activeSlots && shape.hiddenSize > 0 &&
               shape.topK > 0 && launch.input != nullptr && launch.routeIndex != nullptr &&
               launch.output != nullptr && launch.expertId != nullptr && launch.probs != nullptr &&
               finalCombineTiling.coreNum > 0 &&
               finalCombineTiling.moeTokenUnpermuteTilingData.hidden_size == shape.hiddenSize &&
               finalCombineTiling.moeTokenUnpermuteTilingData.top_k == shape.topK &&
               finalCombineTiling.moeTokenUnpermuteTilingData.num_out_tokens == shape.activeSlots;
    }

    __aicore__ inline bool RunFinalCombine() const
    {
        if (!FinalCombineReady()) {
            return false;
        }
        SVDQFinalCombineLaunch launch = BuildFinalCombineLaunch();
        KernelMoeTokenUnpermute<bfloat16_t, int32_t, float, true> kernelMoeTokenUnpermuteOp;
        kernelMoeTokenUnpermuteOp.Init(launch.input, launch.routeIndex, launch.probs, launch.output,
            &tilingData_.finalCombine.moeTokenUnpermuteTilingData);
        kernelMoeTokenUnpermuteOp.Process();
        return true;
    }

private:
    __aicore__ inline void ResolveWorkspaceAddresses()
    {
        workspace_.expandedRowIdx = WorkspaceAddress(SVDQ_REGION_EXPANDED_ROW_IDX);
        workspace_.routedX = WorkspaceAddress(SVDQ_REGION_ROUTED_X);
        workspace_.xQ = WorkspaceAddress(SVDQ_REGION_X_Q);
        workspace_.xScale = WorkspaceAddress(SVDQ_REGION_X_SCALE);
        workspace_.projection1 = WorkspaceAddress(SVDQ_REGION_PROJECTION_1);
        workspace_.accumulator1 = WorkspaceAddress(SVDQ_REGION_ACCUMULATOR_1);
        workspace_.hidden = WorkspaceAddress(SVDQ_REGION_HIDDEN);
        workspace_.hiddenQ = WorkspaceAddress(SVDQ_REGION_HIDDEN_Q);
        workspace_.hiddenScale = WorkspaceAddress(SVDQ_REGION_HIDDEN_SCALE);
        workspace_.projection2 = WorkspaceAddress(SVDQ_REGION_PROJECTION_2);
        workspace_.accumulator2 = WorkspaceAddress(SVDQ_REGION_ACCUMULATOR_2);
        workspace_.lowRankAccumulator1 = WorkspaceAddress(SVDQ_REGION_LOWRANK_ACCUMULATOR_1);
        workspace_.lowRankAccumulator2 = WorkspaceAddress(SVDQ_REGION_LOWRANK_ACCUMULATOR_2);
        workspace_.peerOutput = WorkspaceAddress(SVDQ_REGION_PEER_OUTPUT);
        workspace_.lowRankRank1 = WorkspaceAddress(SVDQ_REGION_LOWRANK_RANK_1);
        workspace_.lowRankRank2 = WorkspaceAddress(SVDQ_REGION_LOWRANK_RANK_2);
    }

    SVDQRuntimeGM runtime_;
    SVDQWorkspaceGM workspace_;
    DispatchFFNCombineW4A8SVDQTilingData tilingData_;
    TPipe pipe_;
    TBuf<> mixedEpilogueUb_;
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // DISPATCH_FFN_COMBINE_W4A8_SVDQ_H
