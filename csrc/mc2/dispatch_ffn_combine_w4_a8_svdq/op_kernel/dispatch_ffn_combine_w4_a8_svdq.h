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
#include "lowrank/svdq_fused_down_up.hpp"

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_FACTOR_COUNT = 5;
constexpr uint32_t SVDQ_BF16_STAGE_COUNT = 7;
constexpr uint32_t SVDQ_MIXED_EPILOGUE_COUNT = 2;
constexpr uint32_t SVDQ_INVALID_ID = 0xffffffffU;
constexpr uint32_t SVDQ_RESIDUAL_WEIGHT1_SLOT = 1;
constexpr uint32_t SVDQ_RESIDUAL_WEIGHT2_SLOT = 2;
constexpr uint32_t SVDQ_RESIDUAL_SCALE1_SLOT = 4;
constexpr uint32_t SVDQ_RESIDUAL_SCALE2_SLOT = 5;

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

struct SVDQFinalCombineContract {
    uint32_t stageId;
    uint32_t inputRegionId;
    uint32_t routeRegionId;
    uint32_t waitOutputFlagId;
    uint32_t waitRouteFlagId;
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
    }

    __aicore__ inline void Process()
    {
        if (!HasCompleteTilingContract()) {
            return;
        }
        if (!RunDispatchRoutingStage()) {
            return;
        }
        if (!RunBF16LowRankStages()) {
            return;
        }
        if (!RunW4A8ResidualStages()) {
            return;
        }
        if (!RunMixedEpilogueStages()) {
            return;
        }
        if (!RunFinalCombine()) {
            return;
        }
    }

    __aicore__ inline bool HasCompleteTilingContract() const
    {
        return tilingData_.info.workspaceBytes > 0 && tilingData_.info.syncFlagCount == SVDQ_SYNC_FLAG_COUNT &&
               tilingData_.info.gateRankOffset == 0 && tilingData_.info.upRankOffset == tilingData_.info.gateRank;
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

    __aicore__ inline SVDQBF16StageShape BF16StageShape(uint32_t stageId) const
    {
        return tilingData_.bf16StageShapes[stageId];
    }

    __aicore__ inline SVDQResidualStageShape ResidualStageShape(uint32_t stageId) const
    {
        return tilingData_.residualStageShapes[stageId];
    }

    __aicore__ inline SVDQDispatchRoutingTiling DispatchRoutingTiling() const
    {
        return tilingData_.dispatchRouting;
    }

    __aicore__ inline GM_ADDR DispatchRoutingTempWorkspace() const
    {
        return runtime_.workspace + tilingData_.info.workspaceBytes;
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
        return runtime_.x != nullptr && runtime_.expertId != nullptr && runtime_.probs != nullptr &&
               runtime_.expertTokenNums != nullptr && WorkspaceAddress(contract.routedOutputRegionId) != nullptr &&
               WorkspaceAddress(contract.routeIndexRegionId) != nullptr &&
               DispatchRoutingTempWorkspace() != nullptr && routingTiling.initRoutingQuantTilingKey != 0 &&
               routingTiling.routingWorkspaceBytes > 0 && routingTiling.aivNum > 0;
    }

    __aicore__ inline bool RunDispatchRoutingStage() const
    {
        if (!DispatchRoutingReady()) {
            return false;
        }
        return false;
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

    __aicore__ inline bool RunW4A8ResidualStages() const
    {
        for (uint32_t stageId = 0; stageId < SVDQ_RESIDUAL_STAGE_COUNT; ++stageId) {
            if (!ResidualStageReady(stageId)) {
                return false;
            }
        }
        return false;
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

    __aicore__ inline bool MixedEpilogueReady(uint32_t epilogueId) const
    {
        SVDQMixedEpilogueContract contract = MixedEpilogueContract(epilogueId);
        if (contract.stageId == SVDQ_INVALID_ID) {
            return false;
        }
        if (WorkspaceAddress(contract.residualRegionId) == nullptr ||
            WorkspaceAddress(contract.lowRankRegionId) == nullptr ||
            WorkspaceAddress(contract.scaleRegionId) == nullptr ||
            WorkspaceAddress(contract.outputRegionId) == nullptr) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool RunMixedEpilogueStages() const
    {
        for (uint32_t epilogueId = 0; epilogueId < SVDQ_MIXED_EPILOGUE_COUNT; ++epilogueId) {
            if (!MixedEpilogueReady(epilogueId)) {
                return false;
            }
        }
        return false;
    }

    __aicore__ inline SVDQFinalCombineContract FinalCombineContract() const
    {
        return {SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_PEER_OUTPUT, SVDQ_REGION_EXPANDED_ROW_IDX,
            SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE, SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE};
    }

    __aicore__ inline bool FinalCombineReady() const
    {
        SVDQFinalCombineContract contract = FinalCombineContract();
        return WorkspaceAddress(contract.inputRegionId) != nullptr &&
               WorkspaceAddress(contract.routeRegionId) != nullptr && runtime_.out != nullptr &&
               runtime_.expertId != nullptr && runtime_.probs != nullptr;
    }

    __aicore__ inline bool RunFinalCombine() const
    {
        if (!FinalCombineReady()) {
            return false;
        }
        return false;
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
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // DISPATCH_FFN_COMBINE_W4A8_SVDQ_H
