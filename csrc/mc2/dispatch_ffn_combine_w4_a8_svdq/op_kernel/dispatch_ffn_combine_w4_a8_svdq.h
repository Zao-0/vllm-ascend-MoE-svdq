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
constexpr uint32_t SVDQ_INVALID_ID = 0xffffffffU;

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
};

struct SVDQBF16StageContract {
    uint32_t stageId;
    uint32_t tilingStageId;
    uint32_t inputRegionId;
    uint32_t outputRegionId;
    uint32_t waitFlagId;
    uint32_t signalFlagId;
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
        if (!RunBF16LowRankStages()) {
            return;
        }
        // The real W4A8 residual stages, mixed epilogues, and final combine are added after the BF16 branch.
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
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_ROUTED_X, SVDQ_REGION_PROJECTION_1,
                    SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_GATE_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_UP_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1,
                    SVDQ_INVALID_ID, SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1};
            case SVDQ_BF16_STAGE_DOWN_L1_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_HIDDEN, SVDQ_REGION_PROJECTION_2,
                    SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_INVALID_ID};
            case SVDQ_BF16_STAGE_DOWN_L2_GEMM:
                return {stageId, SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_PROJECTION_2, SVDQ_REGION_PROJECTION_2,
                    SVDQ_INVALID_ID, SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE};
            default:
                return {SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID, SVDQ_INVALID_ID,
                    SVDQ_INVALID_ID, SVDQ_INVALID_ID};
        }
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
    }

    SVDQRuntimeGM runtime_;
    SVDQWorkspaceGM workspace_;
    DispatchFFNCombineW4A8SVDQTilingData tilingData_;
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // DISPATCH_FFN_COMBINE_W4A8_SVDQ_H
