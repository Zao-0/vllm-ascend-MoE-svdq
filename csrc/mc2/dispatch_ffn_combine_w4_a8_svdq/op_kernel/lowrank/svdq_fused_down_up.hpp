/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SVDQ_FUSED_DOWN_UP_HPP
#define SVDQ_FUSED_DOWN_UP_HPP

#include "kernel_operator.h"
#include "../dispatch_ffn_combine_w4_a8_svdq_tiling.h"
#include "svdq_fused_down_up_tiling.h"

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_LOWRANK_BF16_BYTES = 2;

enum SVDQLowRankStageKind : uint32_t {
    SVDQ_LOWRANK_STAGE_DOWN_PROJECT = 0,
    SVDQ_LOWRANK_STAGE_UP_PROJECT = 1,
    SVDQ_LOWRANK_STAGE_SECOND_UP_PROJECT = 2,
};

struct SVDQLowRankStagePlan {
    uint32_t stageKind;
    uint32_t factorId;
    GM_ADDR factor;
    uint32_t inputColumns;
    uint32_t outputColumns;
    uint32_t inputStrideColumns;
    uint32_t outputStrideColumns;
    uint32_t inputColumnOffset;
    uint32_t outputColumnOffset;
    uint32_t factorColumnOffset;
    bool writesGlobalOutput;

    __aicore__ inline bool HasCompleteContract() const
    {
        return factor != nullptr && factorId != SVDQ_INVALID_ID && inputColumns > 0 && outputColumns > 0;
    }
};

struct SVDQLowRankExpertPlan {
    SVDQLowRankStagePlan stage;
    uint32_t expertId;
    uint32_t tokenStart;
    uint32_t tokenCount;
    GM_ADDR input;
    GM_ADDR factor;
    GM_ADDR output;

    __aicore__ inline bool HasWork() const
    {
        return tokenCount > 0 && input != nullptr && factor != nullptr && output != nullptr &&
               stage.HasCompleteContract();
    }
};

struct SVDQFusedDownUpArgs {
    GM_ADDR input;
    GM_ADDR downFactor;
    GM_ADDR upFactor;
    GM_ADDR secondUpFactor;
    GM_ADDR output;
    GM_ADDR expertTokenNums;
    uint32_t expertPerRank;
    SVDQFusedDownUpTiling tiling;
};

class SVDQFusedDownUp {
public:
    __aicore__ inline SVDQFusedDownUp() {}

    __aicore__ inline void Init(const SVDQFusedDownUpArgs& args)
    {
        args_ = args;
    }

    __aicore__ inline bool HasCompleteContract() const
    {
        return args_.input != nullptr && args_.downFactor != nullptr && args_.upFactor != nullptr &&
               args_.output != nullptr && args_.expertTokenNums != nullptr && args_.tiling.m > 0 &&
               args_.tiling.inputColumns > 0 && args_.tiling.rankColumns > 0 &&
               args_.tiling.outputColumns > 0 && args_.tiling.downFactorId != SVDQ_INVALID_ID &&
               args_.tiling.upFactorId != SVDQ_INVALID_ID &&
               args_.tiling.invocationId < SVDQ_LOWRANK_INVOCATION_COUNT &&
               args_.expertPerRank > 0 &&
               (args_.tiling.secondUpFactorId == SVDQ_INVALID_ID || HasCompleteSecondUpContract());
    }

    __aicore__ inline bool HasIndependentSecondUp() const
    {
        return args_.tiling.secondUpFactorId != SVDQ_INVALID_ID && args_.secondUpFactor != nullptr &&
               args_.tiling.secondRankColumns > 0;
    }

    __aicore__ inline uint32_t TotalRankColumns() const
    {
        return args_.tiling.rankColumns + args_.tiling.secondRankColumns;
    }

    __aicore__ inline uint32_t PrimaryOutputColumns() const
    {
        if (!HasIndependentSecondUp()) {
            return args_.tiling.outputColumns;
        }
        return args_.tiling.secondOutputColumnOffset - args_.tiling.outputColumnOffset;
    }

    __aicore__ inline uint32_t StageCount() const
    {
        return HasIndependentSecondUp() ? 3 : 2;
    }

    __aicore__ inline uint32_t ExpertCount() const
    {
        return args_.expertPerRank;
    }

    __aicore__ inline uint32_t ExpertTokenCount(uint32_t expertId) const
    {
        AscendC::GlobalTensor<int32_t> expertTokenNums;
        expertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_.expertTokenNums));
        const int32_t tokenCount = expertTokenNums(expertId);
        return tokenCount > 0 ? static_cast<uint32_t>(tokenCount) : 0;
    }

    __aicore__ inline uint32_t ExpertTokenStart(uint32_t expertId) const
    {
        uint32_t tokenStart = 0;
        for (uint32_t currentExpertId = 0; currentExpertId < expertId; ++currentExpertId) {
            tokenStart += ExpertTokenCount(currentExpertId);
        }
        return tokenStart;
    }

    __aicore__ inline SVDQLowRankStagePlan StagePlan(uint32_t stageIndex) const
    {
        if (stageIndex == 0) {
            return BuildDownStagePlan();
        }
        if (stageIndex == 1) {
            return BuildPrimaryUpStagePlan();
        }
        return BuildSecondUpStagePlan();
    }

    __aicore__ inline SVDQLowRankExpertPlan ExpertStagePlan(uint32_t stageIndex, uint32_t expertId) const
    {
        const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
        const uint32_t tokenStart = ExpertTokenStart(expertId);
        const uint32_t tokenCount = ExpertTokenCount(expertId);
        GM_ADDR inputBase = args_.input;
        if (stageIndex != 0) {
            inputBase = args_.output;
        }
        return SVDQLowRankExpertPlan{
            stage,
            expertId,
            tokenStart,
            tokenCount,
            MatrixAddress(inputBase, tokenStart, stage.inputStrideColumns, stage.inputColumnOffset),
            FactorAddress(stage, expertId),
            MatrixAddress(args_.output, tokenStart, stage.outputStrideColumns, stage.outputColumnOffset),
        };
    }

    __aicore__ inline bool IsImplemented() const
    {
        return false;
    }

    __aicore__ inline void Process()
    {
        if (!HasCompleteContract()) {
            return;
        }
        for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex) {
            const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
            if (!stage.HasCompleteContract()) {
                return;
            }
            for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId) {
                const SVDQLowRankExpertPlan expertPlan = ExpertStagePlan(stageIndex, expertId);
                if (expertPlan.tokenCount > 0 && !expertPlan.HasWork()) {
                    return;
                }
            }
        }
        // The fused AIC math body will keep the low-rank bottleneck on-chip:
        // input BF16 -> down factor GEMM -> rank tile -> up factor GEMM -> projection BF16 GM.
    }

private:
    __aicore__ inline bool HasCompleteSecondUpContract() const
    {
        return args_.secondUpFactor != nullptr && args_.tiling.secondRankColumns > 0 &&
               args_.tiling.secondInputColumnOffset + args_.tiling.secondRankColumns <= TotalRankColumns() &&
               args_.tiling.secondOutputColumnOffset < args_.tiling.outputColumns &&
               args_.tiling.outputColumnOffset < args_.tiling.secondOutputColumnOffset;
    }

    __aicore__ inline SVDQLowRankStagePlan BuildDownStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_DOWN_PROJECT,
            args_.tiling.downFactorId,
            args_.downFactor,
            args_.tiling.inputColumns,
            TotalRankColumns(),
            args_.tiling.inputColumns,
            args_.tiling.outputColumns,
            args_.tiling.inputColumnOffset,
            0,
            0,
            false,
        };
    }

    __aicore__ inline SVDQLowRankStagePlan BuildPrimaryUpStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_UP_PROJECT,
            args_.tiling.upFactorId,
            args_.upFactor,
            args_.tiling.rankColumns,
            PrimaryOutputColumns(),
            args_.tiling.outputColumns,
            args_.tiling.outputColumns,
            args_.tiling.inputColumnOffset,
            args_.tiling.outputColumnOffset,
            0,
            true,
        };
    }

    __aicore__ inline SVDQLowRankStagePlan BuildSecondUpStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_SECOND_UP_PROJECT,
            args_.tiling.secondUpFactorId,
            args_.secondUpFactor,
            args_.tiling.secondRankColumns,
            args_.tiling.outputColumns - args_.tiling.secondOutputColumnOffset,
            args_.tiling.outputColumns,
            args_.tiling.outputColumns,
            args_.tiling.secondInputColumnOffset,
            args_.tiling.secondOutputColumnOffset,
            0,
            true,
        };
    }

    __aicore__ inline GM_ADDR MatrixAddress(
        GM_ADDR base, uint32_t row, uint32_t strideColumns, uint32_t columnOffset) const
    {
        return base + (static_cast<uint64_t>(row) * strideColumns + columnOffset) * SVDQ_LOWRANK_BF16_BYTES;
    }

    __aicore__ inline GM_ADDR FactorAddress(const SVDQLowRankStagePlan& stage, uint32_t expertId) const
    {
        const uint64_t expertOffset =
            static_cast<uint64_t>(expertId) * stage.inputColumns * stage.outputColumns + stage.factorColumnOffset;
        return stage.factor + expertOffset * SVDQ_LOWRANK_BF16_BYTES;
    }

    SVDQFusedDownUpArgs args_{};
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_FUSED_DOWN_UP_HPP
