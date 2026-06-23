/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SVDQ_LOWRANK_DEBUG_READBACK_H
#define SVDQ_LOWRANK_DEBUG_READBACK_H

#include "kernel_operator.h"
#include "../dispatch_ffn_combine_w4_a8_svdq_tiling.h"
#include "svdq_fused_down_up.hpp"

namespace DispatchFFNCombineW4A8SVDQImpl {

struct SVDQLowRankDebugTilingData {
    uint32_t expertPerRank;
    SVDQFusedDownUpTiling gateUpInvocation;
    SVDQFusedDownUpTiling downInvocation;
};

struct SVDQLowRankDebugRuntimeGM {
    GM_ADDR routedX;
    GM_ADDR hidden;
    GM_ADDR gateUpSvdqL1;
    GM_ADDR gateSvdqL2;
    GM_ADDR upSvdqL2;
    GM_ADDR downSvdqL1;
    GM_ADDR downSvdqL2;
    GM_ADDR expertTokenNums;
    GM_ADDR gateUpOutput;
    GM_ADDR downOutput;
    GM_ADDR gateUpAccumulator;
    GM_ADDR downAccumulator;
};

class SVDQLowRankDebugReadbackKernel {
public:
    __aicore__ inline SVDQLowRankDebugReadbackKernel() {}

    __aicore__ inline void Init(GM_ADDR routedX, GM_ADDR hidden, GM_ADDR gateUpSvdqL1, GM_ADDR gateSvdqL2,
        GM_ADDR upSvdqL2, GM_ADDR downSvdqL1, GM_ADDR downSvdqL2, GM_ADDR expertTokenNums,
        GM_ADDR gateUpOutput, GM_ADDR downOutput, GM_ADDR gateUpAccumulator, GM_ADDR downAccumulator,
        GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(SVDQLowRankDebugTilingData);
        GET_TILING_DATA(tilingData, tilingGM);

        runtime_.routedX = routedX;
        runtime_.hidden = hidden;
        runtime_.gateUpSvdqL1 = gateUpSvdqL1;
        runtime_.gateSvdqL2 = gateSvdqL2;
        runtime_.upSvdqL2 = upSvdqL2;
        runtime_.downSvdqL1 = downSvdqL1;
        runtime_.downSvdqL2 = downSvdqL2;
        runtime_.expertTokenNums = expertTokenNums;
        runtime_.gateUpOutput = gateUpOutput;
        runtime_.downOutput = downOutput;
        runtime_.gateUpAccumulator = gateUpAccumulator;
        runtime_.downAccumulator = downAccumulator;
        tilingData_ = tilingData;
    }

    __aicore__ inline bool HasCompleteTilingContract() const
    {
        return tilingData_.expertPerRank > 0 &&
               tilingData_.gateUpInvocation.invocationId == SVDQ_LOWRANK_INVOCATION_GATE_UP &&
               tilingData_.downInvocation.invocationId == SVDQ_LOWRANK_INVOCATION_DOWN &&
               tilingData_.gateUpInvocation.secondUpFactorId != SVDQ_INVALID_ID &&
               tilingData_.downInvocation.secondUpFactorId == SVDQ_INVALID_ID;
    }

    __aicore__ inline SVDQFusedDownUpArgs BuildGateUpArgs() const
    {
        SVDQFusedDownUpArgs args{};
        args.input = runtime_.routedX;
        args.downFactor = runtime_.gateUpSvdqL1;
        args.upFactor = runtime_.gateSvdqL2;
        args.secondUpFactor = runtime_.upSvdqL2;
        args.output = runtime_.gateUpOutput;
        args.accumulator = runtime_.gateUpAccumulator;
        args.expertTokenNums = runtime_.expertTokenNums;
        args.expertPerRank = tilingData_.expertPerRank;
        args.tiling = tilingData_.gateUpInvocation;
        return args;
    }

    __aicore__ inline SVDQFusedDownUpArgs BuildDownArgs() const
    {
        SVDQFusedDownUpArgs args{};
        args.input = runtime_.hidden;
        args.downFactor = runtime_.downSvdqL1;
        args.upFactor = runtime_.downSvdqL2;
        args.secondUpFactor = nullptr;
        args.output = runtime_.downOutput;
        args.accumulator = runtime_.downAccumulator;
        args.expertTokenNums = runtime_.expertTokenNums;
        args.expertPerRank = tilingData_.expertPerRank;
        args.tiling = tilingData_.downInvocation;
        return args;
    }

    __aicore__ inline bool RunInvocation(const SVDQFusedDownUpArgs& args) const
    {
        SVDQLowRankDebugReadback readback;
        readback.Init(args);
        return readback.Process();
    }

    __aicore__ inline void Process() const
    {
        if (!HasCompleteTilingContract()) {
            return;
        }
        if (!RunInvocation(BuildGateUpArgs())) {
            return;
        }
        (void)RunInvocation(BuildDownArgs());
    }

private:
    SVDQLowRankDebugRuntimeGM runtime_{};
    SVDQLowRankDebugTilingData tilingData_{};
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_LOWRANK_DEBUG_READBACK_H
