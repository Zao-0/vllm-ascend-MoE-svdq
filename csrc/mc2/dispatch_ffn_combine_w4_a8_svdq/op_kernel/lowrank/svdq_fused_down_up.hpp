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

struct SVDQFusedDownUpArgs {
    GM_ADDR input;
    GM_ADDR downFactor;
    GM_ADDR upFactor;
    GM_ADDR secondUpFactor;
    GM_ADDR output;
    GM_ADDR expertTokenNums;
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
               (args_.tiling.secondUpFactorId == SVDQ_INVALID_ID || args_.secondUpFactor != nullptr);
    }

    __aicore__ inline bool HasIndependentSecondUp() const
    {
        return args_.tiling.secondUpFactorId != SVDQ_INVALID_ID && args_.secondUpFactor != nullptr &&
               args_.tiling.secondRankColumns > 0;
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
        // The fused AIC math body will keep the low-rank bottleneck on-chip:
        // input BF16 -> down factor GEMM -> rank tile -> up factor GEMM -> projection BF16 GM.
    }

private:
    SVDQFusedDownUpArgs args_{};
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_FUSED_DOWN_UP_HPP
