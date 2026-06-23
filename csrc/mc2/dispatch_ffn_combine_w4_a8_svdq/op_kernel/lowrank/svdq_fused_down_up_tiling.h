/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SVDQ_FUSED_DOWN_UP_TILING_H
#define SVDQ_FUSED_DOWN_UP_TILING_H

#include <cstdint>

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_LOWRANK_INVOCATION_COUNT = 2;

enum SVDQLowRankInvocationId : uint32_t {
    SVDQ_LOWRANK_INVOCATION_GATE_UP = 0,
    SVDQ_LOWRANK_INVOCATION_DOWN = 1,
};

struct SVDQFusedDownUpTiling {
    uint32_t invocationId;
    uint32_t inputRegionId;
    uint32_t outputRegionId;
    uint32_t downFactorId;
    uint32_t upFactorId;
    uint32_t secondUpFactorId;
    uint32_t m;
    uint32_t inputColumns;
    uint32_t rankColumns;
    uint32_t secondRankColumns;
    uint32_t outputColumns;
    uint32_t inputColumnOffset;
    uint32_t outputColumnOffset;
    uint32_t secondInputColumnOffset;
    uint32_t secondOutputColumnOffset;
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_FUSED_DOWN_UP_TILING_H
