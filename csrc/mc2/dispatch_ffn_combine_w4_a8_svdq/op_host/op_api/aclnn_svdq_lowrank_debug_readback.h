/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SVDQ_LOWRANK_DEBUG_READBACK_
#define OP_API_INC_SVDQ_LOWRANK_DEBUG_READBACK_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQLowRankDebugReadbackGetWorkspaceSize(
    const aclTensor* routedX, const aclTensor* hidden, const aclTensor* gateUpSvdqL1,
    const aclTensor* gateSvdqL2, const aclTensor* upSvdqL2, const aclTensor* downSvdqL1,
    const aclTensor* downSvdqL2, const aclTensor* expertTokenNums, int64_t gateRank,
    int64_t upRank, int64_t downRank, int64_t gateRankOffset, int64_t upRankOffset,
    const aclTensor* gateUpOutput, const aclTensor* downOutput, const aclTensor* gateUpAccumulator,
    const aclTensor* downAccumulator, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQLowRankDebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SVDQ_LOWRANK_DEBUG_READBACK_
