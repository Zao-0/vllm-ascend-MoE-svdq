/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_svdq_mixed_epilogue_debug_readback.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerSVDQMixedEpilogueDebugReadbackGetWorkspaceSize(
    const aclTensor* residualGateUp, const aclTensor* gateUpLowRank, const aclTensor* residualDown,
    const aclTensor* downLowRank, double swigluLimit, const aclTensor* gateUpTotal,
    const aclTensor* hiddenBf16, const aclTensor* hiddenInt8, const aclTensor* hiddenInt4Packed,
    const aclTensor* hiddenScale,
    const aclTensor* downTotal, const aclTensor* outBf16, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerSVDQMixedEpilogueDebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnSVDQMixedEpilogueDebugReadbackGetWorkspaceSize(
    const aclTensor* residualGateUp, const aclTensor* gateUpLowRank, const aclTensor* residualDown,
    const aclTensor* downLowRank, double swigluLimit, const aclTensor* gateUpTotal,
    const aclTensor* hiddenBf16, const aclTensor* hiddenInt8, const aclTensor* hiddenInt4Packed,
    const aclTensor* hiddenScale,
    const aclTensor* downTotal, const aclTensor* outBf16, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerSVDQMixedEpilogueDebugReadbackGetWorkspaceSize(residualGateUp, gateUpLowRank,
        residualDown, downLowRank, swigluLimit, gateUpTotal, hiddenBf16, hiddenInt8,
        hiddenInt4Packed, hiddenScale, downTotal, outBf16, workspaceSize, executor);
}

aclnnStatus aclnnSVDQMixedEpilogueDebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerSVDQMixedEpilogueDebugReadback(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
