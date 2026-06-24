/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_
#define OP_API_INC_SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQMixedEpilogueDebugReadbackGetWorkspaceSize(
    const aclTensor* residualGateUp, const aclTensor* gateUpLowRank, const aclTensor* residualDown,
    const aclTensor* downLowRank, double swigluLimit, const aclTensor* gateUpTotal,
    const aclTensor* hiddenBf16, const aclTensor* hiddenInt8, const aclTensor* hiddenScale,
    const aclTensor* downTotal, const aclTensor* outBf16, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQMixedEpilogueDebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_
