/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowrank/svdq_mixed_epilogue_debug_readback.h"

using namespace AscendC;
using namespace DispatchFFNCombineW4A8SVDQImpl;

extern "C" __global__ __aicore__ void svdq_mixed_epilogue_debug_readback(
    GM_ADDR residualGateUp, GM_ADDR gateUpLowRank, GM_ADDR residualDown, GM_ADDR downLowRank,
    GM_ADDR gateUpTotal, GM_ADDR hiddenBf16, GM_ADDR hiddenInt8, GM_ADDR hiddenInt4Packed, GM_ADDR hiddenScale,
    GM_ADDR downTotal, GM_ADDR outBf16, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)workspaceGM;
    SVDQMixedEpilogueDebugReadbackKernel op;
    op.Init(residualGateUp, gateUpLowRank, residualDown, downLowRank, gateUpTotal, hiddenBf16, hiddenInt8,
        hiddenInt4Packed, hiddenScale, downTotal, outBf16, tilingGM);
    op.Process();
}
