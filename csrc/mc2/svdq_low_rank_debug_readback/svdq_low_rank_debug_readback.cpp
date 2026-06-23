/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if __has_include("../dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback.h")
#include "../dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback.h"
#else
#include "../../../../mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback.h"
#endif

using namespace AscendC;
using namespace DispatchFFNCombineW4A8SVDQImpl;

extern "C" __global__ __aicore__ void svdq_low_rank_debug_readback(
    GM_ADDR routedX, GM_ADDR hidden, GM_ADDR gateUpSvdqL1, GM_ADDR gateSvdqL2, GM_ADDR upSvdqL2,
    GM_ADDR downSvdqL1, GM_ADDR downSvdqL2, GM_ADDR expertTokenNums, GM_ADDR gateUpOutput, GM_ADDR downOutput,
    GM_ADDR gateUpAccumulator, GM_ADDR downAccumulator, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)workspaceGM;
    SVDQLowRankDebugReadbackKernel op;
    op.Init(routedX, hidden, gateUpSvdqL1, gateSvdqL2, upSvdqL2, downSvdqL1, downSvdqL2, expertTokenNums,
        gateUpOutput, downOutput, gateUpAccumulator, downAccumulator, tilingGM);
    op.Process();
}
