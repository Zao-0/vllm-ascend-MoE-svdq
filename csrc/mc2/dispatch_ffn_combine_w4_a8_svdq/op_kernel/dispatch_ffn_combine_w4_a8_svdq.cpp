/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "dispatch_ffn_combine_w4_a8_svdq_tiling.h"
#include "dispatch_ffn_combine_w4_a8_svdq.h"

using namespace AscendC;
using namespace DispatchFFNCombineW4A8SVDQImpl;

extern "C" __global__ __aicore__ void dispatch_ffn_combine_w4_a8_svdq(
    GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR bias1,
    GM_ADDR bias2, GM_ADDR probs, GM_ADDR gateUpSvdqL1, GM_ADDR gateSvdqL2, GM_ADDR upSvdqL2,
    GM_ADDR downSvdqL1, GM_ADDR downSvdqL2, GM_ADDR xActiveMask, GM_ADDR out, GM_ADDR expertTokenNums,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    DispatchFFNCombineW4A8SVDQ op;
    op.Init(x, w1, w2, expertId, scale1, scale2, bias1, bias2, probs, gateUpSvdqL1, gateSvdqL2, upSvdqL2,
        downSvdqL1, downSvdqL2, xActiveMask, out, expertTokenNums, workspaceGM, tilingGM);
    op.Process();
}
