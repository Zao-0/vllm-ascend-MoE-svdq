/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_H
#define SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_H

#include "kernel_operator.h"
#include "svdq_mixed_epilogue_debug_readback_tiling.h"

using namespace AscendC;

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_MIXED_DEBUG_UB_BYTES = 196352;

struct SVDQMixedEpilogueDebugRuntimeGM {
    GM_ADDR residualGateUp;
    GM_ADDR gateUpLowRank;
    GM_ADDR residualDown;
    GM_ADDR downLowRank;
    GM_ADDR gateUpTotal;
    GM_ADDR hiddenBf16;
    GM_ADDR hiddenInt8;
    GM_ADDR hiddenScale;
    GM_ADDR downTotal;
    GM_ADDR outBf16;
};

class SVDQMixedEpilogueDebugReadbackKernel {
public:
    __aicore__ inline SVDQMixedEpilogueDebugReadbackKernel() {}

    __aicore__ inline void Init(GM_ADDR residualGateUp, GM_ADDR gateUpLowRank, GM_ADDR residualDown,
        GM_ADDR downLowRank, GM_ADDR gateUpTotal, GM_ADDR hiddenBf16, GM_ADDR hiddenInt8, GM_ADDR hiddenScale,
        GM_ADDR downTotal, GM_ADDR outBf16, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(SVDQMixedEpilogueDebugTilingData);
        GET_TILING_DATA(tilingData, tilingGM);

        runtime_.residualGateUp = residualGateUp;
        runtime_.gateUpLowRank = gateUpLowRank;
        runtime_.residualDown = residualDown;
        runtime_.downLowRank = downLowRank;
        runtime_.gateUpTotal = gateUpTotal;
        runtime_.hiddenBf16 = hiddenBf16;
        runtime_.hiddenInt8 = hiddenInt8;
        runtime_.hiddenScale = hiddenScale;
        runtime_.downTotal = downTotal;
        runtime_.outBf16 = outBf16;
        tilingData_ = tilingData;

        residualGateUpGm_.SetGlobalBuffer((__gm__ float*)runtime_.residualGateUp);
        gateUpLowRankGm_.SetGlobalBuffer((__gm__ bfloat16_t*)runtime_.gateUpLowRank);
        residualDownGm_.SetGlobalBuffer((__gm__ float*)runtime_.residualDown);
        downLowRankGm_.SetGlobalBuffer((__gm__ bfloat16_t*)runtime_.downLowRank);
        gateUpTotalGm_.SetGlobalBuffer((__gm__ float*)runtime_.gateUpTotal);
        hiddenBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)runtime_.hiddenBf16);
        hiddenInt8Gm_.SetGlobalBuffer((__gm__ int8_t*)runtime_.hiddenInt8);
        hiddenScaleGm_.SetGlobalBuffer((__gm__ float*)runtime_.hiddenScale);
        downTotalGm_.SetGlobalBuffer((__gm__ float*)runtime_.downTotal);
        outBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)runtime_.outBf16);
        pipe_.InitBuffer(ubBuf_, SVDQ_MIXED_DEBUG_UB_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (!HasCompleteContract()) {
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t row = blockIdx; row < tilingData_.rows; row += blockNum) {
            ProcessGateUpRow(row);
            QuantizeHiddenRow(row);
            ProcessDownRow(row);
        }
    }

private:
    __aicore__ inline void SyncMte2ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncVToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2() const
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopyInFloat(LocalTensor<float> dst, const GlobalTensor<float>& src, uint32_t offset,
        uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst, src[offset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyOutFloat(GlobalTensor<float>& dst, uint32_t offset, LocalTensor<float> src,
        uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline void CopyInBf16(LocalTensor<bfloat16_t> dst, const GlobalTensor<bfloat16_t>& src,
        uint32_t offset, uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst, src[offset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyOutBf16(GlobalTensor<bfloat16_t>& dst, uint32_t offset, LocalTensor<bfloat16_t> src,
        uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline void CopyOutInt8(GlobalTensor<int8_t>& dst, uint32_t offset, LocalTensor<int8_t> src,
        uint32_t count) const
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline bool HasCompleteContract() const
    {
        return tilingData_.rows > 0 && tilingData_.intermediateSize > 0 && tilingData_.hiddenSize > 0 &&
               tilingData_.vectorTile > 0 && tilingData_.intermediateSize % tilingData_.vectorTile == 0 &&
               tilingData_.hiddenSize % tilingData_.vectorTile == 0;
    }

    __aicore__ inline void ProcessGateUpRow(uint32_t row)
    {
        LocalTensor<float> ub = ubBuf_.Get<float>();
        LocalTensor<float> gate = ub;
        LocalTensor<float> up = ub[tilingData_.vectorTile];
        LocalTensor<float> tmp = ub[tilingData_.vectorTile * 2];
        LocalTensor<float> hiddenFp32 = ub[tilingData_.vectorTile * 3];
        LocalTensor<bfloat16_t> bf16Ub = ub[tilingData_.vectorTile * 4].template ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> gateLowRankBf16 = bf16Ub;
        LocalTensor<bfloat16_t> upLowRankBf16 = bf16Ub[tilingData_.vectorTile];
        LocalTensor<bfloat16_t> hiddenOut = bf16Ub[tilingData_.vectorTile * 2];

        const uint32_t intermediate = tilingData_.intermediateSize;
        const uint32_t gateUpColumns = intermediate * 2;
        for (uint32_t column = 0; column < intermediate; column += tilingData_.vectorTile) {
            const uint32_t gateOffset = row * gateUpColumns + column;
            const uint32_t upOffset = row * gateUpColumns + intermediate + column;

            CopyInFloat(gate, residualGateUpGm_, gateOffset, tilingData_.vectorTile);
            SyncMte2ToV();
            CopyInBf16(gateLowRankBf16, gateUpLowRankGm_, gateOffset, tilingData_.vectorTile);
            SyncMte2ToV();
            Cast(tmp, gateLowRankBf16, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Add(gate, gate, tmp, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte2();

            CopyInFloat(up, residualGateUpGm_, upOffset, tilingData_.vectorTile);
            SyncMte2ToV();
            CopyInBf16(upLowRankBf16, gateUpLowRankGm_, upOffset, tilingData_.vectorTile);
            SyncMte2ToV();
            Cast(tmp, upLowRankBf16, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Add(up, up, tmp, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte2();

            if (tilingData_.swigluLimit > 0.0f) {
                Maxs(gate, gate, -tilingData_.swigluLimit, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                Mins(gate, gate, tilingData_.swigluLimit, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                Maxs(up, up, -tilingData_.swigluLimit, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                Mins(up, up, tilingData_.swigluLimit, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
            }

            SyncVToMte3();
            CopyOutFloat(gateUpTotalGm_, gateOffset, gate, tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();
            SyncVToMte3();
            CopyOutFloat(gateUpTotalGm_, upOffset, up, tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();

            Muls(tmp, gate, -1.0f, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Exp(tmp, tmp, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Div(hiddenFp32, gate, tmp, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Mul(hiddenFp32, hiddenFp32, up, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Cast(hiddenOut, hiddenFp32, RoundMode::CAST_RINT, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            CopyOutBf16(hiddenBf16Gm_, row * intermediate + column, hiddenOut, tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void QuantizeHiddenRow(uint32_t row)
    {
        LocalTensor<float> ub = ubBuf_.Get<float>();
        LocalTensor<float> hiddenFp32 = ub;
        LocalTensor<float> absHidden = ub[tilingData_.vectorTile];
        LocalTensor<float> reduceTmp = ub[tilingData_.vectorTile * 2];
        LocalTensor<float> scaleLocal = ub[tilingData_.vectorTile * 3];
        LocalTensor<bfloat16_t> hiddenBf16 = ub[tilingData_.vectorTile * 4].template ReinterpretCast<bfloat16_t>();
        LocalTensor<int8_t> hiddenI8 = ub[tilingData_.vectorTile * 5].template ReinterpretCast<int8_t>();
        LocalTensor<half> quantHalf = ub[tilingData_.vectorTile * 6].template ReinterpretCast<half>();

        float maxAbs = 0.0f;
        for (uint32_t column = 0; column < tilingData_.intermediateSize; column += tilingData_.vectorTile) {
            CopyInBf16(hiddenBf16, hiddenBf16Gm_, row * tilingData_.intermediateSize + column,
                tilingData_.vectorTile);
            SyncMte2ToV();
            Cast(hiddenFp32, hiddenBf16, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte2();
            Abs(absHidden, hiddenFp32, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            ReduceMax(reduceTmp, absHidden, scaleLocal, tilingData_.vectorTile);
            event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVToS);
            WaitFlag<HardEvent::V_S>(eventVToS);
            const float chunkMax = reduceTmp.GetValue(0);
            if (chunkMax > maxAbs) {
                maxAbs = chunkMax;
            }
            event_t eventSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSToV);
            WaitFlag<HardEvent::S_V>(eventSToV);
        }

        const float scale = maxAbs / 127.0f;
        scaleLocal.SetValue(0, scale);
        SyncVToMte3();
        CopyOutFloat(hiddenScaleGm_, row, scaleLocal, 1);
        SyncMte3ToV();
        if (scale == 0.0f) {
            for (uint32_t column = 0; column < tilingData_.intermediateSize; column += tilingData_.vectorTile) {
                Duplicate<float>(hiddenFp32, 0.0f, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                Cast(quantHalf, hiddenFp32, RoundMode::CAST_NONE, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                Cast(hiddenI8, quantHalf, RoundMode::CAST_RINT, tilingData_.vectorTile);
                PipeBarrier<PIPE_V>();
                SyncVToMte3();
                CopyOutInt8(hiddenInt8Gm_, row * tilingData_.intermediateSize + column, hiddenI8,
                    tilingData_.vectorTile);
                SyncMte3ToV();
                SyncMte3ToMte2();
            }
            return;
        }

        for (uint32_t column = 0; column < tilingData_.intermediateSize; column += tilingData_.vectorTile) {
            CopyInBf16(hiddenBf16, hiddenBf16Gm_, row * tilingData_.intermediateSize + column,
                tilingData_.vectorTile);
            SyncMte2ToV();
            Cast(hiddenFp32, hiddenBf16, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte2();
            Muls(hiddenFp32, hiddenFp32, 1.0f / scale, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Maxs(hiddenFp32, hiddenFp32, -127.0f, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Mins(hiddenFp32, hiddenFp32, 127.0f, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Cast(quantHalf, hiddenFp32, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Cast(hiddenI8, quantHalf, RoundMode::CAST_RINT, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            CopyOutInt8(hiddenInt8Gm_, row * tilingData_.intermediateSize + column, hiddenI8,
                tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void ProcessDownRow(uint32_t row)
    {
        LocalTensor<float> ub = ubBuf_.Get<float>();
        LocalTensor<float> down = ub;
        LocalTensor<float> lowRankFp32 = ub[tilingData_.vectorTile];
        LocalTensor<bfloat16_t> lowRankBf16 = ub[tilingData_.vectorTile * 2].template ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> outBf16 = lowRankBf16[tilingData_.vectorTile];

        for (uint32_t column = 0; column < tilingData_.hiddenSize; column += tilingData_.vectorTile) {
            const uint32_t offset = row * tilingData_.hiddenSize + column;
            CopyInFloat(down, residualDownGm_, offset, tilingData_.vectorTile);
            CopyInBf16(lowRankBf16, downLowRankGm_, offset, tilingData_.vectorTile);
            SyncMte2ToV();
            Cast(lowRankFp32, lowRankBf16, RoundMode::CAST_NONE, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            Add(down, down, lowRankFp32, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte2();
            SyncVToMte3();
            CopyOutFloat(downTotalGm_, offset, down, tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();
            Cast(outBf16, down, RoundMode::CAST_RINT, tilingData_.vectorTile);
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            CopyOutBf16(outBf16Gm_, offset, outBf16, tilingData_.vectorTile);
            SyncMte3ToV();
            SyncMte3ToMte2();
        }
    }

    TPipe pipe_;
    TBuf<> ubBuf_;
    SVDQMixedEpilogueDebugRuntimeGM runtime_{};
    SVDQMixedEpilogueDebugTilingData tilingData_{};
    GlobalTensor<float> residualGateUpGm_;
    GlobalTensor<bfloat16_t> gateUpLowRankGm_;
    GlobalTensor<float> residualDownGm_;
    GlobalTensor<bfloat16_t> downLowRankGm_;
    GlobalTensor<float> gateUpTotalGm_;
    GlobalTensor<bfloat16_t> hiddenBf16Gm_;
    GlobalTensor<int8_t> hiddenInt8Gm_;
    GlobalTensor<float> hiddenScaleGm_;
    GlobalTensor<float> downTotalGm_;
    GlobalTensor<bfloat16_t> outBf16Gm_;
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_H
