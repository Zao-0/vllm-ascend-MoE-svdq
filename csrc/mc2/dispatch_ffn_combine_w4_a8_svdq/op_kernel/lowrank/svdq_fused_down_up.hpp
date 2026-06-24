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
#include "layout.h"
#include "mem.h"
#include "gm_to_l1_iterator.h"
#include "l1_to_l0_iterator.h"
#include "l0c_to_gm_iterator.h"
#include "mma.h"

namespace DispatchFFNCombineW4A8SVDQImpl {

constexpr uint32_t SVDQ_LOWRANK_BF16_BYTES = 2;
constexpr uint32_t SVDQ_LOWRANK_MMAD_M_TILE = 16;
constexpr uint32_t SVDQ_LOWRANK_MMAD_N_TILE = 64;
constexpr uint32_t SVDQ_LOWRANK_MMAD_K_TILE = 64;
constexpr uint32_t SVDQ_LOWRANK_MMAD_M_ALIGNMENT = 16;
constexpr uint32_t SVDQ_LOWRANK_MMAD_N_ALIGNMENT = 16;
constexpr uint32_t SVDQ_LOWRANK_MMAD_K_ALIGNMENT = 16;
constexpr uint32_t SVDQ_LOWRANK_MMAD_FORMAT_ND = 0;
constexpr uint32_t SVDQ_LOWRANK_MMAD_FORMAT_NZ = 1;
constexpr uint32_t SVDQ_LOWRANK_MMAD_FORMAT_ZN = 2;

enum SVDQLowRankStageKind : uint32_t {
    SVDQ_LOWRANK_STAGE_DOWN_PROJECT = 0,
    SVDQ_LOWRANK_STAGE_UP_PROJECT = 1,
    SVDQ_LOWRANK_STAGE_SECOND_UP_PROJECT = 2,
};

struct SVDQLowRankStagePlan {
    uint32_t stageKind;
    uint32_t factorId;
    GM_ADDR factor;
    uint32_t inputColumns;
    uint32_t outputColumns;
    uint32_t inputStrideColumns;
    uint32_t outputStrideColumns;
    uint32_t inputColumnOffset;
    uint32_t outputColumnOffset;
    uint32_t factorColumnOffset;
    bool writesGlobalOutput;

    __aicore__ inline bool HasCompleteContract() const
    {
        return factor != nullptr && factorId != SVDQ_INVALID_ID && inputColumns > 0 && outputColumns > 0;
    }
};

struct SVDQLowRankExpertPlan {
    SVDQLowRankStagePlan stage;
    uint32_t expertId;
    uint32_t tokenStart;
    uint32_t tokenCount;
    GM_ADDR input;
    GM_ADDR factor;
    GM_ADDR output;

    __aicore__ inline bool HasWork() const
    {
        return tokenCount > 0 && input != nullptr && factor != nullptr && output != nullptr &&
               stage.HasCompleteContract();
    }
};

struct SVDQLowRankCoreTileRange {
    uint32_t tileStart;
    uint32_t tileCount;

    __aicore__ inline bool HasWork() const
    {
        return tileCount > 0;
    }
};

struct SVDQLowRankTilePlan {
    SVDQLowRankExpertPlan expert;
    uint32_t tileId;
    uint32_t rowStart;
    uint32_t rowCount;
    uint32_t outputColumnOffset;
    uint32_t outputColumnCount;
    uint32_t kColumnOffset;
    uint32_t kColumnCount;

    __aicore__ inline bool HasWork() const
    {
        return expert.HasWork() && rowCount > 0 && outputColumnCount > 0 && kColumnCount > 0;
    }
};

struct SVDQLowRankOutputTilePlan {
    SVDQLowRankExpertPlan expert;
    uint32_t tileId;
    uint32_t rowStart;
    uint32_t rowCount;
    uint32_t outputColumnOffset;
    uint32_t outputColumnCount;

    __aicore__ inline bool HasWork() const
    {
        return expert.HasWork() && rowCount > 0 && outputColumnCount > 0;
    }
};

struct SVDQLowRankTileTensorPlan {
    SVDQLowRankTilePlan tile;
    GM_ADDR input;
    GM_ADDR factor;
    GM_ADDR output;
    GM_ADDR accumulator;
    uint32_t inputStrideColumns;
    uint32_t factorStrideColumns;
    uint32_t outputStrideColumns;
    uint32_t accumulatorStrideColumns;
    bool accumulatesFirstKTile;
    bool accumulatesLastKTile;

    __aicore__ inline bool HasWork() const
    {
        return tile.HasWork() && input != nullptr && factor != nullptr && output != nullptr &&
               accumulator != nullptr && inputStrideColumns > 0 && factorStrideColumns > 0 &&
               outputStrideColumns > 0 && accumulatorStrideColumns > 0;
    }
};

struct SVDQLowRankMmadTilePlan {
    SVDQLowRankTileTensorPlan tile;
    uint32_t mActual;
    uint32_t nActual;
    uint32_t kActual;
    uint32_t mRound;
    uint32_t nRound;
    uint32_t kRound;
    bool canUseMmad;

    __aicore__ inline bool HasCompatibleShape() const
    {
        return canUseMmad && tile.HasWork() && mActual > 0 && nActual > 0 && kActual > 0 &&
               mActual <= SVDQ_LOWRANK_MMAD_M_TILE && nActual <= SVDQ_LOWRANK_MMAD_N_TILE &&
               kActual <= SVDQ_LOWRANK_MMAD_K_TILE && mRound > 0 && nRound > 0 && kRound > 0 &&
               mRound <= SVDQ_LOWRANK_MMAD_M_TILE && nRound <= SVDQ_LOWRANK_MMAD_N_TILE &&
               kRound <= SVDQ_LOWRANK_MMAD_K_TILE &&
               tile.inputStrideColumns >= tile.tile.kColumnOffset + kActual &&
               tile.factorStrideColumns >= tile.tile.kColumnOffset + kActual &&
               tile.outputStrideColumns >= tile.tile.outputColumnOffset + nActual &&
               tile.accumulatorStrideColumns >= tile.tile.outputColumnOffset + nActual;
    }
};

struct SVDQLowRankMmadBufferPlan {
    SVDQLowRankMmadTilePlan tile;
    uint32_t inputElementCount;
    uint32_t factorElementCount;
    uint32_t outputElementCount;
    uint32_t l1InputBytes;
    uint32_t l1FactorBytes;
    uint32_t l0ABytes;
    uint32_t l0BBytes;
    uint32_t l0CBytes;
    bool storesAccumulator;
    bool storesOutput;

    __aicore__ inline bool HasCompleteFootprint() const
    {
        return tile.HasCompatibleShape() && inputElementCount == tile.mRound * tile.kRound &&
               factorElementCount == tile.nRound * tile.kRound &&
               outputElementCount == tile.mRound * tile.nRound &&
               l1InputBytes == BytesForBF16Elements(inputElementCount) &&
               l1FactorBytes == BytesForBF16Elements(factorElementCount) &&
               l0ABytes == BytesForBF16Elements(inputElementCount) &&
               l0BBytes == BytesForBF16Elements(factorElementCount) &&
               l0CBytes == BytesForFP32Elements(outputElementCount) &&
               storesAccumulator != storesOutput;
    }

    __aicore__ inline static uint32_t BytesForBF16Elements(uint32_t elementCount)
    {
        return elementCount * SVDQ_LOWRANK_BF16_BYTES;
    }

    __aicore__ inline static uint32_t BytesForFP32Elements(uint32_t elementCount)
    {
        return elementCount * sizeof(float);
    }
};

struct SVDQLowRankMmadPipelinePlan {
    SVDQLowRankMmadBufferPlan buffer;
    uint32_t l1InputOffset;
    uint32_t l1FactorOffset;
    uint32_t l0AOffset;
    uint32_t l0BOffset;
    uint32_t l0COffset;
    uint32_t inputGmFormat;
    uint32_t factorGmFormat;
    uint32_t inputL1Format;
    uint32_t factorL1Format;
    uint32_t l0AFormat;
    uint32_t l0BFormat;
    uint32_t outputFormat;
    bool loadInputGmToL1;
    bool loadFactorGmToL1;
    bool loadInputL1ToL0A;
    bool loadFactorL1ToL0B;
    bool factorLoadsTransposed;
    bool runMmad;
    bool storeL0C;
    bool initAccumulator;
    bool storesAccumulator;
    bool storesOutput;

    __aicore__ inline bool HasCompletePipeline() const
    {
        return buffer.HasCompleteFootprint() && l1InputOffset == 0 &&
               l1FactorOffset == buffer.l1InputBytes && l0AOffset == 0 && l0BOffset == 0 &&
               l0COffset == 0 && inputGmFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND &&
               factorGmFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND &&
               inputL1Format == SVDQ_LOWRANK_MMAD_FORMAT_NZ &&
               factorL1Format == SVDQ_LOWRANK_MMAD_FORMAT_NZ &&
               l0AFormat == SVDQ_LOWRANK_MMAD_FORMAT_NZ &&
               l0BFormat == SVDQ_LOWRANK_MMAD_FORMAT_ZN &&
               outputFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND && loadInputGmToL1 &&
               loadFactorGmToL1 && loadInputL1ToL0A && loadFactorL1ToL0B &&
               factorLoadsTransposed && runMmad && storeL0C &&
               initAccumulator == buffer.tile.tile.accumulatesFirstKTile &&
               storesAccumulator == buffer.storesAccumulator && storesOutput == buffer.storesOutput &&
               storesAccumulator != storesOutput;
    }
};

struct SVDQFusedDownUpArgs {
    GM_ADDR input;
    GM_ADDR downFactor;
    GM_ADDR upFactor;
    GM_ADDR secondUpFactor;
    GM_ADDR rank;
    GM_ADDR output;
    GM_ADDR accumulator;
    GM_ADDR expertTokenNums;
    uint32_t expertPerRank;
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
               args_.rank != nullptr &&
               args_.output != nullptr && args_.accumulator != nullptr && args_.expertTokenNums != nullptr &&
               args_.tiling.m > 0 && args_.tiling.inputColumns > 0 && args_.tiling.rankColumns > 0 &&
               args_.tiling.outputColumns > 0 && args_.tiling.downFactorId != SVDQ_INVALID_ID &&
               args_.tiling.upFactorId != SVDQ_INVALID_ID &&
               args_.tiling.invocationId < SVDQ_LOWRANK_INVOCATION_COUNT &&
               args_.expertPerRank > 0 && args_.tiling.rowTile > 0 &&
               args_.tiling.outputColumnTile > 0 && args_.tiling.kTile > 0 &&
               args_.tiling.coreCount > 0 &&
               (args_.tiling.secondUpFactorId == SVDQ_INVALID_ID || HasCompleteSecondUpContract());
    }

    __aicore__ inline bool HasIndependentSecondUp() const
    {
        return args_.tiling.secondUpFactorId != SVDQ_INVALID_ID && args_.secondUpFactor != nullptr &&
               args_.tiling.secondRankColumns > 0;
    }

    __aicore__ inline uint32_t TotalRankColumns() const
    {
        return args_.tiling.rankColumns + args_.tiling.secondRankColumns;
    }

    __aicore__ inline uint32_t PrimaryOutputColumns() const
    {
        if (!HasIndependentSecondUp()) {
            return args_.tiling.outputColumns;
        }
        return args_.tiling.secondOutputColumnOffset - args_.tiling.outputColumnOffset;
    }

    __aicore__ inline uint32_t StageCount() const
    {
        return HasIndependentSecondUp() ? 3 : 2;
    }

    __aicore__ inline uint32_t ExpertCount() const
    {
        return args_.expertPerRank;
    }

    __aicore__ inline uint32_t ExpertTokenCount(uint32_t expertId) const
    {
        AscendC::GlobalTensor<int32_t> expertTokenNums;
        expertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_.expertTokenNums));
        const int32_t tokenCount = expertTokenNums(expertId);
        return tokenCount > 0 ? static_cast<uint32_t>(tokenCount) : 0;
    }

    __aicore__ inline uint32_t ExpertTokenStart(uint32_t expertId) const
    {
        uint32_t tokenStart = 0;
        for (uint32_t currentExpertId = 0; currentExpertId < expertId; ++currentExpertId) {
            tokenStart += ExpertTokenCount(currentExpertId);
        }
        return tokenStart;
    }

    __aicore__ inline SVDQLowRankStagePlan StagePlan(uint32_t stageIndex) const
    {
        if (stageIndex == 0) {
            return BuildDownStagePlan();
        }
        if (stageIndex == 1) {
            return BuildPrimaryUpStagePlan();
        }
        return BuildSecondUpStagePlan();
    }

    __aicore__ inline SVDQLowRankExpertPlan ExpertStagePlan(uint32_t stageIndex, uint32_t expertId) const
    {
        const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
        const uint32_t tokenStart = ExpertTokenStart(expertId);
        const uint32_t tokenCount = ExpertTokenCount(expertId);
        GM_ADDR inputBase = stageIndex == 0 ? args_.input : args_.rank;
        GM_ADDR outputBase = stageIndex == 0 ? args_.rank : args_.output;
        return SVDQLowRankExpertPlan{
            stage,
            expertId,
            tokenStart,
            tokenCount,
            MatrixAddress(inputBase, tokenStart, stage.inputStrideColumns, stage.inputColumnOffset),
            FactorAddress(stage, expertId),
            MatrixAddress(outputBase, tokenStart, stage.outputStrideColumns, stage.outputColumnOffset),
        };
    }

    __aicore__ inline SVDQLowRankMmadBufferPlan BuildMmadBufferPlan(
        const SVDQLowRankMmadTilePlan& tilePlan) const
    {
        if (!tilePlan.HasCompatibleShape()) {
            return {};
        }
        const uint32_t inputElementCount = tilePlan.mRound * tilePlan.kRound;
        const uint32_t factorElementCount = tilePlan.nRound * tilePlan.kRound;
        const uint32_t outputElementCount = tilePlan.mRound * tilePlan.nRound;
        return SVDQLowRankMmadBufferPlan{
            tilePlan,
            inputElementCount,
            factorElementCount,
            outputElementCount,
            SVDQLowRankMmadBufferPlan::BytesForBF16Elements(inputElementCount),
            SVDQLowRankMmadBufferPlan::BytesForBF16Elements(factorElementCount),
            SVDQLowRankMmadBufferPlan::BytesForBF16Elements(inputElementCount),
            SVDQLowRankMmadBufferPlan::BytesForBF16Elements(factorElementCount),
            SVDQLowRankMmadBufferPlan::BytesForFP32Elements(outputElementCount),
            !tilePlan.tile.accumulatesLastKTile,
            tilePlan.tile.accumulatesLastKTile,
        };
    }

    __aicore__ inline SVDQLowRankMmadPipelinePlan BuildMmadPipelinePlan(
        const SVDQLowRankMmadBufferPlan& bufferPlan) const
    {
        if (!bufferPlan.HasCompleteFootprint()) {
            return {};
        }
        return SVDQLowRankMmadPipelinePlan{
            bufferPlan,
            0,
            bufferPlan.l1InputBytes,
            0,
            0,
            0,
            SVDQ_LOWRANK_MMAD_FORMAT_ND,
            SVDQ_LOWRANK_MMAD_FORMAT_ND,
            SVDQ_LOWRANK_MMAD_FORMAT_NZ,
            SVDQ_LOWRANK_MMAD_FORMAT_NZ,
            SVDQ_LOWRANK_MMAD_FORMAT_NZ,
            SVDQ_LOWRANK_MMAD_FORMAT_ZN,
            SVDQ_LOWRANK_MMAD_FORMAT_ND,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            bufferPlan.tile.tile.accumulatesFirstKTile,
            bufferPlan.storesAccumulator,
            bufferPlan.storesOutput,
        };
    }

    __aicore__ inline uint32_t StageColumnTileCount(const SVDQLowRankStagePlan& stage) const
    {
        return CeilDiv(stage.outputColumns, args_.tiling.outputColumnTile);
    }

    __aicore__ inline uint32_t StageKTileCount(const SVDQLowRankStagePlan& stage) const
    {
        return CeilDiv(stage.inputColumns, args_.tiling.kTile);
    }

    __aicore__ inline uint32_t ExpertRowTileCount(uint32_t tokenCount) const
    {
        return CeilDiv(tokenCount, args_.tiling.rowTile);
    }

    __aicore__ inline uint32_t ExpertTileCount(uint32_t stageIndex, uint32_t expertId) const
    {
        const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
        const uint32_t rowTiles = ExpertRowTileCount(ExpertTokenCount(expertId));
        return rowTiles * StageColumnTileCount(stage);
    }

    __aicore__ inline uint32_t StageTileCount(uint32_t stageIndex) const
    {
        uint32_t tileCount = 0;
        for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId) {
            tileCount += ExpertTileCount(stageIndex, expertId);
        }
        return tileCount;
    }

    __aicore__ inline uint32_t InvocationTileCount() const
    {
        uint32_t tileCount = 0;
        for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex) {
            tileCount += StageTileCount(stageIndex);
        }
        return tileCount;
    }

    __aicore__ inline SVDQLowRankCoreTileRange CoreTileRange(uint32_t coreIdx, uint32_t coreCount) const
    {
        const uint32_t tileCount = InvocationTileCount();
        if (coreIdx >= coreCount || coreCount == 0 || tileCount == 0) {
            return {0, 0};
        }
        const uint32_t baseTiles = tileCount / coreCount;
        const uint32_t remainder = tileCount - baseTiles * coreCount;
        const uint32_t extra = coreIdx < remainder ? 1 : 0;
        const uint32_t tileStart = coreIdx * baseTiles + Min(coreIdx, remainder);
        return {tileStart, baseTiles + extra};
    }

    __aicore__ inline SVDQLowRankCoreTileRange StageCoreTileRange(
        uint32_t stageIndex, uint32_t coreIdx, uint32_t coreCount) const
    {
        const uint32_t tileCount = StageTileCount(stageIndex);
        if (coreIdx >= coreCount || coreCount == 0 || tileCount == 0) {
            return {0, 0};
        }
        const uint32_t baseTiles = tileCount / coreCount;
        const uint32_t remainder = tileCount - baseTiles * coreCount;
        const uint32_t extra = coreIdx < remainder ? 1 : 0;
        const uint32_t tileStart = coreIdx * baseTiles + Min(coreIdx, remainder);
        return {tileStart, baseTiles + extra};
    }

    __aicore__ inline SVDQLowRankOutputTilePlan StageOutputTilePlan(uint32_t stageIndex, uint32_t tileId) const
    {
        uint32_t remainingTile = tileId;
        const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
        const uint32_t columnTiles = StageColumnTileCount(stage);
        const uint32_t tilesPerRow = columnTiles;
        for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId) {
            const SVDQLowRankExpertPlan expertPlan = ExpertStagePlan(stageIndex, expertId);
            const uint32_t rowTiles = ExpertRowTileCount(expertPlan.tokenCount);
            const uint32_t expertTileCount = rowTiles * tilesPerRow;
            if (remainingTile >= expertTileCount) {
                remainingTile -= expertTileCount;
                continue;
            }

            const uint32_t rowTileIndex = remainingTile / tilesPerRow;
            const uint32_t inRowTileOffset = remainingTile - rowTileIndex * tilesPerRow;
            const uint32_t outputTileIndex = inRowTileOffset;
            const uint32_t rowOffset = rowTileIndex * args_.tiling.rowTile;
            const uint32_t outputColumnOffset = outputTileIndex * args_.tiling.outputColumnTile;
            return SVDQLowRankOutputTilePlan{
                expertPlan,
                tileId,
                expertPlan.tokenStart + rowOffset,
                Min(args_.tiling.rowTile, expertPlan.tokenCount - rowOffset),
                outputColumnOffset,
                Min(args_.tiling.outputColumnTile, stage.outputColumns - outputColumnOffset),
            };
        }
        return {};
    }

    __aicore__ inline SVDQLowRankOutputTilePlan OutputTilePlan(uint32_t tileId) const
    {
        uint32_t remainingTile = tileId;
        for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex) {
            const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
            const uint32_t columnTiles = StageColumnTileCount(stage);
            const uint32_t tilesPerRow = columnTiles;
            for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId) {
                const SVDQLowRankExpertPlan expertPlan = ExpertStagePlan(stageIndex, expertId);
                const uint32_t rowTiles = ExpertRowTileCount(expertPlan.tokenCount);
                const uint32_t expertTileCount = rowTiles * tilesPerRow;
                if (remainingTile >= expertTileCount) {
                    remainingTile -= expertTileCount;
                    continue;
                }

                const uint32_t rowTileIndex = remainingTile / tilesPerRow;
                const uint32_t inRowTileOffset = remainingTile - rowTileIndex * tilesPerRow;
                const uint32_t outputTileIndex = inRowTileOffset;
                const uint32_t rowOffset = rowTileIndex * args_.tiling.rowTile;
                const uint32_t outputColumnOffset = outputTileIndex * args_.tiling.outputColumnTile;
                return SVDQLowRankOutputTilePlan{
                    expertPlan,
                    tileId,
                    expertPlan.tokenStart + rowOffset,
                    Min(args_.tiling.rowTile, expertPlan.tokenCount - rowOffset),
                    outputColumnOffset,
                    Min(args_.tiling.outputColumnTile, stage.outputColumns - outputColumnOffset),
                };
            }
        }
        return {};
    }

    __aicore__ inline bool ExecuteStage(uint32_t stageIndex, uint32_t coreIdx, uint32_t coreCount) const
    {
        const SVDQLowRankCoreTileRange tileRange = StageCoreTileRange(stageIndex, coreIdx, coreCount);
        for (uint32_t tileOffset = 0; tileOffset < tileRange.tileCount; ++tileOffset) {
            const SVDQLowRankOutputTilePlan outputTilePlan =
                StageOutputTilePlan(stageIndex, tileRange.tileStart + tileOffset);
            if (!outputTilePlan.HasWork()) {
                return false;
            }
            const uint32_t kTileCount = OutputTileKTileCount(outputTilePlan);
            for (uint32_t kTileIndex = 0; kTileIndex < kTileCount; ++kTileIndex) {
                const SVDQLowRankTilePlan tilePlan = KTilePlan(outputTilePlan, kTileIndex);
                if (!tilePlan.HasWork()) {
                    return false;
                }
                const SVDQLowRankTileTensorPlan tileTensorPlan = BuildTileTensorPlan(tilePlan);
                if (!tileTensorPlan.HasWork()) {
                    return false;
                }
                const SVDQLowRankMmadTilePlan mmaTilePlan = BuildMmadTilePlan(tileTensorPlan);
                if (!mmaTilePlan.HasCompatibleShape()) {
                    return false;
                }
                const SVDQLowRankMmadBufferPlan bufferPlan = BuildMmadBufferPlan(mmaTilePlan);
                if (!bufferPlan.HasCompleteFootprint()) {
                    return false;
                }
                const SVDQLowRankMmadPipelinePlan pipelinePlan = BuildMmadPipelinePlan(bufferPlan);
                if (!pipelinePlan.HasCompletePipeline()) {
                    return false;
                }
                if (!RunPlannedTileBF16(pipelinePlan)) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline uint32_t OutputTileKTileCount(const SVDQLowRankOutputTilePlan& outputTilePlan) const
    {
        if (!outputTilePlan.HasWork()) {
            return 0;
        }
        return StageKTileCount(outputTilePlan.expert.stage);
    }

    __aicore__ inline SVDQLowRankTilePlan KTilePlan(
        const SVDQLowRankOutputTilePlan& outputTilePlan, uint32_t kTileIndex) const
    {
        if (!outputTilePlan.HasWork()) {
            return {};
        }
        const SVDQLowRankStagePlan& stage = outputTilePlan.expert.stage;
        const uint32_t kColumnOffset = kTileIndex * args_.tiling.kTile;
        if (kColumnOffset >= stage.inputColumns) {
            return {};
        }
        return SVDQLowRankTilePlan{
            outputTilePlan.expert,
            outputTilePlan.tileId,
            outputTilePlan.rowStart,
            outputTilePlan.rowCount,
            outputTilePlan.outputColumnOffset,
            outputTilePlan.outputColumnCount,
            kColumnOffset,
            Min(args_.tiling.kTile, stage.inputColumns - kColumnOffset),
        };
    }

    __aicore__ inline SVDQLowRankTileTensorPlan BuildTileTensorPlan(
        const SVDQLowRankTilePlan& tilePlan) const
    {
        if (!tilePlan.HasWork()) {
            return {};
        }
        const SVDQLowRankExpertPlan& expert = tilePlan.expert;
        const SVDQLowRankStagePlan& stage = expert.stage;
        const uint32_t rowOffset = tilePlan.rowStart - expert.tokenStart;
        return SVDQLowRankTileTensorPlan{
            tilePlan,
            MatrixAddress(expert.input, rowOffset, stage.inputStrideColumns, tilePlan.kColumnOffset),
            FactorTileAddress(expert, tilePlan.outputColumnOffset, tilePlan.kColumnOffset),
            MatrixAddress(expert.output, rowOffset, stage.outputStrideColumns, tilePlan.outputColumnOffset),
            AccumulatorAddress(expert, rowOffset, tilePlan.outputColumnOffset),
            stage.inputStrideColumns,
            stage.inputColumns,
            stage.outputStrideColumns,
            stage.outputStrideColumns,
            tilePlan.kColumnOffset == 0,
            tilePlan.kColumnOffset + tilePlan.kColumnCount >= stage.inputColumns,
        };
    }

    __aicore__ inline SVDQLowRankMmadTilePlan BuildMmadTilePlan(
        const SVDQLowRankTileTensorPlan& tilePlan) const
    {
        if (!tilePlan.HasWork()) {
            return {};
        }
        const uint32_t mActual = tilePlan.tile.rowCount;
        const uint32_t nActual = tilePlan.tile.outputColumnCount;
        const uint32_t kActual = tilePlan.tile.kColumnCount;
        const uint32_t mRound = RoundUp(mActual, SVDQ_LOWRANK_MMAD_M_ALIGNMENT);
        const uint32_t nRound = RoundUp(nActual, SVDQ_LOWRANK_MMAD_N_ALIGNMENT);
        const uint32_t kRound = RoundUp(kActual, SVDQ_LOWRANK_MMAD_K_ALIGNMENT);
        const bool canUseMmad = mActual <= SVDQ_LOWRANK_MMAD_M_TILE &&
                                nActual <= SVDQ_LOWRANK_MMAD_N_TILE &&
                                kActual <= SVDQ_LOWRANK_MMAD_K_TILE &&
                                mRound <= SVDQ_LOWRANK_MMAD_M_TILE &&
                                nRound <= SVDQ_LOWRANK_MMAD_N_TILE &&
                                kRound <= SVDQ_LOWRANK_MMAD_K_TILE;
        return SVDQLowRankMmadTilePlan{
            tilePlan,
            mActual,
            nActual,
            kActual,
            mRound,
            nRound,
            kRound,
            canUseMmad,
        };
    }

    __aicore__ inline uint64_t InputElementOffset(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t kOffset) const
    {
        return static_cast<uint64_t>(rowOffset) * tilePlan.inputStrideColumns + kOffset;
    }

    __aicore__ inline uint64_t FactorElementOffset(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t outputOffset, uint32_t kOffset) const
    {
        return static_cast<uint64_t>(outputOffset) * tilePlan.factorStrideColumns + kOffset;
    }

    __aicore__ inline uint64_t OutputElementOffset(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset) const
    {
        return static_cast<uint64_t>(rowOffset) * tilePlan.outputStrideColumns + outputOffset;
    }

    __aicore__ inline uint64_t AccumulatorElementOffset(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset) const
    {
        return static_cast<uint64_t>(rowOffset) * tilePlan.accumulatorStrideColumns + outputOffset;
    }

    __aicore__ inline bfloat16_t LoadInputBF16(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t kOffset) const
    {
        AscendC::GlobalTensor<bfloat16_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.input));
        return input.GetValue(InputElementOffset(tilePlan, rowOffset, kOffset));
    }

    __aicore__ inline bfloat16_t LoadFactorBF16(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t outputOffset, uint32_t kOffset) const
    {
        AscendC::GlobalTensor<bfloat16_t> factor;
        factor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.factor));
        return factor.GetValue(FactorElementOffset(tilePlan, outputOffset, kOffset));
    }

    __aicore__ inline bfloat16_t LoadOutputBF16(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset) const
    {
        AscendC::GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.output));
        return output.GetValue(OutputElementOffset(tilePlan, rowOffset, outputOffset));
    }

    __aicore__ inline float LoadAccumulatorFP32(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset) const
    {
        AscendC::GlobalTensor<float> accumulator;
        accumulator.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tilePlan.accumulator));
        return accumulator.GetValue(AccumulatorElementOffset(tilePlan, rowOffset, outputOffset));
    }

    __aicore__ inline void StoreOutputBF16(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset,
        bfloat16_t value) const
    {
        AscendC::GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.output));
        output.SetValue(OutputElementOffset(tilePlan, rowOffset, outputOffset), value);
    }

    __aicore__ inline void StoreAccumulatorFP32(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset, float value) const
    {
        AscendC::GlobalTensor<float> accumulator;
        accumulator.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tilePlan.accumulator));
        accumulator.SetValue(AccumulatorElementOffset(tilePlan, rowOffset, outputOffset), value);
    }

    __aicore__ inline float AccumulateScalarBF16(
        const SVDQLowRankTileTensorPlan& tilePlan, uint32_t rowOffset, uint32_t outputOffset) const
    {
        float accumulator = tilePlan.accumulatesFirstKTile ? 0.0F :
            LoadAccumulatorFP32(tilePlan, rowOffset, outputOffset);
        for (uint32_t kOffset = 0; kOffset < tilePlan.tile.kColumnCount; ++kOffset) {
            accumulator += static_cast<float>(LoadInputBF16(tilePlan, rowOffset, kOffset)) *
                           static_cast<float>(LoadFactorBF16(tilePlan, outputOffset, kOffset));
        }
        return accumulator;
    }

    __aicore__ inline bool RunScalarTileBF16(const SVDQLowRankTileTensorPlan& tilePlan) const
    {
        if (!tilePlan.HasWork()) {
            return false;
        }
        for (uint32_t rowOffset = 0; rowOffset < tilePlan.tile.rowCount; ++rowOffset) {
            for (uint32_t outputOffset = 0; outputOffset < tilePlan.tile.outputColumnCount; ++outputOffset) {
                const float accumulator = AccumulateScalarBF16(tilePlan, rowOffset, outputOffset);
                if (tilePlan.accumulatesLastKTile) {
                    StoreOutputBF16(tilePlan, rowOffset, outputOffset, static_cast<bfloat16_t>(accumulator));
                } else {
                    StoreAccumulatorFP32(tilePlan, rowOffset, outputOffset, accumulator);
                }
            }
        }
        return true;
    }

    __aicore__ inline bool RunMmadTileBF16(const SVDQLowRankMmadPipelinePlan& pipelinePlan) const
    {
        if (!pipelinePlan.HasCompletePipeline()) {
            return false;
        }
#ifdef __DAV_C220_CUBE__
        const SVDQLowRankMmadTilePlan& tile = pipelinePlan.buffer.tile;
        const SVDQLowRankTileTensorPlan& tensorPlan = tile.tile;

        AsdopsBuffer<ArchType::ASCEND_V220> buffers;
        AscendC::GlobalTensor<bfloat16_t> inputGm;
        AscendC::GlobalTensor<bfloat16_t> factorGm;
        AscendC::GlobalTensor<bfloat16_t> outputGm;
        AscendC::GlobalTensor<float> accumulatorGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.input));
        factorGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.factor));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.output));
        accumulatorGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tensorPlan.accumulator));

        auto l1Input = buffers.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(pipelinePlan.l1InputOffset);
        auto l1Factor = buffers.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(pipelinePlan.l1FactorOffset);
        auto l0A = buffers.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(pipelinePlan.l0AOffset);
        auto l0B = buffers.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(pipelinePlan.l0BOffset);
        auto l0C = buffers.GetBuffer<BufferType::ASCEND_L0C, float>(pipelinePlan.l0COffset);

        gm_to_l1<ArchType::ASCEND_V220, bfloat16_t, DataFormatT::ND, DataFormatT::NZ>(
            l1Input, inputGm, tile.mActual, tile.mRound, tensorPlan.inputStrideColumns,
            tile.kActual, tile.kRound, tensorPlan.inputStrideColumns);
        gm_to_l1<ArchType::ASCEND_V220, bfloat16_t, DataFormatT::ND, DataFormatT::ZN>(
            l1Factor, factorGm, tile.nActual, tile.nRound, tensorPlan.factorStrideColumns,
            tile.kActual, tile.kRound, tensorPlan.factorStrideColumns);
        AscendC::PipeBarrier<PIPE_MTE2>();

        l1_to_l0_a<ArchType::ASCEND_V220, bfloat16_t, false, DataFormatT::NZ, DataFormatT::ZZ>(
            l0A, l1Input, tile.mRound, tile.kRound, tile.mRound, tile.kRound, tile.mRound, tile.kRound);
        l1_to_l0_b<ArchType::ASCEND_V220, bfloat16_t, true, DataFormatT::ZN, DataFormatT::NZ>(
            l0B, l1Factor, tile.nRound, tile.kRound, tile.nRound, tile.kRound, tile.nRound, tile.kRound);
        AscendC::PipeBarrier<PIPE_MTE1>();

        const bool initC = pipelinePlan.initAccumulator;
        (void)mmad<ArchType::ASCEND_V220, bfloat16_t, bfloat16_t, float, false>(
            l0C, l0A, l0B, tile.mActual, tile.nActual, tile.kActual, initC);
        AscendC::PipeBarrier<PIPE_M>();

#ifdef SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK
        (void)l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, float, float>(
            accumulatorGm, l0C, tile.mActual, tile.nActual, tile.nRound,
            tensorPlan.accumulatorStrideColumns);
#endif
        if (!pipelinePlan.storesOutput) {
            (void)accumulatorGm;
            // Production cube execution keeps partial K-loop sums resident in
            // L0C. Defining SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK mirrors the
            // current FP32 accumulator to GM for host-readable validation.
            return true;
        }
        (void)l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, bfloat16_t, float>(
            outputGm, l0C, tile.mActual, tile.nActual, tile.nRound, tensorPlan.outputStrideColumns);
        return true;
#else
        return false;
#endif
    }

    __aicore__ inline bool RunPlannedTileBF16(const SVDQLowRankMmadPipelinePlan& pipelinePlan) const
    {
        if (!pipelinePlan.HasCompletePipeline()) {
            return false;
        }
#ifdef __DAV_C220_CUBE__
        return RunMmadTileBF16(pipelinePlan);
#else
        return RunScalarTileBF16(pipelinePlan.buffer.tile.tile);
#endif
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
        for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex) {
            const SVDQLowRankStagePlan stage = StagePlan(stageIndex);
            if (!stage.HasCompleteContract()) {
                return;
            }
            for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId) {
                const SVDQLowRankExpertPlan expertPlan = ExpertStagePlan(stageIndex, expertId);
                if (expertPlan.tokenCount > 0 && !expertPlan.HasWork()) {
                    return;
                }
            }
        }
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t runtimeCoreCount = AscendC::GetBlockNum();
        const uint32_t scheduledCoreCount = args_.tiling.coreCount <= runtimeCoreCount ? args_.tiling.coreCount : runtimeCoreCount;
        for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex) {
            if (!ExecuteStage(stageIndex, coreIdx, scheduledCoreCount)) {
                return;
            }
            AscendC::SyncAll();
        }
        /*
         * Legacy flat scheduling is kept here only as source history until the
         * production op is enabled. Executable low-rank work must remain
         * stage-ordered so L2 stages cannot read rank workspace before all L1
         * writers have reached the cross-core barrier above.
         *
         * const SVDQLowRankCoreTileRange tileRange = CoreTileRange(coreIdx, scheduledCoreCount);
        for (uint32_t tileOffset = 0; tileOffset < tileRange.tileCount; ++tileOffset) {
            const SVDQLowRankOutputTilePlan outputTilePlan = OutputTilePlan(tileRange.tileStart + tileOffset);
            if (!outputTilePlan.HasWork()) {
                return;
            }
            const uint32_t kTileCount = OutputTileKTileCount(outputTilePlan);
            for (uint32_t kTileIndex = 0; kTileIndex < kTileCount; ++kTileIndex) {
                const SVDQLowRankTilePlan tilePlan = KTilePlan(outputTilePlan, kTileIndex);
                if (!tilePlan.HasWork()) {
                    return;
                }
                const SVDQLowRankTileTensorPlan tileTensorPlan = BuildTileTensorPlan(tilePlan);
                if (!tileTensorPlan.HasWork()) {
                    return;
                }
                const SVDQLowRankMmadTilePlan mmadTilePlan = BuildMmadTilePlan(tileTensorPlan);
                if (!mmadTilePlan.HasCompatibleShape()) {
                    return;
                }
                const SVDQLowRankMmadBufferPlan bufferPlan = BuildMmadBufferPlan(mmadTilePlan);
                if (!bufferPlan.HasCompleteFootprint()) {
                    return;
                }
                const SVDQLowRankMmadPipelinePlan pipelinePlan = BuildMmadPipelinePlan(bufferPlan);
                if (!pipelinePlan.HasCompletePipeline()) {
                    return;
                }
                if (!RunPlannedTileBF16(pipelinePlan)) {
                    return;
                }
            }
        }
        // The fused AIC math body will keep the low-rank bottleneck on-chip:
        // input BF16 -> down factor GEMM -> rank tile -> up factor GEMM -> projection BF16 GM.
         */
    }

private:
    __aicore__ inline bool HasCompleteSecondUpContract() const
    {
        return args_.secondUpFactor != nullptr && args_.tiling.secondRankColumns > 0 &&
               args_.tiling.secondInputColumnOffset + args_.tiling.secondRankColumns <= TotalRankColumns() &&
               args_.tiling.secondOutputColumnOffset < args_.tiling.outputColumns &&
               args_.tiling.outputColumnOffset < args_.tiling.secondOutputColumnOffset;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t RoundUp(uint32_t value, uint32_t alignment) const
    {
        return alignment == 0 ? 0 : CeilDiv(value, alignment) * alignment;
    }

    __aicore__ inline SVDQLowRankStagePlan BuildDownStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_DOWN_PROJECT,
            args_.tiling.downFactorId,
            args_.downFactor,
            args_.tiling.inputColumns,
            TotalRankColumns(),
            args_.tiling.inputColumns,
            TotalRankColumns(),
            args_.tiling.inputColumnOffset,
            0,
            0,
            false,
        };
    }

    __aicore__ inline SVDQLowRankStagePlan BuildPrimaryUpStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_UP_PROJECT,
            args_.tiling.upFactorId,
            args_.upFactor,
            args_.tiling.rankColumns,
            PrimaryOutputColumns(),
            TotalRankColumns(),
            args_.tiling.outputColumns,
            args_.tiling.inputColumnOffset,
            args_.tiling.outputColumnOffset,
            0,
            true,
        };
    }

    __aicore__ inline SVDQLowRankStagePlan BuildSecondUpStagePlan() const
    {
        return SVDQLowRankStagePlan{
            SVDQ_LOWRANK_STAGE_SECOND_UP_PROJECT,
            args_.tiling.secondUpFactorId,
            args_.secondUpFactor,
            args_.tiling.secondRankColumns,
            args_.tiling.outputColumns - args_.tiling.secondOutputColumnOffset,
            TotalRankColumns(),
            args_.tiling.outputColumns,
            args_.tiling.secondInputColumnOffset,
            args_.tiling.secondOutputColumnOffset,
            0,
            true,
        };
    }

    __aicore__ inline GM_ADDR MatrixAddress(
        GM_ADDR base, uint32_t row, uint32_t strideColumns, uint32_t columnOffset) const
    {
        return base + (static_cast<uint64_t>(row) * strideColumns + columnOffset) * SVDQ_LOWRANK_BF16_BYTES;
    }

    __aicore__ inline GM_ADDR FactorAddress(const SVDQLowRankStagePlan& stage, uint32_t expertId) const
    {
        const uint64_t expertOffset =
            static_cast<uint64_t>(expertId) * stage.inputColumns * stage.outputColumns + stage.factorColumnOffset;
        return stage.factor + expertOffset * SVDQ_LOWRANK_BF16_BYTES;
    }

    __aicore__ inline GM_ADDR AccumulatorAddress(
        const SVDQLowRankExpertPlan& expert, uint32_t row, uint32_t columnOffset) const
    {
        return args_.accumulator +
               (static_cast<uint64_t>(expert.tokenStart + row) * expert.stage.outputStrideColumns + columnOffset) *
                   sizeof(float);
    }

    __aicore__ inline GM_ADDR FactorTileAddress(
        const SVDQLowRankExpertPlan& expert, uint32_t outputColumnOffset, uint32_t kColumnOffset) const
    {
        const uint64_t tileOffset =
            static_cast<uint64_t>(outputColumnOffset) * expert.stage.inputColumns + kColumnOffset;
        return expert.factor + tileOffset * SVDQ_LOWRANK_BF16_BYTES;
    }

    SVDQFusedDownUpArgs args_{};
};

class SVDQLowRankDebugReadback {
public:
    __aicore__ inline SVDQLowRankDebugReadback() {}

    __aicore__ inline void Init(const SVDQFusedDownUpArgs& args)
    {
        args_ = args;
    }

    __aicore__ inline bool IsEnabled() const
    {
#ifdef SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK
        return true;
#else
        return false;
#endif
    }

    __aicore__ inline bool HasCompleteContract() const
    {
        SVDQFusedDownUp lowRankOp;
        lowRankOp.Init(args_);
        return IsEnabled() && lowRankOp.HasCompleteContract();
    }

    __aicore__ inline bool Process() const
    {
        if (!HasCompleteContract()) {
            return false;
        }
        SVDQFusedDownUp lowRankOp;
        lowRankOp.Init(args_);
        lowRankOp.Process();
        return true;
    }

private:
    SVDQFusedDownUpArgs args_{};
};

}  // namespace DispatchFFNCombineW4A8SVDQImpl

#endif  // SVDQ_FUSED_DOWN_UP_HPP
