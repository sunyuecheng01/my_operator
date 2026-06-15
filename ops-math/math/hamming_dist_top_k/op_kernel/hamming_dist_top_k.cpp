/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hamming_dist_top_k.cpp
 * \brief HammingDistTopK AscendC kernel.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t INF_DISTANCE = 0x3fffffff;

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

__aicore__ inline int32_t PopCount8(uint8_t x)
{
    uint32_t v = static_cast<uint32_t>(x);
    v = v - ((v >> 1) & 0x55U);
    v = (v & 0x33U) + ((v >> 2) & 0x33U);
    return static_cast<int32_t>((v + (v >> 4)) & 0x0FU);
}

class HammingDistTopKKernel {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k, GM_ADDR seqLen, GM_ADDR chunkSize,
        GM_ADDR keyBlockTable, GM_ADDR indices, GM_ADDR workspace, const HammingDistTopKTilingData* tilingData)
    {
        batch_ = tilingData->batch;
        qHead_ = tilingData->qHead;
        head_ = tilingData->head;
        maxSeqLen_ = tilingData->maxSeqLen;
        dimBytes_ = tilingData->dimBytes;
        outputChunkLen_ = tilingData->outputChunkLen;
        blockCount_ = tilingData->blockCount;
        blockSize_ = tilingData->blockSize;
        totalPairs_ = tilingData->totalPairs;
        topkAttr_ = tilingData->topkAttr;
        hasChunkSize_ = tilingData->hasChunkSize != 0;
        hasKeyBlockTable_ = tilingData->hasKeyBlockTable != 0;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(keyCompressed));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLen));
        if (hasChunkSize_) {
            chunkSizeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunkSize));
        }
        if (hasKeyBlockTable_) {
            keyBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyBlockTable));
        }
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
    }

    __aicore__ inline void Process()
    {
        for (int64_t pairId = static_cast<int64_t>(GetBlockIdx()); pairId < totalPairs_;
             pairId += static_cast<int64_t>(GetBlockNum())) {
            ProcessPair(pairId);
        }
    }

private:
    __aicore__ inline int64_t QueryOffset(int64_t batchIdx, int64_t headIdx) const
    {
        int64_t qHeadIdx = qHead_ == 1 ? 0 : headIdx % qHead_;
        return (batchIdx * qHead_ + qHeadIdx) * dimBytes_;
    }

    __aicore__ inline int64_t KeyOffset(int64_t batchIdx, int64_t headIdx, int64_t tokenIdx)
    {
        if (hasKeyBlockTable_) {
            int64_t logicalBlock = tokenIdx / blockSize_;
            int64_t blockOffset = tokenIdx - logicalBlock * blockSize_;
            int32_t physicalBlock = keyBlockTableGm_.GetValue(batchIdx * blockCount_ + logicalBlock);
            physicalBlock = physicalBlock < 0 ? 0 : physicalBlock;
            return ((static_cast<int64_t>(physicalBlock) * head_ + headIdx) * blockSize_ + blockOffset) * dimBytes_;
        }
        return ((batchIdx * head_ + headIdx) * maxSeqLen_ + tokenIdx) * dimBytes_;
    }

    __aicore__ inline int32_t HammingDistance(int64_t queryOffset, int64_t keyOffset)
    {
        int32_t distance = 0;
        for (int64_t dim = 0; dim < dimBytes_; ++dim) {
            uint8_t q = queryGm_.GetValue(queryOffset + dim);
            uint8_t k = keyGm_.GetValue(keyOffset + dim);
            distance += PopCount8(static_cast<uint8_t>(q ^ k));
        }
        return distance;
    }

    __aicore__ inline int32_t ChunkDistance(int64_t batchIdx, int64_t headIdx, int64_t chunkStart, int64_t chunkLen)
    {
        int64_t queryOffset = QueryOffset(batchIdx, headIdx);
        int32_t best = INF_DISTANCE;
        for (int64_t i = 0; i < chunkLen; ++i) {
            const int64_t keyOffset = KeyOffset(batchIdx, headIdx, chunkStart + i);
            best = Min(best, HammingDistance(queryOffset, keyOffset));
        }
        return best;
    }

    __aicore__ inline bool IsBetter(int32_t score, int32_t index, int32_t curScore, int32_t curIndex)
    {
        return score < curScore || (score == curScore && (curIndex < 0 || index < curIndex));
    }

    __aicore__ inline void InsertTopK(int64_t outBase, int64_t scoreBase, int64_t kLimit, int32_t score, int32_t index)
    {
        if (kLimit <= 0) {
            return;
        }
        int64_t pos = -1;
        for (int64_t i = 0; i < kLimit; ++i) {
            int32_t curScore = scoreGm_.GetValue(scoreBase + i);
            int32_t curIndex = indicesGm_.GetValue(outBase + i);
            if (IsBetter(score, index, curScore, curIndex)) {
                pos = i;
                break;
            }
        }
        if (pos < 0) {
            return;
        }
        for (int64_t i = kLimit - 1; i > pos; --i) {
            scoreGm_.SetValue(scoreBase + i, scoreGm_.GetValue(scoreBase + i - 1));
            indicesGm_.SetValue(outBase + i, indicesGm_.GetValue(outBase + i - 1));
        }
        scoreGm_.SetValue(scoreBase + pos, score);
        indicesGm_.SetValue(outBase + pos, index);
    }

    __aicore__ inline int64_t EffectiveLen(int64_t seqLen, int64_t chunkSize)
    {
        int64_t effectLen = seqLen - (seqLen % chunkSize + 16);
        if (effectLen <= 0) {
            effectLen = seqLen - seqLen % chunkSize;
        }
        return Max<int64_t>(effectLen, 0);
    }

    __aicore__ inline void FillTailChunks(int64_t outBase, int64_t kLimit, int64_t validChunks, int64_t allChunks)
    {
        int64_t tailChunks = allChunks - validChunks;
        if (tailChunks <= 0 || kLimit >= outputChunkLen_) {
            return;
        }
        int64_t writePos = Max(kLimit, outputChunkLen_ - tailChunks);
        for (int64_t chunk = validChunks; chunk < allChunks && writePos < outputChunkLen_; ++chunk, ++writePos) {
            indicesGm_.SetValue(outBase + writePos, static_cast<int32_t>(chunk));
        }
    }

    __aicore__ inline void ProcessPair(int64_t pairId)
    {
        const int64_t batchIdx = pairId / head_;
        const int64_t headIdx = pairId - batchIdx * head_;
        const int64_t outBase = pairId * outputChunkLen_;
        const int64_t scoreBase = pairId * outputChunkLen_;

        for (int64_t i = 0; i < outputChunkLen_; ++i) {
            indicesGm_.SetValue(outBase + i, -1);
            scoreGm_.SetValue(scoreBase + i, INF_DISTANCE);
        }

        int64_t seqLen = seqLenGm_.GetValue(batchIdx);
        seqLen = Min(Max<int64_t>(seqLen, 0), maxSeqLen_);
        int64_t chunkSize = hasChunkSize_ ? chunkSizeGm_.GetValue(batchIdx) : 1;
        chunkSize = chunkSize <= 0 ? 1 : chunkSize;

        int64_t kLimit = kGm_.GetValue(batchIdx);
        kLimit = Min(Max<int64_t>(kLimit, 0), outputChunkLen_);
        if (topkAttr_ > 0) {
            kLimit = Min(kLimit, topkAttr_);
        }

        const int64_t effectLen = EffectiveLen(seqLen, chunkSize);
        const int64_t validChunks = effectLen / chunkSize;
        const int64_t allChunks = (seqLen + chunkSize - 1) / chunkSize;
        for (int64_t chunk = 0; chunk < validChunks; ++chunk) {
            const int64_t start = chunk * chunkSize;
            const int32_t score = ChunkDistance(batchIdx, headIdx, start, chunkSize);
            InsertTopK(outBase, scoreBase, kLimit, score, static_cast<int32_t>(chunk));
        }
        FillTailChunks(outBase, kLimit, validChunks, allChunks);
    }

private:
    GlobalTensor<uint8_t> queryGm_;
    GlobalTensor<uint8_t> keyGm_;
    GlobalTensor<int32_t> kGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> chunkSizeGm_;
    GlobalTensor<int32_t> keyBlockTableGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> scoreGm_;

    int64_t batch_ = 0;
    int64_t qHead_ = 0;
    int64_t head_ = 0;
    int64_t maxSeqLen_ = 0;
    int64_t dimBytes_ = 0;
    int64_t outputChunkLen_ = 0;
    int64_t blockCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalPairs_ = 0;
    int64_t topkAttr_ = 0;
    bool hasChunkSize_ = false;
    bool hasKeyBlockTable_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void hamming_dist_top_k(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k,
    GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    HammingDistTopKKernel op;
    op.Init(query, keyCompressed, k, seqLen, chunkSize, keyBlockTable, indices, userWorkspace, &tilingData);
    op.Process();
}
