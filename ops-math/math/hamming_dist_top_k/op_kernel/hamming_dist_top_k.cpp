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
 * \brief HammingDistTopK AscendC kernel. Similarity is computed by CUBE Matmul.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr int32_t INF_SCORE = 0x3fffffff;
constexpr int32_t NEG_INF_SIM = -0x3fffffff;
constexpr int64_t WORKSPACE_ALIGN = 512;

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

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline int64_t AlignUp(int64_t x, int64_t align)
{
    return align == 0 ? x : (x + align - 1) / align * align;
}

__aicore__ inline int8_t BitToSign(uint8_t value, int64_t bit)
{
    return ((value >> bit) & 1U) == 0 ? static_cast<int8_t>(-1) : static_cast<int8_t>(1);
}

class HammingDistTopKCubeKernel {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k, GM_ADDR seqLen, GM_ADDR chunkSize,
        GM_ADDR keyBlockTable, GM_ADDR indices, GM_ADDR workspace, const HammingDistTopKTilingData* tilingData,
        TPipe* pipe)
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
        tileN2_ = tilingData->tileN2;
        expandedDim_ = tilingData->expandedDim;
        usedCoreNum_ = tilingData->usedCoreNum;
        hasChunkSize_ = tilingData->hasChunkSize != 0;
        hasKeyBlockTable_ = tilingData->hasKeyBlockTable != 0;
        mmTiling_ = &tilingData->mmTiling;
        pipe_ = pipe;

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

        InitWorkspace(workspace);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_, mmTiling_);
        }

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t pairLoops = CeilDiv(totalPairs_, usedCoreNum_);
        const int64_t tileLoops = CeilDiv(maxSeqLen_, tileN2_);

        for (int64_t pairLoop = 0; pairLoop < pairLoops; ++pairLoop) {
            const int64_t pairId = blockIdx + pairLoop * usedCoreNum_;
            const bool active = pairId < totalPairs_;
            PairRuntime runtime;

            if ASCEND_IS_AIV {
                if (active) {
                    InitPair(pairId, runtime);
                    ExpandQuery(runtime.batchIdx, runtime.headIdx);
                }
            }
            SyncAll<false>();

            for (int64_t tileLoop = 0; tileLoop < tileLoops; ++tileLoop) {
                const int64_t tileStart = tileLoop * tileN2_;
                const int64_t curN = Min(tileN2_, maxSeqLen_ - tileStart);
                if ASCEND_IS_AIV {
                    if (active && curN > 0) {
                        ExpandKeyTile(runtime.batchIdx, runtime.headIdx, tileStart, curN);
                    }
                }
                SyncAll<false>();

                if ASCEND_IS_AIC {
                    if (active && curN > 0) {
                        MatmulTile(curN);
                    }
                }
                SyncAll<false>();

                if ASCEND_IS_AIV {
                    if (active && curN > 0) {
                        ReduceTile(runtime, tileStart, curN);
                    }
                }
                SyncAll<false>();
            }

            SyncAll<false>();
        }

        // Barrier: every (batch,head) pair's per-chunk scores are written before the
        // head-agnostic aggregation reads them. Both AIC and AIV must reach it.
        SyncAll<false>();

        // Phase 2: one core per batch sums all heads and selects the final TopK chunks.
        if ASCEND_IS_AIV {
            for (int64_t batchIdx = blockIdx; batchIdx < batch_; batchIdx += usedCoreNum_) {
                FinalizeBatch(batchIdx);
            }
        }

        if ASCEND_IS_AIC {
            mm_.End();
        }
    }

private:
    struct PairRuntime {
        int64_t pairId = 0;
        int64_t batchIdx = 0;
        int64_t headIdx = 0;
        int64_t seqLen = 0;
        int64_t chunkSize = 1;
        int64_t effectLen = 0;
    };

    __aicore__ inline void InitWorkspace(GM_ADDR workspace)
    {
        int64_t offset = 0;
        const int64_t scoreBytes = batch_ * outputChunkLen_ * static_cast<int64_t>(sizeof(int32_t));
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
        offset = AlignUp(offset + scoreBytes, WORKSPACE_ALIGN);

        const int64_t queryBytes = usedCoreNum_ * expandedDim_ * static_cast<int64_t>(sizeof(int8_t));
        queryExpandGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace + offset));
        offset = AlignUp(offset + queryBytes, WORKSPACE_ALIGN);

        const int64_t keyBytes = usedCoreNum_ * tileN2_ * expandedDim_ * static_cast<int64_t>(sizeof(int8_t));
        keyExpandGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace + offset));
        offset = AlignUp(offset + keyBytes, WORKSPACE_ALIGN);

        const int64_t mmOutBytes = usedCoreNum_ * tileN2_ * static_cast<int64_t>(sizeof(int32_t));
        mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
        offset = AlignUp(offset + mmOutBytes, WORKSPACE_ALIGN);

        chunkBestGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
    }

    __aicore__ inline int64_t CoreOffset(int64_t elements) const
    {
        return static_cast<int64_t>(GetBlockIdx()) * elements;
    }

    __aicore__ inline GlobalTensor<int8_t> QueryWorkspace()
    {
        return queryExpandGm_[CoreOffset(expandedDim_)];
    }

    __aicore__ inline GlobalTensor<int8_t> KeyWorkspace()
    {
        return keyExpandGm_[CoreOffset(tileN2_ * expandedDim_)];
    }

    __aicore__ inline GlobalTensor<int32_t> MatmulWorkspace()
    {
        return mmOutGm_[CoreOffset(tileN2_)];
    }

    // Per-chunk best similarity persisted per (batch,head) pair (not reused per core),
    // so phase 2 can sum every head's contribution for a batch.
    __aicore__ inline GlobalTensor<int32_t> ChunkBestForPair(int64_t pairId)
    {
        return chunkBestGm_[pairId * maxSeqLen_];
    }

    __aicore__ inline int64_t QueryOffset(int64_t batchIdx, int64_t headIdx) const
    {
        const int64_t qHeadIdx = qHead_ == 1 ? 0 : headIdx % qHead_;
        return (batchIdx * qHead_ + qHeadIdx) * dimBytes_;
    }

    __aicore__ inline int64_t KeyOffset(int64_t batchIdx, int64_t headIdx, int64_t tokenIdx)
    {
        if (hasKeyBlockTable_) {
            const int64_t logicalBlock = tokenIdx / blockSize_;
            const int64_t blockOffset = tokenIdx - logicalBlock * blockSize_;
            int32_t physicalBlock = keyBlockTableGm_.GetValue(batchIdx * blockCount_ + logicalBlock);
            physicalBlock = physicalBlock < 0 ? 0 : physicalBlock;
            return ((static_cast<int64_t>(physicalBlock) * head_ + headIdx) * blockSize_ + blockOffset) * dimBytes_;
        }
        return ((batchIdx * head_ + headIdx) * maxSeqLen_ + tokenIdx) * dimBytes_;
    }

    __aicore__ inline int64_t EffectiveLen(int64_t seqLen, int64_t chunkSize)
    {
        int64_t effectLen = seqLen - (seqLen % chunkSize + 16);
        if (effectLen <= 0) {
            effectLen = seqLen - seqLen % chunkSize;
        }
        return Max<int64_t>(effectLen, 0);
    }

    __aicore__ inline void InitPair(int64_t pairId, PairRuntime& runtime)
    {
        runtime.pairId = pairId;
        runtime.batchIdx = pairId / head_;
        runtime.headIdx = pairId - runtime.batchIdx * head_;

        // Phase 1 only fills this pair's per-chunk best similarity; the head-agnostic
        // TopK over the summed scores is done per batch in FinalizeBatch (phase 2).
        GlobalTensor<int32_t> chunkBest = ChunkBestForPair(pairId);
        for (int64_t i = 0; i < maxSeqLen_; ++i) {
            chunkBest.SetValue(i, NEG_INF_SIM);
        }

        runtime.seqLen = seqLenGm_.GetValue(runtime.batchIdx);
        runtime.seqLen = Min(Max<int64_t>(runtime.seqLen, 0), maxSeqLen_);
        runtime.chunkSize = hasChunkSize_ ? chunkSizeGm_.GetValue(runtime.batchIdx) : 1;
        runtime.chunkSize = runtime.chunkSize <= 0 ? 1 : runtime.chunkSize;
        runtime.effectLen = EffectiveLen(runtime.seqLen, runtime.chunkSize);
    }

    __aicore__ inline void ExpandQuery(int64_t batchIdx, int64_t headIdx)
    {
        const int64_t queryOffset = QueryOffset(batchIdx, headIdx);
        GlobalTensor<int8_t> queryWorkspace = QueryWorkspace();
        for (int64_t byteIdx = 0; byteIdx < dimBytes_; ++byteIdx) {
            const uint8_t value = queryGm_.GetValue(queryOffset + byteIdx);
            const int64_t bitBase = byteIdx * 8;
            for (int64_t bit = 0; bit < 8; ++bit) {
                queryWorkspace.SetValue(bitBase + bit, BitToSign(value, bit));
            }
        }
    }

    __aicore__ inline void ExpandKeyTile(int64_t batchIdx, int64_t headIdx, int64_t tileStart, int64_t curN)
    {
        GlobalTensor<int8_t> keyWorkspace = KeyWorkspace();
        for (int64_t n = 0; n < curN; ++n) {
            const int64_t keyOffset = KeyOffset(batchIdx, headIdx, tileStart + n);
            for (int64_t byteIdx = 0; byteIdx < dimBytes_; ++byteIdx) {
                const uint8_t value = keyGm_.GetValue(keyOffset + byteIdx);
                const int64_t bitBase = byteIdx * 8;
                for (int64_t bit = 0; bit < 8; ++bit) {
                    const int64_t bitIdx = bitBase + bit;
                    keyWorkspace.SetValue(bitIdx * tileN2_ + n, BitToSign(value, bit));
                }
            }
        }
    }

    __aicore__ inline void MatmulTile(int64_t curN)
    {
        GlobalTensor<int8_t> queryWorkspace = QueryWorkspace();
        GlobalTensor<int8_t> keyWorkspace = KeyWorkspace();
        GlobalTensor<int32_t> mmWorkspace = MatmulWorkspace();
        mm_.SetTensorA(queryWorkspace[0], false);
        mm_.SetTensorB(keyWorkspace[0], false);
        mm_.SetTail(1, curN);
        mm_.IterateAll(mmWorkspace[0]);
    }

    __aicore__ inline void ReduceTile(const PairRuntime& runtime, int64_t tileStart, int64_t curN)
    {
        if (runtime.effectLen <= tileStart) {
            return;
        }
        GlobalTensor<int32_t> mmWorkspace = MatmulWorkspace();
        GlobalTensor<int32_t> chunkBest = ChunkBestForPair(runtime.pairId);
        const int64_t reduceN = Min(curN, runtime.effectLen - tileStart);
        for (int64_t n = 0; n < reduceN; ++n) {
            const int64_t tokenIdx = tileStart + n;
            const int64_t chunkIdx = tokenIdx / runtime.chunkSize;
            const int32_t sim = mmWorkspace.GetValue(n);
            const int32_t current = chunkBest.GetValue(chunkIdx);
            if (sim > current) {
                chunkBest.SetValue(chunkIdx, sim);
            }
        }
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
            const int32_t curScore = scoreGm_.GetValue(scoreBase + i);
            const int32_t curIndex = indicesGm_.GetValue(outBase + i);
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

    __aicore__ inline void FillTailChunks(int64_t outBase, int64_t kLimit, int64_t validChunks, int64_t allChunks)
    {
        const int64_t tailChunks = allChunks - validChunks;
        if (tailChunks <= 0 || kLimit >= outputChunkLen_) {
            return;
        }
        int64_t writePos = Max(kLimit, outputChunkLen_ - tailChunks);
        for (int64_t chunk = validChunks; chunk < allChunks && writePos < outputChunkLen_; ++chunk, ++writePos) {
            indicesGm_.SetValue(outBase + writePos, static_cast<int32_t>(chunk));
        }
    }

    // Phase 2: aggregate every head's per-chunk best similarity for one batch and
    // pick a single, head-agnostic TopK set of chunks. Output is [batch, outputChunkLen].
    __aicore__ inline void FinalizeBatch(int64_t batchIdx)
    {
        const int64_t outBase = batchIdx * outputChunkLen_;
        for (int64_t i = 0; i < outputChunkLen_; ++i) {
            indicesGm_.SetValue(outBase + i, -1);
            scoreGm_.SetValue(outBase + i, INF_SCORE);
        }

        int64_t seqLen = seqLenGm_.GetValue(batchIdx);
        seqLen = Min(Max<int64_t>(seqLen, 0), maxSeqLen_);
        int64_t chunkSize = hasChunkSize_ ? chunkSizeGm_.GetValue(batchIdx) : 1;
        chunkSize = chunkSize <= 0 ? 1 : chunkSize;
        const int64_t effectLen = EffectiveLen(seqLen, chunkSize);
        const int64_t validChunks = effectLen / chunkSize;
        const int64_t allChunks = (seqLen + chunkSize - 1) / chunkSize;

        int64_t kLimit = kGm_.GetValue(batchIdx);
        kLimit = Min(Max<int64_t>(kLimit, 0), outputChunkLen_);
        if (topkAttr_ > 0) {
            kLimit = Min(kLimit, topkAttr_);
        }

        for (int64_t chunk = 0; chunk < validChunks; ++chunk) {
            int32_t agg = 0;
            bool any = false;
            for (int64_t h = 0; h < head_; ++h) {
                GlobalTensor<int32_t> chunkBest = ChunkBestForPair(batchIdx * head_ + h);
                const int32_t sim = chunkBest.GetValue(chunk);
                if (sim != NEG_INF_SIM) {
                    agg += sim;
                    any = true;
                }
            }
            if (any) {
                InsertTopK(outBase, outBase, kLimit, -agg, static_cast<int32_t>(chunk));
            }
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
    GlobalTensor<int8_t> queryExpandGm_;
    GlobalTensor<int8_t> keyExpandGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<int32_t> chunkBestGm_;

    Matmul<AType, BType, CType, BiasType> mm_;
    const TCubeTiling* mmTiling_ = nullptr;
    TPipe* pipe_ = nullptr;

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
    int64_t tileN2_ = 0;
    int64_t expandedDim_ = 0;
    int64_t usedCoreNum_ = 1;
    bool hasChunkSize_ = false;
    bool hasKeyBlockTable_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void hamming_dist_top_k(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k,
    GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    HammingDistTopKCubeKernel op;
    op.Init(query, keyCompressed, k, seqLen, chunkSize, keyBlockTable, indices, userWorkspace, &tilingData, &pipe);
    op.Process();
}
