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
 * \file hash_block_top_k.cpp
 * \brief HashBlockTopK AscendC kernel. Hash bits are expanded to int8 signs and scored by CUBE Matmul.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {
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

class HashBlockTopKCubeKernel {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;

    __aicore__ inline void Init(GM_ADDR hashQ, GM_ADDR hashKCache, GM_ADDR k, GM_ADDR hashKBlockTable,
        GM_ADDR seqLen, GM_ADDR newBlockTable, GM_ADDR workspace, const HashBlockTopKTilingData* tilingData,
        TPipe* pipe)
    {
        batch_ = tilingData->batch;
        qSeqLen_ = tilingData->qSeqLen;
        qHead_ = tilingData->qHead;
        head_ = tilingData->head;
        physicalBlockCount_ = tilingData->physicalBlockCount;
        blockCount_ = tilingData->blockCount;
        blockSize_ = tilingData->blockSize;
        dimBytes_ = tilingData->dimBytes;
        pairCount_ = tilingData->pairCount;
        totalTasks_ = tilingData->totalTasks;
        tileN2_ = tilingData->tileN2;
        expandedDim_ = tilingData->expandedDim;
        usedCoreNum_ = tilingData->usedCoreNum;
        kIsScalar_ = tilingData->kIsScalar != 0;
        seqLenIsScalar_ = tilingData->seqLenIsScalar != 0;
        mmTiling_ = &tilingData->mmTiling;
        pipe_ = pipe;

        hashQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(hashQ));
        hashKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(hashKCache));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hashKBlockTable));
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLen));
        newBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(newBlockTable));

        InitWorkspace(workspace);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_, mmTiling_);
        }

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t taskLoops = CeilDiv(totalTasks_, usedCoreNum_);
        for (int64_t taskLoop = 0; taskLoop < taskLoops; ++taskLoop) {
            const int64_t taskId = blockIdx + taskLoop * usedCoreNum_;
            ProcessTask(taskId);
            SyncAll<false>();
        }

        SyncAll<false>();
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
    struct TaskRuntime {
        bool active = false;
        int64_t batchIdx = 0;
        int64_t logicalBlock = 0;
        int64_t physicalBlock = 0;
        int64_t validTokens = 0;
        int32_t blockScore = NEG_INF_SIM;
    };

    __aicore__ inline void InitWorkspace(GM_ADDR workspace)
    {
        int64_t offset = 0;
        const int64_t blockScoreBytes = batch_ * blockCount_ * static_cast<int64_t>(sizeof(int32_t));
        blockScoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
        offset = AlignUp(offset + blockScoreBytes, WORKSPACE_ALIGN);

        const int64_t queryBytes = usedCoreNum_ * expandedDim_ * static_cast<int64_t>(sizeof(int8_t));
        queryExpandGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace + offset));
        offset = AlignUp(offset + queryBytes, WORKSPACE_ALIGN);

        const int64_t keyBytes = usedCoreNum_ * tileN2_ * expandedDim_ * static_cast<int64_t>(sizeof(int8_t));
        keyExpandGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace + offset));
        offset = AlignUp(offset + keyBytes, WORKSPACE_ALIGN);

        const int64_t mmOutBytes = usedCoreNum_ * tileN2_ * static_cast<int64_t>(sizeof(int32_t));
        mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
        offset = AlignUp(offset + mmOutBytes, WORKSPACE_ALIGN);

        const int64_t tokenScoreBytes = usedCoreNum_ * tileN2_ * static_cast<int64_t>(sizeof(int32_t));
        tokenScoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offset));
        (void)tokenScoreBytes;
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

    __aicore__ inline GlobalTensor<int32_t> TokenScoreWorkspace()
    {
        return tokenScoreGm_[CoreOffset(tileN2_)];
    }

    __aicore__ inline int64_t QueryOffset(int64_t batchIdx, int64_t qSeqIdx, int64_t qHeadIdx) const
    {
        return ((batchIdx * qSeqLen_ + qSeqIdx) * qHead_ + qHeadIdx) * dimBytes_;
    }

    __aicore__ inline int64_t KeyOffset(int64_t physicalBlock, int64_t batchIdx, int64_t tokenInBlock,
        int64_t headIdx) const
    {
        return ((((physicalBlock * batch_ + batchIdx) * blockSize_ + tokenInBlock) * head_ + headIdx) * dimBytes_);
    }

    __aicore__ inline int64_t QHeadForPair(int64_t pairIdx) const
    {
        if (qHead_ >= head_) {
            return pairIdx;
        }
        const int64_t kvPerQ = head_ / qHead_;
        return pairIdx / kvPerQ;
    }

    __aicore__ inline int64_t KvHeadForPair(int64_t pairIdx) const
    {
        if (qHead_ >= head_) {
            const int64_t qPerKv = qHead_ / head_;
            return pairIdx / qPerKv;
        }
        return pairIdx;
    }

    __aicore__ inline int32_t ReadK(int64_t batchIdx) const
    {
        const int32_t kValue = kIsScalar_ ? kGm_.GetValue(0) : kGm_.GetValue(batchIdx);
        return static_cast<int32_t>(Min<int64_t>(Max<int64_t>(kValue, 0), blockCount_));
    }

    __aicore__ inline int64_t ReadSeqLen(int64_t batchIdx) const
    {
        const int32_t seqValue = seqLenIsScalar_ ? seqLenGm_.GetValue(0) : seqLenGm_.GetValue(batchIdx);
        return Min<int64_t>(Max<int64_t>(seqValue, 0), blockCount_ * blockSize_);
    }

    __aicore__ inline TaskRuntime MakeRuntime(int64_t taskId)
    {
        TaskRuntime runtime;
        runtime.active = taskId < totalTasks_;
        if (!runtime.active) {
            return runtime;
        }

        runtime.batchIdx = taskId / blockCount_;
        runtime.logicalBlock = taskId - runtime.batchIdx * blockCount_;
        const int64_t seqLen = ReadSeqLen(runtime.batchIdx);
        const int64_t blockStart = runtime.logicalBlock * blockSize_;
        runtime.validTokens = Min<int64_t>(blockSize_, Max<int64_t>(seqLen - blockStart, 0));
        runtime.physicalBlock = blockTableGm_.GetValue(runtime.batchIdx * blockCount_ + runtime.logicalBlock);
        if (runtime.physicalBlock < 0 || runtime.physicalBlock >= physicalBlockCount_ || runtime.validTokens <= 0) {
            runtime.active = false;
        }
        return runtime;
    }

    __aicore__ inline void ExpandQuery(int64_t batchIdx, int64_t qSeqIdx, int64_t qHeadIdx)
    {
        const int64_t queryOffset = QueryOffset(batchIdx, qSeqIdx, qHeadIdx);
        GlobalTensor<int8_t> queryWorkspace = QueryWorkspace();
        for (int64_t byteIdx = 0; byteIdx < dimBytes_; ++byteIdx) {
            const uint8_t value = hashQGm_.GetValue(queryOffset + byteIdx);
            const int64_t bitBase = byteIdx * 8;
            for (int64_t bit = 0; bit < 8; ++bit) {
                queryWorkspace.SetValue(bitBase + bit, BitToSign(value, bit));
            }
        }
    }

    __aicore__ inline void ExpandKeyTile(const TaskRuntime& runtime, int64_t kvHeadIdx, int64_t tileStart,
        int64_t curN)
    {
        GlobalTensor<int8_t> keyWorkspace = KeyWorkspace();
        for (int64_t n = 0; n < curN; ++n) {
            const int64_t keyOffset = KeyOffset(runtime.physicalBlock, runtime.batchIdx, tileStart + n, kvHeadIdx);
            for (int64_t byteIdx = 0; byteIdx < dimBytes_; ++byteIdx) {
                const uint8_t value = hashKGm_.GetValue(keyOffset + byteIdx);
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

    __aicore__ inline void InitTokenScore(int64_t curN)
    {
        GlobalTensor<int32_t> tokenScore = TokenScoreWorkspace();
        for (int64_t n = 0; n < curN; ++n) {
            tokenScore.SetValue(n, 0);
        }
    }

    __aicore__ inline void AccumulateTokenScore(int64_t curN)
    {
        GlobalTensor<int32_t> tokenScore = TokenScoreWorkspace();
        GlobalTensor<int32_t> mmWorkspace = MatmulWorkspace();
        for (int64_t n = 0; n < curN; ++n) {
            tokenScore.SetValue(n, tokenScore.GetValue(n) + mmWorkspace.GetValue(n));
        }
    }

    __aicore__ inline void UpdateBlockScore(TaskRuntime& runtime, int64_t curN)
    {
        GlobalTensor<int32_t> tokenScore = TokenScoreWorkspace();
        for (int64_t n = 0; n < curN; ++n) {
            runtime.blockScore = Max(runtime.blockScore, tokenScore.GetValue(n));
        }
    }

    __aicore__ inline void ProcessTask(int64_t taskId)
    {
        TaskRuntime runtime = MakeRuntime(taskId);
        if ASCEND_IS_AIV {
            if (taskId < totalTasks_) {
                blockScoreGm_.SetValue(taskId, NEG_INF_SIM);
            }
        }
        SyncAll<false>();

        if (!runtime.active) {
            return;
        }

        const int64_t tileLoops = CeilDiv(runtime.validTokens, tileN2_);
        for (int64_t qSeqIdx = 0; qSeqIdx < qSeqLen_; ++qSeqIdx) {
            for (int64_t tileLoop = 0; tileLoop < tileLoops; ++tileLoop) {
                const int64_t tileStart = tileLoop * tileN2_;
                const int64_t curN = Min(tileN2_, runtime.validTokens - tileStart);
                if ASCEND_IS_AIV {
                    InitTokenScore(curN);
                }
                SyncAll<false>();

                for (int64_t pairIdx = 0; pairIdx < pairCount_; ++pairIdx) {
                    const int64_t qHeadIdx = QHeadForPair(pairIdx);
                    const int64_t kvHeadIdx = KvHeadForPair(pairIdx);
                    if ASCEND_IS_AIV {
                        ExpandQuery(runtime.batchIdx, qSeqIdx, qHeadIdx);
                        ExpandKeyTile(runtime, kvHeadIdx, tileStart, curN);
                    }
                    SyncAll<false>();

                    if ASCEND_IS_AIC {
                        MatmulTile(curN);
                    }
                    SyncAll<false>();

                    if ASCEND_IS_AIV {
                        AccumulateTokenScore(curN);
                    }
                    SyncAll<false>();
                }

                if ASCEND_IS_AIV {
                    UpdateBlockScore(runtime, curN);
                }
                SyncAll<false>();
            }
        }

        if ASCEND_IS_AIV {
            blockScoreGm_.SetValue(taskId, runtime.blockScore);
        }
    }

    __aicore__ inline void FinalizeBatch(int64_t batchIdx)
    {
        const int64_t outBase = batchIdx * blockCount_;
        for (int64_t i = 0; i < blockCount_; ++i) {
            newBlockTableGm_.SetValue(outBase + i, -1);
        }

        const int32_t kLimit = ReadK(batchIdx);
        for (int64_t pos = 0; pos < kLimit; ++pos) {
            int32_t bestScore = NEG_INF_SIM;
            int64_t bestLogicalBlock = -1;
            int32_t bestPhysicalBlock = -1;
            for (int64_t block = 0; block < blockCount_; ++block) {
                const int64_t scoreOffset = batchIdx * blockCount_ + block;
                const int32_t score = blockScoreGm_.GetValue(scoreOffset);
                const int32_t physicalBlock = blockTableGm_.GetValue(scoreOffset);
                const bool betterScore = score > bestScore;
                const bool tieBreak = score == bestScore && bestLogicalBlock >= 0 && block < bestLogicalBlock;
                if (score != NEG_INF_SIM && (betterScore || tieBreak)) {
                    bestScore = score;
                    bestLogicalBlock = block;
                    bestPhysicalBlock = physicalBlock;
                }
            }
            if (bestLogicalBlock < 0) {
                return;
            }
            newBlockTableGm_.SetValue(outBase + pos, bestPhysicalBlock);
            blockScoreGm_.SetValue(batchIdx * blockCount_ + bestLogicalBlock, NEG_INF_SIM);
        }
    }

private:
    GlobalTensor<uint8_t> hashQGm_;
    GlobalTensor<uint8_t> hashKGm_;
    GlobalTensor<int32_t> kGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> newBlockTableGm_;
    GlobalTensor<int32_t> blockScoreGm_;
    GlobalTensor<int8_t> queryExpandGm_;
    GlobalTensor<int8_t> keyExpandGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<int32_t> tokenScoreGm_;

    Matmul<AType, BType, CType, BiasType> mm_;
    const TCubeTiling* mmTiling_ = nullptr;
    TPipe* pipe_ = nullptr;

    int64_t batch_ = 0;
    int64_t qSeqLen_ = 0;
    int64_t qHead_ = 0;
    int64_t head_ = 0;
    int64_t physicalBlockCount_ = 0;
    int64_t blockCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t dimBytes_ = 0;
    int64_t pairCount_ = 0;
    int64_t totalTasks_ = 0;
    int64_t tileN2_ = 0;
    int64_t expandedDim_ = 0;
    int64_t usedCoreNum_ = 1;
    bool kIsScalar_ = true;
    bool seqLenIsScalar_ = true;
};
} // namespace

extern "C" __global__ __aicore__ void hash_block_top_k(GM_ADDR hashQ, GM_ADDR hashKCache, GM_ADDR k,
    GM_ADDR hashKBlockTable, GM_ADDR seqLen, GM_ADDR newBlockTable, GM_ADDR workspace, GM_ADDR tiling)
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
    HashBlockTopKCubeKernel op;
    op.Init(hashQ, hashKCache, k, hashKBlockTable, seqLen, newBlockTable, userWorkspace, &tilingData, &pipe);
    op.Process();
}
