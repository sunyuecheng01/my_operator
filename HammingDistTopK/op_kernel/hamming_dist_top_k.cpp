/**
 * @file hamming_dist_top_k.cpp
 *
 * HammingDistTopK 算子 kernel 侧实现（CANN 8.5.0 AscendC）。
 *
 * 数据流（对照规格“核心算法流程”）：
 *   query(uint8 compressed) --Select+Cast--> int4b_t(unpacked)
 *   key  (uint8 compressed) --Select+Cast--> int4b_t(unpacked)
 *        --Matmul(query^T x key, cube)--> half 距离
 *        --ReduceMax(chunk 级, 当 chunkSize>1)-->
 *        --TopK(TOPK_NORMAL, 取最小 K)-->
 *        --> indices(int32, chunk 级索引)
 *
 * 汉明距离 = POPCOUNT(query XOR key)，在 int4b_t 下用 Matmul 等价计算。
 *
 * tilingKey 约定（见 host）：branch*10 + continFlag
 *   0  = Basic  + 连续存储
 *   1  = Basic  + block_table
 *   10 = SplitS + 连续存储
 *   11 = SplitS + block_table
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/sort/topk/topk.h"
#include "lib/reduce/reduce_common.h"
// 注：OpDef 框架下 GET_TILING_DATA 使用工程自动生成的 tiling 结构；
// 若独立编译，需 include host 侧 "hamming_dist_top_k_tiling.h"。

using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 2;     // double buffer
constexpr int32_t BLOCK_ALIGN = 16;
constexpr int32_t INT4_PER_BYTE = 2;
}

// AType/BType: int4b_t, ND；CType: half；用于 query^T x key
using A_T = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using B_T = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using C_T = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
using BIAS_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;

template <uint32_t BRANCH, uint32_t CONTIN>
class HammingDistTopKKernel {
public:
    __aicore__ inline HammingDistTopKKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k,
                                GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable,
                                GM_ADDR indices, GM_ADDR workspace,
                                const HammingDistTopKTilingData &tiling, TPipe *pipe)
    {
        this->tiling_ = &tiling;
        this->pipe_ = pipe;
        blockIdx_ = GetBlockIdx();

        // 全局 buffer 绑定
        queryGm_.SetGlobalBuffer((__gm__ uint8_t *)query);
        keyGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCompressed);
        kGm_.SetGlobalBuffer((__gm__ int32_t *)k);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t *)seqLen);
        if (chunkSize != nullptr) {
            chunkSizeGm_.SetGlobalBuffer((__gm__ int32_t *)chunkSize);
            hasChunk_ = true;
        }
        if constexpr (CONTIN == 1) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)keyBlockTable);
        }
        indicesGm_.SetGlobalBuffer((__gm__ int32_t *)indices);
        wsGm_.SetGlobalBuffer((__gm__ half *)workspace);

        // 解包 / 距离 / topk 中间 buffer
        pipe_->InitBuffer(qUnpackQue_, BUFFER_NUM, tiling.dimUnpacked * sizeof(int4b_t));
        pipe_->InitBuffer(kUnpackQue_, BUFFER_NUM, tiling.tileN1 * tiling.dimUnpacked * sizeof(int4b_t));
        pipe_->InitBuffer(distQue_, BUFFER_NUM, tiling.tileN1 * sizeof(half));
        pipe_->InitBuffer(chunkBuf_, tiling.maxOutputChunkLen * sizeof(half));
        pipe_->InitBuffer(idxBuf_, tiling.maxOutputChunkLen * sizeof(int32_t));
        pipe_->InitBuffer(castTmpBuf_, tiling.tileN1 * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const auto &t = *tiling_;
        // 本核负责的 (batch*head) 任务区间
        uint32_t totalTask = t.batch * t.head;
        for (uint32_t taskId = blockIdx_; taskId < totalTask; taskId += t.usedCoreNum) {
            uint32_t b = taskId / t.head;
            uint32_t h = taskId % t.head;
            ProcessOne(b, h);
        }
    }

private:
    // 处理单个 (batch b, head h)
    __aicore__ inline void ProcessOne(uint32_t b, uint32_t h)
    {
        const auto &t = *tiling_;

        // ---- 读取该 batch 的运行期标量 ----
        int32_t seqLen = seqLenGm_.GetValue(b);
        int32_t kVal   = kGm_.GetValue(b);
        int32_t chunk  = hasChunk_ ? chunkSizeGm_.GetValue(b) : (int32_t)t.defaultChunkSize;
        if (chunk <= 0) { chunk = 1; }

        // effectLen：跳过末尾不完整 chunk（规格“effectLen 计算”）
        int32_t effectLen = seqLen - (seqLen % chunk + BLOCK_ALIGN);
        if (effectLen < 0) { effectLen = 0; }

        // query 解包：uint8(compressed) -> half -> int4b_t
        LocalTensor<int4b_t> qUnpack = qUnpackQue_.AllocTensor<int4b_t>();
        UnpackQuery(b, h, qUnpack);
        qUnpackQue_.EnQue(qUnpack);
        qUnpack = qUnpackQue_.DeQue<int4b_t>();

        // 距离缓冲（chunk 级最大值）
        LocalTensor<half> chunkDist = chunkBuf_.Get<half>();
        uint32_t chunkCnt = 0;

        // ---- 沿 seqLen 按 tileN1 切块，逐块算距离并做 chunk ReduceMax ----
        for (int32_t n0 = 0; n0 < effectLen; n0 += (int32_t)t.tileN1) {
            int32_t curN = (effectLen - n0) < (int32_t)t.tileN1 ? (effectLen - n0) : (int32_t)t.tileN1;

            // key 解包（按寻址模式取物理地址）
            LocalTensor<int4b_t> kUnpack = kUnpackQue_.AllocTensor<int4b_t>();
            UnpackKey(b, h, n0, curN, kUnpack);
            kUnpackQue_.EnQue(kUnpack);
            kUnpack = kUnpackQue_.DeQue<int4b_t>();

            // Matmul：query^T x key -> half 距离 [1, curN]
            LocalTensor<half> dist = distQue_.AllocTensor<half>();
            ComputeDistance(qUnpack, kUnpack, dist, curN);
            distQue_.EnQue(dist);
            dist = distQue_.DeQue<half>();

            // chunk 级 ReduceMax（chunk>1 时取块内最大汉明距离）
            ReduceChunk(dist, curN, chunk, chunkDist, chunkCnt);

            distQue_.FreeTensor(dist);
            kUnpackQue_.FreeTensor(kUnpack);
        }
        qUnpackQue_.FreeTensor(qUnpack);

        // ---- TopK：取最小的 K 个（汉明距离越小越相似）----
        LocalTensor<int32_t> outIdx = idxBuf_.Get<int32_t>();
        TopKSmallest(chunkDist, chunkCnt, kVal, outIdx);

        // ---- 写回 indices[b, h, :]（chunk 级索引）----
        CopyOutIndices(b, h, outIdx);
    }

    // query: [batch,qHead,1,dim/8] uint8 -> int4b_t[dimUnpacked]
    __aicore__ inline void UnpackQuery(uint32_t b, uint32_t h, LocalTensor<int4b_t> &out)
    {
        const auto &t = *tiling_;
        uint32_t qh = (t.groupNum == 0) ? h : h * t.groupNum;     // GQA：key head -> query head
        uint64_t off = ((uint64_t)b * t.qHead + qh) * t.dimCompressed;
        // SelectCustom + Cast：uint8 压缩位 -> half -> int4b_t。
        // 这里以 GM->UB 拷贝 + 位展开实现；SelectCustom 在 lib 中提供按位选择。
        SelectAndCast(queryGm_[off], t.dimCompressed, out, t.dimUnpacked);
    }

    // key: 按 continFlag 取物理偏移后解包 curN 个 token
    __aicore__ inline void UnpackKey(uint32_t b, uint32_t h, int32_t n0, int32_t curN,
                                     LocalTensor<int4b_t> &out)
    {
        const auto &t = *tiling_;
        for (int32_t i = 0; i < curN; ++i) {
            int32_t logicalTok = n0 + i;
            uint64_t physOff = KeyPhysOffset(b, h, logicalTok);
            SelectAndCast(keyGm_[physOff], t.dimCompressed,
                          out[(uint32_t)i * t.dimUnpacked], t.dimUnpacked);
        }
    }

    // 物理偏移：连续 vs block_table 两种寻址（规格“数据排布与寻址”）
    __aicore__ inline uint64_t KeyPhysOffset(uint32_t b, uint32_t h, int32_t logicalTok)
    {
        const auto &t = *tiling_;
        if constexpr (CONTIN == 1) {
            // 非连续：blockId = logicalTok / blockSize -> 查表得物理块号
            int32_t logicalBlock = logicalTok / (int32_t)t.blockSize;
            int32_t inBlock = logicalTok % (int32_t)t.blockSize;
            int32_t physBlock = blockTableGm_.GetValue(b * t.blockCount + logicalBlock);
            // 物理 key 库按 [physBlock, head, blockSize, dim/8] 排布
            return (((uint64_t)physBlock * t.head + h) * t.blockSize + inBlock) * t.dimCompressed;
        } else {
            // 连续：[batch, head, maxSeqLen, dim/8]
            return (((uint64_t)b * t.head + h) * t.maxSeqLen + logicalTok) * t.dimCompressed;
        }
    }

    // SelectCustom + Cast：把 dimCompressed 个 uint8 的每一位展开为一个 int4b_t（0/1）
    // 汉明距离矩阵乘要求把 query/key 的每个 bit 映射为 int4 操作数。
    __aicore__ inline void SelectAndCast(GlobalTensor<uint8_t> src, uint32_t bytes,
                                         LocalTensor<int4b_t> dst, uint32_t outLen)
    {
        // 上机实现要点（标 TODO，需结合 SelectCustom 位展开指令）：
        //   1) DataCopy uint8 -> UB
        //   2) 按位 Select 生成 half 的 0/1 序列（或 ±1，取决于距离编码）
        //   3) Cast<half, int4b_t> 压成 int4b_t
        // 此处保留接口与数据流；具体位展开指令以 CANN 8.5.0 vec API 为准。
        (void)src; (void)bytes; (void)dst; (void)outLen;
    }

    // Matmul：A=query^T[1,dim] x B=key[dim,curN] -> dist[1,curN] (half)
    __aicore__ inline void ComputeDistance(LocalTensor<int4b_t> &q, LocalTensor<int4b_t> &k,
                                           LocalTensor<half> &dist, int32_t curN)
    {
        const auto &t = *tiling_;
        mm_.SetTensorA(q);
        mm_.SetTensorB(k);
        mm_.SetTail(1, curN, t.dimUnpacked);        // M=1,N=curN,K=dim
        mm_.IterateAll(dist);
        mm_.End();
        // dist[i] = sum_d q_d * k_{i,d}，与 POPCOUNT(q XOR k) 等价（int4 编码下）
    }

    // chunk 级 ReduceMax：chunk>1 时块内取最大汉明距离，写入 chunkDist
    __aicore__ inline void ReduceChunk(LocalTensor<half> &dist, int32_t curN, int32_t chunk,
                                       LocalTensor<half> &chunkDist, uint32_t &chunkCnt)
    {
        if (chunk <= 1) {
            // 无 chunk：逐 token 直接累加
            for (int32_t i = 0; i < curN; ++i) {
                chunkDist.SetValue(chunkCnt++, dist.GetValue(i));
            }
            return;
        }
        for (int32_t i = 0; i + chunk <= curN; i += chunk) {
            half mx = dist.GetValue(i);
            for (int32_t j = 1; j < chunk; ++j) {
                half v = dist.GetValue(i + j);
                mx = v > mx ? v : mx;
            }
            chunkDist.SetValue(chunkCnt++, mx);
        }
        // 注：大规模时应改用向量 ReduceMax 指令（WholeReduceMax）替代标量循环以提速。
    }

    // TopK：TOPK_NORMAL，取“最小的 K 个”。汉明距离越小越相似。
    __aicore__ inline void TopKSmallest(LocalTensor<half> &chunkDist, uint32_t chunkCnt,
                                        int32_t kVal, LocalTensor<int32_t> &outIdx)
    {
        const auto &t = *tiling_;
        uint32_t kk = (uint32_t)kVal;
        if (kk > chunkCnt) { kk = chunkCnt; }
        // 取最小：对距离取负后做 TOPK_NORMAL 求最大，得到最小 K 的索引。
        // 输出为 chunk 级索引（非 token 索引），与规格“输出索引含义”一致。
        TopKInfo topkInfo;
        topkInfo.outter = 1;
        topkInfo.inner  = chunkCnt;
        topkInfo.n      = chunkCnt;
        // TopK<half, int32_t, true, TopKMode::TOPK_NORMAL>(valOut, idxOut, chunkDist, ...)
        // 具体模板参数以 lib/sort/topk/topk.h 为准；此处给出调用骨架。
        (void)topkInfo; (void)kk; (void)t;
        // outIdx 即返回的 chunk 级 Top-K 索引；不足部分按规格填充到输出末尾。
    }

    // 写回 indices[b, h, 0:maxOutputChunkLen]
    __aicore__ inline void CopyOutIndices(uint32_t b, uint32_t h, LocalTensor<int32_t> &outIdx)
    {
        const auto &t = *tiling_;
        uint64_t off = ((uint64_t)b * t.head + h) * t.maxOutputChunkLen;
        DataCopy(indicesGm_[off], outIdx, t.maxOutputChunkLen);
    }

private:
    const HammingDistTopKTilingData *tiling_ = nullptr;
    TPipe *pipe_ = nullptr;
    uint32_t blockIdx_ = 0;
    bool hasChunk_ = false;

    GlobalTensor<uint8_t> queryGm_, keyGm_;
    GlobalTensor<int32_t> kGm_, seqLenGm_, chunkSizeGm_, blockTableGm_, indicesGm_;
    GlobalTensor<half>    wsGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> qUnpackQue_, kUnpackQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> distQue_;
    TBuf<TPosition::VECCALC> chunkBuf_, idxBuf_, castTmpBuf_;

public:
    Matmul<A_T, B_T, C_T, BIAS_T> mm_;   // public：供 REGIST_MATMUL_OBJ 访问
};

// 核函数入口
extern "C" __global__ __aicore__ void hamming_dist_top_k(
    GM_ADDR query, GM_ADDR key_compressed, GM_ADDR k, GM_ADDR seq_len,
    GM_ADDR chunk_size, GM_ADDR key_block_table, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWs = GetUserWorkspace(workspace);

    // 按 tilingKey 分发模板（branch*10 + continFlag）
    if (TILING_KEY_IS(0)) {
        HammingDistTopKKernel<0, 0> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.mmTiling);
        op.Init(query, key_compressed, k, seq_len, chunk_size, key_block_table,
                indices, userWs, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        HammingDistTopKKernel<0, 1> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.mmTiling);
        op.Init(query, key_compressed, k, seq_len, chunk_size, key_block_table,
                indices, userWs, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        HammingDistTopKKernel<1, 0> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.mmTiling);
        op.Init(query, key_compressed, k, seq_len, chunk_size, key_block_table,
                indices, userWs, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        HammingDistTopKKernel<1, 1> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.mmTiling);
        op.Init(query, key_compressed, k, seq_len, chunk_size, key_block_table,
                indices, userWs, tilingData, &pipe);
        op.Process();
    }
}
