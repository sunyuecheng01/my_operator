/**
 * @file hamming_dist_top_k_tiling.h
 *
 * HammingDistTopK 算子 TilingData 定义。
 * 适配 CANN 8.5.0 AscendC OpDef 自定义算子框架。
 *
 * 该结构在 host 侧 TilingFunc 中填充，随后通过 GM 传给 kernel，
 * kernel 侧用 GET_TILING_DATA 解析。
 */
#ifndef HAMMING_DIST_TOP_K_TILING_H
#define HAMMING_DIST_TOP_K_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HammingDistTopKTilingData)
    // ---- 基础形状 ----
    TILING_DATA_FIELD_DEF(uint32_t, batch);             // batch 数
    TILING_DATA_FIELD_DEF(uint32_t, qHead);             // query 的 head 数
    TILING_DATA_FIELD_DEF(uint32_t, head);              // key 的 head 数
    TILING_DATA_FIELD_DEF(uint32_t, dim);               // 原始向量维度（如 128）
    TILING_DATA_FIELD_DEF(uint32_t, dimCompressed);     // dim/8，压缩后字节数（如 16）
    TILING_DATA_FIELD_DEF(uint32_t, dimUnpacked);       // 解包后 int4 元素个数 = dim
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);         // 最大序列长度
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);          // qHead/head，GQA 分组

    // ---- TopK / chunk 相关 ----
    TILING_DATA_FIELD_DEF(uint32_t, topk);              // 属性 topk，与 k 输入配合
    TILING_DATA_FIELD_DEF(uint32_t, topkRatio);         // topk 比例（百分数）
    TILING_DATA_FIELD_DEF(uint32_t, defaultChunkSize);  // 默认 chunk_size（缺省为 1）
    TILING_DATA_FIELD_DEF(uint32_t, maxChunkSize);      // 由 maxSeqLen 推导（1/8/16）
    TILING_DATA_FIELD_DEF(uint32_t, maxOutputChunkLen); // 输出 indices 第三维长度

    // ---- Tiling 分支参数 ----
    TILING_DATA_FIELD_DEF(uint32_t, tileN1);            // N 方向一级切分（254 / 128 / blockSize）
    TILING_DATA_FIELD_DEF(uint32_t, tileN2);            // N 方向二级切分（3328 / 4*1024）
    TILING_DATA_FIELD_DEF(uint32_t, branch);            // 0=Basic/Parallel, 1=SplitS

    // ---- 数据排布 / 寻址 ----
    TILING_DATA_FIELD_DEF(uint32_t, continFlag);        // 1=提供 key_block_table（非连续），0=连续
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);         // 单块 token 数（非连续存储）
    TILING_DATA_FIELD_DEF(uint32_t, blockCount);        // 每 batch 的逻辑块数

    // ---- 核间切分 ----
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);           // 平台 AICore 数
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);       // 实际使用核数
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);      // 每核处理的 (batch*head) 任务数

    // ---- Matmul 高阶 API 所需 tiling ----
    TILING_DATA_FIELD_STRUCT(TCubeTiling, mmTiling);    // Matmul (query^T x key) cube tiling
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HammingDistTopK, HammingDistTopKTilingData)

} // namespace optiling

#endif // HAMMING_DIST_TOP_K_TILING_H
