/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HASH_BLOCK_TOP_K_H_
#define OP_API_INC_HASH_BLOCK_TOP_K_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHashBlockTopK 的第一段接口，根据计算流程获取 workspace 与执行器。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对 hash_q 与 paged hash_k_cache 计算二值哈希相似度，使用 block 内最高 QK 分作为
 * block 分数，为每个 batch 选出 Top-K 个物理 block，写入 new_block_table。
 *
 * @param [in] hashQ: UINT8，shape [batch, seq_len_q, qhead_num, head_dim / 8]。
 * @param [in] hashKCache: UINT8，shape [physical_block_num, batch, block_size, head_num, head_dim / 8]。
 * @param [in] k: INT32，scalar 或 shape [batch]，每个 batch 选出的 block 数。
 * @param [in] hashKBlockTable: INT32，shape [batch, block_num]，逻辑 block 到物理 block 的映射。
 * @param [in] seqLen: INT32，scalar 或 shape [batch]，每个 batch 的有效 KV token 数。
 * @param [out] newBlockTable: INT32，shape [batch, block_num]。前 k 个位置写入选中的物理 block id，其余为 -1。
 * @param [out] workspaceSize: 返回 device workspace 大小。
 * @param [out] executor: 返回 op 执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHashBlockTopKGetWorkspaceSize(const aclTensor* hashQ, const aclTensor* hashKCache,
    const aclTensor* k, const aclTensor* hashKBlockTable, const aclTensor* seqLen, aclTensor* newBlockTable,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHashBlockTopK 的第二段接口，执行计算。
 */
ACLNN_API aclnnStatus aclnnHashBlockTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_HASH_BLOCK_TOP_K_H_
