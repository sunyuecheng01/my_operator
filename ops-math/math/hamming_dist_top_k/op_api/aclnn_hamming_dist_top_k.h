/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HAMMING_DIST_TOP_K_H_
#define OP_API_INC_HAMMING_DIST_TOP_K_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHammingDistTopK的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：用 query 与压缩 key 的二进制哈希码计算汉明距离，按 chunk 选出 Top-K（相似度最大 / 距离最小）的 chunk 索引。
 *
 * @param [in] query: device 侧 aclTensor，dtype 为 UINT8，shape 为 [batch, qHead, 1, dim/8]。
 * @param [in] keyCompressed: device 侧 aclTensor，dtype 为 UINT8。
 *        无 keyBlockTable 时 shape 为 [batch, head, maxSeqLen, dim/8]；
 *        有 keyBlockTable 时 shape 为 [physicalBlockCount, head, blockSize, dim/8]。
 * @param [in] k: device 侧 aclTensor，dtype 为 INT32，shape 为 [batch]，运行期 Top-K 的 chunk 数。
 * @param [in] seqLen: device 侧 aclTensor，dtype 为 INT32，shape 为 [batch]，运行期有效序列长度。
 * @param [in] chunkSizeOptional: 可选，device 侧 aclTensor，dtype 为 INT32，shape 为 [batch]，缺省按 1 处理，可传 nullptr。
 * @param [in] keyBlockTableOptional: 可选，device 侧 aclTensor，dtype 为 INT32，shape 为 [batch, blockCount]，
 *        逻辑块到物理块的映射；不传（nullptr）时按连续 KV 布局处理。
 * @param [in] topk: host 侧 int64，输出容量上限相关的静态属性，默认 128。
 * @param [out] indices: device 侧 aclTensor，dtype 为 INT32，shape 为 [batch, outputChunkLen]。
 *        不区分 head：对所有 head 的相似度求和后选取整体得分最高的一组 chunk。
 * @param [out] workspaceSize: 返回需要在 device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHammingDistTopKGetWorkspaceSize(const aclTensor* query, const aclTensor* keyCompressed,
    const aclTensor* k, const aclTensor* seqLen, const aclTensor* chunkSizeOptional,
    const aclTensor* keyBlockTableOptional, int64_t topk, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnHammingDistTopK的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在 device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 由第一段接口 aclnnHammingDistTopKGetWorkspaceSize 获取的 workspace 大小。
 * @param [in] executor: op 执行器。
 * @param [in] stream: 指定执行任务的 AscendCL Stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHammingDistTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_HAMMING_DIST_TOP_K_H_
