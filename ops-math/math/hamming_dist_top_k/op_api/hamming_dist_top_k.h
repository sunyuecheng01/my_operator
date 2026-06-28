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
 * \file hamming_dist_top_k.h
 * \brief HammingDistTopK level-0 op interface.
 */
#ifndef OP_API_INC_LEVEL0_HAMMING_DIST_TOP_K_H_
#define OP_API_INC_LEVEL0_HAMMING_DIST_TOP_K_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* HammingDistTopK(const aclTensor* query, const aclTensor* keyCompressed, const aclTensor* k,
    const aclTensor* seqLen, const aclTensor* chunkSizeOptional, const aclTensor* keyBlockTableOptional, int64_t topk,
    const aclTensor* indices, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_HAMMING_DIST_TOP_K_H_
