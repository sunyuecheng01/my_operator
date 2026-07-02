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
 * \file hash_block_top_k_tiling.h
 * \brief HashBlockTopK tiling data.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_HASH_BLOCK_TOP_K_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_HASH_BLOCK_TOP_K_H_

#include "platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HashBlockTopKTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, qSeqLen);
TILING_DATA_FIELD_DEF(int64_t, qHead);
TILING_DATA_FIELD_DEF(int64_t, head);
TILING_DATA_FIELD_DEF(int64_t, physicalBlockCount);
TILING_DATA_FIELD_DEF(int64_t, blockCount);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, dimBytes);
TILING_DATA_FIELD_DEF(int64_t, pairCount);
TILING_DATA_FIELD_DEF(int64_t, totalTasks);
TILING_DATA_FIELD_DEF(int64_t, tileN2);
TILING_DATA_FIELD_DEF(int64_t, expandedDim);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kIsScalar);
TILING_DATA_FIELD_DEF(int64_t, seqLenIsScalar);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HashBlockTopK, HashBlockTopKTilingData)

struct HashBlockTopKCompileInfo {
    int32_t coreNum = 0;
    uint64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_HASH_BLOCK_TOP_K_H_
