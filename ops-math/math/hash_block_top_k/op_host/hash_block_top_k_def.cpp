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
 * \file hash_block_top_k_def.cpp
 * \brief HashBlockTopK op definition.
 */
#include "register/op_def_registry.h"

namespace ops {
class HashBlockTopK : public OpDef {
public:
    explicit HashBlockTopK(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> u8 = {ge::DT_UINT8};
        const std::vector<ge::DataType> i32 = {ge::DT_INT32};
        const std::vector<ge::Format> nd = {ge::FORMAT_ND};

        this->Input("hash_q")
            .ParamType(REQUIRED)
            .DataType(u8)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("hash_k_cache")
            .ParamType(REQUIRED)
            .DataType(u8)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(i32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("hash_k_block_table")
            .ParamType(REQUIRED)
            .DataType(i32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Input("seqlen")
            .ParamType(REQUIRED)
            .DataType(i32)
            .Format(nd)
            .UnknownShapeFormat(nd);
        this->Output("new_block_table")
            .ParamType(REQUIRED)
            .DataType(i32)
            .Format(nd)
            .UnknownShapeFormat(nd);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "hash_block_top_k");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(HashBlockTopK);
} // namespace ops
