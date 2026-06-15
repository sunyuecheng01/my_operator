/**
 * @file hamming_dist_top_k.cpp
 *
 * HammingDistTopK 算子 host 侧实现（CANN 8.5.0 AscendC OpDef 框架）：
 *   - OpDef 注册（输入/输出/属性/AICore config）
 *   - InferShape / InferDataType
 *   - TilingFunc（参数计算 + 两条 tiling 分支 + Matmul tiling）
 *
 * 功能：计算 query 与压缩 key 库的汉明距离，返回距离最小的 Top-K 索引（chunk 级）。
 */
#include "hamming_dist_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace {
// 与 kernel 侧保持一致的常量
constexpr uint32_t BITS_PER_BYTE = 8;
constexpr uint32_t BLOCK_ALIGN   = 16;

// tiling 分支阈值（来自规格“Tiling 分支策略”一节）
constexpr uint32_t THRESH_SEQ_26K = 26 * 1024;
constexpr uint32_t THRESH_SEQ_8K  = 8 * 1024;
constexpr uint32_t THRESH_BATCH   = 16;

// 分支切分常量
constexpr uint32_t SPLITS_TILE_N1 = 128;
constexpr uint32_t SPLITS_TILE_N2 = 4 * 1024;
constexpr uint32_t BASIC_TILE_N1  = 254;
constexpr uint32_t BASIC_TILE_N2  = 3328;

// chunk 阈值
constexpr uint32_t SEQ_FOR_CHUNK16 = 1024;
constexpr uint32_t SEQ_FOR_CHUNK8  = 128;
constexpr uint32_t MIN_OUTPUT_CHUNK_LEN = 128;

enum class Branch : uint32_t { BASIC = 0, SPLITS = 1 };
} // namespace

namespace optiling {

// 由 maxSeqLen 推导 chunk 最大尺寸：>=1024 -> 16, >=128 -> 8, else 1
static uint32_t CalcMaxChunkSize(uint32_t maxSeqLen)
{
    if (maxSeqLen >= SEQ_FOR_CHUNK16) {
        return 16U;
    } else if (maxSeqLen >= SEQ_FOR_CHUNK8) {
        return 8U;
    }
    return 1U;
}

// maxOutputChunkLen = max(maxSeqLen/maxChunkSize, 128) * topkRatio/100 + 2，再与 128 取大
static uint32_t CalcMaxOutputChunkLen(uint32_t maxSeqLen, uint32_t maxChunkSize, uint32_t topkRatio)
{
    uint32_t base = maxSeqLen / (maxChunkSize == 0 ? 1U : maxChunkSize);
    base = base > MIN_OUTPUT_CHUNK_LEN ? base : MIN_OUTPUT_CHUNK_LEN;
    uint32_t out = base * topkRatio / 100U + 2U;
    return out > MIN_OUTPUT_CHUNK_LEN ? out : MIN_OUTPUT_CHUNK_LEN;
}

// 选择 tiling 分支并写回 tileN1/tileN2
static Branch SelectBranch(uint32_t batch, uint32_t maxSeqLen, uint32_t continFlag,
                           uint32_t blockSize, uint32_t &tileN1, uint32_t &tileN2)
{
    bool splitS = (maxSeqLen > THRESH_SEQ_26K) ||
                  (batch < THRESH_BATCH && maxSeqLen > THRESH_SEQ_8K);
    if (splitS) {
        tileN1 = SPLITS_TILE_N1;
        tileN2 = SPLITS_TILE_N2;
    } else {
        tileN1 = BASIC_TILE_N1;
        tileN2 = BASIC_TILE_N2;
    }
    // 非连续存储时：tileN1 强制对齐到 blockSize（规格“数据排布与寻址”）
    if (continFlag == 1U && blockSize > 0U) {
        tileN1 = blockSize;
    }
    return splitS ? Branch::SPLITS : Branch::BASIC;
}

// 配置 query^T x key 的 cube tiling（int4 输入，half 输出）
static int32_t SetMatmulTiling(gert::TilingContext *context, HammingDistTopKTilingData &tiling,
                               uint32_t qHead, uint32_t dimUnpacked, uint32_t tileN)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling mm(ascendcPlatform);
    // A: query^T  [M=qHead, K=dim]，B: key [K=dim, N=tileN]
    mm.SetDim(1);
    mm.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    mm.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    mm.SetCType(TPosition::VECCALC, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetShape(1, tileN, dimUnpacked);            // M=1：单查询向量
    mm.SetOrgShape(1, tileN, dimUnpacked);
    mm.SetBias(false);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(tiling.mmTiling) == -1) {
        return -1;
    }
    return 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    HammingDistTopKTilingData tiling;

    // ---------- 1. 读取输入形状 ----------
    auto queryShape = context->GetInputShape(0)->GetStorageShape();           // [batch,qHead,1,dim/8]
    auto keyShape   = context->GetInputShape(1)->GetStorageShape();           // [batch,head,maxSeqLen,dim/8]
    auto blockTable = context->GetOptionalInputShape(5);                      // key_block_table（可选）

    uint32_t batch         = queryShape.GetDim(0);
    uint32_t qHead         = queryShape.GetDim(1);
    uint32_t dimCompressed = queryShape.GetDim(3);
    uint32_t head          = keyShape.GetDim(1);
    uint32_t dim           = dimCompressed * BITS_PER_BYTE;                   // 还原原始维度
    uint32_t dimUnpacked   = dim;                                            // 每 bit 解包为 1 个 int4

    // ---------- 2. continFlag 与 maxSeqLen ----------
    uint32_t continFlag = 0U;
    uint32_t blockSize  = 0U;
    uint32_t blockCount = 0U;
    uint32_t maxSeqLen  = 0U;
    if (blockTable != nullptr && blockTable->GetStorageShape().GetDimNum() >= 2) {
        // 提供 block_table：非连续存储
        continFlag = 1U;
        blockCount = blockTable->GetStorageShape().GetDim(1);
        blockSize  = keyShape.GetDim(2);                                     // 单块 token 数
        maxSeqLen  = blockCount * blockSize;                                 // 规格公式
    } else {
        continFlag = 0U;
        maxSeqLen  = keyShape.GetDim(2);                                     // keyCommitted.shape[2]
    }

    // ---------- 3. 属性 ----------
    auto attrs = context->GetAttrs();
    uint32_t topk = 0U;
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        const int32_t *topkPtr = attrs->GetAttrPointer<int32_t>(0);
        topk = (topkPtr != nullptr) ? static_cast<uint32_t>(*topkPtr) : 0U;
    }
    uint32_t topkRatio = 100U;        // 默认 100%（可按需扩展为属性）
    uint32_t defaultChunkSize = 1U;   // chunk_size 可选输入缺省值

    // ---------- 4. chunk / 输出长度 ----------
    uint32_t maxChunkSize = CalcMaxChunkSize(maxSeqLen);
    uint32_t maxOutputChunkLen = CalcMaxOutputChunkLen(maxSeqLen, maxChunkSize, topkRatio);

    // ---------- 5. tiling 分支 ----------
    uint32_t tileN1 = 0U, tileN2 = 0U;
    Branch branch = SelectBranch(batch, maxSeqLen, continFlag, blockSize, tileN1, tileN2);

    // ---------- 6. 核间切分（按 batch*head 任务粒度）----------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t totalTask = batch * head;
    uint32_t usedCoreNum = (totalTask < coreNum) ? totalTask : coreNum;
    if (usedCoreNum == 0U) { usedCoreNum = 1U; }
    uint32_t batchPerCore = (totalTask + usedCoreNum - 1U) / usedCoreNum;

    // ---------- 7. Matmul tiling ----------
    if (SetMatmulTiling(context, tiling, qHead, dimUnpacked, tileN1) != 0) {
        return ge::GRAPH_FAILED;
    }

    // ---------- 8. 写回 TilingData ----------
    tiling.set_batch(batch);
    tiling.set_qHead(qHead);
    tiling.set_head(head);
    tiling.set_dim(dim);
    tiling.set_dimCompressed(dimCompressed);
    tiling.set_dimUnpacked(dimUnpacked);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_groupNum(head == 0U ? 1U : qHead / head);
    tiling.set_topk(topk);
    tiling.set_topkRatio(topkRatio);
    tiling.set_defaultChunkSize(defaultChunkSize);
    tiling.set_maxChunkSize(maxChunkSize);
    tiling.set_maxOutputChunkLen(maxOutputChunkLen);
    tiling.set_tileN1(tileN1);
    tiling.set_tileN2(tileN2);
    tiling.set_branch(static_cast<uint32_t>(branch));
    tiling.set_continFlag(continFlag);
    tiling.set_blockSize(blockSize);
    tiling.set_blockCount(blockCount);
    tiling.set_coreNum(coreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_batchPerCore(batchPerCore);

    // tilingKey：高位 branch，低位 continFlag，便于 kernel 模板分发
    context->SetTilingKey(static_cast<uint64_t>(branch) * 10U + continFlag);
    context->SetBlockDim(usedCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    // 预留 system workspace + 中间距离/解包缓冲（按最坏情况粗估，可上机再收紧）
    constexpr size_t SYS_WORKSPACE = 16U * 1024U * 1024U;
    workspaces[0] = SYS_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *queryShape = context->GetInputShape(0);   // [batch,qHead,1,dim/8]
    const gert::Shape *keyShape   = context->GetInputShape(1);   // [batch,head,maxSeqLen,dim/8]
    gert::Shape *indicesShape     = context->GetOutputShape(0);  // [batch,head,maxOutputChunkLen]

    int64_t batch     = queryShape->GetDim(0);
    int64_t head      = keyShape->GetDim(1);
    int64_t maxSeqLen = keyShape->GetDim(2);

    // 与 host 侧 CalcMaxChunkSize/CalcMaxOutputChunkLen 一致
    int64_t maxChunkSize = (maxSeqLen >= 1024) ? 16 : ((maxSeqLen >= 128) ? 8 : 1);
    int64_t base = maxSeqLen / maxChunkSize;
    if (base < 128) { base = 128; }
    int64_t maxOutputChunkLen = base + 2;           // topkRatio 默认 100%
    if (maxOutputChunkLen < 128) { maxOutputChunkLen = 128; }

    indicesShape->SetDimNum(3);
    indicesShape->SetDim(0, batch);
    indicesShape->SetDim(1, head);
    indicesShape->SetDim(2, maxOutputChunkLen);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);    // indices 固定 int32
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class HammingDistTopK : public OpDef {
public:
    explicit HammingDistTopK(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_compressed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("seq_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("chunk_size")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("topk").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(HammingDistTopK);

} // namespace ops
