# HammingDistTopK

`HammingDistTopK` computes Hamming distances between one compressed query and a compressed key library, then returns
the chunk-level indices with the smallest distance.

## Inputs

- `query`: `uint8`, shape `[batch, qHead, 1, dim / 8]`.
- `key_compressed`: `uint8`.
  - without `key_block_table`: shape `[batch, head, maxSeqLen, dim / 8]`;
  - with `key_block_table`: shape `[physicalBlockCount, head, blockSize, dim / 8]`.
- `k`: `int32`, shape `[batch]`, runtime Top-K count.
- `seq_len`: `int32`, shape `[batch]`, runtime effective sequence length.
- `chunk_size`: optional `int32`, shape `[batch]`; defaults to 1.
- `key_block_table`: optional `int32`, shape `[batch, blockCount]`, maps logical blocks to physical blocks.

## Output

- `indices`: `int32`, shape `[batch, head, maxOutputChunkLen]`.

The first `min(k[batch], topk, maxOutputChunkLen)` entries contain Top-K chunk indices. Unused entries are `-1`.
Skipped tail chunks are written at the end of the output when there is spare room.

## Notes

- `topk` is an optional attribute and defaults to 128. It is used as the static output-cap and in the repository
  formula for `maxOutputChunkLen`.
- The kernel expands compressed bits to signed `int8` values (`0 -> -1`, `1 -> +1`) and uses CUBE Matmul to compute
  similarity. A larger similarity means a smaller Hamming distance because `similarity = dim - 2 * distance`.
- AIV handles bit expansion, chunk-level reduce, and Top-K; AIC handles the Matmul stage.
