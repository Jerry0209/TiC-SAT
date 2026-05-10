#include "interleavedPipeline.h"

#include "debuggerFunctions.h"
#include "run_mode_config.h"

#if CFG_SIMD
#include <gemm_SVE.h>
#endif

#include <algorithm>
#include <vector>

namespace {

int8_t unpackPackedValueLocal(const uint32_t* row, std::size_t elem_idx) {
    const std::size_t word_idx = elem_idx / 4u;
    const std::size_t byte_idx = elem_idx % 4u;
    return static_cast<int8_t>((row[word_idx] >> (byte_idx * 8u)) & 0xFFu);
}

#if CFG_SIMD
constexpr std::size_t kMatmulRowTile = 8u;
constexpr std::size_t kMatmulColTile = 8u;
constexpr std::size_t kMatmulKTile = 64u;

void unpackInterleavedTileToInt32(const int8_t* src_interleaved,
                                  std::size_t src_cols,
                                  std::size_t row_base,
                                  std::size_t col_base,
                                  std::size_t row_tile,
                                  std::size_t col_tile,
                                  std::size_t learner_count,
                                  int32_t* dst_interleaved) {
    for (std::size_t row = 0; row < row_tile; row++) {
        for (std::size_t col = 0; col < col_tile; col++) {
            const int8_t* src_slot =
                src_interleaved + (((row_base + row) * src_cols + col_base + col) * learner_count);
            int32_t* dst_slot =
                dst_interleaved + ((row * col_tile + col) * learner_count);

            for (std::size_t learner = 0; learner < learner_count; learner++) {
                dst_slot[learner] = static_cast<int32_t>(src_slot[learner]);
            }
        }
    }
}
#endif

void packOneLearner(const int8_t* input_interleaved,
                    std::size_t rows,
                    std::size_t cols,
                    std::size_t learner_count,
                    std::size_t learner,
                    std::vector<uint32_t>& packed) {
    const std::size_t packed_cols = (cols + 3u) / 4u;
    packed.assign(rows * packed_cols, 0u);

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const std::size_t packed_idx = row * packed_cols + (col / 4u);
            const std::size_t byte_idx = col % 4u;
            const uint8_t value = static_cast<uint8_t>(
                input_interleaved[((row * cols) + col) * learner_count + learner]);
            packed[packed_idx] |= static_cast<uint32_t>(value) << (byte_idx * 8u);
        }
    }
}

} // namespace

void interleavePackedLearners4(std::size_t rows,
                               std::size_t cols,
                               uint32_t* const inputs[4],
                               int8_t* output_interleaved) {
    const std::size_t packed_cols = (cols + 3u) / 4u;

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < 4u; learner++) {
                output_interleaved[((row * cols) + col) * 4u + learner] =
                    unpackPackedValueLocal(inputs[learner] + row * packed_cols, col);
            }
        }
    }
}

void interleavePackedLearners2(std::size_t rows,
                               std::size_t cols,
                               uint32_t* const inputs[2],
                               int8_t* output_interleaved) {
    const std::size_t packed_cols = (cols + 3u) / 4u;

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < 2u; learner++) {
                output_interleaved[((row * cols) + col) * 2u + learner] =
                    unpackPackedValueLocal(inputs[learner] + row * packed_cols, col);
            }
        }
    }
}

void packInterleavedLearners4(std::size_t rows,
                              std::size_t cols,
                              const int8_t* input_interleaved,
                              uint32_t* const outputs[4]) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 4u; learner++) {
        packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
        std::copy(packed.begin(), packed.end(), outputs[learner]);
    }
}

void packInterleavedLearners2(std::size_t rows,
                              std::size_t cols,
                              const int8_t* input_interleaved,
                              uint32_t* const outputs[2]) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 2u; learner++) {
        packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
        std::copy(packed.begin(), packed.end(), outputs[learner]);
    }
}

void packInterleavedLearner4(std::size_t rows,
                             std::size_t cols,
                             const int8_t* input_interleaved,
                             std::size_t learner,
                             uint32_t* output) {
    std::vector<uint32_t> packed;
    packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
    std::copy(packed.begin(), packed.end(), output);
}

void packInterleavedLearner2(std::size_t rows,
                             std::size_t cols,
                             const int8_t* input_interleaved,
                             std::size_t learner,
                             uint32_t* output) {
    std::vector<uint32_t> packed;
    packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
    std::copy(packed.begin(), packed.end(), output);
}

void dumpInterleavedLearnerMatrices4(const std::string dump_dirs[4],
                                     const std::string& filename,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 4u; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
        dumpPackedMatrixIfEnabled(dump_dirs[learner], filename, packed.data(), rows, cols);
    }
}

void dumpInterleavedLearnerMatrices2(const std::string dump_dirs[2],
                                     const std::string& filename,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 2u; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
        dumpPackedMatrixIfEnabled(dump_dirs[learner], filename, packed.data(), rows, cols);
    }
}

void transposeInterleavedRowsToCols4(const int8_t* input_interleaved,
                                     int8_t* output_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const int8_t* in_slot = input_interleaved + ((row * cols + col) * 4u);
            int8_t* out_slot = output_interleaved + ((col * rows + row) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void transposeInterleavedRowsToCols2(const int8_t* input_interleaved,
                                     int8_t* output_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const int8_t* in_slot = input_interleaved + ((row * cols + col) * 2u);
            int8_t* out_slot = output_interleaved + ((col * rows + row) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void matmulInterleaved4DToInt8(const int8_t* lhs_interleaved,
                               const int8_t* rhs_by_col_interleaved,
                               std::size_t lhs_rows,
                               std::size_t rhs_cols,
                               std::size_t k_elems,
                               int8_t* output_interleaved) {
#if CFG_SIMD
    std::vector<int32_t> lhs_i32(kMatmulRowTile * kMatmulKTile * 4u, 0);
    std::vector<int32_t> rhs_i32(kMatmulColTile * kMatmulKTile * 4u, 0);
    std::vector<int32_t> partial_acc(kMatmulRowTile * kMatmulColTile * 4u, 0);
    std::vector<int32_t> output_acc(kMatmulRowTile * kMatmulColTile * 4u, 0);

    for (std::size_t row0 = 0; row0 < lhs_rows; row0 += kMatmulRowTile) {
        const std::size_t row_tile =
            std::min(kMatmulRowTile, lhs_rows - row0);

        for (std::size_t col0 = 0; col0 < rhs_cols; col0 += kMatmulColTile) {
            const std::size_t col_tile =
                std::min(kMatmulColTile, rhs_cols - col0);
            const std::size_t out_tile_count = row_tile * col_tile * 4u;

            std::fill(output_acc.begin(), output_acc.begin() + out_tile_count, 0);

            for (std::size_t k0 = 0; k0 < k_elems; k0 += kMatmulKTile) {
                const std::size_t k_tile =
                    std::min(kMatmulKTile, k_elems - k0);

                // Widen only the active row/column/K tile. The output still uses
                // the full rhs_cols stride below, so tiling does not change layout.
                unpackInterleavedTileToInt32(
                    lhs_interleaved,
                    k_elems,
                    row0,
                    k0,
                    row_tile,
                    k_tile,
                    4u,
                    lhs_i32.data());
                unpackInterleavedTileToInt32(
                    rhs_by_col_interleaved,
                    k_elems,
                    col0,
                    k0,
                    col_tile,
                    k_tile,
                    4u,
                    rhs_i32.data());

                sve_gemm_dense_int8_interleaved_4D(
                    lhs_i32.data(),
                    rhs_i32.data(),
                    static_cast<uint32_t>(row_tile),
                    static_cast<uint32_t>(col_tile),
                    static_cast<uint32_t>(k_tile),
                    partial_acc.data());

                for (std::size_t idx = 0; idx < out_tile_count; idx++) {
                    output_acc[idx] += partial_acc[idx];
                }
            }

            for (std::size_t row = 0; row < row_tile; row++) {
                for (std::size_t col = 0; col < col_tile; col++) {
                    const int32_t* acc_slot =
                        output_acc.data() + ((row * col_tile + col) * 4u);
                    int8_t* out_slot =
                        output_interleaved + (((row0 + row) * rhs_cols + col0 + col) * 4u);

                    for (std::size_t learner = 0; learner < 4u; learner++) {
                        out_slot[learner] = static_cast<int8_t>(acc_slot[learner]);
                    }
                }
            }
        }
    }
#else
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            int32_t acc[4] = {0, 0, 0, 0};
            for (std::size_t k = 0; k < k_elems; k++) {
                const int8_t* lhs_slot = lhs_interleaved + ((row * k_elems + k) * 4u);
                const int8_t* rhs_slot = rhs_by_col_interleaved + ((col * k_elems + k) * 4u);
                for (std::size_t learner = 0; learner < 4u; learner++) {
                    acc[learner] += static_cast<int32_t>(lhs_slot[learner]) *
                                    static_cast<int32_t>(rhs_slot[learner]);
                }
            }
            int8_t* out_slot = output_interleaved + ((row * rhs_cols + col) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = static_cast<int8_t>(acc[learner]);
            }
        }
    }
#endif
}

void matmulInterleaved2DToInt8(const int8_t* lhs_interleaved,
                               const int8_t* rhs_by_col_interleaved,
                               std::size_t lhs_rows,
                               std::size_t rhs_cols,
                               std::size_t k_elems,
                               int8_t* output_interleaved) {
#if CFG_SIMD
    std::vector<int32_t> lhs_i32(kMatmulRowTile * kMatmulKTile * 2u, 0);
    std::vector<int32_t> rhs_i32(kMatmulColTile * kMatmulKTile * 2u, 0);
    std::vector<int32_t> partial_acc(kMatmulRowTile * kMatmulColTile * 2u, 0);
    std::vector<int32_t> output_acc(kMatmulRowTile * kMatmulColTile * 2u, 0);

    for (std::size_t row0 = 0; row0 < lhs_rows; row0 += kMatmulRowTile) {
        const std::size_t row_tile =
            std::min(kMatmulRowTile, lhs_rows - row0);

        for (std::size_t col0 = 0; col0 < rhs_cols; col0 += kMatmulColTile) {
            const std::size_t col_tile =
                std::min(kMatmulColTile, rhs_cols - col0);
            const std::size_t out_tile_count = row_tile * col_tile * 2u;

            std::fill(output_acc.begin(), output_acc.begin() + out_tile_count, 0);

            for (std::size_t k0 = 0; k0 < k_elems; k0 += kMatmulKTile) {
                const std::size_t k_tile =
                    std::min(kMatmulKTile, k_elems - k0);

                // Same layout rule as the 4D path: scratch is compact, final
                // stores use full matrix coordinates.
                unpackInterleavedTileToInt32(
                    lhs_interleaved,
                    k_elems,
                    row0,
                    k0,
                    row_tile,
                    k_tile,
                    2u,
                    lhs_i32.data());
                unpackInterleavedTileToInt32(
                    rhs_by_col_interleaved,
                    k_elems,
                    col0,
                    k0,
                    col_tile,
                    k_tile,
                    2u,
                    rhs_i32.data());

                sve_gemm_dense_int8_interleaved_2D(
                    lhs_i32.data(),
                    rhs_i32.data(),
                    static_cast<uint32_t>(row_tile),
                    static_cast<uint32_t>(col_tile),
                    static_cast<uint32_t>(k_tile),
                    partial_acc.data());

                for (std::size_t idx = 0; idx < out_tile_count; idx++) {
                    output_acc[idx] += partial_acc[idx];
                }
            }

            for (std::size_t row = 0; row < row_tile; row++) {
                for (std::size_t col = 0; col < col_tile; col++) {
                    const int32_t* acc_slot =
                        output_acc.data() + ((row * col_tile + col) * 2u);
                    int8_t* out_slot =
                        output_interleaved + (((row0 + row) * rhs_cols + col0 + col) * 2u);

                    for (std::size_t learner = 0; learner < 2u; learner++) {
                        out_slot[learner] = static_cast<int8_t>(acc_slot[learner]);
                    }
                }
            }
        }
    }
#else
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            int32_t acc[2] = {0, 0};
            for (std::size_t k = 0; k < k_elems; k++) {
                const int8_t* lhs_slot = lhs_interleaved + ((row * k_elems + k) * 2u);
                const int8_t* rhs_slot = rhs_by_col_interleaved + ((col * k_elems + k) * 2u);
                for (std::size_t learner = 0; learner < 2u; learner++) {
                    acc[learner] += static_cast<int32_t>(lhs_slot[learner]) *
                                    static_cast<int32_t>(rhs_slot[learner]);
                }
            }
            int8_t* out_slot = output_interleaved + ((row * rhs_cols + col) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = static_cast<int8_t>(acc[learner]);
            }
        }
    }
#endif
}

void copyHeadToMultiheadInterleaved4D(const int8_t* head_interleaved,
                                      int8_t* multihead_interleaved,
                                      std::size_t seq_len,
                                      std::size_t head_idx,
                                      std::size_t head_hidden_size,
                                      std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            const int8_t* in_slot =
                head_interleaved + ((seq * head_hidden_size + feature) * 4u);
            int8_t* out_slot =
                multihead_interleaved + ((seq * multihead_cols + head_col_base + feature) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void copyHeadToMultiheadInterleaved2D(const int8_t* head_interleaved,
                                      int8_t* multihead_interleaved,
                                      std::size_t seq_len,
                                      std::size_t head_idx,
                                      std::size_t head_hidden_size,
                                      std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            const int8_t* in_slot =
                head_interleaved + ((seq * head_hidden_size + feature) * 2u);
            int8_t* out_slot =
                multihead_interleaved + ((seq * multihead_cols + head_col_base + feature) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}
