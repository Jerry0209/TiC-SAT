#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

void interleavePackedLearners4(std::size_t rows,
                               std::size_t cols,
                               uint32_t* const inputs[4],
                               int8_t* output_interleaved);

void packInterleavedLearners4(std::size_t rows,
                              std::size_t cols,
                              const int8_t* input_interleaved,
                              uint32_t* const outputs[4]);

void packInterleavedLearner4(std::size_t rows,
                             std::size_t cols,
                             const int8_t* input_interleaved,
                             std::size_t learner,
                             uint32_t* output);

void dumpInterleavedLearnerMatrices4(const std::string dump_dirs[4],
                                     const std::string& filename,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols);

void transposeInterleavedRowsToCols4(const int8_t* input_interleaved,
                                     int8_t* output_interleaved,
                                     std::size_t rows,
                                     std::size_t cols);

void matmulInterleaved4DToInt8(const int8_t* lhs_interleaved,
                               const int8_t* rhs_by_col_interleaved,
                               std::size_t lhs_rows,
                               std::size_t rhs_cols,
                               std::size_t k_elems,
                               int8_t* output_interleaved);

void copyHeadToMultiheadInterleaved4D(const int8_t* head_interleaved,
                                      int8_t* multihead_interleaved,
                                      std::size_t seq_len,
                                      std::size_t head_idx,
                                      std::size_t head_hidden_size,
                                      std::size_t num_heads);
