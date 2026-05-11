#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "floatCommon.h"

namespace TransformerFloat {

void saveFloatMatrixText(const std::string& path,
                         const float* data,
                         std::size_t rows,
                         std::size_t cols);

void dumpFloatMatrixIfEnabled(const std::string& dump_dir,
                              const std::string& filename,
                              const float* data,
                              std::size_t rows,
                              std::size_t cols);

void dumpInterleavedFloatMatrices(const std::vector<std::string>& dump_dirs,
                                  const std::string& filename,
                                  const float* interleaved,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t learner_count);

void printFloatPreview(const std::string& label,
                       const float* data,
                       std::size_t size);

void printInterleavedFloatPreview(const std::string& label,
                                  const float* interleaved,
                                  std::size_t logical_size,
                                  std::size_t learner_count,
                                  std::size_t learner);

void interleaveLearnerMatrices(const float* const* inputs,
                               std::size_t learner_count,
                               std::size_t rows,
                               std::size_t cols,
                               Matrix& output_interleaved);

void deinterleaveLearnerMatrices(const Matrix& input_interleaved,
                                 std::size_t learner_count,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float* const* outputs);

} // namespace TransformerFloat
