#pragma once

#include <cstddef>
#include <string>

#include "floatCommon.h"

namespace TransformerFloat {

class FloatCodebookDense {
public:
    explicit FloatCodebookDense(std::string layer_name, std::size_t learner = 0);

    void compute(std::size_t rows, const float* input, Matrix& output) const;
    void computeInterleaved(std::size_t learner_count,
                            const float* input_interleaved,
                            std::size_t rows,
                            Matrix& output_interleaved) const;

    std::size_t inputSize() const;
    std::size_t outputSize() const;

private:
    std::string layer_name_;
    std::size_t learner_;
};

} // namespace TransformerFloat
