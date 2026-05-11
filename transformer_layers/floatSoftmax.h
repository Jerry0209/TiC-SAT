#pragma once

#include <cstddef>

namespace TransformerFloat {

class FloatSoftmax {
public:
    FloatSoftmax() = default;

    void compute(float* scores,
                 std::size_t rows,
                 std::size_t cols,
                 float scale) const;

    void computeInterleaved(float* scores,
                            std::size_t rows,
                            std::size_t cols,
                            std::size_t learner_count,
                            float scale) const;
};

} // namespace TransformerFloat
