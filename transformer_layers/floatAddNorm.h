#pragma once

#include <cstddef>

namespace TransformerFloat {

class FloatAddNormalize {
public:
    FloatAddNormalize(std::size_t seq_len, std::size_t input_dim);

    void compute(const float* residual, float* candidate) const;
    void computeInterleaved(const float* residual_interleaved,
                            float* candidate_interleaved,
                            std::size_t learner_count) const;

private:
    std::size_t seq_len_;
    std::size_t input_dim_;
};

} // namespace TransformerFloat
