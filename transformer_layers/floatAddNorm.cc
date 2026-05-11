#include "floatAddNorm.h"

#include <cmath>
#include <vector>

namespace TransformerFloat {

namespace {

constexpr float kLayerNormEps = 1.0e-5f;

} // namespace

FloatAddNormalize::FloatAddNormalize(std::size_t seq_len, std::size_t input_dim)
    : seq_len_(seq_len), input_dim_(input_dim) {}

void FloatAddNormalize::compute(const float* residual, float* candidate) const {
    for (std::size_t row = 0; row < seq_len_; row++) {
        float sum = 0.0f;
        float* out_row = candidate + row * input_dim_;
        const float* residual_row = residual + row * input_dim_;

        for (std::size_t col = 0; col < input_dim_; col++) {
            out_row[col] += residual_row[col];
            sum += out_row[col];
        }

        const float mean = sum / static_cast<float>(input_dim_);
        float variance = 0.0f;
        for (std::size_t col = 0; col < input_dim_; col++) {
            const float diff = out_row[col] - mean;
            variance += diff * diff;
        }

        const float inv_sd =
            1.0f / std::sqrt(variance / static_cast<float>(input_dim_) + kLayerNormEps);
        for (std::size_t col = 0; col < input_dim_; col++) {
            out_row[col] = (out_row[col] - mean) * inv_sd;
        }
    }
}

void FloatAddNormalize::computeInterleaved(const float* residual_interleaved,
                                           float* candidate_interleaved,
                                           std::size_t learner_count) const {
    for (std::size_t row = 0; row < seq_len_; row++) {
        std::vector<float> sum(learner_count, 0.0f);

        for (std::size_t col = 0; col < input_dim_; col++) {
            float* out_slot = candidate_interleaved + ((row * input_dim_ + col) * learner_count);
            const float* residual_slot =
                residual_interleaved + ((row * input_dim_ + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] += residual_slot[learner];
                sum[learner] += out_slot[learner];
            }
        }

        std::vector<float> mean(learner_count, 0.0f);
        std::vector<float> variance(learner_count, 0.0f);
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            mean[learner] = sum[learner] / static_cast<float>(input_dim_);
        }

        for (std::size_t col = 0; col < input_dim_; col++) {
            const float* out_slot =
                candidate_interleaved + ((row * input_dim_ + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                const float diff = out_slot[learner] - mean[learner];
                variance[learner] += diff * diff;
            }
        }

        for (std::size_t col = 0; col < input_dim_; col++) {
            float* out_slot = candidate_interleaved + ((row * input_dim_ + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                const float inv_sd =
                    1.0f / std::sqrt(variance[learner] / static_cast<float>(input_dim_) +
                                      kLayerNormEps);
                out_slot[learner] = (out_slot[learner] - mean[learner]) * inv_sd;
            }
        }
    }
}

} // namespace TransformerFloat
