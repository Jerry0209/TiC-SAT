#include "floatAddNorm.h"

#include <cmath>
#include <vector>

namespace TransformerFloat {

namespace {

// Small constant added to variance before the square root. This prevents
// division by zero for rows whose values are all equal and matches the usual
// layer-normalization epsilon pattern.
constexpr float kLayerNormEps = 1.0e-5f;

} // namespace

FloatAddNormalize::FloatAddNormalize(std::size_t seq_len, std::size_t input_dim)
    : seq_len_(seq_len), input_dim_(input_dim) {}

// Add a residual tensor to candidate in place, then apply row-wise layer norm.
//
// residual is the skip-connection input from an earlier transformer stage.
// candidate is the newly computed stage output and is also the destination. This
// is used after attention projection and after the feed-forward projection.
void FloatAddNormalize::compute(const float* residual, float* candidate) const {
    for (std::size_t row = 0; row < seq_len_; row++) {
        // Step 1: Add the residual values to the candidate row and accumulate
        // the sum needed for the row mean.
        float sum = 0.0f;
        float* out_row = candidate + row * input_dim_;
        const float* residual_row = residual + row * input_dim_;

        for (std::size_t col = 0; col < input_dim_; col++) {
            out_row[col] += residual_row[col];
            sum += out_row[col];
        }

        // Step 2: Compute the row mean, then accumulate the squared distance of
        // each value from that mean.
        const float mean = sum / static_cast<float>(input_dim_);
        float variance = 0.0f;
        for (std::size_t col = 0; col < input_dim_; col++) {
            const float diff = out_row[col] - mean;
            variance += diff * diff;
        }

        // Step 3: Normalize each value to zero mean and unit variance. No learned
        // gamma/beta terms are applied in this implementation.
        const float inv_sd =
            1.0f / std::sqrt(variance / static_cast<float>(input_dim_) + kLayerNormEps);
        for (std::size_t col = 0; col < input_dim_; col++) {
            out_row[col] = (out_row[col] - mean) * inv_sd;
        }
    }
}

// Interleaved version of compute().
//
// The logical tensor is still [seq_len x input_dim] for each learner, but memory
// stores learners together at each element:
//   buffer[(row * input_dim + col) * learner_count + learner]
// Mean and variance are therefore tracked separately for every learner.
void FloatAddNormalize::computeInterleaved(const float* residual_interleaved,
                                           float* candidate_interleaved,
                                           std::size_t learner_count) const {
    for (std::size_t row = 0; row < seq_len_; row++) {
        // Step 1: Add residuals and collect per-learner row sums.
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

        // Step 2: Convert sums to means and compute a separate variance for
        // each learner stream.
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

        // Step 3: Normalize every interleaved learner value in place.
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
