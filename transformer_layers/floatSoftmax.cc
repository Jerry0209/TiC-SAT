#include "floatSoftmax.h"

#include <algorithm>
#include <cmath>

namespace TransformerFloat {

// Apply row-wise softmax to an in-place score matrix.
//
// The attention code passes Q*K scores here. scale is normally
// 1 / sqrt(head_hidden_size), which keeps dot-product magnitudes stable before
// exponentiation. The function overwrites scores with probabilities.
void FloatSoftmax::compute(float* scores,
                           std::size_t rows,
                           std::size_t cols,
                           float scale) const {
    for (std::size_t row = 0; row < rows; row++) {
        // Step 1: Find the scaled row maximum. Subtracting this value before
        // exp() improves numerical stability without changing softmax results.
        float* row_ptr = scores + row * cols;
        float max_value = row_ptr[0] * scale;
        for (std::size_t col = 1; col < cols; col++) {
            max_value = std::max(max_value, row_ptr[col] * scale);
        }

        // Step 2: Exponentiate the shifted scaled scores and accumulate the row
        // sum that will be used for normalization.
        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] = std::exp(row_ptr[col] * scale - max_value);
            sum += row_ptr[col];
        }

        // Step 3: Divide by the sum so the row becomes a probability
        // distribution. A zero sum is guarded even though the shifted exp values
        // should normally make it positive.
        const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] *= inv_sum;
        }
    }
}

// Interleaved row-wise softmax.
//
// scores stores multiple learner matrices in one buffer:
//   scores[(row * cols + col) * learner_count + learner]
// Softmax is still computed independently for each learner and row.
void FloatSoftmax::computeInterleaved(float* scores,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t learner_count,
                                      float scale) const {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            // Step 1: Find the maximum for this row and learner stream.
            float max_value = scores[(row * cols) * learner_count + learner] * scale;
            for (std::size_t col = 1; col < cols; col++) {
                const float value =
                    scores[((row * cols) + col) * learner_count + learner] * scale;
                max_value = std::max(max_value, value);
            }

            // Step 2: Convert shifted scores into exponentials and accumulate
            // the learner-specific row sum.
            float sum = 0.0f;
            for (std::size_t col = 0; col < cols; col++) {
                float& slot = scores[((row * cols) + col) * learner_count + learner];
                slot = std::exp(slot * scale - max_value);
                sum += slot;
            }

            // Step 3: Normalize this learner's row in place.
            const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
            for (std::size_t col = 0; col < cols; col++) {
                scores[((row * cols) + col) * learner_count + learner] *= inv_sum;
            }
        }
    }
}

} // namespace TransformerFloat
