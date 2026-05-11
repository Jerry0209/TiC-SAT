#include "floatSoftmax.h"

#include <algorithm>
#include <cmath>

namespace TransformerFloat {

void FloatSoftmax::compute(float* scores,
                           std::size_t rows,
                           std::size_t cols,
                           float scale) const {
    for (std::size_t row = 0; row < rows; row++) {
        float* row_ptr = scores + row * cols;
        float max_value = row_ptr[0] * scale;
        for (std::size_t col = 1; col < cols; col++) {
            max_value = std::max(max_value, row_ptr[col] * scale);
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] = std::exp(row_ptr[col] * scale - max_value);
            sum += row_ptr[col];
        }

        const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] *= inv_sum;
        }
    }
}

void FloatSoftmax::computeInterleaved(float* scores,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t learner_count,
                                      float scale) const {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            float max_value = scores[(row * cols) * learner_count + learner] * scale;
            for (std::size_t col = 1; col < cols; col++) {
                const float value =
                    scores[((row * cols) + col) * learner_count + learner] * scale;
                max_value = std::max(max_value, value);
            }

            float sum = 0.0f;
            for (std::size_t col = 0; col < cols; col++) {
                float& slot = scores[((row * cols) + col) * learner_count + learner];
                slot = std::exp(slot * scale - max_value);
                sum += slot;
            }

            const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
            for (std::size_t col = 0; col < cols; col++) {
                scores[((row * cols) + col) * learner_count + learner] *= inv_sum;
            }
        }
    }
}

} // namespace TransformerFloat
