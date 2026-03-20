#include "codebookDense.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
int8_t clampToInt8(float value) {
    float rounded = std::round(value);
    if (rounded > static_cast<float>(std::numeric_limits<int8_t>::max())) {
        return std::numeric_limits<int8_t>::max();
    }
    if (rounded < static_cast<float>(std::numeric_limits<int8_t>::min())) {
        return std::numeric_limits<int8_t>::min();
    }
    return static_cast<int8_t>(rounded);
}
}

CodebookDense::CodebookDense(const CodebookDenseConfig &config)
    : input_size_(config.input_size),
      output_size_(config.output_size),
      n_words_row_(config.n_words_row),
      bits_per_cb_(config.bits_per_cb),
      weight_idx_(config.weight_idx),
      codebook_(config.codebook),
      bias_(config.bias),
      input_dequant_scale_(config.input_dequant_scale == 0.0f ? 1.0f : config.input_dequant_scale),
      output_quant_scale_(config.output_quant_scale == 0.0f ? 1.0f : config.output_quant_scale) {
    if (input_size_ == 0 || output_size_ == 0) {
        throw std::invalid_argument("CodebookDense requires non-zero input and output sizes");
    }
    if (bits_per_cb_ == 0 || bits_per_cb_ > 16) {
        throw std::invalid_argument("CodebookDense bits_per_cb must be between 1 and 16");
    }
    if ((input_size_ % 4) != 0 || (output_size_ % 4) != 0) {
        throw std::invalid_argument("CodebookDense expects packed int8 tensors with dimensions divisible by 4");
    }
    std::size_t idxs_per_word = 32u / bits_per_cb_;
    std::size_t required_n_words_row = (input_size_ + idxs_per_word - 1) / idxs_per_word;
    if (n_words_row_ < required_n_words_row) {
        throw std::invalid_argument("CodebookDense n_words_row is too small for the provided input size and bits_per_cb");
    }
    if (weight_idx_ == nullptr || codebook_ == nullptr) {
        throw std::invalid_argument("CodebookDense requires valid weight_idx and codebook pointers");
    }
}

uint32_t CodebookDense::getPackedIndex(const uint32_t *packed_row,
                                       std::size_t elem_idx,
                                       uint8_t bits_per_cb) {
    std::size_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    std::size_t word_idx = elem_idx / idxs_per_word;
    std::size_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    return (packed_row[word_idx] >> offset) & idx_mask;
}

int8_t CodebookDense::unpackInt8(const uint32_t *packed, std::size_t elem_idx) {
    std::size_t word_idx = elem_idx / 4;
    std::size_t byte_idx = elem_idx % 4;
    uint32_t word = packed[word_idx];
    return static_cast<int8_t>((word >> (byte_idx * 8)) & 0xFF);
}

void CodebookDense::packInt8(const std::vector<int8_t> &src, uint32_t *dst) {
    if ((src.size() % 4) != 0) {
        throw std::invalid_argument("CodebookDense output size must be divisible by 4 for packed int8 storage");
    }
    std::size_t packed_size = src.size() / 4;
    for (std::size_t word_idx = 0; word_idx < packed_size; word_idx++) {
        uint32_t packed_word = 0;
        for (std::size_t byte_idx = 0; byte_idx < 4; byte_idx++) {
            uint8_t value = static_cast<uint8_t>(src[word_idx * 4 + byte_idx]);
            packed_word |= static_cast<uint32_t>(value) << (byte_idx * 8);
        }
        dst[word_idx] = packed_word;
    }
}

void CodebookDense::runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const {
    std::vector<int8_t> packed_values(seq_len * output_size_, 0);

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        const uint32_t *input_row = input + seq * (input_size_ / 4);
        for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
            const uint32_t *packed_row = &weight_idx_[out_idx * n_words_row_];
            int sum = 0;

            for (std::size_t in_idx = 0; in_idx < input_size_; in_idx++) {
                int8_t input_value = unpackInt8(input_row, in_idx);
                std::size_t reordered_in_idx = (in_idx & ~static_cast<std::size_t>(3)) + (3 - (in_idx & 3));
                uint32_t cb_idx = getPackedIndex(packed_row, reordered_in_idx, bits_per_cb_);
                int8_t weight_value = clampToInt8(codebook_[cb_idx] * output_quant_scale_);
                sum += static_cast<int>(input_value) * static_cast<int>(weight_value);
            }

            if (bias_ != nullptr) {
                sum += static_cast<int>(std::round(bias_[out_idx] * output_quant_scale_));
            }

            packed_values[seq * output_size_ + out_idx] = static_cast<int8_t>(sum);
        }
    }

    packInt8(packed_values, output);
}

void CodebookDense::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    runCompactGemm(seq_len, input, output);
}
