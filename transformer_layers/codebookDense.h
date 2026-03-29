#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "linearLayer.h"

struct CodebookDenseConfig {
    std::size_t input_size;
    std::size_t output_size;
    std::size_t n_words_row;
    uint8_t bits_per_cb;
    const uint32_t *weight_idx;
    const float *codebook = nullptr;
    const int8_t *codebook_int8 = nullptr;
    const float *bias = nullptr;
    float input_dequant_scale = 1.0f;
    float output_quant_scale = 1.0f;
};

class CodebookDense : public LinearLayer {
public:
    explicit CodebookDense(const CodebookDenseConfig &config);
    ~CodebookDense() override = default;

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) override;

private:
    static uint32_t getPackedIndex(const uint32_t *packed_row,
                                   std::size_t elem_idx,
                                   uint8_t bits_per_cb);
    static int8_t unpackInt8(const uint32_t *packed, std::size_t elem_idx);
    static void packInt8(const std::vector<int8_t> &src, uint32_t *dst);
    static int8_t clampInt32ToInt8(int32_t value);

    void runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const;

    std::size_t input_size_;
    std::size_t output_size_;
    std::size_t n_words_row_;
    uint8_t bits_per_cb_;
    const uint32_t *weight_idx_;
    const float *codebook_; // Redundant
    const float *bias_;

    // Reserved for future mixed-scale path; currently compact int GEMM consumes int8 directly.
    float input_dequant_scale_;
    float output_quant_scale_;

    // For integer weights
    std::vector<int8_t> codebook_q_;
    std::vector<int32_t> bias_q_;
};
