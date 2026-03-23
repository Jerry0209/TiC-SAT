#include "codebookDense.h"

#include "../Full_NN/inc/gemm_exec.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>


namespace {

/**
 * @brief Round a floating-point value to the nearest int8_t and saturate if out of range.
 *
 * This is used when converting floating-point codebook values into quantized int8 values.
 * If the rounded value exceeds the valid int8 range [-128, 127], it is clamped.
 *
 * @param value Input floating-point number.
 * @return Quantized and saturated int8_t value.
 */
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


/**
 * @brief Round a floating-point-like value to the nearest int32_t and saturate if out of range.
 *
 * This is mainly used to quantize floating-point bias values into int32 accumulators.
 * int32 is used because GEMM accumulation results are wider than int8.
 *
 * @param value Input double value.
 * @return Quantized and saturated int32_t value.
 */
int32_t clampToInt32(double value) {
    double rounded = std::round(value);
    if (rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(rounded);
}


/**
 * @brief Reverse the order of elements inside each group of four input values (Endianness coversion).
 *
 * Example:
 *   group [0,1,2,3] becomes [3,2,1,0]
 *   group [4,5,6,7] becomes [7,6,5,4]
 *
 * This is useful when the hardware or packed layout expects bytes in reverse order
 * inside each 32-bit word.
 *
 * @param elem_idx Original element index.
 * @return Reordered element index inside the same group of four.
 */
std::size_t reverseInputGroupOfFour(std::size_t elem_idx) {
    return (elem_idx & ~static_cast<std::size_t>(3)) + (3 - (elem_idx & 3));
}
}


/**
 * @brief Construct a CodebookDense layer from a configuration object.
 *
 * This constructor:
 * - stores the layer dimensions and configuration,
 * - validates input arguments,
 * - quantizes the codebook into int8 form if only float codebook is provided,
 * - quantizes the bias into int32 form if bias exists.
 *
 * Expected assumptions:
 * - input and output tensors are packed int8 values stored in uint32_t arrays
 *   (4 int8 values per uint32_t),
 * - therefore input_size and output_size must both be divisible by 4.
 *
 * @param config Structure containing all layer parameters and data pointers.
 * @throws std::invalid_argument if configuration is invalid.
 */
CodebookDense::CodebookDense(const CodebookDenseConfig &config)
    : input_size_(config.input_size),
      output_size_(config.output_size),
      n_words_row_(config.n_words_row),
      bits_per_cb_(config.bits_per_cb),
      weight_idx_(config.weight_idx),
      codebook_(config.codebook),
      bias_(config.bias),
      input_dequant_scale_(config.input_dequant_scale == 0.0f ? 1.0f : config.input_dequant_scale),
      output_quant_scale_(config.output_quant_scale == 0.0f ? 1.0f : config.output_quant_scale),
      reverse_input_groups_of_4_(config.reverse_input_groups_of_4) {
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
    if (weight_idx_ == nullptr) {
        throw std::invalid_argument("CodebookDense requires a valid weight_idx pointer");
    }

    std::size_t codebook_size = static_cast<std::size_t>(1u) << bits_per_cb_;
    if (config.codebook_int8 != nullptr) {
        codebook_q_.assign(config.codebook_int8, config.codebook_int8 + codebook_size);
    } else if (codebook_ != nullptr) {
        codebook_q_.reserve(codebook_size);
        for (std::size_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
            codebook_q_.push_back(clampToInt8(codebook_[cb_idx] * output_quant_scale_));
        }
    } else {
        throw std::invalid_argument("CodebookDense requires either float or int8 codebook data");
    }

    if (bias_ != nullptr) {
        bias_q_.reserve(output_size_);
        for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
            bias_q_.push_back(clampToInt32(static_cast<double>(bias_[out_idx]) * output_quant_scale_));
        }
    }
}

/**
 * @brief Extract one packed codebook index from a row of packed index words.
 *
 * The weight matrix is not stored directly. Instead, each weight entry is represented
 * by a small integer index into the codebook. Multiple indexes are packed into a uint32_t.
 *
 * This function:
 * - determines which 32-bit word contains the requested index,
 * - computes the bit offset inside that word,
 * - extracts the index using bit shift and masking.
 *
 * @param packed_row Pointer to the packed row of codebook indexes.
 * @param elem_idx Index of the element to extract.
 * @param bits_per_cb Number of bits used per codebook index.
 * @return Extracted codebook index.
 */
uint32_t CodebookDense::getPackedIndex(const uint32_t *packed_row,
                                       std::size_t elem_idx,
                                       uint8_t bits_per_cb) {
    std::size_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    std::size_t word_idx = elem_idx / idxs_per_word;
    std::size_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    return (packed_row[word_idx] >> offset) & idx_mask;
}


/**
 * @brief Unpack one int8 value from a uint32_t-packed tensor.
 *
 * In this storage format, each uint32_t contains 4 signed int8 values:
 *   byte 0 -> element 0
 *   byte 1 -> element 1
 *   byte 2 -> element 2
 *   byte 3 -> element 3
 *
 * This function reads the correct 32-bit word, selects the desired byte,
 * and returns it as signed int8.
 *
 * @param packed Pointer to packed int8 data.
 * @param elem_idx Index of the int8 element to read.
 * @return Unpacked int8 value.
 */
int8_t CodebookDense::unpackInt8(const uint32_t *packed, std::size_t elem_idx) {
    std::size_t word_idx = elem_idx / 4;
    std::size_t byte_idx = elem_idx % 4;
    uint32_t word = packed[word_idx];
    return static_cast<int8_t>((word >> (byte_idx * 8)) & 0xFF);
}


/**
 * @brief Pack a vector of int8 values into uint32_t words.
 *
 * Every 4 int8 values are packed into one uint32_t:
 *   src[0] -> bits [7:0]
 *   src[1] -> bits [15:8]
 *   src[2] -> bits [23:16]
 *   src[3] -> bits [31:24]
 *
 * This matches the storage format expected by the rest of the framework.
 *
 * @param src Source int8 vector. Its size must be divisible by 4.
 * @param dst Destination buffer for packed uint32_t words.
 * @throws std::invalid_argument if src.size() is not divisible by 4.
 */
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

/**
 * @brief Saturate an int32 accumulator value into int8 range.
 *
 * After GEMM, results are accumulated in int32 to avoid overflow during summation.
 * This function compresses the final value back into int8 with saturation.
 *
 * @param value Input int32 value.
 * @return Saturated int8 result.
 */
int8_t CodebookDense::clampInt32ToInt8(int32_t value) {
    if (value > static_cast<int32_t>(std::numeric_limits<int8_t>::max())) {
        return std::numeric_limits<int8_t>::max();
    }
    if (value < static_cast<int32_t>(std::numeric_limits<int8_t>::min())) {
        return std::numeric_limits<int8_t>::min();
    }
    return static_cast<int8_t>(value);
}

/**
 * @brief Execute the dense layer using the compact GEMM backend.
 *
 * Main workflow:
 * 1. Unpack the packed uint32_t input into a plain int8 vector.
 * 2. Optionally reverse each group of 4 input bytes to match hardware/data layout.
 * 3. Call gemm_exec_compact_int() to perform codebook-based matrix multiplication.
 * 4. Clamp the int32 outputs to int8.
 * 5. Pack the int8 output back into uint32_t format.
 *
 * The GEMM backend interprets:
 * - input_unpacked as regular int8 activations,
 * - weight_idx_ as packed codebook indexes,
 * - codebook_q_ as the quantized value lookup table,
 * - bias_q_ as optional int32 bias.
 *
 * @param seq_len Number of input rows / sequence tokens.
 * @param input Pointer to packed int8 input tensor.
 * @param output Pointer to packed int8 output tensor.
 */
void CodebookDense::runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const {
    std::vector<int8_t> input_unpacked(seq_len * input_size_, 0);
    for (std::size_t seq = 0; seq < seq_len; seq++) {
        const uint32_t *input_row = input + seq * (input_size_ / 4); // Input matrix from previous layer
        for (std::size_t in_idx = 0; in_idx < input_size_; in_idx++) {
            std::size_t dst_idx = reverse_input_groups_of_4_ ? reverseInputGroupOfFour(in_idx) : in_idx;
            input_unpacked[seq * input_size_ + dst_idx] = unpackInt8(input_row, in_idx);
        }
    }

    std::vector<int32_t> output_acc(seq_len * output_size_, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

    gemm_exec_compact_int(layer,
                          input_unpacked.data(), // vector, take the head address
                          weight_idx_,
                          codebook_q_.data(), // quantized codebook
                          bias_q_.empty() ? nullptr : bias_q_.data(),
                          output_acc.data(),
                          bits_per_cb_);

    std::vector<int8_t> output_int8(seq_len * output_size_, 0);
    for (std::size_t i = 0; i < output_acc.size(); i++) {
        output_int8[i] = clampInt32ToInt8(output_acc[i]);
    }

    packInt8(output_int8, output); // Pack to 32-bit words and output
}


/**
 * @brief Public inference entry point for the dense layer.
 *
 * This function currently delegates all work to runCompactGemm().
 *
 * @param seq_len Number of sequence elements / rows in the input tensor.
 * @param input Pointer to packed input buffer.
 * @param output Pointer to packed output buffer.
 */
void CodebookDense::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    runCompactGemm(seq_len, input, output);
}
