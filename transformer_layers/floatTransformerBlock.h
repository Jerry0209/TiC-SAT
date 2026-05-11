#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "floatAddNorm.h"
#include "floatCodebookDense.h"
#include "floatCommon.h"
#include "floatSelfAttention.h"
#include "run_mode_config.h"

namespace TransformerFloat {

class FloatTransformerBlock {
public:
    FloatTransformerBlock(std::size_t pre_seq_len,
                          std::size_t input_dim,
                          std::size_t head_hidden_size,
                          std::size_t num_heads,
                          std::size_t ff_size,
                          std::size_t learner_idx = 0,
                          std::string dump_dir = "");

    ~FloatTransformerBlock();

    void compute(std::size_t seq_len, const float* input, float* output);
    static void computeGroup2(std::size_t seq_len,
                              FloatTransformerBlock* blocks[2],
                              const float* const inputs[2],
                              float* const outputs[2]);
    static void computeGroup4(std::size_t seq_len,
                              FloatTransformerBlock* blocks[4],
                              const float* const inputs[4],
                              float* const outputs[4]);

private:
    template <std::size_t LearnerCount>
    static void computeGroupImpl(std::size_t seq_len,
                                 FloatTransformerBlock** blocks,
                                 const float* const* inputs,
                                 float* const* outputs);

#if CFG_FULL_INTERLEAVED_PIPELINE
    template <std::size_t LearnerCount>
    static void computeFullInterleavedBlock(std::size_t seq_len,
                                            FloatTransformerBlock** blocks,
                                            const float* const* inputs,
                                            float* const* outputs);

    static void computeGroup2FullInterleaved(std::size_t seq_len,
                                             FloatTransformerBlock* blocks[2],
                                             const float* const inputs[2],
                                             float* const outputs[2]);
    static void computeGroup4FullInterleaved(std::size_t seq_len,
                                             FloatTransformerBlock* blocks[4],
                                             const float* const inputs[4],
                                             float* const outputs[4]);
#endif

    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    std::size_t learner_idx_;
    std::string dump_dir_;

    std::vector<FloatSingleHeadSelfAttn*> selfatten_;

    Matrix multihead_out_;
    Matrix condense_out_;
    Matrix intermediate_ff_;
    Matrix ff1_out_;

    FloatAddNormalize add_norm_;
    FloatCodebookDense condense_;
    FloatCodebookDense feed_forward0_;
    FloatCodebookDense feed_forward1_;
};

} // namespace TransformerFloat
