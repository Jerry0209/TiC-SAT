#include <feedforward.h>
#include <relu.h>

void feedforward_exec_noCB(feedforward_t ff_layer,
                           const float *in,
                           const float *weights_0,
                           const float *bias_0,
                           const float *weights_1,
                           const float *bias_1,
                           float *hidden,
                           float *out) {
    gemm_exec_noCB(ff_layer.fc1, in, weights_0, bias_0, hidden);
    relu(hidden, ff_layer.fc1.seq_len * ff_layer.fc1.output_size);
    gemm_exec_noCB(ff_layer.fc2, hidden, weights_1, bias_1, out);
}

void feedforward_exec_compact(feedforward_t ff_layer,
                              const float *in,
                              const uint32_t *weight_idx_0,
                              const float *codebook_0,
                              const float *bias_0,
                              const uint32_t *weight_idx_1,
                              const float *codebook_1,
                              const float *bias_1,
                              float *hidden,
                              float *out,
                              uint8_t bits_per_cb) {
    gemm_exec_compact(ff_layer.fc1, in, weight_idx_0, codebook_0, bias_0, hidden,
                      bits_per_cb);
    relu(hidden, ff_layer.fc1.seq_len * ff_layer.fc1.output_size);
    gemm_exec_compact(ff_layer.fc2, hidden, weight_idx_1, codebook_1, bias_1, out,
                      bits_per_cb);
}
