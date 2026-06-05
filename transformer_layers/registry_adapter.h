#pragma once

#include <stdexcept>
#include <string>
#include "../Full_NN/gemm_definitions/generated_codebook_registry.h"
#include "codebookDense.h"

/**
 * Convert a generated registry entry into the runtime CodebookDenseConfig.
 *
 * @param layer_name Name used to find a GeneratedCodebookLayerView in the
 *        generated registry. This must match the exported layer name exactly.
 * @param learner Learner index to read from learner-specific registry arrays.
 *        For SAME_SEQ layers this selects one weight_idx/codebook/bias slice;
 *        for DIFF_SEQ or interleaved execution the full per-learner arrays are
 *        also attached to the config.
 *
 * @return Runtime configuration consumed by CodebookDense.
 *
 * The copied fields describe both layer shape and data layout. Shape fields
 * size the GEMM; learner/layout fields decide which grouped kernel is legal;
 * pointer fields attach generated weight indices, codebooks, and bias tables
 * without copying the generated arrays.
 */
inline CodebookDenseConfig makeCodebookDenseConfigFromRegistry(const char* layer_name,
                                                               std::size_t learner = 0) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(layer_name);
    if (!view) {
        throw std::runtime_error(std::string("Layer not found in registry: ") + layer_name);
    }

    CodebookDenseConfig cfg;
    cfg.input_size = view->input_size; // Logical input features per token row.
    cfg.output_size = view->output_size; // Logical output features per token row.
    cfg.n_words_row = view->n_words_row; // Packed codebook-index words per output row.
    cfg.bits_per_cb = view->bits_per_cb; // Bits used for each compressed codebook index.
    cfg.n_learners = view->n_learners; // Number of learner slices exported for this layer.
    cfg.same_seq = view->same_seq; // True when learners share the same sequence layout.
    cfg.selected_learner = learner; // Learner slice used by non-interleaved execution.
    cfg.weight_idx = getGeneratedCodebookWeightIdx(view, learner); // SAME_SEQ weight indices.
    cfg.weight_idx_by_learner = view->weight_idx_by_learner; // DIFF_SEQ per-learner indices.
    // Weight-index layout consumed directly by grouped interleaved kernels.
    cfg.weight_idx_interleaved = getGeneratedCodebookWeightIdxInterleaved(view);
    cfg.codebook_int8 = getGeneratedCodebookInt8(view, learner); // Selected learner codebook.
    cfg.codebooks_int8 = view->codebooks_int8; // All learner codebooks in registry order.
    // Codebook layout consumed directly by grouped interleaved kernels.
    cfg.codebook_int8_interleaved = getGeneratedCodebookInt8Interleaved(view);
    cfg.bias = getGeneratedCodebookBias(view, learner); // Selected learner bias vector.
    cfg.biases = view->biases; // All learner bias vectors in registry order.
    cfg.bias_interleaved = getGeneratedCodebookBiasInterleaved(view); // Grouped bias layout.
    cfg.input_dequant_scale = 1.0f; // Current int8 path treats inputs as already quantized.
    cfg.output_quant_scale = 1.0f; // Current int8 path emits raw quantized output scale.

    if (cfg.weight_idx == nullptr || cfg.codebook_int8 == nullptr) {
        throw std::runtime_error(std::string("Layer registry entry is missing learner data: ") + layer_name);
    }

    return cfg;
}

/**
 * Return how many learners are stored for a generated CodebookDense layer.
 *
 * @param layer_name Registry key for the generated layer.
 *
 * @return Number of learner slices available for this layer.
 *
 * @throws std::runtime_error if the layer is absent from the generated
 *         registry.
 */
inline std::size_t getCodebookDenseLearnerCount(const char* layer_name) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(layer_name);
    if (!view) {
        throw std::runtime_error(std::string("Layer not found in registry: ") + layer_name);
    }
    return view->n_learners;
}
