#pragma once

#include <stdexcept>
#include <string>
#include "../Full_NN/gemm_definitions/generated_codebook_registry.h"
#include "codebookDense.h"

inline CodebookDenseConfig makeCodebookDenseConfigFromRegistry(const char* layer_name,
                                                               std::size_t learner = 0) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(layer_name);
    if (!view) {
        throw std::runtime_error(std::string("Layer not found in registry: ") + layer_name);
    }

    CodebookDenseConfig cfg;
    cfg.input_size = view->input_size;
    cfg.output_size = view->output_size;
    cfg.n_words_row = view->n_words_row;
    cfg.bits_per_cb = view->bits_per_cb;
    cfg.n_learners = view->n_learners;
    cfg.same_seq = view->same_seq;
    cfg.selected_learner = learner;
    cfg.weight_idx = getGeneratedCodebookWeightIdx(view, learner); // Used for SAME_SEQ
    cfg.weight_idx_by_learner = view->weight_idx_by_learner; // Used for DIFF_SEQ
    cfg.weight_idx_interleaved = getGeneratedCodebookWeightIdxInterleaved(view);
    cfg.codebook_int8 = getGeneratedCodebookInt8(view, learner);
    cfg.codebooks_int8 = view->codebooks_int8;
    cfg.codebook_int8_interleaved = getGeneratedCodebookInt8Interleaved(view);
    cfg.bias = getGeneratedCodebookBias(view, learner);
    cfg.biases = view->biases;
    cfg.bias_interleaved = getGeneratedCodebookBiasInterleaved(view);
    cfg.input_dequant_scale = 1.0f;
    cfg.output_quant_scale = 1.0f;

    if (cfg.weight_idx == nullptr || cfg.codebook_int8 == nullptr) {
        throw std::runtime_error(std::string("Layer registry entry is missing learner data: ") + layer_name);
    }

    return cfg;
}

inline std::size_t getCodebookDenseLearnerCount(const char* layer_name) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(layer_name);
    if (!view) {
        throw std::runtime_error(std::string("Layer not found in registry: ") + layer_name);
    }
    return view->n_learners;
}
