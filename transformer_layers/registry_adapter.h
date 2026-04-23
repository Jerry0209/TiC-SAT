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
    cfg.weight_idx = getGeneratedCodebookWeightIdx(view, learner);
    cfg.codebook_int8 = getGeneratedCodebookInt8(view, learner);
    cfg.bias = getGeneratedCodebookBias(view, learner);
    cfg.input_dequant_scale = 1.0f;
    cfg.output_quant_scale = 1.0f;

    if (cfg.weight_idx == nullptr || cfg.codebook_int8 == nullptr) {
        throw std::runtime_error(std::string("Layer registry entry is missing learner data: ") + layer_name);
    }

    return cfg;
}
