#pragma once

#include <stdexcept>
#include <string>
#include "../Full_NN/gemm_definitions/generated_codebook_registry.h"
#include "codebookDense.h"

inline CodebookDenseConfig makeCodebookDenseConfigFromRegistry(const char* layer_name) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(layer_name);
    if (!view) {
        throw std::runtime_error(std::string("Layer not found in registry: ") + layer_name);
    }

    CodebookDenseConfig cfg;
    cfg.input_size = view->input_size;
    cfg.output_size = view->output_size;
    cfg.n_words_row = view->n_words_row;
    cfg.bits_per_cb = view->bits_per_cb;
    cfg.weight_idx = view->weight_idx;
    cfg.codebook_int8 = view->codebook_int8;
    cfg.bias = view->bias;
    cfg.input_dequant_scale = 1.0f;
    cfg.output_quant_scale = 1.0f;
    return cfg;
}