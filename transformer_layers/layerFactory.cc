#include "layerFactory.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "dense.h"
#include "run_mode_config.h"

#if CFG_USE_CODEBOOK_GEMM
#include "registry_adapter.h"
#include "codebookDense.h"
#endif

namespace {

#if CFG_USE_CODEBOOK_GEMM
CodebookDenseConfig makeCheckedCodebookConfig(
    const char* layer_name,
    std::size_t expected_input_size,
    std::size_t expected_output_size) {

    CodebookDenseConfig cfg = makeCodebookDenseConfigFromRegistry(layer_name);

    if (cfg.input_size != expected_input_size || cfg.output_size != expected_output_size) {
        throw std::invalid_argument(
            "Codebook registry shape mismatch for layer: " + std::string(layer_name));
    }

    return cfg;
}
#endif

} // namespace

LinearLayerBundle LayerFactory::create(
    const std::string& layer_name,
    std::size_t input_size,
    std::size_t output_size,
    uint32_t* fallback_weight) {

    LinearLayerBundle bundle;

#if CFG_USE_CODEBOOK_GEMM
    try {
        CodebookDenseConfig cfg = makeCheckedCodebookConfig(
            layer_name.c_str(), input_size, output_size);

        bundle.main = new CodebookDense(cfg);
        std::cout << "[LayerFactory] Using CodebookDense for " << layer_name << std::endl;
    } catch (const std::exception&) {
        bundle.main = new Dense(input_size, output_size, fallback_weight);
        std::cout << "[LayerFactory] Fallback to Dense for " << layer_name << std::endl;
    }
#else
    bundle.main = new Dense(input_size, output_size, fallback_weight);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    bundle.reference = new Dense(input_size, output_size, fallback_weight);
#endif

    return bundle;
}