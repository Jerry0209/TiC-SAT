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
    std::size_t expected_output_size,
    std::size_t learner) {

    CodebookDenseConfig cfg = makeCodebookDenseConfigFromRegistry(layer_name, learner);

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
    uint32_t* fallback_weight,
    std::size_t learner) {

    LinearLayerBundle bundle;

#if CFG_USE_CODEBOOK_GEMM
    try {
        CodebookDenseConfig cfg = makeCheckedCodebookConfig(
            layer_name.c_str(), input_size, output_size, learner); // Get config from registry and validate shape

        bundle.main = new CodebookDense(cfg);
#if CFG_ENABLE_DEBUG_PRINT
        std::cout << "[LayerFactory] Using CodebookDense for " << layer_name << std::endl;
#endif
    } catch (const std::exception& ex) {
#if CFG_PROFILE_GEMM_ONLY
        throw std::runtime_error(
            "PROFILE_GEMM_ONLY requires a matching CodebookDense registry entry for " +
            layer_name + ": " + ex.what());
#elif CFG_CODEBOOK_ONLY_MODE
        throw std::runtime_error(
            "Codebook-only mode requires a matching CodebookDense registry entry for " +
            layer_name +
            ". Dense fallback weights are intentionally skipped when "
            "ENABLE_CODEBOOK_REFERENCE is disabled: " + ex.what());
#else
        bundle.main = new Dense(input_size, output_size, fallback_weight);
#if CFG_ENABLE_DEBUG_PRINT
        std::cout << "[LayerFactory] Fallback to Dense for " << layer_name << std::endl;
#endif
#endif
    }
#else
    bundle.main = new Dense(input_size, output_size, fallback_weight);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    bundle.reference = new Dense(input_size, output_size, fallback_weight);
#endif

    return bundle;
}
