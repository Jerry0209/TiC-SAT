// #pragma once

// #ifdef RELOAD_WEIGHT
// #define CFG_RELOAD_WEIGHT 1
// #else
// #define CFG_RELOAD_WEIGHT 0
// #endif

// #if CFG_RELOAD_WEIGHT && defined(USE_NOTEBOOK_GENERATED_WEIGHTS)
// #define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 1
// #else
// #define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 0
// #endif

// #if CFG_RELOAD_WEIGHT && defined(USE_CODEBOOK_GEMM)
// #define CFG_USE_CODEBOOK_GEMM 1
// #else
// #define CFG_USE_CODEBOOK_GEMM 0
// #endif

// #if CFG_USE_CODEBOOK_GEMM
// #define CFG_USE_CODEBOOK_REFERENCE 1
// #else
// #define CFG_USE_CODEBOOK_REFERENCE 0
// #endif

// #if CFG_USE_CODEBOOK_GEMM && !CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
// #error "USE_CODEBOOK_GEMM requires USE_NOTEBOOK_GENERATED_WEIGHTS, otherwise Dense reference comparison is meaningless."
// #endif


#pragma once

// --------------------------------------------------
// Base runtime/data-source switches
// --------------------------------------------------

#ifdef RELOAD_WEIGHT
#define CFG_RELOAD_WEIGHT 1
#else
#define CFG_RELOAD_WEIGHT 0
#endif

#if CFG_RELOAD_WEIGHT && defined(USE_NOTEBOOK_GENERATED_WEIGHTS)
#define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 1
#else
#define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 0
#endif

#if CFG_RELOAD_WEIGHT && defined(USE_CODEBOOK_GEMM)
#define CFG_USE_CODEBOOK_GEMM 1
#else
#define CFG_USE_CODEBOOK_GEMM 0
#endif

// --------------------------------------------------
// Validation / profiling switches
// --------------------------------------------------

// Enable Dense reference path and numerical comparison.
// Default: OFF unless explicitly requested.
#if CFG_USE_CODEBOOK_GEMM && defined(ENABLE_CODEBOOK_REFERENCE)
#define CFG_USE_CODEBOOK_REFERENCE 1
#else
#define CFG_USE_CODEBOOK_REFERENCE 0
#endif

// Enable debug prints.
// Default: OFF unless explicitly requested.
#ifdef ENABLE_DEBUG_PRINT
#define CFG_ENABLE_DEBUG_PRINT 1
#else
#define CFG_ENABLE_DEBUG_PRINT 0
#endif

// Optional: a dedicated profiling mode.
// If enabled, force-disable reference and debug printing.
#ifdef PROFILE_GEMM_ONLY
#define CFG_PROFILE_GEMM_ONLY 1
#else
#define CFG_PROFILE_GEMM_ONLY 0
#endif

#if CFG_PROFILE_GEMM_ONLY
#undef CFG_USE_CODEBOOK_REFERENCE
#define CFG_USE_CODEBOOK_REFERENCE 0

#undef CFG_ENABLE_DEBUG_PRINT
#define CFG_ENABLE_DEBUG_PRINT 0
#endif

// --------------------------------------------------
// Sanity checks
// --------------------------------------------------

// Codebook GEMM needs notebook-generated weights / registry-backed data.
#if CFG_USE_CODEBOOK_GEMM && !CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#error "USE_CODEBOOK_GEMM requires USE_NOTEBOOK_GENERATED_WEIGHTS."
#endif

// Reference comparison only makes sense when codebook GEMM is enabled.
#if CFG_USE_CODEBOOK_REFERENCE && !CFG_USE_CODEBOOK_GEMM
#error "ENABLE_CODEBOOK_REFERENCE requires USE_CODEBOOK_GEMM."
#endif


// #if CFG_ENABLE_DEBUG_PRINT
// std::cout << "Head : " << n << std::endl;
// #endif