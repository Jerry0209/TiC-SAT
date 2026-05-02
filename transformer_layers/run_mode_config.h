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

// Emit named gem5 checkpoint dumps for coarse transformer regions.
// Post-processing can subtract adjacent dumpstats snapshots to get each
// region while the last snapshot remains the whole transformer block total.
#ifdef GEM5_PROFILE_REGIONS
#define CFG_GEM5_PROFILE_REGIONS 1
#else
#define CFG_GEM5_PROFILE_REGIONS 0
#endif

// Enable SIMD/SVE implementations when the build defines SIMD.
#ifdef SIMD
#define CFG_SIMD 1
#else
#define CFG_SIMD 0
#endif

// Keep the whole 4-learner transformer block in [seq][feature][learner]
// layout between grouped CodebookDense, attention, softmax, AddNorm, and FFN.
#ifdef FULL_INTERLEAVED_PIPELINE
#define CFG_FULL_INTERLEAVED_PIPELINE 1
#else
#define CFG_FULL_INTERLEAVED_PIPELINE 0
#endif

#if CFG_PROFILE_GEMM_ONLY || CFG_GEM5_PROFILE_REGIONS
#undef CFG_USE_CODEBOOK_REFERENCE
#define CFG_USE_CODEBOOK_REFERENCE 0

#undef CFG_ENABLE_DEBUG_PRINT
#define CFG_ENABLE_DEBUG_PRINT 0
#endif

// Pure codebook mode means the run should rely entirely on the generated
// CodebookDense registry for layer weights. In this mode we do not build Dense
// fallback/reference weights unless reference validation is explicitly enabled.
#if CFG_USE_CODEBOOK_GEMM && !CFG_USE_CODEBOOK_REFERENCE
#define CFG_CODEBOOK_ONLY_MODE 1
#else
#define CFG_CODEBOOK_ONLY_MODE 0
#endif

// --------------------------------------------------
// Sanity checks
// --------------------------------------------------

// Dense fallback/reference runs still need notebook-generated per-learner .bin
// weights. Pure codebook runs skip Dense weights entirely, so they may use the
// generated registry without notebook-generated .bin files.
#if CFG_USE_CODEBOOK_GEMM && !CFG_PROFILE_GEMM_ONLY && \
    !CFG_CODEBOOK_ONLY_MODE && !CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#error "Non-profile Codebook GEMM requires USE_NOTEBOOK_GENERATED_WEIGHTS when Dense fallback/reference is enabled."
#endif

// Reference comparison only makes sense when codebook GEMM is enabled.
#if CFG_USE_CODEBOOK_REFERENCE && !CFG_USE_CODEBOOK_GEMM
#error "ENABLE_CODEBOOK_REFERENCE requires USE_CODEBOOK_GEMM."
#endif

#if CFG_FULL_INTERLEAVED_PIPELINE && !CFG_USE_CODEBOOK_GEMM
#error "FULL_INTERLEAVED_PIPELINE requires USE_CODEBOOK_GEMM."
#endif

#if CFG_FULL_INTERLEAVED_PIPELINE && !CFG_SIMD
#error "FULL_INTERLEAVED_PIPELINE requires SIMD_FLAG=1."
#endif


// #if CFG_ENABLE_DEBUG_PRINT
// std::cout << "Head : " << n << std::endl;
// #endif
