# I2CE Transformer README

This document explains the Transformer experiment implementation in this
repository: which files are required, what each major file does, how the
integer and FP32 Transformer paths execute, how the fully interleaved pipeline
differs from the normal grouped path, and which compile-time macros are useful
for experiments.

The implementation described here is in this project root:

```bash
/home/thu/TiC-SAT
```

The final executable is produced as:

```bash
/home/thu/TiC-SAT/transformer.o
```

Note: the compile script in the repository is named `compile_transformer.sh`
with a lowercase `c`.

## 1. File Map

### 1.1 Generator Files

Generator location:

```text
Full_NN/generators/
```

Important files:

| File | Function |
| --- | --- |
| `Full_NN/generators/Transformer_generator.ipynb` | Main notebook used to generate Transformer inputs, weights, codebook definitions, generated C/C++ headers, and `.bin` weight files used by the C/C++ implementation. |
| `Full_NN/generators/gemm_layer_generator.py` | Helper used by the notebook to generate GEMM layer headers, compact codebook/index data, and TiC-SAT-compatible weight files. |
| `Full_NN/generators/codebooks_defs_generator.py` | Generates global codebook configuration and the generated codebook registry header. |
| `Full_NN/generators/transformer_debug_utils.py` | Python-side reference/debug utilities for comparing notebook-generated tensors with C/C++ output dumps. |

The notebook currently generates Transformer layer names that match the C++
layer factory:

```text
q_h0, k_h0, v_h0
q_h1, k_h1, v_h1
...
condense
ff0
ff1
```



**Codebook GEMM Transformer weights:**

* Used for codebooked GEMM Transformer

| Generated file or directory | Function |
| --- | --- |
| `Full_NN/gemm_definitions/codebooks_def.h` | Global generated configuration: learner count, SVE packing constants, codebook size, bits per codebook index, same-sequence mode, and FP32 mode flag. |
| `Full_NN/gemm_definitions/input_matrix.h` | Generated FP32 input matrix used by the FP32 Transformer path. |
| `Full_NN/gemm_definitions/gemm_header_*.h` | Layer-specific generated compact GEMM data. |
| `Full_NN/gemm_definitions/generated_codebook_registry.h` | Registry that maps layer names such as `q_h0` or `ff1` to generated codebook/index/bias arrays. This is used by `LayerFactory`, `CodebookDense`, and the FP32 implementation. |


**Default dense Transformer weights:**

* Used for default dense Transformer (either used for debug reference or legacy TiC-SAT)


| Generated file or directory | Function |
| --- | --- |
| `weights/generated_from_notebook/learner*/H*.bin` | Notebook-generated dense binary weights and inputs for each learner. These are used when `USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1`. |
| `weights/H-1_L-1.bin` | Legacy/shared generated input file path used by some reload paths. |

Common generated naming convention:

| Logical tensor/layer | Typical generated file |
| --- | --- |
| Input tensor | `H-1_L-1.bin` |
| Query head `h` | `q_h<h>` and usually `H<h>_L0.bin` |
| Key head `h` | `k_h<h>` and usually `H<h>_L1.bin` |
| Value head `h` | `v_h<h>` and usually `H<h>_L2.bin` |
| Multi-head projection | `condense` and usually `H-1_L0.bin` |
| Feed-forward 0 | `ff0` and usually `H-1_L1.bin` |
| Feed-forward 1 | `ff1` and usually `H-1_L2.bin` |

The exact dimensions are controlled by the notebook and must match
`transformer.h` and the generated registry.

### 1.2 GEMM Code

Core GEMM implementation:

```text
Full_NN/inc/gemm_exec.h
Full_NN/src/gemm_exec.c
Full_NN/inc/gemm_SVE.h
Full_NN/src/gemm_SVE.c
```

`Full_NN/inc/gemm_exec.h` defines the shared GEMM descriptor:

```c
typedef struct {
    uint16_t seq_len;
    uint16_t input_size;
    uint16_t output_size;
    uint16_t n_words_row;
} gemm_t;
```

`Full_NN/src/gemm_exec.c` implements the scalar compact GEMM kernels and the
SVE wrapper functions. Important functions:

| Function | Role |
| --- | --- |
| `gemm_exec_noCB` | FP32 dense GEMM without codebook compression. |
| `gemm_exec_compact` | FP32 compact/codebook GEMM. |
| `gemm_exec_noCB_int` | Integer dense GEMM without codebook compression. |
| `gemm_exec_compact_int` | Integer compact/codebook GEMM. |
| `gemm_exec_compact_fp32_interleaved_2D_same_seq` | Scalar FP32 2-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_fp32_interleaved_4D_same_seq` | Scalar FP32 4-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_fp32_interleaved_4D_diff_seq` | Scalar FP32 4-learner interleaved compact GEMM for different per-learner index streams. (Developed but not used in this project)|
| `gemm_exec_compact_int_interleaved_2D_same_seq` | Scalar int8 2-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_int_interleaved_4D_same_seq` | Scalar int8 4-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_int_interleaved_4D_diff_seq` | Scalar int8 4-learner interleaved compact GEMM for different per-learner index streams. (Developed but not used in this project)|
| `gemm_exec_compact_sve` | SVE FP32 compact GEMM wrapper. |
| `gemm_exec_compact_int_sve` | SVE int8 compact GEMM wrapper. |
| `gemm_exec_compact_sve_fp32_interleaved_*` | SVE FP32 interleaved compact GEMM wrappers. |
| `gemm_exec_compact_int_sve_interleaved_*` | SVE int8 interleaved compact GEMM wrappers. |

`Full_NN/src/gemm_SVE.c` contains the low-level SVE row kernels and dense
interleaved attention kernels. Important functions:

| Function | Role |
| --- | --- |
| `sve_gemm_row_compact_int8` | SVE row kernel for single-learner int8 compact GEMM. |
| `sve_gemm_row_compact_fp32` | SVE row kernel for single-learner FP32 compact GEMM. |
| `sve_gemm_row_compact_fp32_interleaved_2D_same_seq` | SVE row kernel for FP32 2D same-sequence interleaving. |
| `sve_gemm_row_compact_fp32_interleaved_4D_same_seq` | SVE row kernel for FP32 4D same-sequence interleaving. |
| `sve_gemm_row_compact_fp32_interleaved_4D_diff_seq` | SVE row kernel for FP32 4D different-sequence interleaving. (Developed but not used in this project)|
| `sve_gemm_row_compact_int8_interleaved_2D_same_seq` | SVE row kernel for int8 2D same-sequence interleaving. |
| `sve_gemm_row_compact_int8_interleaved_4D_same_seq` | SVE row kernel for int8 4D same-sequence interleaving. |
| `sve_gemm_row_compact_int8_interleaved_4D_diff_seq` | SVE row kernel for int8 4D different-sequence interleaving. (Developed but not used in this project)|
| `sve_gemm_dense_int8_interleaved_2D` | SVE dense int8 attention matmul for 2 interleaved learners. |
| `sve_gemm_dense_int8_interleaved_4D` | SVE dense int8 attention matmul for 4 interleaved learners. |

### 1.3 Top-Level Transformer Code

Top-level files:

```text
transformer.cpp
transformer.h
compile_transformer.sh
```

| File | Function |
| --- | --- |
| `transformer.cpp` | Main executable source. Selects runtime mode, loads or generates input/weights, creates Transformer blocks, chooses single learner/multi-learners/INT8/FP32 execution, and calls the block compute functions. |
| `transformer.h` | Defines Transformer dimensions such as `D_Q`, `D_SEQ`, `D_MODEL`, `NUM_HEAD`, and `D_FF`. The active dimensions must match the generated files. |
| `compile_transformer.sh` | Cross-compiles the Transformer binary for AArch64. It selects the compiler, defines experiment macros, includes the generated GEMM headers, and optionally adds SVE support. |

`transformer.cpp` chooses the learner count using this logic:

| Mode | Learner count source |
| --- | --- |
| Dense no-SIMD baseline | `N_LEARNERS` from `Full_NN/gemm_definitions/codebooks_def.h` |
| Codebook GEMM | Generated codebook registry learner count, usually read from layer `q_h0` |
<!-- | Default dense path | `1` | -->

`transformer.cpp` then selects the execution path:

```text
main()
  -> test()
      -> if CFG_USE_FP32_TRANSFORMER:
             TransformerFloat::run(...)
         else:
             build integer TransformerBlock objects
             choose compute(), computeGroup2(), or computeGroup4()
```

### 1.4 Integer Transformer Layer Files

Integer Transformer implementation:

```text
transformer_layers/
```

Important files:

| File | Function |
| --- | --- |
| `transformer_layers/transformerBlock.cc` / `.h` | Integer Transformer block. Owns attention heads, projection layer, feed-forward layers, AddNorm, buffers, and multi-learner/ full-interleaved multi-learner execution. |
| `transformer_layers/selfattention.cc` / `.h` | Integer single-head self-attention. Computes Q/K/V, attention score matmul, softmax, and softmax-times-V. Also contains multi-learner and fully interleaved multi-learner attention entry points. |
| `transformer_layers/codebookDense.cc` / `.h` | Integer codebook-backed dense layer. Converts packed int8 activations, dispatches compact GEMM kernels, and supports multi-learner interleaved GEMM. |
| `transformer_layers/layerFactory.cc` / `.h` | Creates either `CodebookDense` or fallback `Dense` layers based on compile macros and generated registry availability. |
| `transformer_layers/linearLayer.h` | Abstract interface used by `TransformerBlock` and `SingleHeadSelfAttn`: `compute(seq_len, input, output)`. |
| `transformer_layers/addNorm.cc` / `.h` | Integer residual add plus layer normalization. Includes normal packed, BWMA rearranged, 2D interleaved, and 4D interleaved variants. |
| `transformer_layers/softmax.cc` / `.h` | Integer/LUT softmax for attention scores. Includes normal packed, BWMA rearranged, 2D interleaved, and 4D interleaved variants. |
| `transformer_layers/interleavedPipeline.cc` / `.h` | Helpers for the fully interleaved int8 pipeline: interleave, pack, transpose, dense interleaved attention matmul, and copy-head helpers. |
| `transformer_layers/transformerBlockInterleavedHelpers.cc` / `.h` | Validation and helper calls for full int8 interleaved `CodebookDense` execution and optional reference comparison. |
| `transformer_layers/debuggerFunctions.cc` / `.h` | Packed tensor printing, dumping, comparison, weight read/write helpers, and grouped CodebookDense helper functions. |
| `transformer_layers/transpose.cc` / `.h` | Transpose helpers for attention score computation in packed integer layouts. |
| `transformer_layers/dense.cc` / `.h` | Default dense integer layer used as fallback or reference. |

Important integer functions requested explicitly:

| Function | File | Used by |
| --- | --- | --- |
| `void AddNormalize::computeInterleaved4D(...)` | `transformer_layers/addNorm.cc` | Fully interleaved 4-learner int8 Transformer block. |
| `void AddNormalize::computeInterleaved2D(...)` | `transformer_layers/addNorm.cc` | Fully interleaved 2-learner int8 Transformer block. |
| `void Softmax::computeInterleaved2D(int8_t *input, std::size_t seq_len)` | `transformer_layers/softmax.cc` | Fully interleaved 2-learner int8 attention. |
| `void Softmax::computeInterleaved4D(int8_t *input, std::size_t seq_len)` | `transformer_layers/softmax.cc` | Fully interleaved 4-learner int8 attention. |
| `void matmulInterleaved4DToInt8(...)` | `transformer_layers/interleavedPipeline.cc` | Fully interleaved 4-learner int8 attention score and value matmul. |
| `void matmulInterleaved2DToInt8(...)` | `transformer_layers/interleavedPipeline.cc` | Fully interleaved 2-learner int8 attention score and value matmul. |

### 1.5 Floating-Point Transformer Files

FP32 Transformer implementation:

| File | Function |
| --- | --- |
| `transformer_layers/transformerFloat.cc` / `.h` | Top-level FP32 entry point called from `transformer.cpp` when `USE_FP32_TRANSFORMER_FLAG=1`. Loads `input_matrix.h`, builds per-learner FP32 blocks, and selects single/grouped execution. |
| `transformer_layers/floatTransformerBlock.cc` / `.h` | FP32 Transformer block. Implements single, grouped non-interleaved, and fully interleaved block execution. |
| `transformer_layers/floatSelfAttention.cc` / `.h` | FP32 single-head self-attention. Implements normal and interleaved Q/K/V projection, QK matmul, softmax, and softmax-times-V. |
| `transformer_layers/floatCodebookDense.cc` / `.h` | FP32 generated codebook dense layer. Uses generated registry and dispatches FP32 compact GEMM kernels, including 2D and 4D interleaved kernels. |
| `transformer_layers/floatAddNorm.cc` / `.h` | FP32 residual add plus layer normalization for normal and interleaved layouts. |
| `transformer_layers/floatSoftmax.cc` / `.h` | FP32 row-wise softmax for normal and interleaved layouts. |
| `transformer_layers/floatDump.cc` / `.h` | FP32 matrix dumping, previews, and interleave/deinterleave helpers. |
| `transformer_layers/floatCommon.h` | Defines the FP32 `Matrix` alias. |

The FP32 path still uses the generated codebook registry. It does not use the
integer dense fallback/reference layer.

### 1.6 Shared Helper Files

| File | Function |
| --- | --- |
| `transformer_layers/run_mode_config.h` | Centralizes compile-time macro normalization into `CFG_*` macros and performs sanity checks. |
| `transformer_layers/registry_adapter.h` | Converts generated registry (weight .h) entries into `CodebookDenseConfig` objects for integer `CodebookDense`. |
| `transformer_layers/profile.cc` / `.h` | gem5 profiling helpers. Calls `m5 resetstats`, `m5 dumpstats`, or `m5 dumpresetstats` when available. |
| `Full_NN/gemm_definitions/generated_codebook_registry.h` | Generated registry used by integer and FP32 codebook dense layers. |
| `Full_NN/gemm_definitions/codebooks_def.h` | Generated global constants such as `N_LEARNERS`, `N_SVE_LANES`, `CB_SIZE`, `BITS_PER_CB`, `SAME_SEQ`, and `USE_F32`. |

## 2. Generated Configuration

`Full_NN/gemm_definitions/codebooks_def.h` controls generated experiment
metadata. Typical fields:

| Macro | Meaning |
| --- | --- |
| `N_LEARNERS` | Number of learner models generated.  |
| `N_SVE_LANES` | Number of FP32 SVE lanes assumed by generated constants. |
| `N_SVE_BYTE` | Number of int8 values corresponding to the SVE vector length setting. |
| `CB_SIZE` | Codebook size. |
| `BITS_PER_CB` | Number of bits per packed codebook index. |
| `IDXS_PER_WORD` | Number of packed codebook indexes in one 32-bit word. |
| `IDX_MASK` | Mask used to extract one packed codebook index. |
| `SAME_SEQ` | Whether the generated layers use the same weight-index stream across learners. |
| `USE_F32` | Generated flag indicating FP32 generated data support. |

For codebook experiments, regenerate the notebook outputs after changing model
dimensions, learner count, codebook size, or same-sequence/different-sequence
settings.

## 3. Compile Script

Compile from the project root:

```bash
cd /home/thu/TiC-SAT
source ./compile_transformer.sh
```

The script finds an AArch64 compiler in this order:

1. USER CUSTOM compiler path
2. `aarch64-linux-gnu-g++`
3. `aarch64-conda-linux-gnu-g++`

The core compile command includes:

```text
transformer.cpp
transformer_layers/*.cc
Full_NN/src/gemm_exec.c
Full_NN/src/gemm_SVE.c          only when SIMD_FLAG=1
accelerator/smm_gem.cpp
accelerator/systolic_m2m.cc
```

Include paths:

```text
-I.
-IFull_NN/inc
-IFull_NN/gemm_definitions
```

Always-defined compile macros (Legacy from TiC-SAT, do not need to change them):

| Macro | Meaning |
| --- | --- |
| `SA_SIZE=4` | Systolic-array size used by accelerator helpers. |
| `DEVELOP` | Enables development build behavior in the accelerator code. |
| `CORE_NUM=1` | Number of cores requested by this build configuration. |

The output binary is:

```text
transformer.o
```

### 3.1 Experiment Flags and Macros

`compile_transformer.sh` accepts environment variables ending in `_FLAG`. When a
flag is `1`, the script adds the corresponding `-D...` macro.

| Environment flag | Default | Macro | Meaning |
| --- | ---: | --- | --- |
| `RELOAD_WEIGHT_FLAG` | `1` | `RELOAD_WEIGHT` | Enables runtime loading/generation of weights instead of relying only on static/default state. If this is disabled, the weights will be generated during the runtime, otherwise the program will load the weights generated before |
| `USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG` | `1` | `USE_NOTEBOOK_GENERATED_WEIGHTS` | Uses `.bin` weights and input generated by `Transformer_generator.ipynb`. Effective only when `RELOAD_WEIGHT` is enabled. |
| `USE_CODEBOOK_GEMM_FLAG` | `0` | `USE_CODEBOOK_GEMM` | Uses generated `CodebookDense` layers instead of the default dense layer path. With this flag enabled, the program execute codebooked GEMM instead of default dense GEMM. Effective only when `RELOAD_WEIGHT` is enabled. |
| `ENABLE_CODEBOOK_REFERENCE_FLAG` | `0` | `ENABLE_CODEBOOK_REFERENCE` | Builds a dense reference path beside codebook GEMM and compares outputs. Useful for debugging correctness, not profiling. |
| `ENABLE_DEBUG_PRINT_FLAG` | `0` | `ENABLE_DEBUG_PRINT` | Enables debug prints and previews. Automatically disabled by profiling modes. |
| `PROFILE_GEMM_ONLY_FLAG` | `0` | `PROFILE_GEMM_ONLY` | Disables reference/debug overhead so profiling focuses on codebook GEMM. |
| `GEM5_PROFILE_REGIONS_FLAG` | `0` | `GEM5_PROFILE_REGIONS` | Emits named gem5 stats checkpoints for high-level Transformer regions. If this is enabled, the profiling of the program will be divided into 6 parts: `after_mha`, `after_projection`, `after_attn_addnorm`, `after_ff1`, `after_ff2`, `final_total`, so we can see the statistics of different layers. |
| `FULL_INTERLEAVED_PIPELINE_FLAG` | `0` | `FULL_INTERLEAVED_PIPELINE` | Keeps all learner activations interleaved across the whole grouped Transformer block (Our project is focused on fully interleaved path, so it's always enabled). |
| `USE_FP32_TRANSFORMER_FLAG` | `0` | `USE_FP32_TRANSFORMER` | Runs the FP32 Transformer path instead of the integer path. Requires codebook GEMM. |
| `DENSE_NO_SIMD_BASELINE_FLAG` | `0` | `DENSE_NO_SIMD_BASELINE` | Runs the default dense no-SIMD sequential learner baseline. Cannot be combined with SIMD, codebook GEMM, or full interleaving. It's used to execute multiple learners sequentially using default dense Transformer so as to create the baselines of multi-learner inference|
| `SIMD_FLAG` | `0` | `SIMD` | Enables SVE compilation, adds `-march=armv8-a+sve`, and compiles `Full_NN/src/gemm_SVE.c`. |

### 3.2 Normalized `CFG_*` Macros

`transformer_layers/run_mode_config.h` converts raw build macros into normalized
`CFG_*` values:

| Normalized macro | Meaning |
| --- | --- |
| `CFG_RELOAD_WEIGHT` | `1` when `RELOAD_WEIGHT` is defined. |
| `CFG_USE_NOTEBOOK_GENERATED_WEIGHTS` | `1` when reloading weights and `USE_NOTEBOOK_GENERATED_WEIGHTS` is defined. |
| `CFG_USE_CODEBOOK_GEMM` | `1` when reloading weights and `USE_CODEBOOK_GEMM` is defined. |
| `CFG_USE_CODEBOOK_REFERENCE` | `1` when codebook GEMM and `ENABLE_CODEBOOK_REFERENCE` are both enabled. |
| `CFG_ENABLE_DEBUG_PRINT` | `1` when debug printing is enabled and profiling did not disable it. |
| `CFG_PROFILE_GEMM_ONLY` | `1` when GEMM-only profiling is enabled. |
| `CFG_GEM5_PROFILE_REGIONS` | `1` when gem5 region profiling is enabled. |
| `CFG_SIMD` | `1` when `SIMD` is defined. |
| `CFG_DENSE_NO_SIMD_BASELINE` | `1` for the dense no-SIMD sequential learner baseline. |
| `CFG_FULL_INTERLEAVED_PIPELINE` | `1` for the full grouped interleaved pipeline. |
| `CFG_USE_FP32_TRANSFORMER` | `1` for the FP32 Transformer path. |
| `CFG_CODEBOOK_ONLY_MODE` | `1` when codebook GEMM is enabled without dense reference. This skips dense fallback/reference weights. |

<!-- Weight-source note:

```text
In profiling/codebook-only mode, USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG mainly
affects the input tensor, because layer weights come from the generated
codebook registry.

In dense fallback/reference modes, USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG affects
both the input tensor and the dense .bin weights.
``` -->

Important sanity checks:

| Rule | Why |
| --- | --- |
| `ENABLE_CODEBOOK_REFERENCE` requires `USE_CODEBOOK_GEMM`. | Reference comparison only makes sense against codebook execution. |
| `DENSE_NO_SIMD_BASELINE` requires `SIMD_FLAG=0`. | This baseline is intentionally scalar/sequential. |
| `DENSE_NO_SIMD_BASELINE` requires `USE_CODEBOOK_GEMM_FLAG=0`. | It benchmarks dense execution, not codebook execution. |
| `DENSE_NO_SIMD_BASELINE` cannot combine with `FULL_INTERLEAVED_PIPELINE`. | Full interleaving is a grouped codebook pipeline. |
| `FULL_INTERLEAVED_PIPELINE` requires `USE_CODEBOOK_GEMM`. | Full interleaving relies on codebook interleaved kernels. |
| Integer `FULL_INTERLEAVED_PIPELINE` requires `SIMD_FLAG=1`. | The int8 full pipeline is intended for SVE interleaved execution. |
| `USE_FP32_TRANSFORMER` requires `USE_CODEBOOK_GEMM`. | The FP32 path uses the generated codebook registry. |
| `USE_FP32_TRANSFORMER` cannot combine with `ENABLE_CODEBOOK_REFERENCE`. | The dense reference checker is for the integer codebook path. |

## 4. Function Call Maps

This section shows which functions are called in each major run mode.

### 4.1 Top-Level Dispatch

```text
main()
  -> test()
      -> print dimensions and CFG_* flags
      -> learner_count = getTransformerLearnerCount()
      -> if CFG_USE_FP32_TRANSFORMER:
             TransformerFloat::run(learner_count, dump_dir)
             return
      -> load or generate packed int8 input
      -> create TransformerBlock objects
      -> if learner_count == 4:
             TransformerBlock::computeGroup4(...)
         else if learner_count == 2:
             TransformerBlock::computeGroup2(...)
         else:
             TransformerBlock::compute(...)
```

### 4.2 Layer Creation

All integer Transformer dense/projection layers are created through
`LayerFactory`:

```text
TransformerBlock constructor
  -> LayerFactory::create("condense", ...)
  -> LayerFactory::create("ff0", ...)
  -> LayerFactory::create("ff1", ...)

SingleHeadSelfAttn constructor
  -> LayerFactory::create("q_hX", ...)
  -> LayerFactory::create("k_hX", ...)
  -> LayerFactory::create("v_hX", ...)
```

`LayerFactory::create(...)` behavior:

```text
if CFG_USE_CODEBOOK_GEMM:
    look up layer name in generated_codebook_registry.h
    if found and shape matches:
        return CodebookDense
    if profile/codebook-only mode:
        throw error if missing
    otherwise:
        fall back to Dense
else:
    return Dense
```

If `CFG_USE_CODEBOOK_REFERENCE=1`, the codebook path also creates dense
reference layers for comparison.

### 4.3 Integer Single-Learner or Default Non-Grouped Path

Used when learner count is `1`, or when the default dense path is selected:

```text
TransformerBlock::compute(...)
  -> TransformerBlock::computeWithStatsLabel("single_transformer_block")
      -> TransformerBlock::computeBody(...)
```

Inside `computeBody(...)`:

```text
for each attention head:
    SingleHeadSelfAttn::compute(...)
        -> query_layer_->compute(...)
             -> Dense::compute(...) or CodebookDense::compute(...)
                 -> CodebookDense::runCompactGemm(...)
                     -> gemm_exec_compact_int(...)
                     -> or gemm_exec_compact_int_sve(...) when SIMD
        -> key_layer_->compute(...)
        -> value_layer_->compute(...)
        -> Transpose::transpose(...) or Transpose::transpose_rearranged(...)
        -> attention Q*K matmul:
             -> smmComputeRWMA(...) or simdComputeRWMA(...)
             -> or smmComputeBWMA(...) or simdComputeBWMA(...) for BWMA
        -> Softmax::compute(...)
             -> or Softmax::computeRearranged(...) for BWMA
        -> attention softmax*V matmul:
             -> smmComputeRWMA(...) or simdComputeRWMA(...)
             -> or smmComputeBWMA(...) or simdComputeBWMA(...) for BWMA
        -> Softmax::post_softmax(...)

Transpose::multihead_transpose(...) if needed

condense->compute(...)
    -> Dense::compute(...) or CodebookDense::compute(...)

AddNormalize::compute(...)
    -> or AddNormalize::computeRearranged(...) for BWMA

feedForward0->compute(...)
feedForward1->compute(...)

AddNormalize::compute(...)
```

The important point: in this path, activations are normal per-learner packed
integer buffers. Interleaved `AddNorm`, `Softmax`, and `matmulInterleaved*`
functions are not used.

### 4.4 Integer Grouped Path, Not Fully Interleaved

Used when learner count is `2` or `4` and `FULL_INTERLEAVED_PIPELINE_FLAG=0`.

Entry:

```text
TransformerBlock::computeGroup2(...)
  -> TransformerBlock::computeGroupImpl<2>(...)

TransformerBlock::computeGroup4(...)
  -> TransformerBlock::computeGroupImpl<4>(...)
```

Attention:

```text
for each head:
    SingleHeadSelfAttn::computeGroup2(...)
        -> SingleHeadSelfAttn::computeGroupImpl<2>(...)

    or

    SingleHeadSelfAttn::computeGroup4(...)
        -> SingleHeadSelfAttn::computeGroupImpl<4>(...)
```

For Q/K/V layers, grouped non-full interleaving tries to use grouped
CodebookDense kernels:

```text
tryComputeGroupedCodebookDense2(...)
  -> CodebookDense::computeInterleaved2DSameSeq(...)
      -> gemm_exec_compact_int_interleaved_2D_same_seq(...)
      -> or gemm_exec_compact_int_sve_interleaved_2D_same_seq(...)

tryComputeGroupedCodebookDense4(...)
  -> CodebookDense::computeInterleaved4DDiffSeq(...)
      -> gemm_exec_compact_int_interleaved_4D_same_seq(...)
      -> or gemm_exec_compact_int_interleaved_4D_diff_seq(...)
      -> or SVE versions when SIMD
```

If grouped CodebookDense is unsupported for a layer, the code falls back to
per-learner `layer->compute(...)`.

After Q/K/V projection, attention itself is still per learner:

```text
for each learner:
    Transpose::transpose(...)
    smmComputeRWMA(...) or simdComputeRWMA(...)
    Softmax::compute(...)
    smmComputeRWMA(...) or simdComputeRWMA(...)
    Softmax::post_softmax(...)
```

After attention heads:

```text
for each learner:
    copy/transpose multi-head output

condense:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner condense->compute(...)

for each learner:
    AddNormalize::compute(...)

ff0:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner feedForward0->compute(...)

ff1:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner feedForward1->compute(...)

for each learner:
    AddNormalize::compute(...)
```

The important point: this path can use interleaved CodebookDense GEMM kernels,
but it does not keep the whole Transformer block interleaved. It repeatedly
converts between separate packed learner buffers and temporary interleaved GEMM
buffers. `Softmax::computeInterleaved*`, `AddNormalize::computeInterleaved*`,
and `matmulInterleaved*ToInt8` are not used.

### 4.5 Integer Fully Interleaved Path

Used when:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
SIMD_FLAG=1
learner_count is 2 or 4
USE_FP32_TRANSFORMER_FLAG=0
```

Entry:

```text
TransformerBlock::computeGroup2(...)
  -> TransformerBlock::computeGroupImpl<2>(...)
      -> TransformerBlock::computeGroup2FullInterleaved(...)

TransformerBlock::computeGroup4(...)
  -> TransformerBlock::computeGroupImpl<4>(...)
      -> TransformerBlock::computeGroup4FullInterleaved(...)
```

Full 2D/4D interleaved layout:

```text
int8_t buffer[(seq * feature + col) * learner_count + learner]
```

That is logically:

```text
[seq][feature][learner]
```

Full int8 2D path:

```text
TransformerBlock::computeGroup2FullInterleaved(...)
  -> interleavePackedLearners2(...)

  for each head:
      SingleHeadSelfAttn::computeInterleaved2D(...)
          -> computeCodebookDenseInterleaved2D("q_hX", ...)
              -> CodebookDense::computeInterleaved2DToInt8(...)
                  -> gemm_exec_compact_int_sve_interleaved_2D_same_seq(...)
          -> computeCodebookDenseInterleaved2D("k_hX", ...)
          -> computeCodebookDenseInterleaved2D("v_hX", ...)
          -> matmulInterleaved2DToInt8(...)        // Q * K
              -> sve_gemm_dense_int8_interleaved_2D(...) when SIMD
          -> Softmax::computeInterleaved2D(...)
          -> transposeInterleavedRowsToCols2(...)  // V layout for matmul
          -> matmulInterleaved2DToInt8(...)        // softmax * V
          -> Softmax::post_softmax_interleaved2D(...)
      -> copyHeadToMultiheadInterleaved2D(...)

  -> computeCodebookDenseInterleaved2D("condense", ...)
      -> CodebookDense::computeInterleaved2DToInt8(...)

  -> AddNormalize::computeInterleaved2D(...)

  -> computeCodebookDenseInterleaved2D("ff0", ...)
  -> computeCodebookDenseInterleaved2D("ff1", ...)

  -> AddNormalize::computeInterleaved2D(...)
  -> packInterleavedLearners2(...)
```

Full int8 4D path:

```text
TransformerBlock::computeGroup4FullInterleaved(...)
  -> interleavePackedLearners4(...)

  for each head:
      SingleHeadSelfAttn::computeInterleaved4D(...)
          -> computeCodebookDenseInterleaved4D("q_hX", ...)
              -> CodebookDense::computeInterleaved4DToInt8(...)
                  -> gemm_exec_compact_int_sve_interleaved_4D_same_seq(...)
                  -> or gemm_exec_compact_int_sve_interleaved_4D_diff_seq(...)
          -> computeCodebookDenseInterleaved4D("k_hX", ...)
          -> computeCodebookDenseInterleaved4D("v_hX", ...)
          -> matmulInterleaved4DToInt8(...)        // Q * K
              -> sve_gemm_dense_int8_interleaved_4D(...) when SIMD
          -> Softmax::computeInterleaved4D(...)
          -> transposeInterleavedRowsToCols4(...)  // V layout for matmul
          -> matmulInterleaved4DToInt8(...)        // softmax * V
          -> Softmax::post_softmax_interleaved4D(...)
      -> copyHeadToMultiheadInterleaved4D(...)

  -> computeCodebookDenseInterleaved4D("condense", ...)
      -> CodebookDense::computeInterleaved4DToInt8(...)

  -> AddNormalize::computeInterleaved4D(...)

  -> computeCodebookDenseInterleaved4D("ff0", ...)
  -> computeCodebookDenseInterleaved4D("ff1", ...)

  -> AddNormalize::computeInterleaved4D(...)
  -> packInterleavedLearners4(...)
```

The important point: in the full interleaved path, the learner data is
interleaved once at the beginning of the block and packed back into per-learner
buffers only at the end. CodebookDense, attention matmul, softmax, AddNorm, and
feed-forward all operate on the shared interleaved layout.

### 4.6 FP32 Single and Grouped Paths

Used when:

```text
USE_FP32_TRANSFORMER_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
```

Top-level:

```text
TransformerFloat::run(...)
  -> load input_matrix from Full_NN/gemm_definitions/input_matrix.h
  -> create FloatTransformerBlock for each learner
  -> choose:
       FloatTransformerBlock::compute(...)
       FloatTransformerBlock::computeGroup2(...)
       FloatTransformerBlock::computeGroup4(...)
```

FP32 single path:

```text
FloatTransformerBlock::compute(...)
  for each head:
      FloatSingleHeadSelfAttn::compute(...)
          -> FloatCodebookDense::compute(...) for Q
              -> gemm_exec_compact(...)
              -> or gemm_exec_compact_sve(...) when SIMD
          -> FloatCodebookDense::compute(...) for K
          -> FloatCodebookDense::compute(...) for V
          -> matmulTransposedRhs(...)       // Q * K^T
          -> FloatSoftmax::compute(...)
          -> matmulRows(...)               // softmax * V
      -> copyHeadToMultihead(...)

  -> FloatCodebookDense::compute(...)       // condense
  -> FloatAddNormalize::compute(...)
  -> FloatCodebookDense::compute(...)       // ff0
  -> FloatCodebookDense::compute(...)       // ff1
  -> FloatAddNormalize::compute(...)
```

FP32 grouped but not fully interleaved:

```text
FloatTransformerBlock::computeGroup2/4(...)
  -> FloatTransformerBlock::computeGroupImpl<2/4>(...)
      -> if FULL_INTERLEAVED_PIPELINE:
             delegate to full interleaved path
         else:
             run FloatSingleHeadSelfAttn::computeGroup2/4(...)
                 -> currently loops over per-learner compute(...)
             run condense/ff0/ff1 per learner
             run FloatAddNormalize::compute(...) per learner
```

### 4.7 FP32 Fully Interleaved Path

Used when:

```text
USE_FP32_TRANSFORMER_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
FULL_INTERLEAVED_PIPELINE_FLAG=1
learner_count is 2 or 4
```

FP32 interleaved layout:

```text
float buffer[(row * col_count + col) * learner_count + learner]
```

That is also logically:

```text
[row][col][learner]
```

Call map:

```text
FloatTransformerBlock::computeGroup2/4(...)
  -> FloatTransformerBlock::computeFullInterleavedBlock<2/4>(...)
      -> interleaveLearnerMatrices(...)

      for each head:
          FloatSingleHeadSelfAttn::computeInterleaved2D/4D(...)
              -> FloatCodebookDense::computeInterleaved(...) for Q
                  -> gemm_exec_compact_fp32_interleaved_2D_same_seq(...)
                  -> or gemm_exec_compact_fp32_interleaved_4D_same_seq(...)
                  -> or gemm_exec_compact_fp32_interleaved_4D_diff_seq(...)
                  -> SVE versions when SIMD
              -> FloatCodebookDense::computeInterleaved(...) for K
              -> FloatCodebookDense::computeInterleaved(...) for V
              -> matmulInterleavedTransposedRhs(...)  // Q * K^T
              -> FloatSoftmax::computeInterleaved(...)
              -> matmulInterleavedRows(...)           // softmax * V
          -> copyHeadToMultiheadInterleaved(...)

      -> FloatCodebookDense::computeInterleaved(...)   // condense
      -> FloatAddNormalize::computeInterleaved(...)
      -> FloatCodebookDense::computeInterleaved(...)   // ff0
      -> FloatCodebookDense::computeInterleaved(...)   // ff1
      -> FloatAddNormalize::computeInterleaved(...)

      -> deinterleaveLearnerMatrices(...)
```

## 5. Fully Interleaved vs Not Fully Interleaved

### 5.1 Quick Comparison

| Area | Not fully interleaved | Fully interleaved |
| --- | --- | --- |
| Main activation layout | Separate per-learner buffers. Integer path uses packed `uint32_t` int8 values. FP32 path uses one `float` matrix per learner. | One shared `[seq or row][feature or col][learner]` buffer. Integer path uses `int8_t`; FP32 path uses `float`. |
| When interleaving happens | Only around selected grouped CodebookDense calls, if supported. | Once at the beginning of the block, then maintained through attention, projection, AddNorm, FFN, and final output. |
| Q/K/V projection | May use `CodebookDense::computeInterleaved2DSameSeq` or `computeInterleaved4DDiffSeq`, then returns to per-learner packed buffers. | Uses `CodebookDense::computeInterleaved2DToInt8` or `computeInterleaved4DToInt8` in int8, or `FloatCodebookDense::computeInterleaved` in FP32. |
| Attention QK matmul | Per learner: `smmComputeRWMA`, `simdComputeRWMA`, or BWMA variants. | Integer: `matmulInterleaved2DToInt8` or `matmulInterleaved4DToInt8`. FP32: `matmulInterleavedTransposedRhs`. |
| Softmax | Integer: `Softmax::compute` or `computeRearranged`. FP32: `FloatSoftmax::compute`. | Integer: `Softmax::computeInterleaved2D` or `computeInterleaved4D`. FP32: `FloatSoftmax::computeInterleaved`. |
| Attention output matmul | Per learner: `smmComputeRWMA`, `simdComputeRWMA`, or BWMA variants. | Integer: `matmulInterleaved2DToInt8` or `matmulInterleaved4DToInt8`. FP32: `matmulInterleavedRows`. |
| AddNorm | Integer: `AddNormalize::compute` or `computeRearranged`. FP32: `FloatAddNormalize::compute`. | Integer: `AddNormalize::computeInterleaved2D` or `computeInterleaved4D`. FP32: `FloatAddNormalize::computeInterleaved`. |
| Projection and FFN | May use grouped CodebookDense, but stores outputs per learner between stages. | Projection and FFN consume and produce interleaved buffers. |
| Final conversion | Not needed; outputs are already per learner. | Integer calls `packInterleavedLearners2/4`. FP32 calls `deinterleaveLearnerMatrices`. |

### 5.2 Functions Used Only by Integer Fully Interleaved Mode

These functions are part of the full int8 interleaved pipeline and are not used
by the normal grouped non-full path:

```text
TransformerBlock::computeGroup2FullInterleaved
TransformerBlock::computeGroup4FullInterleaved
SingleHeadSelfAttn::computeInterleaved2D
SingleHeadSelfAttn::computeInterleaved4D
computeCodebookDenseInterleaved2D
computeCodebookDenseInterleaved4D
CodebookDense::computeInterleaved2DToInt8
CodebookDense::computeInterleaved4DToInt8
interleavePackedLearners2
interleavePackedLearners4
packInterleavedLearners2
packInterleavedLearners4
transposeInterleavedRowsToCols2
transposeInterleavedRowsToCols4
matmulInterleaved2DToInt8
matmulInterleaved4DToInt8
copyHeadToMultiheadInterleaved2D
copyHeadToMultiheadInterleaved4D
Softmax::computeInterleaved2D
Softmax::computeInterleaved4D
Softmax::post_softmax_interleaved2D
Softmax::post_softmax_interleaved4D
AddNormalize::computeInterleaved2D
AddNormalize::computeInterleaved4D
```

### 5.3 Functions Used by Integer Grouped Non-Full Interleaved Mode

These functions are used when grouped learners are enabled but the full
interleaved pipeline is off:

```text
TransformerBlock::computeGroup2
TransformerBlock::computeGroup4
TransformerBlock::computeGroupImpl<2>
TransformerBlock::computeGroupImpl<4>
SingleHeadSelfAttn::computeGroup2
SingleHeadSelfAttn::computeGroup4
SingleHeadSelfAttn::computeGroupImpl<2>
SingleHeadSelfAttn::computeGroupImpl<4>
tryComputeGroupedCodebookDense2
tryComputeGroupedCodebookDense4
CodebookDense::computeInterleaved2DSameSeq
CodebookDense::computeInterleaved4DDiffSeq
gemm_exec_compact_int_interleaved_2D_same_seq
gemm_exec_compact_int_interleaved_4D_same_seq
gemm_exec_compact_int_interleaved_4D_diff_seq
gemm_exec_compact_int_sve_interleaved_2D_same_seq
gemm_exec_compact_int_sve_interleaved_4D_same_seq
gemm_exec_compact_int_sve_interleaved_4D_diff_seq
Softmax::compute
AddNormalize::compute
```

The grouped non-full path may use interleaved GEMM kernels, but it does not use
interleaved softmax, interleaved AddNorm, or interleaved attention matmul.

### 5.4 Functions Used by Non-Interleaved Single-Learner Integer Mode

```text
TransformerBlock::compute
TransformerBlock::computeBody
SingleHeadSelfAttn::compute
Dense::compute
CodebookDense::compute
CodebookDense::runCompactGemm
gemm_exec_compact_int
gemm_exec_compact_int_sve
Transpose::transpose
Transpose::multihead_transpose
smmComputeRWMA
simdComputeRWMA
Softmax::compute
Softmax::post_softmax
AddNormalize::compute
```

### 5.5 Functions Used by FP32 Fully Interleaved Mode

```text
TransformerFloat::run
FloatTransformerBlock::computeGroup2
FloatTransformerBlock::computeGroup4
FloatTransformerBlock::computeFullInterleavedBlock<2>
FloatTransformerBlock::computeFullInterleavedBlock<4>
FloatSingleHeadSelfAttn::computeInterleaved2D
FloatSingleHeadSelfAttn::computeInterleaved4D
FloatCodebookDense::computeInterleaved
gemm_exec_compact_fp32_interleaved_2D_same_seq
gemm_exec_compact_fp32_interleaved_4D_same_seq
gemm_exec_compact_fp32_interleaved_4D_diff_seq
gemm_exec_compact_sve_fp32_interleaved_2D_same_seq
gemm_exec_compact_sve_fp32_interleaved_4D_same_seq
gemm_exec_compact_sve_fp32_interleaved_4D_diff_seq
FloatSoftmax::computeInterleaved
FloatAddNormalize::computeInterleaved
interleaveLearnerMatrices
deinterleaveLearnerMatrices
```

## 6. Common Compile Commands

Run all commands from:

```bash
cd /home/thu/TiC-SAT
```

### 6.1 Dense No-SIMD Sequential Baseline

Use this to measure the scalar dense baseline over `N_LEARNERS` learners.

```bash
DENSE_NO_SIMD_BASELINE_FLAG=1 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

For gem5 region profiling:

```bash
DENSE_NO_SIMD_BASELINE_FLAG=1 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

### 6.2 Dense SIMD Baseline

Use this to compile the default dense path with SVE enabled.

```bash
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

### 6.3 Codebook GEMM, Grouped but Not Fully Interleaved

Use this for the normal codebook grouped path. This can use grouped interleaved
CodebookDense kernels but does not keep the full Transformer block interleaved.

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

### 6.4 Codebook GEMM, Fully Interleaved Int8 Pipeline

Use this for the full int8 interleaved Transformer pipeline.

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

Requirements:

```text
N_LEARNERS should be 2 or 4.
USE_CODEBOOK_GEMM_FLAG must be 1.
SIMD_FLAG must be 1 for the int8 full interleaved path.
```

### 6.5 Codebook Debug With Dense Reference

Use this to validate codebook output against dense reference output. This is a
debug/correctness mode, not a performance mode.

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=1 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

For full interleaved correctness debugging:

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=1 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

### 6.6 Codebook GEMM Profiling

Use this for cleaner profiling of codebook GEMM without dense reference or
debug-print overhead.

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

For full interleaved profiling:

```bash
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

### 6.7 FP32 Codebook Transformer

Use this to run the FP32 Transformer implementation with generated codebook
registry data.

```bash
USE_FP32_TRANSFORMER_FLAG=1 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

FP32 full interleaved:

```bash
USE_FP32_TRANSFORMER_FLAG=1 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

FP32 requirements:

```text
USE_CODEBOOK_GEMM_FLAG must be 1.
ENABLE_CODEBOOK_REFERENCE_FLAG must be 0.
The generated registry must include the requested learners and FP32-compatible data.
```

## 7. Running Under QEMU SVE

After compiling `transformer.o`, QEMU SVE vector length can be selected with
`sve-default-vector-length`.

Examples from the compile script comments:

```bash
# 128-bit SVE
-cpu max,sve=on,sve-default-vector-length=16

# 256-bit SVE
-cpu max,sve=on,sve-default-vector-length=32

# 512-bit SVE
-cpu max,sve=on,sve-default-vector-length=64
```

Make sure the QEMU vector length, generated `N_SVE_*` constants, and experiment
expectations are consistent.

## 8. Practical Experiment Checklist

Before compiling:

1. Confirm the active dimensions in `transformer.h`.
2. Regenerate notebook artifacts if model dimensions, learner count, codebook
   size, same-sequence mode, or FP32 support changed.
3. Confirm `Full_NN/gemm_definitions/generated_codebook_registry.h` contains
   all required layer names:

```text
q_h0/k_h0/v_h0 ... q_hN/k_hN/v_hN
condense
ff0
ff1
```

4. Confirm `Full_NN/gemm_definitions/codebooks_def.h` has the intended
   `N_LEARNERS`.
5. For full int8 interleaving, use `N_LEARNERS=2` or `N_LEARNERS=4`,
   `USE_CODEBOOK_GEMM_FLAG=1`, and `SIMD_FLAG=1`.
6. For FP32 Transformer, use `USE_FP32_TRANSFORMER_FLAG=1`,
   `USE_CODEBOOK_GEMM_FLAG=1`, and `ENABLE_CODEBOOK_REFERENCE_FLAG=0`.
7. For performance profiling, keep `ENABLE_DEBUG_PRINT_FLAG=0` and
   `ENABLE_CODEBOOK_REFERENCE_FLAG=0`.

## 9. Troubleshooting

### Missing generated registry layer

If `LayerFactory` or `FloatCodebookDense` reports that a layer is missing, rerun
`Full_NN/generators/Transformer_generator.ipynb` and confirm that the generated
registry includes all Q/K/V heads plus `condense`, `ff0`, and `ff1`.

### Shape mismatch

If a generated layer shape does not match the C++ Transformer dimensions, check:

```text
transformer.h
Full_NN/gemm_definitions/input_matrix.h
Full_NN/gemm_definitions/generated_codebook_registry.h
Full_NN/gemm_definitions/codebooks_def.h
```

The generator and C++ code must agree on `D_SEQ`, `D_MODEL`, `D_Q`, `NUM_HEAD`,
and `D_FF`.

### Full interleaved compile failure

For int8 full interleaving, the expected flags are:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
SIMD_FLAG=1
USE_FP32_TRANSFORMER_FLAG=0
```

For FP32 full interleaving, the expected flags are:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
USE_FP32_TRANSFORMER_FLAG=1
ENABLE_CODEBOOK_REFERENCE_FLAG=0
```

### Profiling includes too much overhead

Use:

```text
ENABLE_CODEBOOK_REFERENCE_FLAG=0
ENABLE_DEBUG_PRINT_FLAG=0
PROFILE_GEMM_ONLY_FLAG=1
```

`run_mode_config.h` also automatically disables debug print and reference paths
when profiling modes are enabled.

### Notebook weights are not found

Check that this directory exists and contains per-learner files:

```text
weights/generated_from_notebook/learner0/
weights/generated_from_notebook/learner1/
weights/generated_from_notebook/learner2/
weights/generated_from_notebook/learner3/
```

The number of learner directories should match the generated learner count used
by the experiment.
