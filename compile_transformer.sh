# aarch64-linux-gnu-g++ transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc   -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2
# aarch64-linux-gnu-g++ -static transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2

unset CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CPPFLAGS LDFLAGS


DEFAULT_A64CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
if [ -z "${A64CXX:-}" ]; then
  if [ -x "$DEFAULT_A64CXX" ]; then
    A64CXX="$DEFAULT_A64CXX"
  elif command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    A64CXX="$(command -v aarch64-linux-gnu-g++)"
  elif command -v aarch64-conda-linux-gnu-g++ >/dev/null 2>&1; then
    A64CXX="$(command -v aarch64-conda-linux-gnu-g++)"
  else
    echo "No aarch64 C++ compiler found. Set A64CXX to your cross compiler path." >&2
    exit 1
  fi
fi
export A64CXX
export A64SYSROOT="$($A64CXX -print-sysroot 2>/dev/null || true)"


EXTRA_DEFS=""
EXTRA_CXXFLAGS=""
GEMM_SVE_SRC=""

if [ "${RELOAD_WEIGHT_FLAG:-1}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DRELOAD_WEIGHT"
fi

if [ "${USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG:-1}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DUSE_NOTEBOOK_GENERATED_WEIGHTS"
fi

if [ "${USE_CODEBOOK_GEMM_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DUSE_CODEBOOK_GEMM"
fi

if [ "${ENABLE_CODEBOOK_REFERENCE_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DENABLE_CODEBOOK_REFERENCE"
fi

if [ "${ENABLE_DEBUG_PRINT_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DENABLE_DEBUG_PRINT"
fi

if [ "${PROFILE_GEMM_ONLY_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DPROFILE_GEMM_ONLY"
fi

if [ "${GEM5_PROFILE_REGIONS_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DGEM5_PROFILE_REGIONS"
fi

if [ "${FULL_INTERLEAVED_PIPELINE_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DFULL_INTERLEAVED_PIPELINE"
fi

if [ "${SIMD_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DSIMD"
  EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -march=armv8-a+sve"
  GEMM_SVE_SRC="Full_NN/src/gemm_SVE.c"
fi

echo "Compile options:"
echo "  RELOAD_WEIGHT_FLAG=${RELOAD_WEIGHT_FLAG:-1}"
echo "  USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=${USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG:-1}"
echo "  USE_CODEBOOK_GEMM_FLAG=${USE_CODEBOOK_GEMM_FLAG:-0}"
echo "  ENABLE_CODEBOOK_REFERENCE_FLAG=${ENABLE_CODEBOOK_REFERENCE_FLAG:-0}"
echo "  ENABLE_DEBUG_PRINT_FLAG=${ENABLE_DEBUG_PRINT_FLAG:-0}"
echo "  PROFILE_GEMM_ONLY_FLAG=${PROFILE_GEMM_ONLY_FLAG:-0}"
echo "  GEM5_PROFILE_REGIONS_FLAG=${GEM5_PROFILE_REGIONS_FLAG:-0}"
echo "  FULL_INTERLEAVED_PIPELINE_FLAG=${FULL_INTERLEAVED_PIPELINE_FLAG:-0}"
echo "  SIMD_FLAG=${SIMD_FLAG:-0}"

# Symbolic link to required library
# mkdir -p "$A64SYSROOT/lib"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgomp.so.1" \
#        "$A64SYSROOT/lib/libgomp.so.1"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libstdc++.so.6" \
#        "$A64SYSROOT/lib/libstdc++.so.6"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgcc_s.so.1" \
#        "$A64SYSROOT/lib/libgcc_s.so.1"

"$A64CXX" -std=c++17 -O2 -Wall \
  $EXTRA_CXXFLAGS \
  transformer.cpp \
  transformer_layers/*.cc \
  Full_NN/src/gemm_exec.c \
  $GEMM_SVE_SRC \
  accelerator/smm_gem.cpp \
  accelerator/systolic_m2m.cc \
  $EXTRA_DEFS \
  -I. \
  -IFull_NN/inc \
  -IFull_NN/gemm_definitions \
  -DSA_SIZE=16 \
  -DDEVELOP \
  -DCORE_NUM=1 \
  -fopenmp \
  -static \
  -o transformer.o \
  -Wl,--start-group \
  -lgomp \
  -ldl \
  -Wl,--end-group \
  -pthread

# conda activate gem5_env
# source compile_transformer.sh 
# USE_CODEBOOK=1 source compile_transformer.sh

#   -DDEBUG_SMALL_MODEL \
#   -DRELOAD_WEIGHT \
#   -DUSE_NOTEBOOK_GENERATED_WEIGHTS \
#   -DUSE_CODEBOOK_GEMM \


# New weights
# RELOAD_WEIGHT_FLAG=0 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 source compile_transformer.sh

# Old weights, original
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 source compile_transformer.sh

# Old weights, notebook generated
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=0 source compile_transformer.sh

# Old weights, codebooked GEMM
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 source compile_transformer.sh



# Run config v2.0
# New weights, default dense
# RELOAD_WEIGHT_FLAG=0 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, original, default dense
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, notebook generated, default dense
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, codebooked GEMM, reference mode
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=1 ENABLE_DEBUG_PRINT_FLAG=1 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Only codebook GEMM, clean mode
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Profiling mode
# RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1 source compile_transformer.sh

# Run config v3.0 with SIMD
# New weights, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=0 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, original, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, notebook generated, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, codebooked GEMM, reference mode, SIMD enabled (if in codebooked GEMM, automatically choose multiple learner implementation)
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=1 ENABLE_DEBUG_PRINT_FLAG=1 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Only codebook GEMM, clean mode, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Profiling mode, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1 source compile_transformer.sh

# Run config v4.0 with SIMD
# New weights, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=0 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, original, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, notebook generated, default dense, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=0 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, codebooked GEMM, reference mode, SIMD enabled (Used for debugging)
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=1 ENABLE_DEBUG_PRINT_FLAG=1 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh
# FULL_INTERLEAVED_PIPELINE_FLAG=1 SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=1 ENABLE_DEBUG_PRINT_FLAG=1 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh
# FULL_INTERLEAVED_PIPELINE_FLAG=0 SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=1 ENABLE_DEBUG_PRINT_FLAG=1 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Old weights, codebooked GEMM (no reference), SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Only codebook GEMM, clean mode, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=0 source compile_transformer.sh

# Profiling mode, CodebookDense registry only, SIMD enabled
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=0 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1 source compile_transformer.sh
# SIMD_FLAG=1 RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1 source compile_transformer.sh

# Under profiling mode: USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG only affects the input tensor
# Under other modes: USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG affects the input tensor + other weights

# RELOAD_WEIGHT_FLAG: generate new .bin weigths by the transformer c code
# USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG: use .bin weights generated by the notebook or not
# USE_CODEBOOK_GEMM_FLAG: use default dense or codebook dense (enable USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG first to use codebook computation)
# ENABLE_CODEBOOK_REFERENCE_FLAG: parallel load .bin and compute default dense as reference (only effective when USE_CODEBOOK_GEMM_FLAG = 1)

# Run this command to run Transformer with aarch64 on eslsrv12
# qemu-aarch64 -L "$A64SYSROOT" ./transformer.o 
# cp ~/TiC-SAT-Jerry/transformer.o /home/jerry/gem5/shared_folder/
# cp ~/TiC-SAT/transformer.o ~/gem5/shared_folder/



# conda activate gem5_env
# SYSROOT=$(aarch64-conda-linux-gnu-g++ -print-sysroot)

# $HOME/opt/qemu-sve/bin/qemu-aarch64 \
#   -cpu max,sve=on,sve-default-vector-length=16 \
#   -L "$SYSROOT" \
#   /tmp/test_single_layer_SVE_aarch64 q_h0 2 \
#   < /tmp/test_single_layer_SVE_aarch64


# QEMU SVE
# SYSROOT=$(/home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++ -print-sysroot)

# HOME/thu/opt/qemu-sve/bin/qemu-aarch64 \
#   -cpu max,sve=on,sve-default-vector-length=16 \
#   -L "$SYSROOT" \
#   ./transformer.o \
#   < ./transformer.o

# /home/thu/opt/qemu-sve/bin/qemu-aarch64 -cpu max,sve=on,sve-default-vector-length=16 -L "$SYSROOT" ./transformer.o < ./transformer.o


# # 128-bit
# -cpu max,sve=on,sve-default-vector-length=16

# # 256-bit
# -cpu max,sve=on,sve-default-vector-length=32

# # 512-bit
# -cpu max,sve=on,sve-default-vector-length=64
