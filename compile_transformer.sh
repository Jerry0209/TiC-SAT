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

if [ "${RELOAD_WEIGHT_FLAG:-1}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DRELOAD_WEIGHT"
fi

if [ "${USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG:-1}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DUSE_NOTEBOOK_GENERATED_WEIGHTS"
fi

if [ "${USE_CODEBOOK_GEMM_FLAG:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DUSE_CODEBOOK_GEMM"
fi

echo "Compile options:"
echo "  RELOAD_WEIGHT_FLAG=${RELOAD_WEIGHT_FLAG:-1}"
echo "  USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=${USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG:-1}"
echo "  USE_CODEBOOK_GEMM_FLAG=${USE_CODEBOOK_GEMM_FLAG:-0}"

# Symbolic link to required library
# mkdir -p "$A64SYSROOT/lib"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgomp.so.1" \
#        "$A64SYSROOT/lib/libgomp.so.1"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libstdc++.so.6" \
#        "$A64SYSROOT/lib/libstdc++.so.6"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgcc_s.so.1" \
#        "$A64SYSROOT/lib/libgcc_s.so.1"

"$A64CXX" -std=c++17 -O2 -Wall \
  transformer.cpp \
  transformer_layers/*.cc \
  Full_NN/src/gemm_exec.c \
  accelerator/smm_gem.cpp \
  accelerator/systolic_m2m.cc \
  $EXTRA_DEFS \
  -I. \
  -IFull_NN/inc \
  -IFull_NN/gemm_definitions \
  -DSA_SIZE=4 \
  -DDEVELOP \
  -DCORE_NUM=1 \
  -fopenmp \
  -o transformer.o

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



# qemu-aarch64 -L "$A64SYSROOT" ./transformer.o # Run this command to run Transformer with aarch64 on eslsrv12
# cp ~/TiC-SAT-Jerry/transformer.o /home/jerry/gem5/shared_folder/

