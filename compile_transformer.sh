# aarch64-linux-gnu-g++ transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc   -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2
# aarch64-linux-gnu-g++ -static transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2

unset CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CPPFLAGS LDFLAGS

export A64CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
export A64SYSROOT="$($A64CXX -print-sysroot)"

EXTRA_DEFS=""
if [ "${USE_CODEBOOK:-0}" = "1" ]; then
  EXTRA_DEFS="$EXTRA_DEFS -DUSE_CODEBOOK"
  echo "Compiling with USE_CODEBOOK enabled"
else
  echo "Compiling with notebook/generated FFN bin loading enabled"
fi

# Symbolic link to required library
# mkdir -p "$A64SYSROOT/lib"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgomp.so.1" \
#        "$A64SYSROOT/lib/libgomp.so.1"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libstdc++.so.6" \
#        "$A64SYSROOT/lib/libstdc++.so.6"

# ln -sf "$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/13.4.0/libgcc_s.so.1" \
#        "$A64SYSROOT/lib/libgcc_s.so.1"

"$A64CXX" -O2 -Wall \
  transformer.cpp \
  transformer_layers/*.cc \
  accelerator/smm_gem.cpp \
  accelerator/systolic_m2m.cc \
  -DSA_SIZE=4 \
  -DDEVELOP \
  -DRELOAD_WEIGHT \
  -DDEBUG_SMALL_MODEL \
  -DCORE_NUM=1 \
  $EXTRA_DEFS \
  -IFull_NN/gemm_definitions \
  -fopenmp \
  -o transformer.o
  # -DRELOAD_WEIGHT \

# conda activate gem5_env
# source compile_transformer.sh 
# qemu-aarch64 -L "$A64SYSROOT" ./transformer.o # Run this command to run Transformer with aarch64 on eslsrv12
# cp ~/TiC-SAT-Jerry/transformer.o /home/jerry/gem5/shared_folder/



