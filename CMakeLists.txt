cmake_minimum_required(VERSION 3.20)
project(FvllMontiTransformer)

set(CMAKE_CXX_STANDARD 17)
option(USE_CODEBOOK "Enable codebook-backed FFN layers" OFF)

set(GCC_COVERAGE_COMPILE_FLAGS "-DDEVELOP -DREARRANGE -DSA -DSA_SIZE=16 -DRELOAD_WEIGHT -DCORE_NUM=4 -fopenmp -O2")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB transformer_SRC
        "transformer_layers/*.h"
        "transformer_layers/*.cc"
        "Full_NN/src/gemm_exec.c"
        "accelerator/smm_gem.cpp"
        "accelerator/systolic_m2m.cc"
        )

add_executable(FvllMontiTransformer transformer.cpp ${transformer_SRC})
target_include_directories(FvllMontiTransformer PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Full_NN/inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/Full_NN/gemm_definitions")

if (USE_CODEBOOK)
    target_compile_definitions(FvllMontiTransformer PRIVATE USE_CODEBOOK)
endif()
