#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "accelerator/smm_gem.h"
#include <algorithm>
#include <fstream>
#include <cstring>

#include "transformer_layers/run_mode_config.h"

// #ifndef RELOAD_WEIGHT
#include <filesystem>

#include "transformer_layers/debuggerFunctions.h"

#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/4)


void fill_kernel(uint32_t *kernel, int kernel_size) {
    for (int i = 0; i < kernel_size; i++) {
        uint32_t result = 0;
        for (int j = 0; j < 4; j++) {
            result |= ((uint8_t) (rand() % 5 - 2)) << (8 * j);
        }
        kernel[i] = result;
    }
}

void fill_weight(uint32_t *kernel, int n_row, int n_col) { // ？
    uint32_t *kernel_ptr = kernel;
    for (int i = 0; i < n_row / KERNEL_DIM; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            for (int ii = 0; ii < KERNEL_DIM; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t result = 0;
                    for (int k = 0; k < 4; k++) {
                        result |= ((uint8_t) (rand() % 5 - 2)) << (8 * k);
                    }
                    *kernel_ptr = result;
                    kernel_ptr++;
                }
            }
        }
    }
}

void saveWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    // Write the kernel array to file
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
                           std::to_string(qkv) + ".bin";
    std::ofstream fout(filename);
    if (fout.is_open()) {
        for (int i = 0; i < size; i++) {
            fout << array[i] << " ";
        }
        fout.close();
    }
}


bool tryLoadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        return false;
    }

    for (int i = 0; i < size; i++) {
        if (!(fin >> array[i])) {
            std::cout << "Weight file size mismatch: " << filename
                      << " (expected " << size << " words)" << std::endl;
            fin.close();
            return false;
        }
    }

    uint32_t extra;
    if (fin >> extra) {
        std::cout << "Weight file has extra data: " << filename
                  << " (expected exactly " << size << " words)" << std::endl;
        fin.close();
        return false;
    }

    fin.close();
    return true;
}

void loadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    if (!tryLoadWeight(n_head, qkv, size, array, dir_name)) {
        std::cout << filename + " Not loaded" << std::endl;
    }
}


void test() {
    std::cout << "Welcome to TiC-SAT" << std::endl;
    
#ifdef BWMA
    std::cout << "BWMA method" << std::endl;
#else
    std::cout<<"RWMA method" << std::endl;
#endif
    std::cout << "SA_SIZE = " << SA_SIZE << std::endl;
    std::cout << "KERNEL_DIM = " << KERNEL_DIM << std::endl;
    std::cout << "MAX_COL = " << MAX_COL << std::endl;
    std::cout << "CFG_RELOAD_WEIGHT = " << CFG_RELOAD_WEIGHT << std::endl;
    std::cout << "CFG_USE_NOTEBOOK_GENERATED_WEIGHTS = " << CFG_USE_NOTEBOOK_GENERATED_WEIGHTS << std::endl;
    std::cout << "CFG_USE_CODEBOOK_GEMM = " << CFG_USE_CODEBOOK_GEMM << std::endl;
    std::cout << "CFG_USE_CODEBOOK_REFERENCE = " << CFG_USE_CODEBOOK_REFERENCE << std::endl;
    std::cout << "CFG_ENABLE_DEBUG_PRINT = " << CFG_ENABLE_DEBUG_PRINT << std::endl;
    std::cout << "CFG_PROFILE_GEMM_ONLY = " << CFG_PROFILE_GEMM_ONLY << std::endl;

    // Prefer the host-side project path and fall back to the 9p mount in gem5.
    std::string dir_name = "/home/thu/TiC-SAT/weights";
    if (!std::filesystem::exists(dir_name)) {
        dir_name = "/mnt/weights";
    }
    std::string notebook_weights_dir = dir_name + "/generated_from_notebook";
#if !CFG_RELOAD_WEIGHT
    std::filesystem::create_directories(dir_name);
#endif


    uint32_t *tensor_in = new uint32_t[D_SEQ * D_MODEL >> 2];
#if CFG_RELOAD_WEIGHT
    // Load the tensor input from file
    // We assign -1 and -1 to n_head and qkv to indicate that we are not loading the weight
#if CFG_ENABLE_DEBUG_PRINT
    printf("Input matrix: loading weights of input matrix\n");
#endif

    loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, dir_name);
#else
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    // Save the tensor input to file
    // We assign -1 and -1 to n_head and qkv to indicate that we are not saving the weight
    saveWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, dir_name);
#endif

#ifndef BWMA
    uint32_t tensorInRowWise[D_SEQ * D_MODEL >> 2];
    // By default, the saved tensor is in block-wise format
    // We need to convert it to row-wise format
    blockWise2RowWise(tensor_in, tensorInRowWise, D_SEQ, D_MODEL >> 2);
    tensor_in = tensorInRowWise;
#endif

    uint32_t *out = new uint32_t[D_SEQ * D_MODEL >> 2]();
    uint32_t *weightVec[3 * NUM_HEAD + 3]; // Array of pointers: each head has Q, K, V matrices
    int head_qkv_size = D_Q * D_MODEL >> 2;

    for (int n = 0; n < NUM_HEAD; n++) {
        volatile auto query_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
        volatile auto key_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
        volatile auto value_kernel = new uint32_t[D_Q * D_MODEL >> 2]();

    // for (int n = 0; n < NUM_HEAD; n++) {
    //     uint32_t* query_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
    //     uint32_t* key_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
    //     uint32_t* value_kernel = new uint32_t[D_Q * D_MODEL >> 2]();

        bool q_loaded_from_notebook = false;
        bool k_loaded_from_notebook = false;
        bool v_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
        printf("Head %d : trying notebook-generated Q/K/V weights\n", n);
#endif

        q_loaded_from_notebook = tryLoadWeight(
            n, 0, head_qkv_size, query_kernel, notebook_weights_dir);

        k_loaded_from_notebook = tryLoadWeight(
            n, 1, head_qkv_size, key_kernel, notebook_weights_dir);

        v_loaded_from_notebook = tryLoadWeight(
            n, 2, head_qkv_size, value_kernel, notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        if (q_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated Q weights for head "
                    << n << " from " << notebook_weights_dir << std::endl;
        }
        if (k_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated K weights for head "
                    << n << " from " << notebook_weights_dir << std::endl;
        }
        if (v_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated V weights for head "
                    << n << " from " << notebook_weights_dir << std::endl;
        }
#endif
#endif

#if CFG_RELOAD_WEIGHT
#if CFG_ENABLE_DEBUG_PRINT
        printf("Head %d : Q, K, V weight matrix: loading weights of Q, K, V kernels\n", n);
#endif
        if (!q_loaded_from_notebook) {
            loadWeight(n, 0, head_qkv_size, query_kernel, dir_name);
        }
        if (!k_loaded_from_notebook) {
            loadWeight(n, 1, head_qkv_size, key_kernel, dir_name);
        }
        if (!v_loaded_from_notebook) {
            loadWeight(n, 2, head_qkv_size, value_kernel, dir_name);
        }
#else
        fill_weight(query_kernel, D_MODEL, D_Q >> 2);
        fill_weight(key_kernel, D_MODEL, D_Q >> 2);
        fill_weight(value_kernel, D_MODEL, D_Q >> 2);

        saveWeight(n, 0, head_qkv_size, query_kernel, dir_name);
        saveWeight(n, 1, head_qkv_size, key_kernel, dir_name);
        saveWeight(n, 2, head_qkv_size, value_kernel, dir_name);
#endif

#ifndef BWMA
        // By default, the saved weights are in block-wise format
        // We need to convert them to row-wise format
        // Dense and reference paths expect row-wise layout in RWMA mode
        uint32_t* queryRowWise = new uint32_t [D_MODEL * D_Q >> 2];
        blockWise2RowWise(query_kernel, queryRowWise, D_MODEL, D_Q >> 2);
        query_kernel = queryRowWise;
        uint32_t* keyRowWise = new uint32_t [D_MODEL * D_Q >> 2];
        blockWise2RowWise(key_kernel, keyRowWise, D_MODEL, D_Q >> 2);
        key_kernel = keyRowWise;
        uint32_t* valueRowWise = new uint32_t [D_MODEL * D_Q >> 2];
        blockWise2RowWise(value_kernel, valueRowWise, D_MODEL, D_Q >> 2);
        value_kernel = valueRowWise;
#endif

        weightVec[n * 3] = query_kernel;
        weightVec[n * 3 + 1] = key_kernel;
        weightVec[n * 3 + 2] = value_kernel;
    }

    volatile auto condense_kernel = new uint32_t[NUM_HEAD * D_Q * D_MODEL >> 2]();
    volatile auto ff0_kernel = new uint32_t[D_MODEL * D_FF >> 2]();
    volatile auto ff1_kernel = new uint32_t[D_FF * D_MODEL >> 2]();


    // Load .bin weights generated from the notebook
    int n = -1; // n=-1 means that we are not saving/loading a head
    bool condense_loaded_from_notebook = false;
    bool ff0_loaded_from_notebook = false;
    bool ff1_loaded_from_notebook = false;

    #if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
    #if CFG_ENABLE_DEBUG_PRINT
        printf("Condense/projection layer: trying notebook-generated condense weights\n");
    #endif
        condense_loaded_from_notebook = tryLoadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, notebook_weights_dir);

    #if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 0: trying notebook-generated FF0 weights\n");
    #endif
        ff0_loaded_from_notebook = tryLoadWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, notebook_weights_dir);

    #if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 1: trying notebook-generated FF1 weights\n");
    #endif
        ff1_loaded_from_notebook = tryLoadWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, notebook_weights_dir);

    #if CFG_ENABLE_DEBUG_PRINT
        if (condense_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated condense weights from " << notebook_weights_dir << std::endl;
        }
        if (ff0_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF0 weights from " << notebook_weights_dir << std::endl;
        }
        if (ff1_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF1 weights from " << notebook_weights_dir << std::endl;
        }
    #endif
    #endif

#if CFG_RELOAD_WEIGHT

    if (!condense_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
        printf("Condense/projection layer: loading original condense weights\n");
#endif
        loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, dir_name);
    }

    if (!ff0_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 0: loading original FF0 weights\n");
#endif
        loadWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, dir_name);
    }

    if (!ff1_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 1: loading original FF1 weights\n");
#endif
        loadWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, dir_name);
    }

#else

    // Fresh run after dimension change: generate everything from C code
    // fill_weight(condense_kernel, D_MODEL, NUM_HEAD * D_Q >> 2); // This is the original statement. Probably the dimension is incorrect, requiring further checking
    fill_weight(condense_kernel, NUM_HEAD * D_Q, D_MODEL >> 2); // Modified by Jerry, suggested by ChatGPT
    fill_weight(ff0_kernel, D_MODEL, D_FF >> 2);
    fill_weight(ff1_kernel, D_FF, D_MODEL >> 2);

    saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, dir_name);
    saveWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, dir_name);
    saveWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, dir_name);

#endif

#ifndef BWMA
    // By default, the saved weights are in block-wise format
    // We need to convert them to row-wise format
    // printf("blockWise2RowWise: condense\n");
    uint32_t* condenseRowWise = new uint32_t [NUM_HEAD * D_Q * D_MODEL >> 2];
    blockWise2RowWise(condense_kernel, condenseRowWise, NUM_HEAD * D_Q, D_MODEL >> 2);
    condense_kernel = condenseRowWise;

    // printf("blockWise2RowWise: ff0\n");
    uint32_t* ff0RowWise = new uint32_t [D_MODEL * D_FF >> 2];
    blockWise2RowWise(ff0_kernel, ff0RowWise, D_MODEL, D_FF >> 2);
    ff0_kernel = ff0RowWise;

    // printf("blockWise2RowWise: ff1\n");
    uint32_t* ff1RowWise = new uint32_t [D_FF * D_MODEL >> 2];
    blockWise2RowWise(ff1_kernel, ff1RowWise, D_FF, D_MODEL >> 2);
    ff1_kernel = ff1RowWise;

#endif

    weightVec[NUM_HEAD * 3] = condense_kernel;
    weightVec[NUM_HEAD * 3 + 1] = ff0_kernel;
    weightVec[NUM_HEAD * 3 + 2] = ff1_kernel;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec, KERNEL_DIM, MAX_COL);
    selfatten.compute(D_SEQ, tensor_in, out); // Compute loop from here

}

int main() {
    test();
    return 0;
}