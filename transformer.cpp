#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "accelerator/smm_gem.h"
#include <algorithm>
#include <fstream>
#include <cstring>
#include <vector>

#include "transformer_layers/run_mode_config.h"
#include <codebooks_def.h>
#if CFG_USE_CODEBOOK_GEMM
#include "transformer_layers/registry_adapter.h"
#endif

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

std::size_t getTransformerLearnerCount() {
#if CFG_USE_CODEBOOK_GEMM
    try {
        return getCodebookDenseLearnerCount("q_h0");
    } catch (const std::exception&) {
        return 1;
    }
#else
    return 1;
#endif
}

std::string getNotebookWeightsDirForLearner(const std::string& notebook_weights_dir,
                                            std::size_t learner_idx) {
    return notebook_weights_dir + "/learner" + std::to_string(learner_idx);
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
    std::cout << "SVE_LENGTH = " << (N_SVE_BYTE * 8) << " bits" << std::endl;

    std::cout << "D_Q = " << D_Q << std::endl;
    std::cout << "D_SEQ = " << D_SEQ << std::endl;
    std::cout << "D_MODEL = " << D_MODEL << std::endl;
    std::cout << "NUM_HEAD = " << NUM_HEAD << std::endl;
    std::cout << "D_FF = " << D_FF << std::endl;

    std::cout << "N_LEARNERS = " << N_LEARNERS << std::endl;
    std::cout << "CODEBOOK_SIZE = " << CB_SIZE << std::endl;
    std::cout << "TILE_SIZE = " << TILE_SIZE << std::endl;
    std::cout << "USE_F32 = " << USE_F32 << std::endl;

    std::cout << "CFG_RELOAD_WEIGHT = " << CFG_RELOAD_WEIGHT << std::endl;
    std::cout << "CFG_USE_NOTEBOOK_GENERATED_WEIGHTS = " << CFG_USE_NOTEBOOK_GENERATED_WEIGHTS << std::endl;
    std::cout << "CFG_USE_CODEBOOK_GEMM = " << CFG_USE_CODEBOOK_GEMM << std::endl;
    std::cout << "CFG_USE_CODEBOOK_REFERENCE = " << CFG_USE_CODEBOOK_REFERENCE << std::endl;
    std::cout << "CFG_ENABLE_DEBUG_PRINT = " << CFG_ENABLE_DEBUG_PRINT << std::endl;
    std::cout << "CFG_PROFILE_GEMM_ONLY = " << CFG_PROFILE_GEMM_ONLY << std::endl;
    std::cout << "CFG_SIMD = " << CFG_SIMD << std::endl;
    
    
// #ifdef USE_F32
//     std::cout << "USE_F32 = " << USE_F32 << std::endl;
// #else
//     std::cout << "USE_F32 = 0" << std::endl;
// #endif

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
    bool input_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
    printf("Input matrix: trying notebook-generated input\n");
#endif
    input_loaded_from_notebook = tryLoadWeight(
        -1, -1, D_SEQ * D_MODEL >> 2, tensor_in, notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
    if (input_loaded_from_notebook) {
        std::cout << "Loaded notebook-generated input matrix from "
                  << notebook_weights_dir << std::endl;
    }
#endif
#endif

#if CFG_ENABLE_DEBUG_PRINT
    if (!input_loaded_from_notebook) {
        printf("Input matrix: loading weights of input matrix\n");
    }
#endif
    if (!input_loaded_from_notebook) {
        loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, dir_name);
    }
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

    const std::size_t learner_count = getTransformerLearnerCount();
    std::cout << "CODEBOOK_REGISTRY_N_LEARNERS = " << learner_count << std::endl;

    const std::string multiple_learner_output_root = dir_name + "/multiple_learner_outputs/c";
    if (learner_count > 1) {
        std::filesystem::create_directories(multiple_learner_output_root);
    }

    auto buildTransformerBlockForLearner =
        [&](std::size_t learner_idx,
            const std::string& learner_notebook_weights_dir,
            const std::string& learner_dump_dir) -> TransformerBlock* {
        std::vector<uint32_t*> weightVec(3 * NUM_HEAD + 3, nullptr);
#if !CFG_CODEBOOK_ONLY_MODE
        const int head_qkv_size = D_Q * D_MODEL >> 2;
#endif

        for (int n = 0; n < NUM_HEAD; n++) {
            uint32_t* query_kernel = nullptr;
            uint32_t* key_kernel = nullptr;
            uint32_t* value_kernel = nullptr;

#if !CFG_CODEBOOK_ONLY_MODE
            query_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
            key_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
            value_kernel = new uint32_t[D_Q * D_MODEL >> 2]();

            bool q_loaded_from_notebook = false;
            bool k_loaded_from_notebook = false;
            bool v_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
            printf("Head %d : trying notebook-generated Q/K/V weights\n", n);
#endif

            q_loaded_from_notebook = tryLoadWeight(
                n, 0, head_qkv_size, query_kernel, learner_notebook_weights_dir);

            k_loaded_from_notebook = tryLoadWeight(
                n, 1, head_qkv_size, key_kernel, learner_notebook_weights_dir);

            v_loaded_from_notebook = tryLoadWeight(
                n, 2, head_qkv_size, value_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
            if (q_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated Q weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
            }
            if (k_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated K weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
            }
            if (v_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated V weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
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
            uint32_t* queryRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(query_kernel, queryRowWise, D_MODEL, D_Q >> 2);
            query_kernel = queryRowWise;

            uint32_t* keyRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(key_kernel, keyRowWise, D_MODEL, D_Q >> 2);
            key_kernel = keyRowWise;

            uint32_t* valueRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(value_kernel, valueRowWise, D_MODEL, D_Q >> 2);
            value_kernel = valueRowWise;
#endif
#endif

            weightVec[n * 3] = query_kernel;
            weightVec[n * 3 + 1] = key_kernel;
            weightVec[n * 3 + 2] = value_kernel;
        }

        uint32_t* condense_kernel = nullptr;
        uint32_t* ff0_kernel = nullptr;
        uint32_t* ff1_kernel = nullptr;

#if !CFG_CODEBOOK_ONLY_MODE
        condense_kernel = new uint32_t[NUM_HEAD * D_Q * D_MODEL >> 2]();
        ff0_kernel = new uint32_t[D_MODEL * D_FF >> 2]();
        ff1_kernel = new uint32_t[D_FF * D_MODEL >> 2]();

        int n = -1;
        bool condense_loaded_from_notebook = false;
        bool ff0_loaded_from_notebook = false;
        bool ff1_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
        printf("Condense/projection layer: trying notebook-generated condense weights\n");
#endif
        condense_loaded_from_notebook = tryLoadWeight(
            n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 0: trying notebook-generated FF0 weights\n");
#endif
        ff0_loaded_from_notebook = tryLoadWeight(
            n, 1, D_MODEL * D_FF >> 2, ff0_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 1: trying notebook-generated FF1 weights\n");
#endif
        ff1_loaded_from_notebook = tryLoadWeight(
            n, 2, D_FF * D_MODEL >> 2, ff1_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        if (condense_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated condense weights from "
                      << learner_notebook_weights_dir << std::endl;
        }
        if (ff0_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF0 weights from "
                      << learner_notebook_weights_dir << std::endl;
        }
        if (ff1_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF1 weights from "
                      << learner_notebook_weights_dir << std::endl;
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
        fill_weight(condense_kernel, NUM_HEAD * D_Q, D_MODEL >> 2);
        fill_weight(ff0_kernel, D_MODEL, D_FF >> 2);
        fill_weight(ff1_kernel, D_FF, D_MODEL >> 2);

        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, dir_name);
        saveWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, dir_name);
        saveWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, dir_name);
#endif

#ifndef BWMA
        uint32_t* condenseRowWise = new uint32_t[NUM_HEAD * D_Q * D_MODEL >> 2];
        blockWise2RowWise(condense_kernel, condenseRowWise, NUM_HEAD * D_Q, D_MODEL >> 2);
        condense_kernel = condenseRowWise;

        uint32_t* ff0RowWise = new uint32_t[D_MODEL * D_FF >> 2];
        blockWise2RowWise(ff0_kernel, ff0RowWise, D_MODEL, D_FF >> 2);
        ff0_kernel = ff0RowWise;

        uint32_t* ff1RowWise = new uint32_t[D_FF * D_MODEL >> 2];
        blockWise2RowWise(ff1_kernel, ff1RowWise, D_FF, D_MODEL >> 2);
        ff1_kernel = ff1RowWise;
#endif
#endif

        weightVec[NUM_HEAD * 3] = condense_kernel;
        weightVec[NUM_HEAD * 3 + 1] = ff0_kernel;
        weightVec[NUM_HEAD * 3 + 2] = ff1_kernel;

        return new TransformerBlock(
            D_SEQ,
            D_MODEL,
            D_Q,
            NUM_HEAD,
            D_FF,
            weightVec.data(),
            KERNEL_DIM,
            MAX_COL,
            learner_idx,
            learner_dump_dir);
    };

    // The grouped execution path is enabled when the registry exposes a learner
    // count that has an interleaved GEMM backend. With 2 learners, CodebookDense
    // only groups same-sequence layers; with 4 learners, it chooses same-seq or
    // diff-seq internally from the registry metadata.
    if (learner_count == 2) {
        TransformerBlock* grouped_blocks[2] = {nullptr, nullptr};
        uint32_t* grouped_inputs[2] = {tensor_in, tensor_in};
        uint32_t* grouped_outputs[2] = {
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
        };

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            std::cout << "\n=============== LEARNER " << learner_idx
                      << " ===============\n" << std::endl;

            const std::string learner_notebook_weights_dir =
                getNotebookWeightsDirForLearner(notebook_weights_dir, learner_idx);
            const std::string learner_dump_dir =
                multiple_learner_output_root + "/learner" + std::to_string(learner_idx);

            std::filesystem::create_directories(learner_dump_dir);
            grouped_blocks[learner_idx] = buildTransformerBlockForLearner(
                learner_idx,
                learner_notebook_weights_dir,
                learner_dump_dir);
        }

        TransformerBlock::computeGroup2(D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            delete[] grouped_outputs[learner_idx];
            delete grouped_blocks[learner_idx];
        }

        return;
    }

    if (learner_count == 4) {
        TransformerBlock* grouped_blocks[4] = {nullptr, nullptr, nullptr, nullptr};
        uint32_t* grouped_inputs[4] = {tensor_in, tensor_in, tensor_in, tensor_in};
        uint32_t* grouped_outputs[4] = {
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
        };

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            std::cout << "\n=============== LEARNER " << learner_idx
                      << " ===============\n" << std::endl;

            const std::string learner_notebook_weights_dir =
                getNotebookWeightsDirForLearner(notebook_weights_dir, learner_idx);
            const std::string learner_dump_dir =
                multiple_learner_output_root + "/learner" + std::to_string(learner_idx);

            std::filesystem::create_directories(learner_dump_dir);
            grouped_blocks[learner_idx] = buildTransformerBlockForLearner(
                learner_idx,
                learner_notebook_weights_dir,
                learner_dump_dir);
        }

        // This call does not jump to GEMM directly. It enters the grouped
        // transformer path, where each attention/FFN dense layer first tries the
        // 4-learner CodebookDense fast path and only then dispatches to either the
        // SVE or scalar interleaved GEMM backend.
        TransformerBlock::computeGroup4(D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            delete[] grouped_outputs[learner_idx];
            delete grouped_blocks[learner_idx];
        }

        return;
    }

    for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
        if (learner_count > 1) {
            std::cout << "\n=============== LEARNER " << learner_idx
                      << " ===============\n" << std::endl;
        }

        const std::string learner_notebook_weights_dir =
            getNotebookWeightsDirForLearner(notebook_weights_dir, learner_idx);
        const std::string learner_dump_dir = (learner_count > 1)
            ? (multiple_learner_output_root + "/learner" + std::to_string(learner_idx))
            : std::string();

        if (!learner_dump_dir.empty()) {
            std::filesystem::create_directories(learner_dump_dir);
        }

        uint32_t *out = new uint32_t[D_SEQ * D_MODEL >> 2]();
        TransformerBlock* selfatten = buildTransformerBlockForLearner(
            learner_idx,
            learner_notebook_weights_dir,
            learner_dump_dir);
        selfatten->compute(D_SEQ, tensor_in, out);
        delete selfatten;

        delete[] out;
    }

}

int main() {
    test();
    return 0;
}
