#include "floatDump.h"

#include "run_mode_config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace TransformerFloat {

void saveFloatMatrixText(const std::string& path,
                         const float* data,
                         std::size_t rows,
                         std::size_t cols) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::cout << path << " Not saved" << std::endl;
        return;
    }

    fout << std::setprecision(9);
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            if (col != 0u) {
                fout << " ";
            }
            fout << data[row * cols + col];
        }
        fout << "\n";
    }
}

void dumpFloatMatrixIfEnabled(const std::string& dump_dir,
                              const std::string& filename,
                              const float* data,
                              std::size_t rows,
                              std::size_t cols) {
#if CFG_PROFILE_GEMM_ONLY || CFG_GEM5_PROFILE_REGIONS
    (void)dump_dir;
    (void)filename;
    (void)data;
    (void)rows;
    (void)cols;
#else
    if (dump_dir.empty()) {
        return;
    }
    std::filesystem::create_directories(dump_dir);
    const std::string path = dump_dir + "/" + filename;
    saveFloatMatrixText(path, data, rows, cols);
    std::cout << "[DUMP] float matrix -> " << path << std::endl;
#endif
}

void dumpInterleavedFloatMatrices(const std::vector<std::string>& dump_dirs,
                                  const std::string& filename,
                                  const float* interleaved,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t learner_count) {
    Matrix tmp(rows * cols, 0.0f);
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        for (std::size_t row = 0; row < rows; row++) {
            for (std::size_t col = 0; col < cols; col++) {
                tmp[row * cols + col] =
                    interleaved[((row * cols) + col) * learner_count + learner];
            }
        }
        dumpFloatMatrixIfEnabled(dump_dirs[learner], filename, tmp.data(), rows, cols);
    }
}

void printFloatPreview(const std::string& label,
                       const float* data,
                       std::size_t size) {
#if CFG_ENABLE_DEBUG_PRINT
    const std::size_t preview = std::min<std::size_t>(size, 8u);
    std::cout << label << " preview (first " << preview << " float values):" << std::endl;
    std::cout << std::setprecision(6);
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << label << "[" << idx << "] = " << data[idx] << std::endl;
    }
#else
    (void)label;
    (void)data;
    (void)size;
#endif
}

void printInterleavedFloatPreview(const std::string& label,
                                  const float* interleaved,
                                  std::size_t logical_size,
                                  std::size_t learner_count,
                                  std::size_t learner) {
#if CFG_ENABLE_DEBUG_PRINT
    const std::size_t preview = std::min<std::size_t>(logical_size, 8u);
    std::cout << label << " preview (first " << preview << " float values):" << std::endl;
    std::cout << std::setprecision(6);
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << label << "[" << idx << "] = "
                  << interleaved[idx * learner_count + learner] << std::endl;
    }
#else
    (void)label;
    (void)interleaved;
    (void)logical_size;
    (void)learner_count;
    (void)learner;
#endif
}

void interleaveLearnerMatrices(const float* const* inputs,
                               std::size_t learner_count,
                               std::size_t rows,
                               std::size_t cols,
                               Matrix& output_interleaved) {
    output_interleaved.assign(rows * cols * learner_count, 0.0f);
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                output_interleaved[((row * cols) + col) * learner_count + learner] =
                    inputs[learner][row * cols + col];
            }
        }
    }
}

void deinterleaveLearnerMatrices(const Matrix& input_interleaved,
                                 std::size_t learner_count,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float* const* outputs) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                outputs[learner][row * cols + col] =
                    input_interleaved[((row * cols) + col) * learner_count + learner];
            }
        }
    }
}

} // namespace TransformerFloat
