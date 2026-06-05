#include "floatDump.h"

#include "run_mode_config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace TransformerFloat {

/**
 * Save a row-major float matrix as plain text.
 *
 * @param path Destination file path. Parent directories must already exist.
 * @param data Pointer to rows * cols float values in row-major order.
 * @param rows Number of matrix rows to write.
 * @param cols Number of matrix columns per row.
 */
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

/**
 * Dump a float matrix unless profiling mode has disabled debug artifacts.
 *
 * @param dump_dir Directory where the dump file should be created. An empty
 *        string disables dumping for this matrix.
 * @param filename File name to create inside dump_dir.
 * @param data Pointer to rows * cols float values in row-major order.
 * @param rows Number of logical rows in the matrix.
 * @param cols Number of logical columns in the matrix.
 */
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

/**
 * Split an interleaved multi-learner matrix and dump each learner separately.
 *
 * @param dump_dirs Per-learner output directories. dump_dirs[learner] controls
 *        whether that learner is dumped and where the file is written.
 * @param filename File name reused for every learner-specific dump.
 * @param interleaved Pointer to data laid out as [row][col][learner].
 * @param rows Number of rows in each learner's logical matrix.
 * @param cols Number of columns in each learner's logical matrix.
 * @param learner_count Number of learner values stored for every row/column
 *        element in the interleaved input.
 */
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

/**
 * Print the first few values of a float buffer when debug printing is enabled.
 *
 * @param label Prefix printed before each preview value.
 * @param data Pointer to the contiguous float buffer to preview.
 * @param size Number of float values available in data.
 */
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

/**
 * Print one learner's view of an interleaved float buffer.
 *
 * @param label Prefix printed before each preview value.
 * @param interleaved Pointer to data laid out as [logical index][learner].
 * @param logical_size Number of non-learner elements to preview from the
 *        selected learner, usually rows * cols.
 * @param learner_count Number of learner lanes stored per logical element.
 * @param learner Learner lane to print from the interleaved buffer.
 */
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

/**
 * Combine separate learner matrices into one interleaved matrix.
 *
 * @param inputs Array of learner_count input matrix pointers. Each matrix is
 *        row-major with rows * cols values.
 * @param learner_count Number of learner matrices to interleave.
 * @param rows Number of rows in every input matrix.
 * @param cols Number of columns in every input matrix.
 * @param output_interleaved Destination vector. It is resized and filled with
 *        values in [row][col][learner] order.
 */
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

/**
 * Split an interleaved matrix back into separate learner matrices.
 *
 * @param input_interleaved Source vector laid out as [row][col][learner].
 * @param learner_count Number of learner lanes stored per matrix element.
 * @param rows Number of rows in each learner's output matrix.
 * @param cols Number of columns in each learner's output matrix.
 * @param outputs Array of learner_count destination matrix pointers. The caller
 *        must allocate rows * cols floats for each destination.
 */
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
