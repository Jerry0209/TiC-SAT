#include "../gemm_definitions/generated_codebook_registry.h"

#include <gemm_exec.h>
#include <gemm_SVE.h>

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void print_vect_f32(svfloat32_t v) {
    const std::size_t lanes = svcntw();
    std::vector<float> buf(lanes, 0.0f);

    svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0),
                                static_cast<uint64_t>(lanes));
    svst1_f32(pg, buf.data(), v);

    for (std::size_t i = 0; i < lanes; i++) {
        std::printf("[%zu]\t%f\n", i, buf[i]);
    }
    std::printf("\n");
}

static void print_predicate_w(svbool_t pg) {
    const std::size_t lanes = svcntw();
    std::vector<uint32_t> buf(lanes, 0U);

    svint32_t indices = svindex_s32(0, 1);
    for (std::size_t i = 0; i < lanes; i++) {
        svbool_t p =
            svcmpeq_n_s32(svptrue_b32(), indices, static_cast<int32_t>(i));
        buf[i] = svptest_any(p, pg) ? 1U : 0U;
    }

    std::printf("Predicate Vector (W):\n");
    for (std::size_t i = 0; i < lanes; i++) {
        std::printf("[%zu]\t%u\n", i, buf[i]);
    }
    std::printf("\n");
}

static void runHelloSVEDemo() {
    constexpr std::size_t N_ELEMS = 10;

    static const float input_A[N_ELEMS] = {
        0.30593392294314126f, 0.9729489210930778f, 0.6967651289479858f,
        0.45340254562143867f, 0.7722622902191596f, 0.4631572250189654f,
        0.7058883294981203f, 0.8284812740863892f, 0.25400633077082535f,
        0.7025260272018509f};

    static const float input_B[N_ELEMS] = {
        0.9799316649760836f,  0.036416962363738814f, 0.38098218215107194f,
        0.9862879481587331f,  0.37151641123862167f,  0.7611771045148974f,
        0.3870005424619666f,  0.7599391197169654f,   0.7615588261600846f,
        0.8027245750657497f};

    std::printf("========================\n");
    std::printf("Hello, SVE World! [Merged]\n");
    std::printf("========================\n\n");

    int n_8bits_elems = static_cast<int>(svcntb());
    int n_16bits_elems = static_cast<int>(svcnth());
    int n_32bits_elems = static_cast<int>(svcntw());

    std::printf("-------------------------\n");
    std::printf("N 8-bits elements = %d\n", n_8bits_elems);
    std::printf("N 16-bits elements = %d\n", n_16bits_elems);
    std::printf("N 32-bits elements = %d\n", n_32bits_elems);
    std::printf("-------------------------\n\n");

    svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(4));
    print_predicate_w(pg);

    svfloat32_t va = svld1(pg, input_A);
    std::printf("Vector va:\n");
    print_vect_f32(va);

    svfloat32_t vb = svld1(pg, input_B);
    std::printf("Vector vb:\n");
    print_vect_f32(vb);

    svfloat32_t add_result = svadd_f32_m(pg, va, vb);
    std::printf("Adding result:\n");
    print_vect_f32(add_result);

    std::printf("\n-------------------------\n");
    std::printf("Starting the loop....\n");
    std::printf("-------------------------\n\n");

    int loop_cnt = 0;
    svfloat32_t mult_result = svdup_f32(0.0f);

    for (std::size_t i = 0; i < N_ELEMS; i += static_cast<std::size_t>(n_32bits_elems)) {
        std::printf("\nIteration [%d]  |  number of processed elements = %zu / %zu\n",
                    loop_cnt, i, N_ELEMS);

        pg = svwhilelt_b32(static_cast<uint64_t>(i),
                           static_cast<uint64_t>(N_ELEMS));
        print_predicate_w(pg);

        va = svld1(pg, &input_A[i]);
        vb = svld1(pg, &input_B[i]);

        mult_result = svmad_f32_x(pg, va, vb, mult_result);
        loop_cnt++;
    }

    std::printf("Vector result of the MAC:\n");
    print_vect_f32(mult_result);

    pg = svwhilelt_b32(static_cast<uint64_t>(0),
                       static_cast<uint64_t>(N_ELEMS));
    float final_mac_result = svaddv_f32(pg, mult_result);

    std::printf("\nFINAL MAC RESULT = %f\n\n", final_mac_result);
}

static std::vector<int8_t> makeDefaultInput(std::size_t rows, std::size_t cols) {
    std::vector<int8_t> input(rows * cols, 0);

    for (std::size_t c = 0; c < cols; c++) {
        input[c] = static_cast<int8_t>(static_cast<int>(c % 8) - 4);
    }

    if (rows >= 2) {
        for (std::size_t c = 0; c < cols; c++) {
            input[cols + c] = static_cast<int8_t>(static_cast<int>(c % 5) - 2);
        }
    }

    for (std::size_t r = 2; r < rows; r++) {
        for (std::size_t c = 0; c < cols; c++) {
            input[r * cols + c] =
                static_cast<int8_t>(static_cast<int>((r + c) % 7) - 3);
        }
    }

    return input;
}

static bool compareOutputs(const std::vector<int32_t>& expected,
                           const std::vector<int32_t>& actual) {
    if (expected.size() != actual.size()) {
        std::cout << "sve_vs_scalar_size_match = false\n";
        return false;
    }

    for (std::size_t idx = 0; idx < expected.size(); idx++) {
        if (expected[idx] != actual[idx]) {
            std::cout << "sve_vs_scalar_match = false\n";
            std::cout << "first_mismatch_index = " << idx << "\n";
            std::cout << "expected_scalar = " << expected[idx] << "\n";
            std::cout << "actual_sve = " << actual[idx] << "\n";
            return false;
        }
    }

    std::cout << "sve_vs_scalar_match = true\n";
    return true;
}

static void printPreview(const std::vector<int32_t>& values,
                         std::size_t cols,
                         const std::string& label) {
    const std::size_t preview = (cols < 8) ? cols : 8;
    std::cout << label << "_row0_first_" << preview << " = [";
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << values[idx];
        if (idx + 1 != preview) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

static bool runSingleLayerSVE(const std::string& layer_name, std::size_t seq_len) {
    const GeneratedCodebookLayerView* layer =
        findGeneratedCodebookLayer(layer_name.c_str());
    if (layer == nullptr) {
        throw std::runtime_error("Layer not found in generated registry: " + layer_name);
    }
    if (layer->codebook_int8 == nullptr) {
        throw std::runtime_error("Layer has no int8 codebook: " + layer_name);
    }
    if ((layer->input_size > UINT16_MAX) || (layer->output_size > UINT16_MAX) ||
        (layer->n_words_row > UINT16_MAX) || (seq_len > UINT16_MAX)) {
        throw std::runtime_error("Layer dimensions exceed gemm_t uint16_t fields");
    }

    gemm_t gemm_layer;
    gemm_layer.seq_len = static_cast<uint16_t>(seq_len);
    gemm_layer.input_size = static_cast<uint16_t>(layer->input_size);
    gemm_layer.output_size = static_cast<uint16_t>(layer->output_size);
    gemm_layer.n_words_row = static_cast<uint16_t>(layer->n_words_row);

    std::vector<int8_t> input = makeDefaultInput(seq_len, layer->input_size);
    std::vector<int32_t> scalar_out(seq_len * layer->output_size, 0);
    std::vector<int32_t> sve_out(seq_len * layer->output_size, 0);

    gemm_exec_compact_int(gemm_layer,
                          input.data(),
                          layer->weight_idx,
                          layer->codebook_int8,
                          nullptr,
                          scalar_out.data(),
                          layer->bits_per_cb);

    gemm_exec_compact_int_sve(gemm_layer,
                              input.data(),
                              layer->weight_idx,
                              layer->codebook_int8,
                              nullptr,
                              sve_out.data(),
                              layer->bits_per_cb);

    std::cout << "layer_name = \"" << layer_name << "\"\n";
    std::cout << "seq_len = " << seq_len << "\n";
    std::cout << "input_size = " << layer->input_size << "\n";
    std::cout << "output_size = " << layer->output_size << "\n";
    std::cout << "n_words_row = " << layer->n_words_row << "\n";
    std::cout << "bits_per_cb = " << static_cast<int>(layer->bits_per_cb) << "\n";
    printPreview(scalar_out, layer->output_size, "scalar_acc");
    printPreview(sve_out, layer->output_size, "sve_acc");

    return compareOutputs(scalar_out, sve_out);
}

int main(int argc, char** argv) {
    try {
        std::string layer_name = "q_h0";
        std::size_t seq_len = 2;

        if (argc >= 2) {
            layer_name = argv[1];
        }
        if (argc >= 3) {
            seq_len = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
        }

        runHelloSVEDemo();
        return runSingleLayerSVE(layer_name, seq_len) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}