import math
from string import Template

from dense_layer_generator import (
    gen_biases_strings,
    gen_codebooks,
    generate_dense_indexes,
    generate_weights_no_CB,
)


def generate_template_gemm(
    same_seq,
    filename,
    layer_ID,
    n_learners,
    codebook_size,
    tile_size,
    in_size,
    out_size,
    use_f16,
    use_codebooks,
):
    words_bitlen = 32

    if n_learners <= 4:
        groups_of_4_learners = 1
    else:
        groups_of_4_learners = int(n_learners / 4)

    biases_strings, biases_values_learners = gen_biases_strings(
        layer_ID, n_learners, out_size, groups_of_4_learners, use_f16
    )

    if use_codebooks:
        idxs_bits = math.ceil(math.log2(codebook_size))
        idxs_per_word = int(math.modf(words_bitlen / idxs_bits)[1])
        words_per_row = math.ceil(in_size / idxs_per_word)

        cb_string, cb_interl_string, codebooks_ensembles = gen_codebooks(
            layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners
        )
        indexes_string, values, indexes_tiled_packed_string = generate_dense_indexes(
            same_seq,
            codebooks_ensembles,
            n_learners,
            layer_ID,
            codebook_size,
            in_size,
            out_size,
            words_per_row,
            tile_size,
            words_bitlen,
            use_f16,
        )

        no_cb_weights_string = ""
        bias_string_tot = biases_strings[0] + "\n\n\n" + biases_strings[1]
    else:
        no_cb_weights_string, values = generate_weights_no_CB(
            layer_ID, n_learners, in_size, out_size, use_f16
        )

        cb_string = ""
        cb_interl_string = ""
        indexes_string = ""
        indexes_tiled_packed_string = ""
        bias_string_tot = biases_strings[0]

    with open("./templates/gemm_weights_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            ID=layer_ID,
            input_size=in_size,
            otuput_size=out_size,
            codebook_string=cb_string,
            codebook_string_interleaved=cb_interl_string,
            indexes_packed_string=indexes_string,
            indexes_packed_tiled_string=indexes_tiled_packed_string,
            no_cb_weights_string=no_cb_weights_string,
            bias_strings=bias_string_tot,
        )

    with open(filename, "w") as f:
        f.write(header)

    return values, biases_values_learners


def generate_gemm_data_file(filename, n_layers):
    header = "#ifndef _GEMM_DATA_H_\n"
    header += "#define _GEMM_DATA_H_\n\n"

    header += "#include <gemm_exec.h>\n\n"
    header += "#include <./input_matrix.h>\n"

    for layer_id in range(n_layers):
        header += "#include <./gemm_header_{}.h>\n".format(layer_id)

    header += "\n#define N_GEMM_LAYERS {}\n\n".format(n_layers)

    for layer_id in range(n_layers):
        header += "static const gemm_t gemm_{} = {{\n".format(layer_id)
        header += "\t.seq_len = GEMM_M,\n"
        header += "\t.input_size = INPUT_SIZE_{},\n".format(layer_id)
        header += "\t.output_size = OUTPUT_SIZE_{},\n".format(layer_id)
        header += "\t.n_words_row = N_WORDS_ROW_{}\n".format(layer_id)
        header += "};\n\n"

    header += "#endif\n"

    with open(filename, "w") as f:
        f.write(header)
