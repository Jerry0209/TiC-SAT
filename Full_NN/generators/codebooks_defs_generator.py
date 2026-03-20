
from ast import If
import math
from string import Template




def generate_cb_definitions(filename, n_learners, codebook_size, SVE_lanes, use_bias, use_f16, same_seq, use_codebooks):

    idxs_bits = math.ceil(math.log2(codebook_size)) # How many bits needed to represent codebook_size unique values (codewords)
    # for example, if codebook_size=16, we need 4 bits to represent 16 unique values (0 to 15). If codebook_size=256, we need 8 bits to represent 256 unique values (0 to 255).
    
    # Create a binary mask with idxs_bits number of 1's. This will be used to mask out the relevant bits when extracting codeword indices from packed bit representations.
    idx_mask = "0b{}".format("1" * idxs_bits)
    # If idxs_bits=4 → idx_mask = "0b1111"
    # If idxs_bits=5 → idx_mask = "0b11111"
    
    n_regs_cb = math.ceil(codebook_size / SVE_lanes)  # How many SVE registers to hold a full codebook
    n_regs_per_cb_string = "#define N_SVE_REG_CB_{}\t{}\n".format(n_regs_cb, n_regs_cb)

    n_regs_cb_f16 = math.ceil(codebook_size / (SVE_lanes * 2))  # How many SVE registers to hold a full codebook (using fp16 quantization)
    n_regs_per_cb_f16_string = "#define N_SVE_REG_CB_F16_{}\t{}\n".format(n_regs_cb_f16, n_regs_cb_f16)
    
    n_regs_per_cb_string += n_regs_per_cb_f16_string

    idxs_bitlen = 32

    n_lanes = SVE_lanes
    n_halflanes = SVE_lanes * 2
    sve_bytes = n_halflanes * 2

    if use_bias:
        use_bias_def_string = "#define USE_BIAS"
    else:
        use_bias_def_string = ""

    if not use_f16:
        word_bitlen = 32
        # n_lanes = SVE_lanes
    else:
        word_bitlen = 16
        # n_lanes = int(SVE_lanes * 2)

    if same_seq:
        seq_type = "SAME_SEQ"
    else:
        seq_type = "DIFF_SEQ"

    if n_learners == 8:
        learners_8_def = "#define LEARNERS_8\t1"
    else:
        learners_8_def = ""

    if not use_codebooks:
        no_cb_def = "#define NO_CODEBOOKS\t1"
    else:
        no_cb_def = ""
        

    with open("./templates/codebooks_defs_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            n_learners = n_learners,
            sve_size = n_lanes,
            sve_halfwords = n_halflanes,
            sve_bytes = sve_bytes,
            codebook_size = codebook_size,
            bits_per_codeword = idxs_bits,
            index_bin_mask = idx_mask,
            word_bitlen = idxs_bitlen,
            string_n_regs_per_cb = n_regs_per_cb_string,
            bitwidth_used = word_bitlen,
            use_bias_def = use_bias_def_string,
            sequence_type = seq_type,
            learners_8_def = learners_8_def,
            no_cb_def = no_cb_def
        )
    
    with open(filename, "w") as f:
        f.write(header)