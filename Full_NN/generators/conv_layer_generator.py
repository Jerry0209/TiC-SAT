
import numpy as np
import math
from string import Template

def to_bin_N_digits(num, N_digits=2):
    binary = bin(num)
    if len(binary.split("b")[1]) < N_digits:
        n_to_insert = N_digits - len(binary.split("b")[1])
        for i in range(n_to_insert):
            binary = binary[:2] + "0" + binary[2:]
    return binary[2:]

def compute_CB_parameters(codebook_size, in_ch, out_ch, k_size, words_bitlen=32):
    idxs_bits = math.ceil(math.log2(codebook_size))
    idx_per_word = int(math.modf(words_bitlen / idxs_bits)[1])

    idx_per_channel = (in_ch * k_size * k_size)
    words_per_channel = math.ceil(idx_per_channel / idx_per_word)
    N_words_of_indexes_perCH = (out_ch * words_per_channel)

    return N_words_of_indexes_perCH, idxs_bits, idx_per_channel, idx_per_word




"""
    Generates the C-formatted strings for the header file.
    This includes the generation of the codebook values (both per learner and in interleaved way).
    Also generates the access indexes to the codebook and the relative values from them.
    
    Returns the values of the codebooks per each learner and the values of the kernels, as taken from the codebooks according to the generated indexes
"""
def generate_kernel_header_file(same_seq, filename, layer_ID, n_learners, codebook_size, out_ch, in_ch, k_size, stride, padding, tile_L2_size, tile_L1_size, use_f16, use_codebooks):

    if use_codebooks:
        # generate the codebooks
        cb_strings, codebook_values_learners  = gen_kernel_strings(layer_ID, n_learners, codebook_size, use_f16) 

        # generate the indexes
        k_values, k_indexes = generate_kernel_indexes(same_seq, codebook_values_learners, n_learners, out_ch, in_ch, k_size, codebook_size)


        no_cb_weights_string_tiled = ""
        # no_cb_weights_string = gen_non_cb_weights_string(layer_ID, k_values)

        # format the indexes separated per channels
        k_indexes_per_ch_string, bin_kernel = generate_kernel_indexes_per_CH(same_seq, layer_ID, n_learners, codebook_size, in_ch, out_ch, k_size, k_indexes, use_f16)

        # if not use_f16:
        words_bitlen = 32
        # else:
        #     words_bitlen = 16

        _, _, _, idx_per_word = compute_CB_parameters(codebook_size, in_ch, out_ch, k_size, words_bitlen=words_bitlen)
        bin_kernel_tiled = tile_kernel(same_seq, n_learners, bin_kernel, k_size, in_ch, out_ch, idx_per_word, tile_L1_size, layer_ID, use_f16)
        
        words_per_k_def_string = ""
        no_cb_weights_string_tiled = ""

    else:
        words_per_k_def_string, weights_string, weights_values = generate_kernel_no_CB(layer_ID, n_learners, in_ch, out_ch, k_size, use_f16)

        no_cb_weights_string_tiled = tile_kernel_no_CB(n_learners, weights_values, k_size, in_ch, out_ch, tile_L1_size, layer_ID, use_f16)

        cb_strings = ["", ""]
        k_indexes_per_ch_string = ""
        bin_kernel_tiled = ""

        # Return values
        codebook_values_learners = []
        k_values = weights_values


    # reusue the same function to generate the codebooks for the biases
    biases_strings, biases_values_learners = gen_biases_strings(layer_ID, n_learners, out_ch, use_f16)
    # biases_values, biases_indexes = generate_biases_indexes(biases_values_learners, n_learners, out_ch, codebook_size)
    # biase_indexes_string = generate_biases_indexes_string(layer_ID, codebook_size, in_ch, out_ch, k_size, biases_indexes)

    with open("./templates/conv_weights_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            ID = layer_ID,
            in_ch = in_ch,
            out_ch = out_ch,
            stride = stride,
            padding = padding,
            n_learners = n_learners,
            tile_l2_size = tile_L2_size,
            tile_l1_size = tile_L1_size,
            k_size = k_size,
            codebook_string = cb_strings[0],
            codebook_string_interleaved = cb_strings[1],
            kernel_indexes_perCH_string = k_indexes_per_ch_string,
            kernel_indexes_tiled_string = bin_kernel_tiled,
            non_cb_weights_string = words_per_k_def_string + no_cb_weights_string_tiled,
            bias_strings = biases_strings[0] + "\n\n\n" + biases_strings[1]
            # codebook_string_interleaved_f16 = cb_strings[2],
            # codebook_biases_interl_string = biases_strings[1],
            # biases_indexes_perCH_string = biase_indexes_string
        )

    
    with open(filename, "w") as f:
        f.write(header)

    return codebook_values_learners, k_values, biases_values_learners





"""
    Generates the codebooks (normal and interleaved) for all the learners.
    Returns the C header string representation of the codebooks (normal and interleaved),
    together with a list with the codebook values per each learner
"""
def gen_kernel_strings(layer_ID, n_learners, codebook_size, use_f16):
    codebook_string = "static const float{} codebook{}_{}[N_LEARNERS][CB_SIZE] = {{\n"

    # In case there are more than 4 learners, group them 4 by 4
    if n_learners <= 4:
        codebook_string_interleaved = "static const float{} codebook_interleaved{}_{}[N_LEARNERS * CB_SIZE] = {{\n"
    else:
        groups_of_4_learners = int(n_learners / 4)
        codebook_string_interleaved = "#define GROUPS_OF_4_LEARNERS {}\n\n".format(groups_of_4_learners)
        codebook_string_interleaved += "static const float{} codebook_interleaved{}_{}[GROUPS_OF_4_LEARNERS][N_LEARNERS * CB_SIZE] = {{\n"


    if not use_f16:
        codebook_string = codebook_string.format("", "", layer_ID)
        codebook_string_interleaved = codebook_string_interleaved.format("", "", layer_ID)
    else:
        codebook_string = codebook_string.format("16_t", "_f16", layer_ID)
        codebook_string_interleaved = codebook_string_interleaved.format("16_t", "_f16", layer_ID)
        

    codebooks_ensembles = [] # Holds all the codebooks of the learners

    # Generate the codebook for all the learners
    for i in range(n_learners):
        codebooks_ensembles.append(np.random.uniform(0.0, 0.9, codebook_size))

    # Fill the codebooks in C format
    for ens_cb in codebooks_ensembles:
        codebook_string += "\t{\n"
        for v in ens_cb:
            codebook_string += "\t\t{},\n".format(v)
        codebook_string += "\t},\n"
    codebook_string += "};"


    if n_learners <= 4:
        for i in range(codebook_size):
            for ens in range(n_learners):
                codebook_string_interleaved += "\t{},\n".format(codebooks_ensembles[ens][i])
        codebook_string_interleaved += "};"
    else:
        for group in range(groups_of_4_learners):
            codebook_string_interleaved += "\t{\n"
            for i in range(codebook_size):
                for ens in range(4):
                    codebook_string_interleaved += "\t\t{},\n".format(codebooks_ensembles[(group * 4) + ens][i])
            codebook_string_interleaved += "\t},\n"
        codebook_string_interleaved += "};\n"

    return (codebook_string, codebook_string_interleaved), codebooks_ensembles




#######################################################
# Generates the indexes and the values of the KERNELS #
#######################################################
def generate_kernel_indexes(same_seq, codebooks_ensembles, n_learners, out_ch, in_ch, kernel_size, codebook_size):
    kernel_values = [[] for _ in range(n_learners)]      # Raw kernel values (one list per learner)
    kernel_indexes = []     # Kernel indexs (one list per learner)

    if same_seq:
        for i in range(out_ch * in_ch * kernel_size * kernel_size):
            new_idx = np.random.randint(0, codebook_size)       # New index
            kernel_indexes.append(new_idx)

            for ens in range(n_learners):
                new_val = codebooks_ensembles[ens][new_idx]     # New value from each codebook
                kernel_values[ens].append(new_val)
    else:
        for ens in range(n_learners):
            learner_idxs = []
            for i in range(out_ch * in_ch * kernel_size * kernel_size):
                new_idx = np.random.randint(0, codebook_size)       # New index
                learner_idxs.append(new_idx)

                new_val = codebooks_ensembles[ens][new_idx]     # New value from each codebook
                kernel_values[ens].append(new_val)
            kernel_indexes.append(learner_idxs)


    return kernel_values, kernel_indexes    # Return both values and indexes


def tile_compact_kernel(w_size, h_size, bin_kernel, l1_tile_size):
    tiled_string = ""
    tiled_bin_kernel = []

    for wtiles in range(0, w_size, l1_tile_size):
        for htiles in range(0, h_size, l1_tile_size):

            # print("Tile: {} {}".format(htiles, wtiles))
            tile = []

            for th in range(l1_tile_size): 
                for tw in range(l1_tile_size):
                    hindex = htiles + th
                    windex = wtiles + tw

                    # print("{} {}".format(hindex, windex))

                    if (hindex < h_size) and (windex < w_size):
                        index = (hindex * w_size) + windex
                        # print(index)
                        tile.append(bin_kernel[index])
                        tiled_string += "\t{},".format(bin_kernel[index])
                        tiled_bin_kernel.append(bin_kernel[index])
            tiled_string += "\n" 

    return tiled_string, tiled_bin_kernel




def merge_bin_kernels(bin_kernel):
    
    orig_len = len(bin_kernel[0])
    n_learners = len(bin_kernel)

    res = []

    for i in range(orig_len):
        for learn in range(n_learners):
            res.append(bin_kernel[learn][i])
    return res





#####################################################################
# Formats the kernel indexes in a tiled manner (used for L1 tiling) #
#####################################################################

def tile_kernel(same_seq, n_learners, bin_kernel, k_size, in_ch, out_ch, idx_per_word, tile_L1_size, layer_ID, use_f16):

    w_size = math.ceil((k_size * k_size * in_ch) / idx_per_word)
    h_size = out_ch

    if same_seq:
        if not use_f16:
            bin_kernel_tiled_string = "static uint{}_t kernel_compact_tiled{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        else:
            bin_kernel_tiled_string = "static uint{}_t kernel_compact_tiled{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "_f16", layer_ID, layer_ID)
            # bin_kernel_tiled_string = "static uint{}_t kernel_compact_tiled{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID)

        bin_kernel_tiled_string += tile_compact_kernel(w_size, h_size, bin_kernel, tile_L1_size)[0]
        bin_kernel_tiled_string += "};\n"

    else:
        if not use_f16:
            bin_kernel_tiled_string = "static uint{}_t kernel_compact_tiled{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        else:
            bin_kernel_tiled_string = "static uint{}_t kernel_compact_tiled{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID)

        # test
        # bin_kernel_tiled_string_test = "static uint{}_t kernel_compact_tiled_test{}_{}[N_LEARNERS * N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        # w_size_merged = math.ceil((k_size * n_learners * k_size * in_ch) / idx_per_word)
        # merged_bin_kernels = merge_bin_kernels(bin_kernel)
        # bin_kernel_tiled_string += tile_compact_kernel(w_size_merged, h_size, merged_bin_kernels, tile_L1_size)[0]
        # bin_kernel_tiled_string += "};\n"   
        # bin_kernel_tiled_string += "\n\n\nstatic uint{}_t kernel_compact_tiled{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        bin_kernels_tiled = []
        # end test

        for ens in range(n_learners):
            bin_kernel_tiled_string += "\t{\n\t"
            tiled_s, tiled_bin_k = tile_compact_kernel(w_size, h_size, bin_kernel[ens], tile_L1_size)
            bin_kernel_tiled_string += tiled_s
            bin_kernels_tiled.append(tiled_bin_k)
            bin_kernel_tiled_string += "\t},\n"
        bin_kernel_tiled_string += "};\n"


        merged_tiled_bin_k = merge_bin_kernels(bin_kernels_tiled)
        bin_kernel_tiled_string_test = "static uint{}_t kernel_compact_tiled_merged{}_{}[N_LEARNERS * N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        for e in merged_tiled_bin_k:
            bin_kernel_tiled_string_test += "\t{},\n".format(e)
        bin_kernel_tiled_string_test += "};\n"
    
        bin_kernel_tiled_string += "\n\n" + bin_kernel_tiled_string_test

    return bin_kernel_tiled_string




def tile_kernel_no_CB(n_learners, kernel_weights, k_size, in_ch, out_ch, tile_L1_size, layer_ID, use_f16):

    w_size = (k_size * k_size * in_ch)
    h_size = out_ch

    weights_no_CB_tiled_string = "static float{} weights_no_CB_tiled{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n"

    if not use_f16:
        weights_no_CB_tiled_string = weights_no_CB_tiled_string.format("", "", layer_ID, layer_ID)
    else:
        weights_no_CB_tiled_string = weights_no_CB_tiled_string.format("_f16", "_f16", layer_ID, layer_ID)

    for ens in range(n_learners):
        weights_no_CB_tiled_string += "\t{\n\t"
        weights_no_CB_tiled_string += tile_compact_kernel(w_size, h_size, kernel_weights[ens], tile_L1_size)[0]
        weights_no_CB_tiled_string += "\t},\n"
    weights_no_CB_tiled_string += "};\n\n"

    return weights_no_CB_tiled_string





def gen_idxs_string_for_learner(indexes, idxs_bits, idx_per_channel, idx_per_word, use_f16=False):
 
    binary_indexes_string = ""
    bin_kernel = []

    # Format the indexes in C, the indexes of a single channels are isolated from the ones of a different channel
    idxs_inserted = 0   # Counts the indexes inserted in a word
    channel_idxs_inserted = 0   # Counts the indexes inserted for the channel --> we split the indexe of each channel

    word_of_indexes = ""

    # For the f16 implementation, if we have 3 bits per index, we want to align the indexes so that there is no index 
    # split by the lowest and highest half of a word
    if (idxs_bits == 3) and (use_f16):
        pad_bits = True
        n_0_bits_pad = 1
        n_idxs_before_pad = 5
    else:
        pad_bits = False

    # Format the indexes in rows of words.
    # The indexes are separated per each channel.
    for i, idx in enumerate(indexes):
        bin_idx = to_bin_N_digits(idx, N_digits=idxs_bits)

        if channel_idxs_inserted < idx_per_channel:
            if idxs_inserted < idx_per_word:
                word_of_indexes = bin_idx + word_of_indexes
                idxs_inserted += 1
            else:
                word_of_indexes = "0b" + word_of_indexes
                binary_indexes_string += "\t{}, ".format(word_of_indexes)
                bin_kernel.append(word_of_indexes)
                idxs_inserted = 0
                word_of_indexes = bin_idx
                idxs_inserted += 1
                
            channel_idxs_inserted += 1
        else:
            word_of_indexes = "0b" + word_of_indexes
            binary_indexes_string += "\t{},\n".format(word_of_indexes)
            bin_kernel.append(word_of_indexes)
            idxs_inserted = 0
            channel_idxs_inserted = 0
            word_of_indexes = bin_idx
            idxs_inserted += 1
            channel_idxs_inserted += 1
        
        if pad_bits:
            if idxs_inserted == n_idxs_before_pad:
                word_of_indexes = ("0" * n_0_bits_pad) + word_of_indexes

    word_of_indexes = "0b" + word_of_indexes
    binary_indexes_string += "\t{},\n".format(word_of_indexes)
    bin_kernel.append(word_of_indexes)

    # print(binary_indexes_string)
    # print(bin_kernel)
    return binary_indexes_string, bin_kernel



##################################################################################
# Formats the kernel indexes in COMPACT form, keeping them separated per channel # 
##################################################################################

def generate_kernel_indexes_per_CH(same_seq, layer_ID, n_learners, codebook_size, in_ch, out_ch, k_size, k_indexes, use_f16):

    # if not use_f16:
    words_bitlen = 32
    # else:
    #     words_bitlen = 16

    N_words_of_indexes_perCH, idxs_bits, idx_per_channel, idx_per_word = compute_CB_parameters(codebook_size, in_ch, out_ch, k_size, words_bitlen=words_bitlen)

    binary_kernel_string_perCH = "#define N_WORDS_KERNEL_PER_CH_{}  {}\n\n".format(layer_ID, N_words_of_indexes_perCH)
    
    if same_seq:

        if not use_f16:
            binary_kernel_string_perCH += "static const uint{}_t kernel_compact_perCH{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        else:
            binary_kernel_string_perCH += "static const uint{}_t kernel_compact_perCH{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "_f16", layer_ID, layer_ID)
            # binary_kernel_string_perCH += "static const uint{}_t kernel_compact_perCH{}_{}[N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID)

        indexes, bin_kernel = gen_idxs_string_for_learner(k_indexes, idxs_bits, idx_per_channel, idx_per_word, use_f16=use_f16)
        binary_kernel_string_perCH += indexes
        binary_kernel_string_perCH += "};\n\n"

    else:
        if not use_f16:
            binary_kernel_string_perCH += "static const uint{}_t kernel_compact_perCH{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("32", "", layer_ID, layer_ID)
        else:
            binary_kernel_string_perCH += "static const uint{}_t kernel_compact_perCH{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID)

        bin_kernel = []

        for ens in range(n_learners):
            binary_kernel_string_perCH += "\t{\n"
            learner_idxs, learner_bin_kernel = gen_idxs_string_for_learner(k_indexes[ens], idxs_bits, idx_per_channel, idx_per_word, use_f16=use_f16)
            binary_kernel_string_perCH += "\t" + learner_idxs
            binary_kernel_string_perCH += "\t},\n"
            bin_kernel.append(learner_bin_kernel)
        binary_kernel_string_perCH += "};\n\n"

    return binary_kernel_string_perCH, bin_kernel




# generate_kernel_no_CB(layer_ID, n_learners, in_ch, out_ch, k_size, k_indexes, use_f16)
def generate_kernel_no_CB(layer_ID, n_learners, in_ch, out_ch, k_size, use_f16):

    weights_per_out_ch = (in_ch * k_size * k_size)
    N_words_of_indexes_perCH = (out_ch * weights_per_out_ch)
    words_per_k_def_string = "#define N_WORDS_KERNEL_PER_CH_{}  {}\n\n".format(layer_ID, N_words_of_indexes_perCH)
    weights_no_CB_string = "static const float{} kernel_weights{}_{}[N_LEARNERS][N_WORDS_KERNEL_PER_CH_{}] = {{\n"

    if not use_f16:
        weights_no_CB_string = weights_no_CB_string.format("", "", layer_ID, layer_ID)
    else:
        weights_no_CB_string = weights_no_CB_string.format("_f16", "_f16", layer_ID, layer_ID)
    
    raw_weights = np.random.uniform(-1.0, 1.0, size=(n_learners, N_words_of_indexes_perCH))

    for ens in range(n_learners):
        weights_no_CB_string += "\t{\n"
        for w in raw_weights[ens]:
            weights_no_CB_string += "\t\t{},\n".format(w)
        weights_no_CB_string += "\t},\n"
    weights_no_CB_string += "};\n\n"

    return words_per_k_def_string, weights_no_CB_string, raw_weights







# def gen_non_cb_weights_string(layer_ID, k_values):

#     weights_string = "#define N_WEIGHTS_{}  (N_K_ELEMS_PER_CHANNEL_{} * OUT_CH_{})\n\n".format(layer_ID, layer_ID, layer_ID)
#     weights_string += "static float non_cb_weights_{}[N_LEARNERS][N_WEIGHTS_{}] = {{\n".format(layer_ID, layer_ID)

#     for ens in k_values:
#         weights_string += "\t{\n"

#         for k in ens:
#             weights_string += "\t\t{},\n".format(k)

#         weights_string += "\t},\n"

#     weights_string += "};\n" 
#     return weights_string





def gen_biases_strings(layer_ID, n_learners, out_ch, use_f16):

    bias_string = "static float{} bias_{} [N_LEARNERS][OUT_CH_{}] = {{\n"

    # In case there are more than 4 learners, group them 4 by 4
    if n_learners <= 4:
        bias_string_interleaved = "static float{} bias_interleaved_{}[N_LEARNERS * OUT_CH_{}] = {{\n"
    else:
        bias_string_interleaved = "static float{} bias_interleaved_{}[GROUPS_OF_4_LEARNERS][N_LEARNERS * OUT_CH_{}] = {{\n"



    if not use_f16:
        bias_string = bias_string.format("", layer_ID, layer_ID)
        bias_string_interleaved = bias_string_interleaved.format("", layer_ID, layer_ID)
    else:
        bias_string = bias_string.format("16_t", layer_ID, layer_ID)
        bias_string_interleaved = bias_string_interleaved.format("16_t", layer_ID, layer_ID)


    bias_ensembles = [] # Holds all the bias of the learners

    # Generate the codebook for all the learners
    for i in range(n_learners):
        bias_ensembles.append(np.random.uniform(0.0, 0.9, out_ch))

    # Fill the bias string in C format
    for bias_learn in bias_ensembles:
        bias_string += "\t{\n"
        for v in bias_learn:
            bias_string += "\t\t{},\n".format(v)
        bias_string += "\t},\n"
    bias_string += "};"


    if n_learners <= 4:
        for i in range(out_ch):
            for ens in range(n_learners):
                bias_string_interleaved += "\t{},\n".format(bias_ensembles[ens][i])
        bias_string_interleaved += "};"
    else:
        for group in range(int(n_learners / 4)):
            bias_string_interleaved += "\t{\n"
            for i in range(out_ch):
                for ens in range(4):
                    bias_string_interleaved += "\t\t{},\n".format(bias_ensembles[(group * 4) + ens][i])
            bias_string_interleaved += "\t},\n"
        bias_string_interleaved += "};"


    return (bias_string, bias_string_interleaved), bias_ensembles


#######################################################
# Generates the indexes and the values of the BIASES  #
#######################################################
def generate_biases_indexes(codebooks_ensembles, n_learners, out_ch, codebook_size):
    biases_values = [[] for _ in range(n_learners)]      # Raw biases values (one list per learner)
    biases_indexes = []     # biases indexs (one list per learner)

    for i in range(out_ch):
        new_idx = np.random.randint(0, codebook_size)       # New index
        biases_indexes.append(new_idx)

        for ens in range(n_learners):
            new_val = codebooks_ensembles[ens][new_idx]     # New value from each codebook
            biases_values[ens].append(new_val)

    return biases_values, biases_indexes    # Return both values and indexes


##################################################################################
# Formats the kernel indexes in COMPACT form, keeping them separated per channel # 
##################################################################################

def generate_biases_indexes_string(layer_ID, codebook_size, in_ch, out_ch, k_size, n_indexes):

    _, idxs_bits, _, idx_per_word = compute_CB_parameters(codebook_size, in_ch, out_ch, k_size, words_bitlen=32)

    biases_indexes_string = "#define N_WORDS_BIASES_PER_CH_{}  {}\n\n".format(layer_ID, math.ceil(out_ch / idx_per_word))
    biases_indexes_string += "static const uint32_t biases_compact_indexes_{}[N_WORDS_BIASES_PER_CH_{}] = {{\n".format(layer_ID, layer_ID)

    # Format the indexes in C, the indexes of a single channels are isolated from the ones of a different channel
    idxs_inserted = 0   # Counts the indexes inserted in a word

    word_of_indexes = ""

    # Format the indexes in rows of words.
    # The indexes are separated per each channel.
    for i, idx in enumerate(n_indexes):
        bin_idx = to_bin_N_digits(idx, N_digits=idxs_bits)

        # if channel_idxs_inserted < idx_per_channel:
        if idxs_inserted < idx_per_word:
            word_of_indexes = bin_idx + word_of_indexes
            idxs_inserted += 1
        else:
            word_of_indexes = "0b" + word_of_indexes
            biases_indexes_string += "\t{}, ".format(word_of_indexes)
            idxs_inserted = 0
            word_of_indexes = bin_idx
            idxs_inserted += 1
            
        # channel_idxs_inserted += 1
        # else:
        #     word_of_indexes = "0b" + word_of_indexes
        #     binary_kernel_string_perCH += "\t{},\n".format(word_of_indexes)
        #     idxs_inserted = 0
        #     channel_idxs_inserted = 0
        #     word_of_indexes = bin_idx
        #     idxs_inserted += 1
        #     channel_idxs_inserted += 1

    word_of_indexes = "0b" + word_of_indexes
    biases_indexes_string += "\t{},\n".format(word_of_indexes)
    biases_indexes_string += "};\n\n"

    return biases_indexes_string