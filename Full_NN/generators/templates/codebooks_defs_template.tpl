#ifndef _CODEBOOKS_DEF_H_
#define _CODEBOOKS_DEF_H_

#define N_LEARNERS      ${n_learners}

#define N_SVE_LANES     ${sve_size}
#define N_SVE_HALF      ${sve_halfwords}
#define N_SVE_BYTE      ${sve_bytes}

#define CB_SIZE         ${codebook_size}
#define TILE_L1_SIZE    ${tile_l1_size}
#define TILE_L2_SIZE    ${tile_l2_size}
#define BITS_PER_CB		${bits_per_codeword}
#define IDX_MASK		a${index_bin_mask}

// Number of indexes packed per each ${word_bitlen}-bits word
#define IDXS_PER_WORD ((${word_bitlen}) / (BITS_PER_CB))
#define IDXS_PER_WORD_16 ((16) / (BITS_PER_CB))


// Number of indexes packed per each vector register
#define IDX_PER_VECT	((N_SVE_LANES) * (IDXS_PER_WORD))

${string_n_regs_per_cb}

#define USE_F${bitwidth_used}   1

${use_bias_def}

#define ${sequence_type}    1

${learners_8_def}

${no_cb_def}

#endif
