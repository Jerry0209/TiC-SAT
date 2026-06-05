#ifndef _LENET_CONV_${ID}_H_
#define _LENET_CONV_${ID}_H_


#include <inttypes.h>
#include <arm_sve.h>
#include <codebooks_def.h>

#define IN_CH_${ID}     ${in_ch}
#define OUT_CH_${ID}	${out_ch}

#define STRIDE_${ID}    ${stride}
#define PADDING_${ID}	${padding}

#define N_LEARNERS  ${n_learners}

#define PADDED_WIDTH_${ID}	(IN_W + (2 * PADDING_${ID}))
#define PADDED_HEIGHT_${ID} 	(IN_H + (2 * PADDING_${ID}))
#define N_ELEMS_SLICE_${ID}	(PADDED_WIDTH_${ID} * PADDED_HEIGHT_${ID})	// How many elements are in a depth-wise slice


#define TILE_L2_SIZE    ${tile_l2_size} // Size of the L2 tile
#define TILE_L1_SIZE    ${tile_l1_size} // Size of the L1 tile


#define K_D_${ID} IN_CH_${ID}
#define K_H_${ID} ${k_size}
#define K_W_${ID} ${k_size}

#define N_K_ELEMS_PER_CHANNEL_${ID}	(K_D_${ID} * K_H_${ID} * K_W_${ID})	// How many elements are in a channel of the kernel

// Number of words of compact indexes per kernel channel
#define N_WORDS_IDX_PER_CH_${ID}	(((N_K_ELEMS_PER_CHANNEL_${ID}) + (IDXS_PER_WORD) - 1) / (IDXS_PER_WORD))


${codebook_string}


${codebook_string_interleaved}



${kernel_indexes_perCH_string}

${kernel_indexes_tiled_string}


${non_cb_weights_string}


${bias_strings}


#endif