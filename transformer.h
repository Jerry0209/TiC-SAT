//
// Created by alireza on 10/25/21.
//

#ifndef FVLLMONTITRANSFORMER_TRANSFORMER_H
#define FVLLMONTITRANSFORMER_TRANSFORMER_H

#ifdef DEBUG_SMALL_MODEL

// Extra Small Menu (Pass)
// #define D_Q 4
// #define D_SEQ 2
// #define D_MODEL 8
// #define NUM_HEAD 2
// #define D_FF 4

// Smaller Smaller Menu (Match notebook)
// #define D_Q 4
// #define D_SEQ 2
// #define D_MODEL 8
// #define NUM_HEAD 2
// #define D_FF 8

// Smaller Menu (Pass)
// #define D_Q 4
// #define D_SEQ 4
// #define D_MODEL 8
// #define NUM_HEAD 2
// #define D_FF 4

// Small Menu (Match notebook)
// #define D_Q 4
// #define D_SEQ 8
// #define D_MODEL 8
// #define NUM_HEAD 2
// #define D_FF 16

// Medium Menu (Match notebook)
#define D_Q 8
#define D_SEQ 8
#define D_MODEL 16
#define NUM_HEAD 2
#define D_FF 32

// Large Menu (Pass)
// #define D_Q 16
// #define D_SEQ 16
// #define D_MODEL 32
// #define NUM_HEAD 2
// #define D_FF 4

#else
#define D_Q 64
#define D_SEQ 512
#define D_MODEL 768
#define NUM_HEAD 12
#define D_FF 3072
#endif

#endif //FVLLMONTITRANSFORMER_TRANSFORMER_H
