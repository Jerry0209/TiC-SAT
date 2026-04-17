
#include <stdio.h>
#include <conv_def.h>
#include <stdlib.h>

void print_volume(float *vol, dim3D_t *dims){

    printf("---------------\n");
    printf("%d %d %d\n", dims->depth, dims->height, dims->width);
    for(int d=0; d<dims->depth; d++){
        for(int i=0; i<dims->height; i++){
            for(int j=0; j<dims->width; j++){
                // printf("[%d -> %d -> %d = %d] %f, ",    (d*dims->height*dims->width),
                //                                         (i*dims->width),
                //                                         j, 
                //                                         (d*dims->height*dims->width) + (i*dims->width) + j,
                //                                         vol[(d*dims->height*dims->width) + (i*dims->width) + j]);
                printf("%f, ", vol[(d*dims->height*dims->width) + (i*dims->width) + j]);
            }
            printf("\n");
        }
        printf("\n");
        // break;
    }
    printf("---------------\n");

}



void print_volume_f16(float16_t *vol, dim3D_t *dims){

    printf("---------------\n");
    printf("%d %d %d\n", dims->depth, dims->height, dims->width);
    for(int d=0; d<dims->depth; d++){
        for(int i=0; i<dims->height; i++){
            for(int j=0; j<dims->width; j++){
                // printf("[%d -> %d -> %d = %d] %f, ",    (d*dims->height*dims->width),
                //                                         (i*dims->width),
                //                                         j, 
                //                                         (d*dims->height*dims->width) + (i*dims->width) + j,
                //                                         vol[(d*dims->height*dims->width) + (i*dims->width) + j]);
                printf("%f, ", vol[(d*dims->height*dims->width) + (i*dims->width) + j]);
            }
            printf("\n");
        }
        printf("\n");
        // break;
    }
    printf("---------------\n");

}



void print_interleaved_out(tensor3D_t *out_interleaved, uint8_t n_interleaved){

    int nelems = out_interleaved->dim.depth * out_interleaved->dim.height * (out_interleaved->dim.width / n_interleaved);
    float *out_l = (float*)malloc(nelems * sizeof(float));

    printf("out l dim: %d\n", nelems);

    dim3D_t single_learner_dim_out = {
        .depth = out_interleaved->dim.depth,
        .height = out_interleaved->dim.height,
        .width = (out_interleaved->dim.width / n_interleaved)
    };

    for(int i=0; i<n_interleaved; i++){

        printf("\n------ Interl %d\n", i);

        for(int n=0; n<nelems; n++){
            out_l[n] = out_interleaved->tensor[(n * n_interleaved) + i];
        }

        print_volume(out_l, &single_learner_dim_out);
        // break;

    }

    free(out_l);
}


void print_interleaved_out_f16(tensor3D_f16_t *out_interleaved, uint8_t n_interleaved){

    int nelems = out_interleaved->dim.depth * out_interleaved->dim.height * (out_interleaved->dim.width / n_interleaved);
    float16_t *out_l = (float16_t*)malloc(nelems * sizeof(float16_t));

    printf("out l dim: %d\n", nelems);

    dim3D_t single_learner_dim_out = {
        .depth = out_interleaved->dim.depth,
        .height = out_interleaved->dim.height,
        .width = (out_interleaved->dim.width / n_interleaved)
    };

    for(int i=0; i<n_interleaved; i++){

        for(int n=0; n<nelems; n++){
            out_l[n] = out_interleaved->tensor[(n * n_interleaved) + i];
        }

        print_volume_f16(out_l, &single_learner_dim_out);
        break;

    }

    free(out_l);
}




/**
 * Merges the 2 input arrays in an interleaved way.
 */
void merge_interleaved_2D(float *in0, float *in1, uint32_t len, float *out){

    for(uint32_t i=0; i<len; i++){
        out[(i*2) +  0] = in0[i];
        out[(i*2) +  1] = in1[i];
    }
}



/**
 * Merges the 4 input arrays in an interleaved way.
 */
void merge_interleaved_4D(float *in0, float *in1, float *in2, float *in3, uint32_t len, float *out){

    for(uint32_t i=0; i<len; i++){
        out[(i*4) +  0] = in0[i];
        out[(i*4) +  1] = in1[i];
        out[(i*4) +  2] = in2[i];
        out[(i*4) +  3] = in3[i];
    }
}

void merge_interleaved_4D_uint32(uint32_t *in0, uint32_t *in1, uint32_t *in2, uint32_t *in3, uint32_t len, uint32_t *out){

    for(uint32_t i=0; i<len; i++){
        out[(i*4) +  0] = in0[i];
        out[(i*4) +  1] = in1[i];
        out[(i*4) +  2] = in2[i];
        out[(i*4) +  3] = in3[i];
    }
}



void merge_interleaved_4D_uint16(uint16_t *in0, uint16_t *in1, uint16_t *in2, uint16_t *in3, uint16_t len, uint16_t *out){

    for(uint16_t i=0; i<len; i++){
        out[(i*4) +  0] = in0[i];
        out[(i*4) +  1] = in1[i];
        out[(i*4) +  2] = in2[i];
        out[(i*4) +  3] = in3[i];
    }
}



void merge_interleaved_4D_f16(float16_t *in0, float16_t *in1, float16_t *in2, float16_t *in3, uint32_t len, float16_t *out){

    for(uint32_t i=0; i<len; i++){
        out[(i*4) +  0] = in0[i];
        out[(i*4) +  1] = in1[i];
        out[(i*4) +  2] = in2[i];
        out[(i*4) +  3] = in3[i];
    }
}