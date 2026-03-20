#include <stdio.h>
#include <stdlib.h>

#include <SVE_implementations.h>


#ifdef LENET
#include <LeNet.h>
#endif

#ifdef ALEXNET
#include <AlexNet.h>
#endif

#ifdef VGG16
#include <VGG16.h>
#endif

#ifdef RESNET18
#include <ResNet18.h>
#endif

#ifdef TRANSFORMER
#include <Transformer.h>
#endif


#include <print_functs.h>


// #define LEARN_BY_LEARN
// #define LEARN_BY_LEARN_TILED_L2
// #define TOGETHER_NO_SIMD_L2L1
// #define INTERL_2D
// #define INTERL_4D
// #define INTERL_4D_TILED_L2L1_CBMEM


// #define LEARN_BY_LEARN_TILED_L2_DIFF_SEQ
// #define LEARN_BY_LEARN_TILED_L2L1
// #define INTERL_4D_TILED_L2L1_NOLANESLOOP



///////////////////
// RELEVANT ONES //
///////////////////

// #define LEARN_BY_LEARN_TILED_L2L1_NO_CB
#define LEARN_BY_LEARN_TILED_L2L1_DIFF_SEQ
// #define LEARN_BY_LEARN_TILED_L2L1_SIMD   // Both f16 or f32
// #define INTERL_4D_TILED_L2L1_DIFF_SEQ
// #define INTERL_ND_TILED_L2L1
// #define INTERL_4D_TILED_F16

// #define INTERL_4D_TILED_L2L1_DIFF_SEQ_MERGED


#define START_ITERS 0
#define ITERATIONS  1

// float16_t val0 = 10.1;
// float16_t val1 = 0.9;

// float16_t val2 = 4.6;
// float16_t val3 = 1.4;

// float16_t vals[8] = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};

int main(){


    // tensor3D_f16_t res;
    // float16_t *t = (float16_t*)malloc(10 * sizeof(float16_t));
    // res.tensor = t;

    // float16_t current_val = res.tensor[0];
    // printf("0\n");
    // svfloat16_t row_res_vect0 = svld1_f16(svwhilelt_b16(0, CB_SIZE), vals);
    // float16_t tmp0 = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
    // printf("1\n");
    // float16_t new_val = current_val + tmp0;              // Perform addition
    // printf("2\n");
    // res.tensor[0] = new_val;
    // printf("3\n");

    system("m5 checkpoint");

    //////////////////////////////////////////
    //      Learner by learner              //
    //////////////////////////////////////////

#ifdef LEARN_BY_LEARN

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner();
        #endif

        #ifdef TRANSFORMER
        exec_Transformer_learner_by_learner();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner();
        #endif

        #ifdef TRANSFORMER
        exec_Transformer_learner_by_learner();
        #endif
    }

    system("m5 exit");

    
#endif



    //////////////////////////////////////////
    //      Learner by learner (TILED)      //
    //////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2();
        #endif
    }

    system("m5 exit");

    
#endif




    ///////////////////////////////////////////////////////////
    //      Learner by learner (TILED) - diff sequences      //
    ///////////////////////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2_DIFF_SEQ

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2_diff_seq();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2_diff_seq();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2_diff_seq();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2_diff_seq();
        #endif
    }

    system("m5 exit");

    
#endif





    //////////////////////////////////////////
    //      Learner by learner (TILED)      //
    //////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2L1_NO_CB

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        printf("Not yet implemented! Exiting...");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1_no_CB();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        printf("Not yet implemented! Exiting...");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1_no_CB();
        #endif
    }

    system("m5 exit");

    
#endif







    //////////////////////////////////////////
    //      Learner by learner (TILED)      //
    //////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2L1

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1();
        #endif
    }

    system("m5 exit");

    
#endif






    ///////////////////////////////////////////////
    //      Learner by learner (TILED + DIFF SEQ)//
    ///////////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2L1_DIFF_SEQ

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef VGG16
        exec_VGG16_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef RESNET18
        exec_ResNet18_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef TRANSFORMER
        exec_Transformer_learner_by_learner_tiled_l2l1_diff_seq();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef VGG16
        exec_VGG16_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef RESNET18
        exec_ResNet18_learner_by_learner_tiled_l2l1_diffSeq();
        #endif

        #ifdef TRANSFORMER
        exec_Transformer_learner_by_learner_tiled_l2l1_diff_seq();
        #endif
    }

    system("m5 exit");

    
#endif







    /////////////////////////////////////////////
    //      Learner by learner (TILED) + SIMD  //
    /////////////////////////////////////////////

    #ifdef LEARN_BY_LEARN_TILED_L2L1_SIMD

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET

        #ifdef USE_F16
        exec_AlexNet_learner_by_learner_tiled_l2l1_SVE_f16();
        #else
        exec_AlexNet_learner_by_learner_tiled_l2l1_SVE();
        #endif

        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET

        #ifdef USE_F16
        exec_AlexNet_learner_by_learner_tiled_l2l1_SVE_f16();
        #else
        exec_AlexNet_learner_by_learner_tiled_l2l1_SVE();
        #endif

        #endif
    }

    system("m5 exit");

    
#endif









    //////////////////////////////////////////
    //      Together no SIMD (tile L2L1)    //
    //////////////////////////////////////////

    #ifdef TOGETHER_NO_SIMD_L2L1

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_together_diff_seq_l2l1();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        // exec_LeNet5_learner_by_learner_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_together_diff_seq_l2l1();
        #endif
    }

    system("m5 exit");

    
#endif



    //////////////////////////////////////////
    //      Interleaved 2D                  //
    //////////////////////////////////////////

#ifdef INTERL_2D    

    if(N_LEARNERS != 2){
        printf("ERROR: Interleaved 2D: N_LEARNERS != 2\n");
        printf("Exiting...\n");
        exit(1);
    }

 
    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_interleaved_2D();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_2D();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_interleaved_2D();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_2D();
        #endif
    }

    system("m5 exit");

#endif


    //////////////////////////////////////////
    //      Interleaved 4D                  //
    //////////////////////////////////////////

#ifdef INTERL_4D    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D();
        #endif
    }

    system("m5 exit");

#endif





    //////////////////////////////////////////
    //      Interleaved ND (TILED)          //
    //////////////////////////////////////////

#ifdef INTERL_ND_TILED_L2L1

    switch (N_LEARNERS)
    {
    case 2:
        //....................//
        //  N_LEARNERS == 2   //
        //....................//

        for(int si=0; si<START_ITERS; si++){

            #ifdef LENET
            printf("LeNet Not implemented!\n");
            // exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_2D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_2D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_2D_tiled_l2l1();
            #endif
        }

        system("m5 resetstats");

        for(int it=0; it<ITERATIONS; it++){

            #ifdef LENET
            printf("LeNet Not implemented!\n");
            // exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_2D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_2D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_2D_tiled_l2l1();
            #endif
        }
        break;
    
    case 4:
        //....................//
        //  N_LEARNERS == 4   //
        //....................//

        for(int si=0; si<START_ITERS; si++){

            #ifdef LENET
            exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_4D_tiled_l2l1();
            #endif
        }


        system("m5 resetstats");

        for(int it=0; it<ITERATIONS; it++){

            #ifdef LENET
            exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_4D_tiled_l2l1();
            #endif
        }
        break;
    
    case 8:
        //....................//
        //  N_LEARNERS == 8   //
        //....................//

        for(int si=0; si<START_ITERS; si++){

            #ifdef LENET
            printf("LeNet Not implemented!\n");
            // exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_8D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_8D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_8D_tiled_l2l1();
            #endif
        }


        system("m5 resetstats");

        for(int it=0; it<ITERATIONS; it++){

            #ifdef LENET
            printf("LeNet Not implemented!\n");
            // exec_LeNet5_interleaved_4D_tiled_l2l1();
            #endif

            #ifdef ALEXNET
            exec_AlexNet_interleaved_8D_tiled_l2l1();
            #endif

            #ifdef VGG16
            exec_VGG16_interleaved_8D_tiled_l2l1();
            #endif

            #ifdef RESNET18
            exec_ResNet18_interleaved_8D_tiled_l2l1();
            #endif
        }
        break;
    
    default:
        printf("ERROR: Interleaved ND: N_LEARNERS = %d not implemented! Exiting...\n", N_LEARNERS);
        exit(1);
        break;
    }
 

    system("m5 exit");

#endif





    //////////////////////////////////////////
    //      Interleaved 4D (TILED)          //
    //////////////////////////////////////////

#ifdef INTERL_4D_TILED_L2L1_NOLANESLOOP    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_l2l1_no_lanes_loop();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_l2l1_no_lanes_loop();
        #endif
    }

    system("m5 exit");

#endif






    //////////////////////////////////////////
    //      Interleaved 4D (TILED)          //
    //////////////////////////////////////////

#ifdef INTERL_4D_TILED_L2L1_CBMEM    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_l2l1_mem();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_l2l1_mem();
        #endif
    }

    system("m5 exit");

#endif

    //////////////////////////////////////////////
    //      Interleaved 4D (TILED + QUANTIZED)  //
    //////////////////////////////////////////////

#ifdef INTERL_4D_TILED_F16    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_f16();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_f16();
        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        exec_LeNet5_interleaved_4D_tiled_f16();
        #endif

        #ifdef ALEXNET
        exec_AlexNet_interleaved_4D_tiled_f16();
        #endif
    }

    system("m5 exit");

#endif




    //////////////////////////////////////////
    //      Interleaved 4D (TILED) DIFF SEQ //
    //////////////////////////////////////////

#ifdef INTERL_4D_TILED_L2L1_DIFF_SEQ    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET

        #ifdef USE_F16
        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq_f16();
        #else
        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif

        #ifdef VGG16

        #ifdef USE_F16
        printf("ERROR : VGG 16 not implemented in f16!");
        exit(1);
        #else
        exec_VGG16_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif

        #ifdef RESNET18

        #ifdef USE_F16
        printf("ERROR : VGG 16 not implemented in f16!");
        exit(1);
        #else
        exec_ResNet18_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif
    }


    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef LENET
        printf("Not implemented!\n");
        exit(1);
        exec_LeNet5_interleaved_4D_tiled_l2l1();
        #endif

        #ifdef ALEXNET

        #ifdef USE_F16
        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq_f16();
        #else
        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif

        #ifdef VGG16

        #ifdef USE_F16
        printf("ERROR : VGG 16 not implemented in f16!");
        exit(1);
        #else
        exec_VGG16_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif

        #ifdef RESNET18

        #ifdef USE_F16
        printf("ERROR : VGG 16 not implemented in f16!");
        exit(1);
        #else
        exec_ResNet18_interleaved_4D_tiled_l2l1_diff_seq();
        #endif

        #endif
    }

    system("m5 exit");

#endif









    //////////////////////////////////////////
    //      Interleaved 4D (TILED) DIFF SEQ //
    //////////////////////////////////////////

#ifdef INTERL_4D_TILED_L2L1_DIFF_SEQ_MERGED    
 
    if(N_LEARNERS != 4){
        printf("ERROR: Interleaved 4D: N_LEARNERS != 4\n");
        printf("Exiting...\n");
        exit(1);
    }

    for(int si=0; si<START_ITERS; si++){

        #ifdef ALEXNET

        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq_merged();

        #endif
    }

    printf("MERGED\n");

    system("m5 resetstats");

    for(int it=0; it<ITERATIONS; it++){

        #ifdef ALEXNET

        exec_AlexNet_interleaved_4D_tiled_l2l1_diff_seq_merged();

        #endif
    }

    system("m5 exit");

#endif




    return 0;
}

