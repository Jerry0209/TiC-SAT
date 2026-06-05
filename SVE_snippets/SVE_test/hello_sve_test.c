#include <stdio.h>
#include <arm_sve.h>

void print_vect_f32(svfloat32_t v){
    int lanes = svcntw();
    float buf[lanes];

    svbool_t pg = svwhilelt_b32((uint64_t)0, svcntw());

    svst1_f32(pg, buf, v);

    for(int i = 0; i < svcntw(); i++){
        printf("[%d]\t%f\n", i, buf[i]);
    }
    printf("\n");
}

void print_predicate_w(svbool_t pg) {
    int lanes = svcntw();
    uint32_t buf[lanes];

    for (int i = 0; i < lanes; i++) {
        svint32_t indices = svindex_s32(0, 1);
        svbool_t p = svcmpeq_n_s32(svptrue_b32(), indices, i);
        buf[i] = svptest_any(p, pg) ? 1 : 0;
    }

    printf("Predicate Vector (W):\n");
    for (int i = 0; i < lanes; i++) {
        printf("[%d]\t%d\n", i, buf[i]);
    }
    printf("\n");
}

#define N_ELEMS 10

float input_A[N_ELEMS] = {
    0.30593392294314126, 0.9729489210930778, 0.6967651289479858,
    0.45340254562143867, 0.7722622902191596, 0.4631572250189654,
    0.7058883294981203, 0.8284812740863892, 0.25400633077082535,
    0.7025260272018509
};

float input_B[N_ELEMS] = {
    0.9799316649760836, 0.036416962363738814, 0.38098218215107194,
    0.9862879481587331, 0.37151641123862167, 0.7611771045148974,
    0.3870005424619666, 0.7599391197169654, 0.7615588261600846,
    0.8027245750657497
};

int main() {
    printf("========================\n");
    printf("Hello, SVE World! [Stage2]\n");
    printf("========================\n\n");

    int n_8bits_elems = svcntb();
    int n_16bits_elems = svcnth();
    int n_32bits_elems = svcntw();

    printf("-------------------------\n");
    printf("N 8-bits elements = %d\n", n_8bits_elems);
    printf("N 16-bits elements = %d\n", n_16bits_elems);
    printf("N 32-bits elements = %d\n", n_32bits_elems);
    printf("-------------------------\n\n");

    svbool_t pg = svwhilelt_b32(0, 4);
    print_predicate_w(pg);

    svfloat32_t va;
    svfloat32_t vb;

    va = svld1(pg, input_A);
    printf("Vector va:\n");
    print_vect_f32(va);

    vb = svld1(pg, input_B);
    printf("Vector vb:\n");
    print_vect_f32(vb);

    svfloat32_t add_result = svadd_f32_m(pg, va, vb);
    printf("Adding result:\n");
    print_vect_f32(add_result);

    printf("\n-------------------------\n");
    printf("Starting the loop....\n");
    printf("-------------------------\n\n");

    int loop_cnt = 0;
    svfloat32_t mult_result = svdup_f32(0.0);

    for(size_t i = 0; i < N_ELEMS; i += n_32bits_elems){
        printf("\nIteration [%d]  |  number of processed elements = %d / %d\n",
               loop_cnt, i, N_ELEMS);

        pg = svwhilelt_b32((int32_t)i, N_ELEMS);
        print_predicate_w(pg);

        va = svld1(pg, &input_A[i]);
        vb = svld1(pg, &input_B[i]);

        mult_result = svmad_f32_x(pg, va, vb, mult_result);

        loop_cnt++;
    }

    printf("Vector result of the MAC:\n");
    print_vect_f32(mult_result);

    pg = svwhilelt_b32(0, N_ELEMS);
    float final_mac_result = svaddv_f32(pg, mult_result);

    printf("\nFINAL MAC RESULT = %f\n", final_mac_result);

    return 0;
}