#include <stdio.h>
#include <arm_sve.h>



void print_vect_f32(svfloat32_t v){
    int lanes = svcntw();
    float buf[lanes];

    svbool_t pg = svwhilelt_b32((uint64_t)0, svcntw());

    svst1_f32(pg, buf, v);

    for(int i=0; i<svcntw(); i++){
        printf("[%d]\t%f\n", i, buf[i]);
    }
    printf("\n");
}


void print_predicate_w(svbool_t pg) {
    int lanes = svcntw();  // Get the number of lanes in the predicate vector (in bytes)
    uint32_t buf[lanes];

    // Iterate over the lanes of the predicate vector
    for (int i = 0; i < lanes; i++) {
        // Each lane has an incrementing index (0, 1, 2, ...)
        svint32_t indices = svindex_s32(0, 1); // svindex_s32(base, stride): incrementing indices for each lane (0, 1, 2, ...) 
        // Compare indices and i element wise (activate only the lane with the same index as the loop counter)
        svbool_t p = svcmpeq_n_s32(svptrue_b32(), indices, i); // svptrue_b32() creates a predicate with all lanes active, svcmpeq_n_s32 compares the indices with the loop counter i, returning a predicate with only the lane that matches i active (1) and the rest inactive (0)
        // i = 0
        // [0==0, 1==0, 2==0, 3==0]
        // → [1, 0, 0, 0]

        // Test if the lane is active (1) or inactive (0)
        buf[i] = svptest_any(p, pg) ? 1 : 0; // Compare the predicate p with the input predicate pg, if the lane is active in both p and pg, set buf[i] to 1, otherwise set it to 0
    }

    // Print the predicate vector
    printf("Predicate Vector (W):\n");
    for (int i = 0; i < lanes; i++) {
        printf("[%d]\t%d\n", i, buf[i]);  // Print each lane (1 for active, 0 for inactive)
    }
    printf("\n");
}





#define N_ELEMS 10

float input_A[N_ELEMS] = {0.30593392294314126, 0.9729489210930778, 0.6967651289479858, 0.45340254562143867, 0.7722622902191596, 0.4631572250189654, 0.7058883294981203, 0.8284812740863892, 0.25400633077082535, 0.7025260272018509};
float input_B[N_ELEMS] = {0.9799316649760836, 0.036416962363738814, 0.38098218215107194, 0.9862879481587331, 0.37151641123862167, 0.7611771045148974, 0.3870005424619666, 0.7599391197169654, 0.7615588261600846, 0.8027245750657497};

int main() {

    printf("========================\n");
    printf("Hello, SVE World!\n");
    printf("========================\n\n");

    // Count the number of elements (bytes / half-words / words) in a vector-register
    int n_8bits_elems = svcntb();
    int n_16bits_elems = svcnth();
    int n_32bits_elems = svcntw();

    printf("-------------------------\n");
    printf("N 8-bits elements = %d\n", n_8bits_elems);
    printf("N 16-bits elements = %d\n", n_16bits_elems);
    printf("N 32-bits elements = %d\n", n_32bits_elems);
    printf("-------------------------\n\n");


    // Create a PREDICATE vector register //
    svbool_t pg = svwhilelt_b32(0, 4); // While less than 4 (activate the first 4 lanes), each lane is 32 bits (4 bytes) wide
    print_predicate_w(pg);


    // Declare two vector registers with float32 elements
    svfloat32_t va;
    svfloat32_t vb;

    // Load from input_A and input_B some elements (basedon on predicate pg) inside the va and vb registers
    va = svld1(pg, input_A);
    printf("Vector va:\n");
    print_vect_f32(va);

    vb = svld1(pg, input_B);
    printf("Vector vb:\n");
    print_vect_f32(vb);

    // Adding them 
    svfloat32_t add_result = svadd_f32_m(pg, va, vb); // merge the two vectors based on the predicate pg, if the lane is active (1) take the sum of the two vectors, otherwise (0) take the value from the first vector (va)
    printf("Adding result:\n");
    print_vect_f32(add_result);


    printf("\n-------------------------\n");
    printf("Starting the loop....\n");
    printf("-------------------------\n\n");

    int loop_cnt = 0;

    // Result of multiplication, initilize to 0.0
    svfloat32_t mult_result = svdup_f32(0.0);


    // Add the two vectors
    for(size_t i=0; i<N_ELEMS; i+=n_32bits_elems){

        printf("\nIteration [%d]  |  number of processed elements = %d / %d\n", loop_cnt, i, N_ELEMS);

        pg = svwhilelt_b32((int32_t)i, N_ELEMS);
        print_predicate_w(pg);

        // Load the elements from the arrays
        va = svld1(pg, &input_A[i]); // Load the elements from input_A starting at index i into the vector register va, based on the predicate pg (which determines how many lanes to activate)
        vb = svld1(pg, &input_B[i]);

        // Multiply them and accumulate the result in `mult_result`
        mult_result = svmad_f32_x(pg, va, vb, mult_result); // mult_result = (va * vb) + mult_result, but only for the active lanes determined by the predicate pg, for inactive lanes, keep the value in mult_result unchanged

        loop_cnt++;
    }

    printf("Vector result of the MAC:\n");
    print_vect_f32(mult_result);

    // Accumulate the final result by adding the elements in the vector register
    pg = svwhilelt_b32(0, N_ELEMS);
    float final_mac_result = svaddv_f32(pg, mult_result); //add each element in the vector register mult_result based on the predicate pg (which determines how many lanes to activate), and return the final result as a single float value

    printf("\nFINAL MAC RESULT = %f\n", final_mac_result);



    return 0;
}


// conda activate gem5_env
// SYSROOT=$(aarch64-conda-linux-gnu-g++ -print-sysroot)
	
		
// aarch64-conda-linux-gnu-g++ \
// -x c++ -std=c++17 -O2 -Wall -march=armv8-a+sve \
// hello_sve.c \
// -o /tmp/hello_sve


// aarch64-conda-linux-gnu-gcc \
// -O2 -Wall -march=armv8-a+sve \
// hello_sve.c \
// -o /tmp/hello_sve
		
		
// $HOME/opt/qemu-sve/bin/qemu-aarch64 \
// -cpu max,sve=on,sve-default-vector-length=16 \
// -L "$SYSROOT" \
// /tmp/hello_sve \
// < /tmp/hello_sve
