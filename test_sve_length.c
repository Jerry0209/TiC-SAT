#include <arm_sve.h>
#include <stdio.h>

int
main(void)
{
    const size_t sve_bytes = svcntb();
    const size_t sve_int32_lanes = svcntw();

    printf("SVE length = %zu bytes = %zu bits\n",
           sve_bytes, sve_bytes * 8);
    printf("float lanes = %zu, int32 lanes = %zu\n",
           sve_int32_lanes, sve_int32_lanes);

    return 0;
}
