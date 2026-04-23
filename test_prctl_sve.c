#include <stdio.h>
#include <sys/prctl.h>
#include <errno.h>
#include <string.h>

#ifndef PR_SVE_GET_VL
#define PR_SVE_SET_VL 50
#define PR_SVE_GET_VL 51
#endif

int main() {
    int vl = prctl(PR_SVE_GET_VL);
    if (vl < 0) {
        printf("PR_SVE_GET_VL failed: errno=%d (%s)\n", errno, strerror(errno));
        return 1;
    }
    printf("SVE VL reported by kernel: %d\n", vl);
    return 0;
}