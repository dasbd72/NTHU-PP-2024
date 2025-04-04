#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    // cpu_set_t cpuset;
    // sched_getaffinity(0, sizeof(cpuset), &cpuset);
    // unsigned long long ncpus = CPU_COUNT(&cpuset);

    for (unsigned long long x = 0; x < r; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        pixels += y;
        pixels %= k;
    }
    printf("%llu\n", (4 * pixels) % k);
}
