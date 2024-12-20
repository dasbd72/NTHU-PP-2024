#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <ans_file> <out_file> [eps]\n", argv[0]);
        return 1;
    }
    char *ans_file_path = argv[1];
    char *out_file_path = argv[2];
    float eps = 1e-3;
    if (argc >= 4) {
        eps = atof(argv[3]);
    }

    FILE *ans_file = fopen(ans_file_path, "rb");
    if (ans_file == NULL) {
        printf("Cannot open file %s\n", ans_file_path);
        return 1;
    }
    FILE *out_file = fopen(out_file_path, "rb");
    if (out_file == NULL) {
        printf("Cannot open file %s\n", out_file_path);
        return 1;
    }

    // Check file size
    int num_floats = 0;
    fseek(ans_file, 0, SEEK_END);
    num_floats = ftell(ans_file) / sizeof(float);

    fseek(out_file, 0, SEEK_END);
    if (num_floats != int((float)ftell(out_file) / sizeof(float))) {
        printf("Different number of floats\n");
        return 1;
    }
    printf("Number of floats: %d\n", num_floats);

    // Check file content
    fseek(ans_file, 0, SEEK_SET);
    fseek(out_file, 0, SEEK_SET);
    float *ans = (float *)malloc(num_floats * sizeof(float));
    float *out = (float *)malloc(num_floats * sizeof(float));
    fread(ans, sizeof(float), num_floats, ans_file);
    fread(out, sizeof(float), num_floats, out_file);
    int diff_count = 0;
    for (int i = 0; i < num_floats; i++) {
        if (abs(ans[i] - out[i]) > eps) {
            if (diff_count < 10) {
                printf("Different value at index %d, ans: %f, out: %f\n", i, ans[i], out[i]);
            }
            diff_count++;
        }
    }

    if (diff_count == 0) {
        printf("All values are the same\n");
    } else {
        printf("Number of different values: %d\n", diff_count);
    }

    free(ans);
    free(out);
    return 0;
}