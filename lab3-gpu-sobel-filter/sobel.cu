#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define ADJUST_X ((MASK_X % 2) ? 1 : 0)
#define ADJUST_Y ((MASK_Y % 2) ? 1 : 0)
#define MASK_ADJ_X (MASK_X - ADJUST_X)
#define MASK_ADJ_Y (MASK_Y - ADJUST_Y)
#define START_X (MASK_X / 2)
#define START_Y (MASK_Y / 2)
#define BLOCK_X 16
#define BLOCK_Y 8

#define CLAMP_8BIT(x) (x > 255.0 ? 255 : x)

__constant__ short mask[MASK_N][MASK_X][MASK_Y] = {
    {{-1, -4, -6, -4, -1},
     {-2, -8, -12, -8, -2},
     {0, 0, 0, 0, 0},
     {2, 8, 12, 8, 2},
     {1, 4, 6, 4, 1}},
    {{-1, -2, 0, 2, 1},
     {-4, -8, 0, 8, 4},
     {-6, -12, 0, 12, 6},
     {-4, -8, 0, 8, 4},
     {-1, -2, 0, 2, 1}}};

void read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels, unsigned* height_pad, unsigned* width_pad) {
    unsigned char sig[8];
    FILE* fp = fopen(filename, "rb");
    fread(sig, 1, 8, fp);
    assert(png_check_sig(sig, 8));
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);
    *height_pad = (*height % BLOCK_Y == 0) ? *height : (*height / BLOCK_Y + 1) * BLOCK_Y;
    *width_pad = (*width % BLOCK_X == 0) ? *width : (*width / BLOCK_X + 1) * BLOCK_X;
    size_t size = *channels * (*width_pad + MASK_ADJ_X) * (*height_pad + MASK_ADJ_Y) * sizeof(unsigned char);
    *image = (unsigned char*)malloc(size);
    assert(*image);
    for (png_uint_32 i = 0; i < *height; ++i) {
        row_pointers[i] = *image + (i + START_Y) * *channels * (*width_pad + MASK_ADJ_X) + START_X * *channels;
    }
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels, const unsigned height_pad, const unsigned width_pad) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + (i + START_Y) * channels * (width_pad + MASK_ADJ_X) * sizeof(unsigned char) + START_X * channels;
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, unsigned height_pad, unsigned width_pad) {
    int x, y, i, v, u;
    short color[3];
    float val[3], total[3] = {0.0};

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
#pragma unroll 2
    for (i = 0; i < MASK_N; ++i) {
        val[2] = 0.0;
        val[1] = 0.0;
        val[0] = 0.0;

#pragma unroll 5
        for (v = 0; v < MASK_Y; ++v) {
#pragma unroll 5
            for (u = 0; u < MASK_X; ++u) {
                color[2] = s[channels * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 2];
                color[1] = s[channels * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 1];
                color[0] = s[channels * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 0];
                val[2] += color[2] * mask[i][u][v];
                val[1] += color[1] * mask[i][u][v];
                val[0] += color[0] * mask[i][u][v];
            }
        }

        total[2] += val[2] * val[2];
        total[1] += val[1] * val[1];
        total[0] += val[0] * val[0];
    }
    total[2] = sqrtf(total[2]) / SCALE;
    total[1] = sqrtf(total[1]) / SCALE;
    total[0] = sqrtf(total[0]) / SCALE;
    t[channels * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 2] = CLAMP_8BIT(total[2]);
    t[channels * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 1] = CLAMP_8BIT(total[1]);
    t[channels * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 0] = CLAMP_8BIT(total[0]);
}

int main(int argc, char** argv) {
    assert(argc == 3);
    char* input_filename = argv[1];
    char* output_filename = argv[2];

    size_t size;
    unsigned height, width, channels, height_pad, width_pad;
    unsigned char* host_s = NULL;
    unsigned char* host_t = NULL;
    unsigned char* dev_s = NULL;
    unsigned char* dev_t = NULL;
    dim3 blk, grid;

    read_png(input_filename, &host_s, &height, &width, &channels, &height_pad, &width_pad);

    blk = dim3(BLOCK_X, BLOCK_Y);
    grid = dim3(width_pad / BLOCK_X, height_pad / BLOCK_Y);
    size = channels * (width_pad + MASK_ADJ_X) * (height_pad + MASK_ADJ_Y) * sizeof(unsigned char);
    host_t = (unsigned char*)malloc(size);
    cudaMalloc(&dev_s, size);
    cudaMalloc(&dev_t, size);
    cudaMemset(dev_t, 0, size);

    cudaMemcpy(dev_s, host_s, size, cudaMemcpyHostToDevice);
    sobel<<<grid, blk>>>(dev_s, dev_t, height, width, channels, height_pad, width_pad);
    cudaMemcpyAsync(host_t, dev_t, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    write_png(output_filename, host_t, height, width, channels, height_pad, width_pad);

    free(host_s);
    free(host_t);
    cudaFree(dev_s);
    cudaFree(dev_t);
    return 0;
}
