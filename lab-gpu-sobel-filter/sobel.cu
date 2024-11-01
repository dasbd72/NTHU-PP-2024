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
#define SHARED_X (BLOCK_X + MASK_ADJ_X)
#define SHARED_Y (BLOCK_Y + MASK_ADJ_Y)

#define CLAMP_FLOAT2UCHAR(x) (x > 255.0 ? (unsigned char)255U : (unsigned char)x)

#ifdef PROFILING
#include <nvtx3/nvtx3.hpp>
#define NVTX_RANGE_START(arg) \
    nvtxRangePushA(#arg);
#define NVTX_RANGE_END() \
    nvtxRangePop();
#else
#define NVTX_RANGE_START(arg) \
    {}
#define NVTX_RANGE_END() \
    {}
#endif  // PROFILING

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

typedef struct read_png_t {
    // Input
    const char* filename;
    // Output
    size_t size;
    unsigned height;
    unsigned width;
    unsigned height_pad;
    unsigned width_pad;
    // Internal
    FILE* fp;
    png_structp png_ptr;
    png_infop info_ptr;
} read_png_t;

typedef struct write_png_t {
    // Input
    const char* filename;
    size_t size;
    unsigned height;
    unsigned width;
    unsigned height_pad;
    unsigned width_pad;
    // Internal
    FILE* fp;
    png_structp png_ptr;
    png_infop info_ptr;
} write_png_t;

void read_png_init(read_png_t* data) {
    NVTX_RANGE_START(read_png_init);
    unsigned char sig[8];
    data->fp = fopen(data->filename, "rb");
    fread(sig, 1, 8, data->fp);
    assert(png_check_sig(sig, 8));
    data->png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(data->png_ptr);
    data->info_ptr = png_create_info_struct(data->png_ptr);
    assert(data->info_ptr);
    png_init_io(data->png_ptr, data->fp);
    png_set_sig_bytes(data->png_ptr, 8);
    png_read_info(data->png_ptr, data->info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(data->png_ptr, data->info_ptr, &data->width, &data->height, &bit_depth, &color_type, NULL, NULL, NULL);
    png_read_update_info(data->png_ptr, data->info_ptr);
    unsigned channels = (int)png_get_channels(data->png_ptr, data->info_ptr);
    assert(channels == 3);
    data->height_pad = (data->height % BLOCK_Y == 0) ? data->height : (data->height / BLOCK_Y + 1) * BLOCK_Y;
    data->width_pad = (data->width % BLOCK_X == 0) ? data->width : (data->width / BLOCK_X + 1) * BLOCK_X;
    data->size = 3 * (data->width_pad + MASK_ADJ_X) * (data->height_pad + MASK_ADJ_Y) * sizeof(unsigned char);
    NVTX_RANGE_END();
}

void read_png_rows(read_png_t* data, unsigned char* image, unsigned y, unsigned n) {
    NVTX_RANGE_START(read_png_rows);
    png_bytep row_pointers[n];
    for (png_uint_32 i = 0; i < n; ++i) {
        row_pointers[i] = image + (y + i + START_Y) * 3 * (data->width_pad + MASK_ADJ_X) + START_X * 3;
    }
    png_read_rows(data->png_ptr, row_pointers, NULL, n);
    NVTX_RANGE_END();
}

void read_png_end(read_png_t* data) {
    NVTX_RANGE_START(read_png_end);
    png_read_end(data->png_ptr, NULL);
    NVTX_RANGE_END();
}

void read_png_destroy(read_png_t* data) {
    NVTX_RANGE_START(read_png_destroy);
    png_destroy_read_struct(&data->png_ptr, &data->info_ptr, NULL);
    fclose(data->fp);
    NVTX_RANGE_END();
}

void write_png_init(write_png_t* data) {
    NVTX_RANGE_START(write_png_init);
    data->fp = fopen(data->filename, "wb");
    data->png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    data->info_ptr = png_create_info_struct(data->png_ptr);
    png_init_io(data->png_ptr, data->fp);
    png_set_IHDR(data->png_ptr, data->info_ptr, data->width, data->height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(data->png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(data->png_ptr, data->info_ptr);
    png_set_compression_level(data->png_ptr, 0);
    NVTX_RANGE_END();
}

void write_png_rows(write_png_t* data, unsigned char* image, unsigned y, unsigned n) {
    NVTX_RANGE_START(write_png_rows);
    png_bytep row_pointers[n];
    for (int i = 0; i < n; ++i) {
        row_pointers[i] = image + (y + i + START_Y) * 3 * (data->width_pad + MASK_ADJ_X) * sizeof(unsigned char) + START_X * 3;
    }
    png_write_rows(data->png_ptr, row_pointers, n);
    NVTX_RANGE_END();
}

void write_png_end(write_png_t* data) {
    NVTX_RANGE_START(write_png_end);
    png_write_end(data->png_ptr, NULL);
    NVTX_RANGE_END();
}

void write_png_destroy(write_png_t* data) {
    NVTX_RANGE_START(write_png_destroy);
    png_destroy_write_struct(&data->png_ptr, &data->info_ptr);
    fclose(data->fp);
    NVTX_RANGE_END();
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned height_pad, unsigned width_pad) {
#ifdef SOBEL_SMEM_ENABLED
    int txy;
    __shared__ unsigned char shared_s[3 * SHARED_X * SHARED_Y];
    __shared__ short shared_mask[MASK_N][MASK_X][MASK_Y];
#endif  // SOBEL_SMEM_ENABLED
    int x, y, i, v, u;
    short color[3];
    float val[3], total[3] = {0.0};

#ifdef SOBEL_SMEM_ENABLED
    // Set up shared memory
    txy = threadIdx.x * blockDim.y + threadIdx.y;
    // Load source to shared memory
    for (i = txy; i < 3 * SHARED_X * SHARED_Y; i += blockDim.x * blockDim.y) {
        int row = i / (3 * SHARED_X);
        int col = (i % (3 * SHARED_X)) / 3;
        int offset = i % 3;
        shared_s[i] = s[3 * ((width_pad + MASK_ADJ_X) * (blockIdx.y * blockDim.y + row) + (blockIdx.x * blockDim.x + col)) + offset];
    }
    // Load mask to shared memory
    if (txy < MASK_N * (MASK_X * MASK_Y)) {
        shared_mask[txy / (MASK_X * MASK_Y)][txy % (MASK_X * MASK_Y) / MASK_Y][txy % MASK_Y] = mask[txy / (MASK_X * MASK_Y)][txy % (MASK_X * MASK_Y) / MASK_Y][txy % MASK_Y];
    }
    __syncthreads();
#endif  // SOBEL_SMEM_ENABLED

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
#ifdef SOBEL_SMEM_ENABLED
                color[2] = shared_s[3 * (SHARED_X * (threadIdx.y + v) + (threadIdx.x + u)) + 2];
                color[1] = shared_s[3 * (SHARED_X * (threadIdx.y + v) + (threadIdx.x + u)) + 1];
                color[0] = shared_s[3 * (SHARED_X * (threadIdx.y + v) + (threadIdx.x + u)) + 0];
                val[2] += color[2] * shared_mask[i][u][v];
                val[1] += color[1] * shared_mask[i][u][v];
                val[0] += color[0] * shared_mask[i][u][v];
#else
                color[2] = s[3 * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 2];
                color[1] = s[3 * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 1];
                color[0] = s[3 * ((width_pad + MASK_ADJ_X) * (y + v) + (x + u)) + 0];
                val[2] += color[2] * mask[i][u][v];
                val[1] += color[1] * mask[i][u][v];
                val[0] += color[0] * mask[i][u][v];
#endif  // SOBEL_SMEM_ENABLED
            }
        }

        total[2] += val[2] * val[2];
        total[1] += val[1] * val[1];
        total[0] += val[0] * val[0];
    }
    total[2] = sqrtf(total[2]) / SCALE;
    total[1] = sqrtf(total[1]) / SCALE;
    total[0] = sqrtf(total[0]) / SCALE;
    t[3 * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 2] = CLAMP_FLOAT2UCHAR(total[2]);
    t[3 * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 1] = CLAMP_FLOAT2UCHAR(total[1]);
    t[3 * ((width_pad + MASK_ADJ_X) * (y + START_Y) + (x + START_X)) + 0] = CLAMP_FLOAT2UCHAR(total[0]);
}

int main(int argc, char** argv) {
    NVTX_RANGE_START(main);
    assert(argc == 3);
    char* input_filename = argv[1];
    char* output_filename = argv[2];

    read_png_t read_data;
    write_png_t write_data;
    size_t size;
    unsigned height, width, height_pad, width_pad;
    unsigned char* host_s = NULL;
    unsigned char* host_t = NULL;
    unsigned char* dev_s = NULL;
    unsigned char* dev_t = NULL;
    dim3 blk, grid;

    // Get image info and initialize output image
    read_data.filename = input_filename;
    read_png_init(&read_data);
    size = write_data.size = read_data.size;
    height = write_data.height = read_data.height;
    width = write_data.width = read_data.width;
    height_pad = write_data.height_pad = read_data.height_pad;
    width_pad = write_data.width_pad = read_data.width_pad;
    write_data.filename = output_filename;
    write_png_init(&write_data);

    blk = dim3(BLOCK_X, BLOCK_Y);
    grid = dim3(width_pad / BLOCK_X, height_pad / BLOCK_Y);

    NVTX_RANGE_START(allocate);
    host_s = (unsigned char*)malloc(size);
    memset(host_s, 0, size);
    cudaHostRegister(host_s, size, cudaHostRegisterDefault);
    host_t = (unsigned char*)malloc(size);
    cudaHostRegister(host_t, size, cudaHostRegisterDefault);
    cudaMalloc(&dev_s, size);
    cudaMalloc(&dev_t, size);
    NVTX_RANGE_END();

    read_png_rows(&read_data, host_s, 0, height);
    cudaMemcpy(dev_s, host_s, size, cudaMemcpyHostToDevice);
    sobel<<<grid, blk>>>(dev_s, dev_t, height, width, height_pad, width_pad);
    cudaMemcpyAsync(host_t, dev_t, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    write_png_rows(&write_data, host_t, 0, write_data.height);

    read_png_end(&read_data);
    write_png_end(&write_data);

    NVTX_RANGE_START(free);
    read_png_destroy(&read_data);
    write_png_destroy(&write_data);
    free(host_s);
    free(host_t);
    cudaFree(dev_s);
    cudaFree(dev_t);
    NVTX_RANGE_END();
    NVTX_RANGE_END();
    return 0;
}
