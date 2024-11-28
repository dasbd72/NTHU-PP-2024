#include <sys/time.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef PROFILING
#include <nvtx3/nvtx3.hpp>
#define NVTX_RANGE_START(arg) \
    nvtxRangePushA(#arg);
#define NVTX_RANGE_END() \
    nvtxRangePop();
#define NVTX_RANGE_FUNC() \
    NVTX3_FUNC_RANGE()
#else
#define NVTX_RANGE_START(arg) \
    {}
#define NVTX_RANGE_END() \
    {}
#define NVTX_RANGE_FUNC() \
    {}
#endif  // PROFILING

constexpr int BLOCK_C = 32;
constexpr int BLOCK_R = 32;

struct Data {
    char *input_filename;
    char *output_filename;
    int B, N, d;
    float *Q, *K, *V, *O;
};

void input(Data *data);
void output(Data *data);

namespace flash_attention {
void flash_attention(Data *data);
void flash_attention_block(float *o, float *q, float *k, float *v, int N, int d);
void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar);
void row_max(float *mij, float *sij);
void minus_max_and_exp(float *pij, float *sij, float *mij);
void row_sum(float *lij, float *pij);
void update_mlo(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int d);
};  // namespace flash_attention

namespace fused_flash_attention {
void flash_attention(Data *data);
void flash_attention_block(float *o, float *q, float *k, float *v, int N, int d);
void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar);
void update_mlo(float *mi, float *li, float *oi, float *sij, float *vj, int d);
};  // namespace fused_flash_attention

int main(int argc, char *argv[]) {
    NVTX_RANGE_FUNC();
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    Data data;

    data.input_filename = argv[1];
    data.output_filename = argv[2];

    input(&data);

    flash_attention::flash_attention(&data);

    output(&data);

    return 0;
}

void input(Data *data) {
    NVTX_RANGE_FUNC();
    FILE *file;
    int B, N, d;
    float *Q, *K, *V, *O;

    file = fopen(data->input_filename, "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    data->B = B;
    data->N = N;
    data->d = d;
    data->Q = Q;
    data->K = K;
    data->V = V;
    data->O = O;

    fclose(file);
}

void output(Data *data) {
    NVTX_RANGE_FUNC();
    FILE *file;
    int B = data->B;
    int N = data->N;
    int d = data->d;
    float *Q = data->Q;
    float *K = data->K;
    float *V = data->V;
    float *O = data->O;

    file = fopen(data->output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);

    fclose(file);
    free(Q);
    free(K);
    free(V);
    free(O);
}

namespace flash_attention {
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;
    float *Q = data->Q;
    float *K = data->K;
    float *V = data->V;
    float *O = data->O;

    for (int i = 0; i < B; i++) {
        flash_attention_block(
            O + (i * N * d),
            Q + (i * N * d),
            K + (i * N * d),
            V + (i * N * d),
            N,
            d);
    }
}

void flash_attention_block(float *o, float *q, float *k, float *v, int N, int d) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int tr = N / BLOCK_R, tc = N / BLOCK_C;
    float *oi = (float *)malloc(d * BLOCK_R * sizeof(float));
    float *qi = (float *)malloc(d * BLOCK_R * sizeof(float));
    float *kj = (float *)malloc(d * BLOCK_C * sizeof(float));
    float *vj = (float *)malloc(d * BLOCK_C * sizeof(float));
    float *li = (float *)malloc(BLOCK_R * sizeof(float));
    float *mi = (float *)malloc(BLOCK_R * sizeof(float));

    float *sij = (float *)malloc(BLOCK_R * BLOCK_C * sizeof(float));
    float *pij = (float *)malloc(BLOCK_R * BLOCK_C * sizeof(float));
    float *lij = (float *)malloc(BLOCK_R * sizeof(float));
    float *mij = (float *)malloc(BLOCK_R * sizeof(float));

    for (int j = 0; j < tc; j++) {
        memcpy(kj, k + j * d * BLOCK_C, d * BLOCK_C * sizeof(float));
        memcpy(vj, v + j * d * BLOCK_C, d * BLOCK_C * sizeof(float));
        for (int i = 0; i < tr; i++) {
            memcpy(oi, o + i * d * BLOCK_R, d * BLOCK_R * sizeof(float));
            memcpy(qi, q + i * d * BLOCK_R, d * BLOCK_R * sizeof(float));
            memcpy(li, l + i * BLOCK_R, BLOCK_R * sizeof(float));
            memcpy(mi, m + i * BLOCK_R, BLOCK_R * sizeof(float));

            qk_dot_and_scalar(sij, qi, kj, d, 1.0 / sqrtf(d));
            row_max(mij, sij);
            minus_max_and_exp(pij, sij, mij);
            row_sum(lij, pij);

            update_mlo(mi, li, oi, mij, lij, pij, vj, d);

            memcpy(o + i * d * BLOCK_R, oi, d * BLOCK_R * sizeof(float));
            memcpy(l + i * BLOCK_R, li, BLOCK_R * sizeof(float));
            memcpy(m + i * BLOCK_R, mi, BLOCK_R * sizeof(float));
        }
    }

    free(sij);
    free(pij);
    free(lij);
    free(mij);

    free(kj);
    free(vj);
    free(qi);
    free(oi);
    free(li);
    free(mi);

    free(l);
    free(m);
}

void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar) {
    NVTX_RANGE_FUNC();
    for (int i = 0; i < BLOCK_R; i++) {
        for (int j = 0; j < BLOCK_C; j++) {
            out[i * BLOCK_C + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * BLOCK_C + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * BLOCK_C + j] *= scalar;
        }
    }
}

void row_max(float *mij, float *sij) {
    NVTX_RANGE_FUNC();
    for (int i = 0; i < BLOCK_R; i++) {
        mij[i] = sij[i * BLOCK_C];
        for (int j = 0; j < BLOCK_C; j++) {
            mij[i] = std::max(mij[i], sij[i * BLOCK_C + j]);
        }
    }
}

void minus_max_and_exp(float *pij, float *sij, float *mij) {
    NVTX_RANGE_FUNC();
    for (int i = 0; i < BLOCK_R; i++) {
        for (int j = 0; j < BLOCK_C; j++) {
            pij[i * BLOCK_C + j] = expf(sij[i * BLOCK_C + j] - mij[i]);
        }
    }
}

void row_sum(float *lij, float *pij) {
    NVTX_RANGE_FUNC();
    for (int i = 0; i < BLOCK_R; i++) {
        lij[i] = 0.0F;
        for (int j = 0; j < BLOCK_C; j++) {
            lij[i] += pij[i * BLOCK_C + j];
        }
    }
}

void update_mlo(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int d) {
    NVTX_RANGE_FUNC();
    float *mi_new = (float *)malloc(BLOCK_R * sizeof(float));
    float *li_new = (float *)malloc(BLOCK_R * sizeof(float));
    float val0, val1;
    float pv;

    for (int i = 0; i < BLOCK_R; i++) {
        mi_new[i] = std::max(mi[i], mij[i]);
        val0 = expf(mi[i] - mi_new[i]) * li[i];
        val1 = expf(mij[i] - mi_new[i]);
        li_new[i] = val0 + val1 * lij[i];
        for (int j = 0; j < d; j++) {
            pv = 0.0F;
            for (int t = 0; t < BLOCK_C; t++) {
                pv += pij[i * BLOCK_C + t] * vj[t * d + j];
            }
            oi[i * d + j] = (val0 * oi[i * d + j] + val1 * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, BLOCK_R * sizeof(float));
    memcpy(li, li_new, BLOCK_R * sizeof(float));

    free(mi_new);
    free(li_new);
}
};  // namespace flash_attention

namespace fused_flash_attention {
void flash_attention(Data *data) {
    NVTX_RANGE_FUNC();
    int B = data->B;
    int N = data->N;
    int d = data->d;
    float *Q = data->Q;
    float *K = data->K;
    float *V = data->V;
    float *O = data->O;

    for (int i = 0; i < B; i++) {
        flash_attention_block(
            O + (i * N * d),
            Q + (i * N * d),
            K + (i * N * d),
            V + (i * N * d),
            N,
            d);
    }
}

void flash_attention_block(float *o, float *q, float *k, float *v, int N, int d) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int tr = N / BLOCK_R, tc = N / BLOCK_C;
    float *oi = (float *)malloc(d * BLOCK_R * sizeof(float));
    float *qi = (float *)malloc(d * BLOCK_R * sizeof(float));
    float *kj = (float *)malloc(d * BLOCK_C * sizeof(float));
    float *vj = (float *)malloc(d * BLOCK_C * sizeof(float));
    float *li = (float *)malloc(BLOCK_R * sizeof(float));
    float *mi = (float *)malloc(BLOCK_R * sizeof(float));

    float *sij = (float *)malloc(BLOCK_R * BLOCK_C * sizeof(float));

    for (int j = 0; j < tc; j++) {
        memcpy(kj, k + j * d * BLOCK_C, d * BLOCK_C * sizeof(float));
        memcpy(vj, v + j * d * BLOCK_C, d * BLOCK_C * sizeof(float));
        for (int i = 0; i < tr; i++) {
            memcpy(oi, o + i * d * BLOCK_R, d * BLOCK_R * sizeof(float));
            memcpy(qi, q + i * d * BLOCK_R, d * BLOCK_R * sizeof(float));
            memcpy(li, l + i * BLOCK_R, BLOCK_R * sizeof(float));
            memcpy(mi, m + i * BLOCK_R, BLOCK_R * sizeof(float));

            qk_dot_and_scalar(sij, qi, kj, d, 1.0 / sqrtf(d));
            update_mlo(mi, li, oi, sij, vj, d);

            memcpy(o + i * d * BLOCK_R, oi, d * BLOCK_R * sizeof(float));
            memcpy(l + i * BLOCK_R, li, BLOCK_R * sizeof(float));
            memcpy(m + i * BLOCK_R, mi, BLOCK_R * sizeof(float));
        }
    }

    free(sij);

    free(kj);
    free(vj);
    free(qi);
    free(oi);
    free(li);
    free(mi);

    free(l);
    free(m);
}

void qk_dot_and_scalar(float *out, float *q, float *k, int d, float scalar) {
    flash_attention::qk_dot_and_scalar(out, q, k, d, scalar);
}

void update_mlo(float *mi, float *li, float *oi, float *sij, float *vj, int d) {
    NVTX_RANGE_FUNC();
    float *mi_new = (float *)malloc(BLOCK_R * sizeof(float));
    float *li_new = (float *)malloc(BLOCK_R * sizeof(float));
    float *pij = (float *)malloc(BLOCK_C * sizeof(float));
    float lij, mij;
    float val0, val1;
    float pv;

    for (int i = 0; i < BLOCK_R; i++) {
        mij = sij[i * BLOCK_C];
        for (int j = 0; j < BLOCK_C; j++) {
            mij = std::max(mij, sij[i * BLOCK_C + j]);
        }
        lij = 0.0F;
        for (int j = 0; j < BLOCK_C; j++) {
            pij[j] = expf(sij[i * BLOCK_C + j] - mij);
            lij += pij[j];
        }
        mi_new[i] = std::max(mi[i], mij);
        val0 = expf(mi[i] - mi_new[i]) * li[i];
        val1 = expf(mij - mi_new[i]);
        li_new[i] = val0 + val1 * lij;
        for (int j = 0; j < d; j++) {
            pv = 0.0F;
            for (int t = 0; t < BLOCK_C; t++) {
                pv += pij[t] * vj[t * d + j];
            }
            oi[i * d + j] = (val0 * oi[i * d + j] + val1 * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, BLOCK_R * sizeof(float));
    memcpy(li, li_new, BLOCK_R * sizeof(float));

    free(mi_new);
    free(li_new);
    free(pij);
}
};  // namespace fused_flash_attention