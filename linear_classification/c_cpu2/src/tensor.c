#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"

TensorViewF32 tensor_at(TensorViewF32 view, const int *indexes, int rank) {
  int idx = 0;
  for (int i = 0; i < rank; i++)
    idx += indexes[i] * view.strides[i];

  TensorViewF32 new_view = {
      .data = view.data + idx,
      .data_len = view.strides[rank - 1],
      .rank = view.rank - rank,
  };

  memcpy(new_view.sizes, view.sizes + rank, sizeof(int) * new_view.rank);
  memcpy(new_view.strides, view.strides + rank, sizeof(int) * new_view.rank);
  return new_view;
}

void tensor_describe(TensorViewF32 view) {
  printf("sizes: %d", view.sizes[0]);
  for (int i = 1; i < view.rank; i++)
    printf(", %d", view.sizes[i]);

  printf("\nstrides: %d", view.strides[0]);
  for (int i = 1; i < view.rank; i++)
    printf(", %d", view.strides[i]);

  printf("\ndata_len: %d", view.data_len);

  printf("\ndata: \n");
  if (view.rank == 0) {
    printf("%4.1f\n", view.data[0]);
    return;
  }

  printf("[");
  for (int i = 1; i < view.rank; i++)
    printf("[");

  int idxs[MAX_DIMS] = {0};
  int pos = 0;
  int dim;
  do {
    if (idxs[view.rank - 1] > 0)
      printf(" ");
    printf("%f", view.data[pos]);

    dim = incr_pos_at_dim(&pos, idxs, view.sizes, view.strides, view.rank);
    for (int i = dim; i < view.rank; i++)
      printf("]");

    if (dim > 0 && dim < view.rank) {
      printf("\n");
      int i = 0;
      for (; i < dim; i++)
        printf(" ");
      for (; i < view.rank; i++)
        printf("[");
    }
  } while (dim > 0);

  printf("\n");
}

#define DEFINE_TENSOR_VIEW(T1, DATA_TYPE, OP_SUFFIX)                           \
  T1 tensor_view_##OP_SUFFIX(DATA_TYPE *data, const int *sizes, int rank) {    \
    T1 view = {                                                                \
        .data = data,                                                          \
        .desc.rank = rank,                                                     \
    };                                                                         \
    memcpy(view.desc.sizes, sizes, sizeof(int) * rank);                        \
    int stride = 1;                                                            \
    for (int i = rank; i > 0; i--) {                                           \
      view.desc.strides[i - 1] = stride;                                       \
      stride *= view.desc.sizes[i - 1];                                        \
    }                                                                          \
    view.desc.data_len = stride;                                               \
    return view;                                                               \
  }

#define DEFINE_TENSOR_UNARY(OP, OP_SUFFIX)                                     \
  void tensor_##OP_SUFFIX(TensorViewF32 a) {                                   \
    for (int i = 0; i < a.desc.data_len; i++)                                  \
      a.data[i] = OP(a.data[i]);                                               \
  }

#define DEFINE_TENSOR_ELEMENTWISE(OP, OP_SUFFIX)                               \
  void tensor_##OP_SUFFIX(TensorViewF32 a, TensorViewF32 b) {                  \
    tensor_align_to(&b, a.desc.rank);                                          \
    tensor_broadcast_to(&b, a.desc.sizes, a.desc.rank);                        \
    int pos_a = 0, idxs_a[MAX_DIMS] = {0}, pos_b = 0, idxs_b[MAX_DIMS] = {0};  \
    do {                                                                       \
      a.data[pos_a] = a.data[pos_a] OP b.data[pos_b];                          \
    } while (incr_pos_at_dim(&pos_a, idxs_a, a.desc, a.rank) &&                \
             incr_pos_at_dim(&pos_b, idxs_b, b.desc, b.rank));                 \
  }

#define DEFINE_TENSOR_CONTRACTION(OP, INITIAL_VALUE, OP_SUFFIX)                \
  TensorViewF32 tensor_##OP_SUFFIX(float *a_data, TensorViewF32 b,             \
                                   const int *dims, int n) {                   \
    int a_sizes[MAX_DIMS];                                                     \
    memcpy(a_sizes, b.desc.sizes, sizeof(int) * b.rank);                       \
    for (int i = 0; i < n; i++)                                                \
      a_sizes[dims[i]] = 1;                                                    \
    TensorViewF32 a = tensor_view_f32(a_data, a_sizes, b.desc.rank);           \
    tensor_broadcast_to(&a, b.desc.sizes, b.desc.rank);                        \
    int pos_a = 0, idxs_a[MAX_DIMS] = {0}, pos_b = 0, idxs_b[MAX_DIMS] = {0};  \
    for (int i = 0; i < a.desc.data_len; i++)                                  \
      a.data[i] = INITIAL_VALUE;                                               \
    do {                                                                       \
      a.data[pos_a] = OP(a.data[pos_a], b.data[pos_b]);                        \
    } while (incr_pos_at_dim(&pos_a, idxs_a, a.desc, a.rank) &&                \
             incr_pos_at_dim(&pos_b, idxs_b, b.desc, b.rank));                 \
    tensor_unbroadcast(&a);                                                    \
    return a;                                                                  \
  }

#define DEFINE_TENSOR_MATMUL(T1, PLUS_OP, MUL_OP, INITIAL_VALUE, OP_SUFFIX)    \
  TensorViewF32 tensor_matmul_##OP_SUFFIX(float *a_data, T1 b,                 \
                                          TensorViewF32 c) {                   \
    int dim = b.desc.rank - 2;                                                 \
    tensor_align_to(&c, b.desc.rank);                                          \
    tensor_broadcast_to(&c, b.desc.sizes, dim);                                \
    int N = b.desc.sizes[dim], K = c.desc.sizes[dim],                          \
        M = c.desc.sizes[dim + 1];                                             \
    assert(b.desc.sizes[dim + 1] == K);                                        \
    int a_sizes[MAX_DIMS];                                                     \
    memcpy(a_sizes, b.desc.sizes, sizeof(int) * dim);                          \
    a_sizes[dim] = N;                                                          \
    a_sizes[dim + 1] = M;                                                      \
    TensorViewF32 a = tensor_view_f32(a_data, a_sizes, b.desc.rank);           \
    int pos_a = 0, idxs_a[MAX_DIMS] = {0};                                     \
    int pos_b = 0, idxs_b[MAX_DIMS] = {0};                                     \
    int pos_c = 0, idxs_c[MAX_DIMS] = {0};                                     \
    do {                                                                       \
      for (int i = 0; i < N; i++) {                                            \
        int a_i = i * a.desc.strides[dim], b_i = i * b.desc.strides[dim];      \
        for (int j = 0; j < M; j++) {                                          \
          int a_j = j * a.desc.strides[dim + 1],                               \
              c_j = j * c.desc.strides[dim + 1];                               \
          auto res = &a.data[pos_a + a_i + a_j];                               \
          *res = INITIAL_VALUE;                                                \
          for (int k = 0; k < K; k++) {                                        \
            int b_k = k * b.desc.strides[dim + 1],                             \
                c_k = k * c.desc.strides[dim];                                 \
            *res = *res PLUS_OP b.data[pos_b + b_i + b_k] MUL_OP c             \
                        .data[pos_c + c_k + c_j];                              \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } while (incr_pos_at_dim(&pos_a, idxs_a, a.desc, dim) &&                   \
             incr_pos_at_dim(&pos_b, idxs_b, b.desc, dim) &&                   \
             incr_pos_at_dim(&pos_c, idxs_c, c.desc, dim));                    \
    return a;                                                                  \
  }

float fsum(float a, float b) { return a + b; }


int incr_pos_at_dim(int *pos, int *idxs, const DataDesc *desc, int dim) {
  while (dim > 0) {
    *pos += desc->strides[dim - 1];
    idxs[dim - 1] += 1;
    if (idxs[dim - 1] != desc->sizes[dim - 1])
      break;

    *pos -= idxs[dim - 1] * desc->strides[dim - 1];
    idxs[dim - 1] = 0;
    dim -= 1;
  }

  return dim;
}

DEFINE_TENSOR_VIEW(F32, float, f32)
DEFINE_TENSOR_DESCRIBE(F32, float, f32)
DEFINE_TENSOR_AT(F32, float, f32)
DEFINE_TENSOR_UNARY(expf, exp_f32)
DEFINE_TENSOR_UNARY(logf, log_f32)
DEFINE_TENSOR_UNARY(-, neg_f32)
DEFINE_TENSOR_ELEMENTWISE(+, add_f32)
DEFINE_TENSOR_ELEMENTWISE(-, sub_f32)
DEFINE_TENSOR_ELEMENTWISE(*, mul_f32)
DEFINE_TENSOR_ELEMENTWISE(/, div_f32)
DEFINE_TENSOR_CONTRACTION(fsum, 0, sum_f32)
DEFINE_TENSOR_CONTRACTION(fmax, -FLT_MAX, max_f32)
DEFINE_TENSOR_CONTRACTION(fmin, +FLT_MAX, min_f32)
DEFINE_TENSOR_MATMUL(F32, +, *, 0, f32xf32)


void tensor_align_to(DataDef *a, int rank) {
  int rank_diff = rank - a->rank;
  for (int i = a->rank; i > 0; --i) {
    a->sizes[i - 1 + rank_diff] = a->sizes[i - 1];
    a->strides[i - 1 + rank_diff] = a->strides[i - 1];
  }

  for (int i = 0; i < rank_diff; i++) {
    a->sizes[i] = 1;
    a->strides[i] = a->data_len;
  }

  a->rank = rank;
}

void tensor_unalign(DataDef *a) {
  int rank_diff = 0;
  while (rank_diff < a->rank && a->sizes[rank_diff] == 1)
    rank_diff += 1;

  for (int i = rank_diff; i < a->rank; ++i) {
    a->sizes[i - rank_diff] = a->sizes[i];
    a->strides[i - rank_diff] = a->strides[i];
  }

  a->rank -= rank_diff;
}

void tensor_squeeze_all(DataDef *a) {
  int i = 0, j = 0;
  for (; j < a->rank; i++, j++) {
    if (a->sizes[j] == 1)
      while (j < a->rank && a->sizes[j] == 1)
        j++;

    a->sizes[i] = a->sizes[j];
    a->strides[i] = a->strides[j];
  }
  a->rank -= j - i;
}

void tensor_unbroadcast(DataDef *a) {
  for (int i = 0; i < a->rank; ++i) {
    if (a->strides[i] == 0) {
      a->strides[i] = 1;
      a->sizes[i] = 1;
    }
  }
}

void tensor_broadcast_to(DataDef *a, const int *sizes, int rank) {
  for (int i = 0; i < rank; i++) {
    if (a->sizes[i] == 1 && sizes[i] > 1) {
      a->sizes[i] = sizes[i];
      a->strides[i] = 0;
    }
  }
}

