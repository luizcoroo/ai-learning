#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"

FTensor ftensor_softmax(FTensor a, float_t *tmp_data) {
  ftensor_subexp(a, ftensor_max(tmp_data, a, (int[]){1}, 1));
  ftensor_div(a, ftensor_sum(tmp_data, a, (int[]){1}, 1));
  return a;
}

// #define DEFINE_TENSOR_AT(TENSOR_SUFFIX, OP_SUFFIX)                             \
//   TensorView##TENSOR_SUFFIX tensor_at_##OP_SUFFIX(                             \
//       TensorView##TENSOR_SUFFIX view, const int *indexes, int rank) {          \
//     int idx = 0;                                                               \
//     for (int i = 0; i < rank; i++)                                             \
//       idx += indexes[i] * view.desc.strides[i];                                \
//     TensorView##TENSOR_SUFFIX new_view = {                                     \
//         .data = view.data + idx,                                               \
//         .desc.data_len = view.desc.strides[rank - 1],                          \
//         .desc.rank = view.desc.rank - rank,                                    \
//     };                                                                         \
//     memcpy(new_view.desc.sizes, view.desc.sizes + rank,                        \
//            sizeof(int) * new_view.desc.rank);                                  \
//     memcpy(new_view.desc.strides, view.desc.strides + rank,                    \
//            sizeof(int) * new_view.desc.rank);                                  \
//     return new_view;                                                           \
//   }
//
// #define DEFINE_TENSOR_DESCRIBE(TENSOR_SUFFIX, OP_SUFFIX) \
//   void tensor_describe_##OP_SUFFIX(TensorView##TENSOR_SUFFIX view) { \
//     printf("sizes: %d", view.desc.sizes[0]); \
//     for (int i = 1; i < view.desc.rank; i++) \
//       printf(", %d", view.desc.sizes[i]); \
//     printf("\nstrides: %d", view.desc.strides[0]); \
//     for (int i = 1; i < view.desc.rank; i++) \
//       printf(", %d", view.desc.strides[i]); \
//     printf("\ndata_len: %d", view.desc.data_len); \
//     printf("\ndata: \n"); \
//     if (view.desc.rank == 0) { \
//       printf("%4.1lf\n", (double)view.data[0]); \
//       return; \
//     } \
//     printf("["); \
//     for (int i = 1; i < view.desc.rank; i++) \
//       printf("["); \
//     int pos = 0, idxs[MAX_DIMS] = {0}, dim; \
//     do { \
//       if (idxs[view.desc.rank - 1] > 0) \
//         printf(" "); \
//       printf("%4.1lf", (double)view.data[pos]); \
//       dim = incr_pos_at_dim(&pos, idxs, &view.desc, view.desc.rank); \
//       for (int i = dim; i < view.desc.rank; i++) \
//         printf("]"); \
//       if (dim > 0 && dim < view.desc.rank) { \
//         printf("\n"); \
//         int i = 0; \
//         for (; i < dim; i++) \
//           printf(" "); \
//         for (; i < view.desc.rank; i++) \
//           printf("["); \
//       } \
//     } while (dim > 0); \
//     printf("\n"); \
//   }
//
// #define DEFINE_TENSOR_VIEW(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
//   TensorView##TENSOR_SUFFIX tensor_view_##OP_SUFFIX( \
//       DATA_TYPE *data, const int *sizes, int rank) { \
//     TensorView##TENSOR_SUFFIX view = { \
//         .data = data, \
//         .desc.rank = rank, \
//     }; \
//     memcpy(view.desc.sizes, sizes, sizeof(int) * rank); \
//     int stride = 1; \
//     for (int i = rank; i > 0; i--) { \
//       view.desc.strides[i - 1] = stride; \
//       stride *= view.desc.sizes[i - 1]; \
//     } \
//     view.desc.data_len = stride; \
//     return view; \
//   }
//
// #define DEFINE_TENSOR_ASSIGN(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
//   void tensor_assign_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a, DATA_TYPE x) {
//   \
//     for (int i = 0; i < a.desc.data_len; i++) \
//       a.data[i] = x; \
//   }
//
// #define DEFINE_TENSOR_UNARY(TENSOR_SUFFIX, OP, OP_SUFFIX) \
//   void tensor_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a) { \
//     for (int i = 0; i < a.desc.data_len; i++) \
//       a.data[i] = OP(a.data[i]); \
//   }
//
// #define DEFINE_TENSOR_ELEMENTWISE(TENSOR_SUFFIX, OP, OP_SUFFIX) \
//   void tensor_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a, \
//                           TensorView##TENSOR_SUFFIX b) { \
//     tensor_align_to(&b.desc, a.desc.rank); \
//     tensor_broadcast_to(&b.desc, a.desc.sizes, a.desc.rank); \
//     int pos_a = 0, idxs_a[MAX_DIMS] = {0}, pos_b = 0, idxs_b[MAX_DIMS] = {0};
//     \
//     do { \
//       a.data[pos_a] = a.data[pos_a] OP b.data[pos_b]; \
//     } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) && \
//              incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank)); \
//   }
//
// #define DEFINE_TENSOR_CONTRACTION(TENSOR_SUFFIX, DATA_TYPE, OP,
// INITIAL_VALUE, \
//                                   OP_SUFFIX, VIEW_SUFFIX) \
//   TensorView##TENSOR_SUFFIX tensor_##OP_SUFFIX(DATA_TYPE *a_data, \
//                                                TensorView##TENSOR_SUFFIX b, \
//                                                const int *dims, int n) { \
//     int a_sizes[MAX_DIMS]; \
//     memcpy(a_sizes, b.desc.sizes, sizeof(int) * b.desc.rank); \
//     for (int i = 0; i < n; i++) \
//       a_sizes[dims[i]] = 1; \
//     TensorView##TENSOR_SUFFIX a = \
//         tensor_view_##VIEW_SUFFIX(a_data, a_sizes, b.desc.rank); \
//     tensor_broadcast_to(&a.desc, b.desc.sizes, b.desc.rank); \
//     int pos_a = 0, idxs_a[MAX_DIMS] = {0}, pos_b = 0, idxs_b[MAX_DIMS] = {0};
//     \
//     for (int i = 0; i < a.desc.data_len; i++) \
//       a.data[i] = INITIAL_VALUE; \
//     do { \
//       a.data[pos_a] = OP(a.data[pos_a], b.data[pos_b]); \
//     } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) && \
//              incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank)); \
//     tensor_unbroadcast(&a.desc); \
//     return a; \
//   }
//
// #define DEFINE_TENSOR_ONEHOTENC(DATA_TYPE1, TENSOR_SUFFIX2, TENSOR_SUFFIX1, \
//                                 OP_SUFFIX, VIEW_SUFFIX) \
//   TensorView##TENSOR_SUFFIX1 tensor_onehotenc_##OP_SUFFIX( \
//       DATA_TYPE1 *a_data, TensorView##TENSOR_SUFFIX2 b, int classes) { \
//     int a_sizes[MAX_DIMS]; \
//     int dim = b.desc.rank; \
//     memcpy(a_sizes, b.desc.sizes, sizeof(int) * dim); \
//     a_sizes[b.desc.rank] = classes; \
//     TensorView##TENSOR_SUFFIX1 a = \
//         tensor_view_##VIEW_SUFFIX(a_data, a_sizes, dim + 1); \
//     tensor_align_to(&b.desc, dim + 1); \
//     tensor_broadcast_to(&b.desc, a.desc.sizes, dim + 1); \
//     int pos_a = 0, idxs_a[MAX_DIMS] = {0}; \
//     int pos_b = 0, idxs_b[MAX_DIMS] = {0}; \
//     do { \
//       for (int j = 0; j < classes; j++) { \
//         int a_i = j * a.desc.strides[dim]; \
//         int b_i = j * b.desc.strides[dim]; \
//         a.data[pos_a + a_i] = b.data[pos_b + b_i] == j; \
//       } \
//     } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim) && \
//              incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim)); \
//     return a; \
//   }
//
// #define DEFINE_TENSOR_CROSSENTROPYSUM(DATA_TYPE2, TENSOR_SUFFIX1, \
//                                       TENSOR_SUFFIX2, LOG_OP, OP_SUFFIX) \
//   DATA_TYPE2 tensor_crossentropysum_##OP_SUFFIX( \
//       TensorView##TENSOR_SUFFIX1 a, TensorView##TENSOR_SUFFIX2 b) { \
//     a.desc.sizes[a.desc.rank] = 1, a.desc.strides[a.desc.rank] = 0; \
//     a.desc.rank++; \
//     int dim = b.desc.rank - 1; \
//     tensor_align_to(&a.desc, dim); \
//     tensor_broadcast_to(&a.desc, b.desc.sizes, dim); \
//     int pos_a = 0, idxs_a[MAX_DIMS] = {0}; \
//     int pos_b = 0, idxs_b[MAX_DIMS] = {0}; \
//     DATA_TYPE2 sum = 0; \
//     do { \
//       for (int j = 0; j < b.sizes[dim]; j++) { \
//         DATA_TYPE2 prob = a.data[j * a.desc.strides[dim]] == j; \
//         sum += prob * LOG_OP(b.data[pos_b + j * b.desc.strides[dim]]); \
//       } \
//     } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim) && \
//              incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim)); \
//     return sum; \
//   }
//
// #define DEFINE_TENSOR_MATMUL(DATA_TYPE1, TENSOR_SUFFIX2, TENSOR_SUFFIX3, \
//                              PLUS_OP, MUL_OP, INITIAL_VALUE, OP_SUFFIX, \
//                              VIEW_SUFFIX) \
//   TensorView##TENSOR_SUFFIX3 tensor_matmul_##OP_SUFFIX( \
//       DATA_TYPE1 *a_data, TensorView##TENSOR_SUFFIX2 b, \
//       TensorView##TENSOR_SUFFIX3 c) { \
//     int dim = b.desc.rank - 2; \
//     tensor_align_to(&c.desc, b.desc.rank); \
//     tensor_broadcast_to(&c.desc, b.desc.sizes, dim); \
//     int N = b.desc.sizes[dim], K = c.desc.sizes[dim], \
//         M = c.desc.sizes[dim + 1]; \
//     assert(b.desc.sizes[dim + 1] == K); \
//     int a_sizes[MAX_DIMS]; \
//     memcpy(a_sizes, b.desc.sizes, sizeof(int) * dim); \
//     a_sizes[dim] = N; \
//     a_sizes[dim + 1] = M; \
//     TensorView##TENSOR_SUFFIX3 a = \
//         tensor_view_##VIEW_SUFFIX(a_data, a_sizes, b.desc.rank); \
//     int pos_a = 0, idxs_a[MAX_DIMS] = {0}; \
//     int pos_b = 0, idxs_b[MAX_DIMS] = {0}; \
//     int pos_c = 0, idxs_c[MAX_DIMS] = {0}; \
//     do { \
//       for (int i = 0; i < N; i++) { \
//         int a_i = i * a.desc.strides[dim], b_i = i * b.desc.strides[dim]; \
//         for (int j = 0; j < M; j++) { \
//           int a_j = j * a.desc.strides[dim + 1], \
//               c_j = j * c.desc.strides[dim + 1]; \
//           auto res = &a.data[pos_a + a_i + a_j]; \
//           *res = INITIAL_VALUE; \
//           for (int k = 0; k < K; k++) { \
//             int b_k = k * b.desc.strides[dim + 1], \
//                 c_k = k * c.desc.strides[dim]; \
//             *res = *res PLUS_OP b.data[pos_b + b_i + b_k] MUL_OP c \
//                         .data[pos_c + c_k + c_j]; \
//           } \
//         } \
//       } \
//     } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim) && \
//              incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim) && \
//              incr_pos_at_dim(&pos_c, idxs_c, &c.desc, dim)); \
//     return a; \
//   }
//
// #define SUM(a, b) \
//   ({ \
//     __typeof__(a) _a = (a); \
//     __typeof__(b) _b = (b); \
//     _a + _b; \
//   })
//
// #define MAX(a, b) \
//   ({ \
//     __typeof__(a) _a = (a); \
//     __typeof__(b) _b = (b); \
//     _a > _b ? _a : _b; \
//   })
//
// #define MIN(a, b) \
//   ({ \
//     __typeof__(a) _a = (a); \
//     __typeof__(b) _b = (b); \
//     _a < _b ? _a : _b; \
//   })
//
// int incr_pos_at_dim(int *pos, int *idxs, const DataDesc *desc, int dim) {
//   while (dim > 0) {
//     *pos += desc->strides[dim - 1];
//     idxs[dim - 1] += 1;
//     if (idxs[dim - 1] != desc->sizes[dim - 1])
//       break;
//
//     *pos -= idxs[dim - 1] * desc->strides[dim - 1];
//     idxs[dim - 1] = 0;
//     dim -= 1;
//   }
//
//   return dim;
// }
//
// #define DEFINE_TENSOR_BASIC(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
//   DEFINE_TENSOR_AT(TENSOR_SUFFIX, OP_SUFFIX) \
//   DEFINE_TENSOR_DESCRIBE(TENSOR_SUFFIX, OP_SUFFIX) \
//   DEFINE_TENSOR_VIEW(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
//   DEFINE_TENSOR_ASSIGN(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
//   DEFINE_TENSOR_ELEMENTWISE(TENSOR_SUFFIX, +, add_##OP_SUFFIX) \
//   DEFINE_TENSOR_ELEMENTWISE(TENSOR_SUFFIX, -, sub_##OP_SUFFIX) \
//   DEFINE_TENSOR_ELEMENTWISE(TENSOR_SUFFIX, *, mul_##OP_SUFFIX) \
//   DEFINE_TENSOR_ELEMENTWISE(TENSOR_SUFFIX, /, div_##OP_SUFFIX)
//
// DEFINE_TENSOR_BASIC(U8, uint8_t, u8)
// DEFINE_TENSOR_UNARY(U8, expf, exp_u8)
// DEFINE_TENSOR_UNARY(U8, logf, log_u8)
// DEFINE_TENSOR_UNARY(U8, -, neg_u8)
// DEFINE_TENSOR_CONTRACTION(U8, uint8_t, SUM, 0, sum_u8, u8)
// DEFINE_TENSOR_CONTRACTION(U8, uint8_t, MAX, 0, max_u8, u8)
// DEFINE_TENSOR_CONTRACTION(U8, uint8_t, MIN, UINT8_MAX, min_u8, u8)
//
// DEFINE_TENSOR_BASIC(U16, uint16_t, u16)
// DEFINE_TENSOR_UNARY(U16, expf, exp_u16)
// DEFINE_TENSOR_UNARY(U16, logf, log_u16)
// DEFINE_TENSOR_UNARY(U16, -, neg_u16)
// DEFINE_TENSOR_CONTRACTION(U16, uint16_t, SUM, 0, sum_u16, u16)
// DEFINE_TENSOR_CONTRACTION(U16, uint16_t, MAX, 0, max_u16, u16)
// DEFINE_TENSOR_CONTRACTION(U16, uint16_t, MIN, UINT16_MAX, min_u16, u16)
//
// DEFINE_TENSOR_BASIC(U32, uint32_t, u32)
// DEFINE_TENSOR_UNARY(U32, expf, exp_u32)
// DEFINE_TENSOR_UNARY(U32, logf, log_u32)
// DEFINE_TENSOR_UNARY(U32, -, neg_u32)
// DEFINE_TENSOR_CONTRACTION(U32, uint32_t, SUM, 0, sum_u32, u32)
// DEFINE_TENSOR_CONTRACTION(U32, uint32_t, MAX, 0, max_u32, u32)
// DEFINE_TENSOR_CONTRACTION(U32, uint32_t, MIN, UINT32_MAX, min_u32, u32)
//
// DEFINE_TENSOR_BASIC(U64, uint64_t, u64)
// DEFINE_TENSOR_UNARY(U64, exp, exp_u64)
// DEFINE_TENSOR_UNARY(U64, log, log_u64)
// DEFINE_TENSOR_UNARY(U64, -, neg_u64)
// DEFINE_TENSOR_CONTRACTION(U64, uint64_t, SUM, 0, sum_u64, u64)
// DEFINE_TENSOR_CONTRACTION(U64, uint64_t, MAX, 0, max_u64, u64)
// DEFINE_TENSOR_CONTRACTION(U64, uint64_t, MIN, UINT64_MAX, min_u64, u64)
//
// DEFINE_TENSOR_BASIC(F16, _Float16, f16)
// DEFINE_TENSOR_UNARY(F16, expf, exp_f16)
// DEFINE_TENSOR_UNARY(F16, logf, log_f16)
// DEFINE_TENSOR_UNARY(F16, -, neg_f16)
// DEFINE_TENSOR_CONTRACTION(F16, _Float16, SUM, 0, sum_f16, f16)
// DEFINE_TENSOR_CONTRACTION(F16, _Float16, MAX, -FLT_MAX, max_f16, f16)
// DEFINE_TENSOR_CONTRACTION(F16, _Float16, MIN, FLT_MAX, min_f16, f16)
// DEFINE_TENSOR_ONEHOTENC(_Float16, U8, F16, f16u8, f16)
// DEFINE_TENSOR_CROSSENTROPYSUM(_Float16, U8, F16, logf, u8f16)
// DEFINE_TENSOR_MATMUL(_Float16, U8, F16, +, *, 0, u8f16, f16)
// DEFINE_TENSOR_ONEHOTENC(_Float16, U16, F16, f16u16, f16)
// DEFINE_TENSOR_CROSSENTROPYSUM(_Float16, U16, F16, logf, u16f16)
// DEFINE_TENSOR_MATMUL(_Float16, U16, F16, +, *, 0, u16f16, f16)
// DEFINE_TENSOR_ONEHOTENC(_Float16, U32, F16, f16u32, f16)
// DEFINE_TENSOR_CROSSENTROPYSUM(_Float16, U32, F16, logf, u32f16)
// DEFINE_TENSOR_MATMUL(_Float16, U32, F16, +, *, 0, u32f16, f16)
// DEFINE_TENSOR_ONEHOTENC(_Float16, U64, F16, f16u64, f16)
// DEFINE_TENSOR_CROSSENTROPYSUM(_Float16, U64, F16, logf, u64f16)
// DEFINE_TENSOR_MATMUL(_Float16, U64, F16, +, *, 0, u64f16, f16)
// DEFINE_TENSOR_MATMUL(_Float16, F16, F16, +, *, 0, f16f16, f16)
//
// DEFINE_TENSOR_BASIC(F32, float, f32)
// DEFINE_TENSOR_UNARY(F32, expf, exp_f32)
// DEFINE_TENSOR_UNARY(F32, logf, log_f32)
// DEFINE_TENSOR_UNARY(F32, -, neg_f32)
// DEFINE_TENSOR_CONTRACTION(F32, float, SUM, 0, sum_f32, f32)
// DEFINE_TENSOR_CONTRACTION(F32, float, MAX, -FLT_MAX, max_f32, f32)
// DEFINE_TENSOR_CONTRACTION(F32, float, MIN, FLT_MAX, min_f32, f32)
// DEFINE_TENSOR_ONEHOTENC(float, U8, F32, f32u8, f32)
// DEFINE_TENSOR_CROSSENTROPYSUM(float, U8, F32, logf, u8f32)
// DEFINE_TENSOR_MATMUL(float, U8, F32, +, *, 0, u8f32, f32)
// DEFINE_TENSOR_ONEHOTENC(float, U16, F32, f32u16, f32)
// DEFINE_TENSOR_CROSSENTROPYSUM(float, U16, F32, logf, u16f32)
// DEFINE_TENSOR_MATMUL(float, U16, F32, +, *, 0, u16f32, f32)
// DEFINE_TENSOR_ONEHOTENC(float, U32, F32, f32u32, f32)
// DEFINE_TENSOR_CROSSENTROPYSUM(float, U32, F32, logf, u32f32)
// DEFINE_TENSOR_MATMUL(float, U32, F32, +, *, 0, u32f32, f32)
// DEFINE_TENSOR_ONEHOTENC(float, U64, F32, f32u64, f32)
// DEFINE_TENSOR_CROSSENTROPYSUM(float, U64, F32, logf, u64f32)
// DEFINE_TENSOR_MATMUL(float, U64, F32, +, *, 0, u64f32, f32)
// DEFINE_TENSOR_MATMUL(float, F16, F32, +, *, 0, f16f32, f32)
// DEFINE_TENSOR_MATMUL(float, F32, F32, +, *, 0, f32f32, f32)
//
// DEFINE_TENSOR_BASIC(F64, double, f64)
// DEFINE_TENSOR_UNARY(F64, exp, exp_f64)
// DEFINE_TENSOR_UNARY(F64, log, log_f64)
// DEFINE_TENSOR_UNARY(F64, -, neg_f64)
// DEFINE_TENSOR_CONTRACTION(F64, double, SUM, 0, sum_f64, f64)
// DEFINE_TENSOR_CONTRACTION(F64, double, MAX, -DBL_MAX, max_f64, f64)
// DEFINE_TENSOR_CONTRACTION(F64, double, MIN, DBL_MAX, min_f64, f64)
// DEFINE_TENSOR_ONEHOTENC(double, U8, F64, f64u8, f64)
// DEFINE_TENSOR_CROSSENTROPYSUM(double, U8, F64, log, u8f64)
// DEFINE_TENSOR_MATMUL(double, U8, F64, +, *, 0, u8f64, f64)
// DEFINE_TENSOR_ONEHOTENC(double, U16, F64, f64u16, f64)
// DEFINE_TENSOR_CROSSENTROPYSUM(double, U16, F64, log, u16f64)
// DEFINE_TENSOR_MATMUL(double, U16, F64, +, *, 0, u16f64, f64)
// DEFINE_TENSOR_ONEHOTENC(double, U32, F64, f64u32, f64)
// DEFINE_TENSOR_CROSSENTROPYSUM(double, U32, F64, log, u32f64)
// DEFINE_TENSOR_MATMUL(double, U32, F64, +, *, 0, u32f64, f64)
// DEFINE_TENSOR_ONEHOTENC(double, U64, F64, f64u64, f64)
// DEFINE_TENSOR_CROSSENTROPYSUM(double, U64, F64, log, u64f64)
// DEFINE_TENSOR_MATMUL(double, U64, F64, +, *, 0, u64f64, f64)
// DEFINE_TENSOR_MATMUL(double, F16, F64, +, *, 0, f16f64, f64)
// DEFINE_TENSOR_MATMUL(double, F32, F64, +, *, 0, f32f64, f64)
// DEFINE_TENSOR_MATMUL(double, F64, F64, +, *, 0, f64f64, f64)
//
// void tensor_align_to(DataDesc *a, int rank) {
//   int rank_diff = rank - a->rank;
//   for (int i = a->rank; i > 0; --i) {
//     a->sizes[i - 1 + rank_diff] = a->sizes[i - 1];
//     a->strides[i - 1 + rank_diff] = a->strides[i - 1];
//   }
//
//   for (int i = 0; i < rank_diff; i++) {
//     a->sizes[i] = 1;
//     a->strides[i] = a->data_len;
//   }
//
//   a->rank = rank;
// }
//
// void tensor_unalign(DataDesc *a) {
//   int rank_diff = 0;
//   while (rank_diff < a->rank && a->sizes[rank_diff] == 1)
//     rank_diff += 1;
//
//   for (int i = rank_diff; i < a->rank; ++i) {
//     a->sizes[i - rank_diff] = a->sizes[i];
//     a->strides[i - rank_diff] = a->strides[i];
//   }
//
//   a->rank -= rank_diff;
// }
//
// void tensor_squeeze_all(DataDesc *a) {
//   int i = 0, j = 0;
//   for (; j < a->rank; i++, j++) {
//     if (a->sizes[j] == 1)
//       while (j < a->rank && a->sizes[j] == 1)
//         j++;
//
//     a->sizes[i] = a->sizes[j];
//     a->strides[i] = a->strides[j];
//   }
//   a->rank -= j - i;
// }
//
// void tensor_unbroadcast(DataDesc *a) {
//   for (int i = 0; i < a->rank; ++i) {
//     if (a->strides[i] == 0) {
//       a->strides[i] = 1;
//       a->sizes[i] = 1;
//     }
//   }
// }
//
// void tensor_broadcast_to(DataDesc *a, const int *sizes, int rank) {
//   for (int i = 0; i < rank; i++) {
//     if (a->sizes[i] == 1 && sizes[i] > 1) {
//       a->sizes[i] = sizes[i];
//       a->strides[i] = 0;
//     }
//   }
// }
