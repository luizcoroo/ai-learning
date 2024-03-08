#include <stdint.h>

#define MAX_DIMS 8

typedef struct {
  int sizes[MAX_DIMS];
  int strides[MAX_DIMS];
  int data_len;
  int rank;
} DataDesc;

#define DECLARE_TENSOR_STRUCT(TENSOR_SUFFIX, DATA_TYPE)                        \
  typedef struct {                                                             \
    DATA_TYPE *data;                                                           \
    DataDesc desc;                                                             \
  } TensorView##TENSOR_SUFFIX;

#define DECLARE_TENSOR_SIMPLE_OPS(TENSOR_SUFFIX, OP_SUFFIX)                    \
  TensorView##TENSOR_SUFFIX tensor_at_##OP_SUFFIX(                             \
      TensorView##TENSOR_SUFFIX view, const int *indexes, int rank);           \
  void tensor_exp_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a);                    \
  void tensor_log_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a);                    \
  void tensor_neg_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a);                    \
  void tensor_add_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a,                     \
                              TensorView##TENSOR_SUFFIX b);                    \
  void tensor_sub_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a,                     \
                              TensorView##TENSOR_SUFFIX b);                    \
  void tensor_mul_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a,                     \
                              TensorView##TENSOR_SUFFIX b);                    \
  void tensor_div_##OP_SUFFIX(TensorView##TENSOR_SUFFIX a,                     \
                              TensorView##TENSOR_SUFFIX b);                    \
  void tensor_describe_##OP_SUFFIX(TensorView##TENSOR_SUFFIX view);

#define DECLARE_TENSOR_DATA_DEPENDENT_OPS(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX) \
  TensorView##TENSOR_SUFFIX tensor_view_##OP_SUFFIX(                           \
      DATA_TYPE *data, const int *sizes, int rank);                            \
  TensorView##TENSOR_SUFFIX tensor_sum_##OP_SUFFIX(                            \
      DATA_TYPE *a_data, TensorView##TENSOR_SUFFIX b, const int *dims, int n); \
  TensorView##TENSOR_SUFFIX tensor_max_##OP_SUFFIX(                            \
      DATA_TYPE *a_data, TensorView##TENSOR_SUFFIX b, const int *dims, int n); \
  TensorView##TENSOR_SUFFIX tensor_min_##OP_SUFFIX(                            \
      DATA_TYPE *a_data, TensorView##TENSOR_SUFFIX b, const int *dims, int n);

#define DECLARE_TENSOR_MATMUL_OP(DATA_TYPE1, TENSOR_SUFFIX2, TENSOR_SUFFIX3,   \
                                 OP_SUFFIX)                                    \
  TensorView##TENSOR_SUFFIX3 tensor_matmul_##OP_SUFFIX(                        \
      DATA_TYPE1 *a_data, TensorView##TENSOR_SUFFIX2 b,                        \
      TensorView##TENSOR_SUFFIX3 c);

#define DECLARE_TENSOR_BASIC(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX)              \
  DECLARE_TENSOR_STRUCT(TENSOR_SUFFIX, DATA_TYPE)                              \
  DECLARE_TENSOR_SIMPLE_OPS(TENSOR_SUFFIX, OP_SUFFIX)                          \
  DECLARE_TENSOR_DATA_DEPENDENT_OPS(TENSOR_SUFFIX, DATA_TYPE, OP_SUFFIX)

DECLARE_TENSOR_BASIC(U8, uint8_t, u8)
DECLARE_TENSOR_BASIC(U16, uint16_t, u16)
DECLARE_TENSOR_BASIC(U32, uint32_t, u32)
DECLARE_TENSOR_BASIC(U64, uint64_t, u64)

DECLARE_TENSOR_BASIC(F16, _Float16, f16)
DECLARE_TENSOR_MATMUL_OP(_Float16, U8, F16, u8xf16)
DECLARE_TENSOR_MATMUL_OP(_Float16, U16, F16, u16xf16)
DECLARE_TENSOR_MATMUL_OP(_Float16, U32, F16, u32xf16)
DECLARE_TENSOR_MATMUL_OP(_Float16, U64, F16, u64xf16)
DECLARE_TENSOR_MATMUL_OP(_Float16, F16, F16, f16xf16)

DECLARE_TENSOR_BASIC(F32, float, f32)
DECLARE_TENSOR_MATMUL_OP(float, U8, F32, u8xf32)
DECLARE_TENSOR_MATMUL_OP(float, U16, F32, u16xf32)
DECLARE_TENSOR_MATMUL_OP(float, U32, F32, u32xf32)
DECLARE_TENSOR_MATMUL_OP(float, U64, F32, u64xf32)
DECLARE_TENSOR_MATMUL_OP(float, F16, F32, f16xf32)
DECLARE_TENSOR_MATMUL_OP(float, F32, F32, f32xf32)

DECLARE_TENSOR_BASIC(F64, double, f64)
DECLARE_TENSOR_MATMUL_OP(double, U8, F64, u8xf64)
DECLARE_TENSOR_MATMUL_OP(double, U16, F64, u16xf64)
DECLARE_TENSOR_MATMUL_OP(double, U32, F64, u32xf64)
DECLARE_TENSOR_MATMUL_OP(double, U64, F64, u64xf64)
DECLARE_TENSOR_MATMUL_OP(double, F16, F64, f16xf64)
DECLARE_TENSOR_MATMUL_OP(double, F32, F64, f32xf64)
DECLARE_TENSOR_MATMUL_OP(double, F64, F64, f64xf64)

void tensor_align_to(DataDesc *d, int rank);
void tensor_unalign(DataDesc *a);
void tensor_broadcast_to(DataDesc *a, const int *sizes, int rank);
void tensor_unbroadcast(DataDesc *a);
void tensor_squeeze_all(DataDesc *a);
