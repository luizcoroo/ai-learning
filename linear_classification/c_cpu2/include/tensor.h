#define MAX_DIMS 8

typedef struct {
  float *data;
  int sizes[MAX_DIMS];
  int strides[MAX_DIMS];
  int data_len;
  int rank;
} TensorViewF32;

typedef struct {
  unsigned char *data;
  int sizes[MAX_DIMS];
  int strides[MAX_DIMS];
  int data_len;
  int rank;
} TensorViewU8;

TensorViewF32 tensor_at(TensorViewF32 view, const int *indexes, int rank);

void tensor_align_to(TensorViewF32 *a, int rank);
void tensor_unalign(TensorViewF32 *a);
void tensor_broadcast_to(TensorViewF32 *a, const int *sizes, int rank);
void tensor_unbroadcast(TensorViewF32 *a);
void tensor_squeeze_all(TensorViewF32 *a);

TensorViewF32 tensor_view_f32(float *data, const int *sizes, int rank);
TensorViewU8 tensor_view_u8(unsigned char *data, const int *sizes, int rank);

void tensor_exp_f32(TensorViewF32 a);
void tensor_log_f32(TensorViewF32 a);
void tensor_neg_f32(TensorViewF32 a);
void tensor_add_f32(TensorViewF32 a, TensorViewF32 b);
void tensor_sub_f32(TensorViewF32 a, TensorViewF32 b);
void tensor_mul_f32(TensorViewF32 a, TensorViewF32 b);
void tensor_div_f32(TensorViewF32 a, TensorViewF32 b);
TensorViewF32 tensor_sum_f32(float *a_data, TensorViewF32 b, const int *dims, int n);
TensorViewF32 tensor_max_f32(float *a_data, TensorViewF32 b, const int *dims, int n);
TensorViewF32 tensor_min_f32(float *a_data, TensorViewF32 b, const int *dims, int n);
TensorViewF32 tensor_matmul_f32xf32(float *a_data, TensorViewF32 b, TensorViewF32 c);
TensorViewF32 tensor_matmul_u8xf32(float *a_data, TensorViewU8 b, TensorViewF32 c);

void tensor_describe(TensorViewF32 view);
