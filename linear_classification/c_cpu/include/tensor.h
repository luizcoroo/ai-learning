#define MAX_DIMS 8

typedef struct {
  float *data;
  int sizes[MAX_DIMS];
  int strides[MAX_DIMS];
  int data_len;
  int rank;
} TensorView;

TensorView tensor_view(float *data, const int *sizes, int rank);
TensorView tensor_at(TensorView view, const int *indexes, int rank);

void tensor_align_to(TensorView *a, int rank);
void tensor_unalign(TensorView *a);
void tensor_broadcast_to(TensorView *a, const int *sizes, int rank);
void tensor_unbroadcast(TensorView *a);
void tensor_squeeze_all(TensorView *a);


void tensor_exp(TensorView a);
void tensor_neg(TensorView a);

void tensor_add(TensorView a, TensorView b);
void tensor_mul(TensorView a, TensorView b);

TensorView tensor_sum(float *a_data, TensorView b, const int *dims, int n);
TensorView tensor_max(float *a_data, TensorView b, const int *dims, int n);
TensorView tensor_matmul(float *a_data, TensorView b, TensorView c);

void tensor_describe(TensorView view);
