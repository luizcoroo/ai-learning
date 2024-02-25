#include <cuda_runtime_api.h>
#include <string.h>

__global__ void kernel_forward(const float *x, const float *w, const float *b,
                               float *y, int n_rows, int n_columns) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_rows) {
    float sum = *b;
    for (int j = 0; j < n_columns; j++)
      sum += x[i * n_columns + j] * w[j];

    y[i] = sum;
  }
}

extern "C" void model_cuda_forward(const float *x, const float *w,
                                   const float *b, float *y, int n_rows,
                                   int n_columns) {
  int block_size = 256;
  int num_blocks = (n_rows + block_size - 1) / block_size;
  kernel_forward<<<num_blocks, block_size>>>(x, w, b, y, n_rows, n_columns);
}

__global__ void kernel_evaluate(const float *y_hat, const float *y, int n_rows,
                                float *loss) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_rows) {
    float diff = y_hat[i] - y[i];
    atomicAdd(loss, diff * diff / 2.0);
  }
}

extern "C" float model_cuda_evaluate(const float *y_hat, const float *y,
                                     int n_rows) {

  float loss = 0;
  float *loss_d;

  cudaMalloc((void **)&loss_d, sizeof(float));
  cudaMemcpy(loss_d, &loss, sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int num_blocks = (n_rows + block_size - 1) / block_size;
  kernel_evaluate<<<num_blocks, block_size>>>(y_hat, y, n_rows, loss_d);

  cudaMemcpy(&loss, loss_d, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(loss_d);

  return loss / n_rows;
}

__global__ void kernel_backward(const float *x, const float *y_hat,
                                const float *y, float *grads, int n_rows,
                                int n_columns) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n_columns + 1) {
    float sum = 0;
    for (int i = 0; i < n_rows; ++i) {
      float diff = y_hat[i] - y[i];
      if (j < n_columns)
        sum += x[i * n_columns + j] * diff;
      else
        sum += diff;
    }
    sum /= n_rows;

    grads[j] = sum;
  }
}

extern "C" void model_cuda_backward(const float *x, const float *y_hat,
                                    const float *y, float *grads, int n_rows,
                                    int n_columns) {
  int block_size = 256;
  int num_blocks = ((n_columns + 1) + block_size - 1) / block_size;
  kernel_backward<<<num_blocks, block_size>>>(x, y_hat, y, grads, n_rows,
                                              n_columns);
}

__global__ void kernel_update(const float *grads, float *w, float *b,
                              int n_columns, float learning_rate,
                              float weight_decay) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n_columns + 1) {
    if (j < n_columns) {
      float regularization = 1.0 - learning_rate * weight_decay;
      w[j] = regularization * w[j] - learning_rate * grads[j];
    } else {
      *b -= learning_rate * grads[j];
    }
  }
}

extern "C" void model_cuda_update(const float *grads, float *w, float *b,
                                  int n_columns, float learning_rate,
                                  float weight_decay) {

  int block_size = 256;
  int num_blocks = ((n_columns + 1) + block_size - 1) / block_size;
  kernel_update<<<num_blocks, block_size>>>(grads, w, b, n_columns,
                                            learning_rate, weight_decay);
}
