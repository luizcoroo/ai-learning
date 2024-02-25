#include <cuda_runtime_api.h>
#include <string.h>

__global__ void kernel_forward(const float *x, const float *w, const float *b,
                               float *y, int n_rows, int n_columns) {
  int i = threadIdx.x;
  if (i < n_rows) {
    float sum = *b;
    for (int j = 0; j < n_columns; j++)
      sum += x[i * n_columns + j] * w[j];

    y[i] = sum;
  }
}

extern "C" void model_cuda_forward(const float *x, const float *w, const float *b,
                                   float *y, int n_rows, int n_columns) {

  kernel_forward<<<1, n_rows>>>(x, w, b, y, n_rows, n_columns);
}

__global__ void kernel_evaluate(const float *y_hat, const float *y, int n_rows,
                                float *loss) {
  int i = threadIdx.x;
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

  kernel_evaluate<<<1, n_rows>>>(y_hat, y, n_rows, loss_d);

  cudaMemcpy(&loss, loss_d, sizeof(float), cudaMemcpyDeviceToHost);

  return loss / n_rows;
}

__global__ void kernel_backward(const float *x, const float *y_hat,
                                const float *y, float *grads, int n_rows,
                                int n_columns) {
  int j = threadIdx.x;
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
  kernel_backward<<<1, n_columns + 1>>>(x, y_hat, y, grads, n_rows, n_columns);
}

__global__ void kernel_update(const float *grads, float *w, float *b,
                              int n_columns, float learning_rate,
                              float weight_decay) {
  int j = threadIdx.x;
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

  kernel_update<<<1, n_columns + 1>>>(grads, w, b, n_columns, learning_rate,
                                      weight_decay);
}
