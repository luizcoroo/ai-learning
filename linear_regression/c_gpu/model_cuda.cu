#include <cuda_runtime_api.h>
#include <string.h>

__global__ void kernel_forward(const float *x, const float *w, float *y,
                               int n_rows, int n_columns) {
  int i = threadIdx.x;
  float sum = 0;
  if (i < n_rows)
    for (int j = 0; j < n_columns; j++)
      sum += x[i * n_columns + j] * w[j];

  y[i] = sum;
}

extern "C" void model_cuda_forward(const float *x, const float *w, float *y,
                                   int n_rows, int n_columns) {

  unsigned long x_size = sizeof(float) * n_rows * n_columns;
  unsigned long w_size = sizeof(float) * n_columns;
  unsigned long y_size = sizeof(float) * n_rows;

  float *d_x, *d_w, *d_y;
  cudaMalloc((void **)&d_x, x_size);
  cudaMalloc((void **)&d_w, w_size);
  cudaMalloc((void **)&d_y, y_size);

  cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, y_size, cudaMemcpyHostToDevice);

  kernel_forward<<<1, n_rows>>>(d_x, d_w, d_y, n_rows, n_columns);
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, y_size, cudaMemcpyDeviceToHost);
}
