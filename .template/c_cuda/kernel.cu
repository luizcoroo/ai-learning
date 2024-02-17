#include <cuda_runtime_api.h>

__global__ void kernel(const float *a, const float *b, float *c, int n) {
  int i = threadIdx.x;
  float sum = 0;
  if (i < n)
    sum = b[i] + c[i];

  c[i] = sum;
}

extern "C" void kcall(const float *a, const float *b, float *c, int n) {
  kernel<<<1, n>>>(a, b, c, n);
  cudaDeviceSynchronize();
}
