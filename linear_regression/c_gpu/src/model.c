#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "model_cuda.h"
#include "util.h"

Model model_init(ModelDesc desc) {

  Model m = {
      .width = desc.width,
      .learning_rate = desc.learning_rate,
      .weight_decay = desc.weight_decay,
  };

  int bytes = sizeof(float) * 2 * (desc.width + 1);
  cudaMallocHost((void **)&m.cpu_data, bytes);
  cudaMalloc((void **)&m.gpu_data, bytes);

  m.w = m.gpu_data;
  m.b = m.w + m.width;
  m.grads = m.b + 1;

  for (int j = 0; j < m.width; j++)
    m.w[j] = randnf() * desc.noise;

  m.b[0] = 0.001;

  return m;
}

void model_deinit(const Model *m) {
  cudaFreeHost(m->cpu_data);
  cudaFree(m->gpu_data);
}

void model_forward(const Model *m, const float *x, float *y_hat, int n) {
  model_cuda_forward(x, m->w, m->b[0], y_hat, n, m->width);
}

float model_evaluate(const Model *, const float *y_hat, const float *y, int n) {
  return model_cuda_evaluate(y_hat, y, n);
}

void model_backward(Model *m, const float *x, const float *y_hat,
                    const float *y, int n) {

  model_cuda_backward(x, y_hat, y, m->grads, n, m->width);
}

void model_update(Model *m) {
  model_cuda_update(m->grads, m->w, m->b, m->width, m->learning_rate,
                    m->weight_decay);
}
