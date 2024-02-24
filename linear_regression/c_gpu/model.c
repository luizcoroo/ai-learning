#include <string.h>

#include "model.h"
#include "model_cuda.h"
#include "util.h"

Model model_init(ModelDesc desc) {
  Model m = {
      .w = desc.w,
      .width = desc.width,
      .weight_decay = desc.weight_decay,
      .learning_rate = desc.learning_rate,
  };

  for (int j = 0; j < m.width - 1; j++)
    m.w[j] = rand_normal() * desc.noise;
  m.w[m.width - 1] = 0;

  return m;
}

void model_deinit(const Model *) {}

void model_forward(const Model *m, const DataDesc *d, float *y_hat) {
  model_cuda_forward(d->x, m->w, y_hat, d->size, m->width);
}

float model_evaluate(const Model *, const DataDesc *d, const float *y_hat) {
  float loss = 0;
  for (int i = 0; i < d->size; i++) {
    float diff = y_hat[i] - d->y[i];
    loss += diff * diff / 2.0;
  }

  return loss / d->size;
}

void model_compute_grad(const Model *m, const DataDesc *d, const float *y_hat,
                        float *grad) {
  memset(grad, 0, sizeof(float) * m->width);
  for (int i = 0; i < d->size; i++) {
    float diff = y_hat[i] - d->y[i];
    for (int j = 0; j < m->width; j++)
      grad[j] += d->x[i * m->width + j] * diff;
  }

  for (int j = 0; j < m->width; j++)
    grad[j] /= d->size;
}

void model_backward(const Model *m, const float *grad) {
  float regularization = 1 - m->learning_rate * m->weight_decay;
  for (int j = 0; j < m->width - 1; j++)
    m->w[j] = regularization * m->w[j] - m->learning_rate * grad[j];

  m->w[m->width - 1] -= m->learning_rate * grad[m->width - 1];
}
