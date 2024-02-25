#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "model_cuda.h"
#include "util.h"

Model model_init(ModelDesc desc) {
  Model m = {
      .w = malloc(sizeof(float) * desc.width),
      .b = 0.001,
      .grads = malloc(sizeof(float) * (desc.width + 1)),
      .width = desc.width,
      .learning_rate = desc.learning_rate,
      .weight_decay = desc.weight_decay,
  };

  for (int j = 0; j < m.width; j++)
    m.w[j] = randnf() * desc.noise;

  return m;
}

void model_deinit(const Model *m) {
  free(m->w);
  free(m->grads);
}

void model_forward(const Model *m, const float *x, float *y_hat, int n) {
  model_cuda_forward(x, m->w, m->b, y_hat, n, m->width);
}

float model_evaluate(const Model *, const float *y_hat, const float *y, int n) {
  float loss = 0;
  for (int i = 0; i < n; i++) {
    float diff = y_hat[i] - y[i];
    loss += diff * diff / 2.0;
  }

  return loss / n;
}

void model_backward(Model *m, const float *x, const float *y_hat,
                    const float *y, int n) {
  memset(m->grads, 0, sizeof(float) * (m->width + 1));
  for (int i = 0; i < n; i++) {
    float diff = y_hat[i] - y[i];
    for (int j = 0; j < m->width; j++)
      m->grads[j] += x[i * m->width + j] * diff;

    m->grads[m->width] += diff;
  }

  for (int j = 0; j < m->width + 1; j++)
    m->grads[j] /= n;
}

void model_update(Model *m) {
  float regularization = 1 - m->learning_rate * m->weight_decay;
  for (int j = 0; j < m->width; j++)
    m->w[j] = regularization * m->w[j] - m->learning_rate * m->grads[j];

  m->b -= m->learning_rate * m->grads[m->width];
}
