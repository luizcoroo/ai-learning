#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "util.h"

Model model_init(ModelDesc desc) {
  Model m = {
      .w = malloc(sizeof(float) * desc.input_width * desc.output_width),
      .b = malloc(sizeof(float) * desc.output_width),
      .grads =
          malloc(sizeof(float) * desc.output_width * (desc.input_width + 1)),
      .input_width = desc.input_width,
      .output_width = desc.output_width,
      .learning_rate = desc.learning_rate,
      .weight_decay = desc.weight_decay,
  };

  for (int j = 0; j < m.input_width * m.output_width; j++)
    m.w[j] = randnf() * desc.noise;

  for (int j = 0; j < m.output_width; j++)
    m.b[j] = 0.001;

  return m;
}

void model_deinit(const Model *m) {
  free(m->w);
  free(m->b);
  free(m->grads);
}

void model_forward(Model *m, const ubyte *x, float *log_y_hat, int n) {
  float *o = m->grads;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->output_width; j++) {
      o[i * m->output_width + j] = 0;
      for (int k = 0; k < m->input_width; k++)
        o[i * m->output_width + j] +=
            x[i * m->input_width + k] * m->w[k * m->output_width + j] + m->b[j];
    }

    float max = -FLT_MAX;
    for (int j = 0; j < m->output_width; j++)
      if (o[i * m->output_width + j] > max)
        max = o[i * m->output_width + j];

    float sum = 0;
    for (int j = 0; j < m->output_width; j++)
      sum += expf(o[i * m->output_width + j] - max);

    for (int j = 0; j < m->output_width; j++)
      log_y_hat[i * m->output_width + j] =
          o[i * m->output_width + j] - max - logf(sum);
  }
}

float model_evaluate(const Model *m, const float *log_y_hat, const ubyte *y,
                     int n) {
  float loss = 0;
  for (int i = 0; i < n; i++)
    loss += -log_y_hat[i * m->output_width + y[i]];

  return loss / n;
}

void model_backward(Model *m, const ubyte *x, const float *log_y_hat,
                    const ubyte *y, int n) {
  int parameters_len = m->output_width * (m->input_width + 1);
  memset(m->grads, 0, sizeof(float) * parameters_len);

  float *w_grads = m->grads;
  float *b_grads = m->grads + m->input_width * m->output_width;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->output_width; j++) {
      float y_prob = j == y[i];
      float diff = expf(log_y_hat[i * m->output_width + j]) - y_prob;
      for (int k = 0; k < m->input_width; k++)
        w_grads[k * m->output_width + j] += x[i * m->input_width + k] * diff;

      b_grads[j] += diff;
    }
  }

  for (int j = 0; j < parameters_len; j++)
    m->grads[j] /= n;
}

void model_update(Model *m) {
  int w_len = m->input_width * m->output_width;
  int b_len = m->output_width;

  float regularization = 1 - m->learning_rate * m->weight_decay;
  for (int j = 0; j < w_len; j++)
    m->w[j] = regularization * m->w[j] - m->learning_rate * m->grads[j];

  for (int j = 0; j < b_len; j++)
    m->b[j] -= m->learning_rate * m->grads[w_len + j];
}
