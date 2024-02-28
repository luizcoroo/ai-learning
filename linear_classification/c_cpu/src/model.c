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
      .o = malloc(sizeof(float) * desc.output_width),
      .grads =
          malloc(sizeof(float) * desc.output_width * (desc.input_width + 2)),
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
  free(m->o);
  free(m->grads);
}

void model_forward(Model *m, const float *x, float *o, float *log_y_hat,
                   int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->output_width; j++) {
      o[i * m->output_width + j] = 0;
      for (int k = 0; k < m->input_width; k++)
        o[i * m->output_width + j] +=
            x[i * m->input_width + k] * m->w[m->input_width * k + j] + m->b[j];
    }
  }

  for (int i = 0; i < n; i++) {
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

float model_evaluate(const Model *m, const float *log_y_hat, const float *y,
                     int n) {
  float loss = 0;
  for (int i = 0; i < n; i++) {
    float tmp = 0;
    for (int j = 0; j < m->output_width; j++)
      tmp += y[i * m->output_width + j] * log_y_hat[i * m->output_width + j];

    loss += -tmp;
  }

  return loss / n;
}

void model_backward(Model *m, const float *x, const float *o,
                    const float *log_y_hat, const float *y, int n) {
  int parameters_len = m->output_width * (m->input_width + 1);
  memset(m->grads, 0, sizeof(float) * parameters_len);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->output_width; j++) {
      float diff = expf(m->o[i * m->output_width + j]) - y[i];
      for (int k = 0; k < m->input_width; k++)
        m->grads[k] += x[i * m->input_width + k] * diff;

      m->grads[m->input_width] += diff;
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
