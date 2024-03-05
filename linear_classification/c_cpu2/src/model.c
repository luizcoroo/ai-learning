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

  m.wv = tensor_view(m.w, (int[]){m.input_width, m.output_width}, 2);
  m.bv = tensor_view(m.b, (int[]){m.output_width}, 1);

  return m;
}

void model_deinit(const Model *m) {
  free(m->w);
  free(m->b);
  free(m->grads);
}

float ov_data[10000];
float max_data[10000];
float sum_data[10000];

void model_forward(Model *m, ubyte *x, float *y_hat, int n) {
  TensorView ov = tensorint_matmul(
      y_hat, tensorint_view(x, (int[]){n, m->input_width}, 2), m->wv);
  tensor_add(ov, m->bv);

  tensor_sub(ov, tensor_max(max_data, ov, (int[]){1}, 1));
  tensor_exp(ov);

  tensor_div(ov, tensor_sum(sum_data, ov, (int[]){1}, 1));
}

float model_evaluate(const Model *m, const float *y_hat, const ubyte *y,
                     int n) {
  float loss = 0;
  for (int i = 0; i < n; i++)
    loss += -log1pf(y_hat[i * m->output_width + y[i]]);

  return loss / n;
}

void model_backward(Model *m, const ubyte *x, const float *y_hat,
                    const ubyte *y, int n) {
  int parameters_len = m->output_width * (m->input_width + 1);
  memset(m->grads, 0, sizeof(float) * parameters_len);

  float *w_grads = m->grads;
  float *b_grads = m->grads + m->input_width * m->output_width;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->output_width; j++) {
      float y_prob = j == y[i];
      float diff = y_hat[i * m->output_width + j] - y_prob;
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
