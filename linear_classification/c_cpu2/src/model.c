#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "util.h"

Model model_init(ModelDesc desc) {
  int w_len = desc.input_width * desc.output_width;
  int b_len = desc.output_width;
  Model m = {
      .params = malloc(sizeof(float) * (w_len + b_len)),
      .grads = malloc(sizeof(float) * (w_len + b_len)),
      .desc = desc,
  };

  int j = 0;
  for (; j < w_len; j++)
    m.params[j] = randnf();

  for (; j < w_len + b_len; j++)
    m.params[j] = 0.001;

  m.wv = tensor_view_f32(m.params, (int[]){desc.input_width, desc.output_width},
                         2);
  m.bv = tensor_view_f32(m.params + w_len, (int[]){desc.output_width}, 1);

  return m;
}

void model_deinit(const Model *m) {
  free(m->params);
  free(m->grads);
}

void model_forward(Model *m, ubyte *x, float *y_hat, int n) {
  TensorViewF32 ov = tensor_matmul_u8xf32(
      y_hat, tensor_view_u8(x, (int[]){n, m->desc.input_width}, 2), m->wv);
  tensor_add_f32(ov, m->bv);

  float *tmp = m->grads;
  tensor_sub_f32(ov, tensor_max_f32(tmp, ov, (int[]){1}, 1));
  tensor_exp_f32(ov);
  tensor_div_f32(ov, tensor_sum_f32(tmp, ov, (int[]){1}, 1));
}

float model_evaluate(const Model *m, const float *y_hat, const ubyte *y,
                     int n) {
  float loss = 0;
  for (int i = 0; i < n; i++)
    loss += -logf(y_hat[i * m->desc.output_width + y[i]] + 0.0001);

  return loss / n;
}

void model_backward(Model *m, const ubyte *x, const float *y_hat,
                    const ubyte *y, int n) {
  int w_len = m->desc.input_width * m->desc.output_width;
  int b_len = m->desc.output_width;
  memset(m->grads, 0, sizeof(float) * (w_len + b_len));

  float *w_grads = m->grads;
  float *b_grads = m->grads + w_len;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m->desc.output_width; j++) {
      float y_prob = j == y[i];
      float diff = y_hat[i * m->desc.output_width + j] - y_prob;
      for (int k = 0; k < m->desc.input_width; k++)
        w_grads[k * m->desc.output_width + j] +=
            x[i * m->desc.input_width + k] * diff;

      b_grads[j] += diff;
    }
  }

  for (int j = 0; j < w_len + b_len; j++)
    m->grads[j] /= n;
}

void model_update(Model *m) {
  int w_len = m->desc.input_width * m->desc.output_width;
  int b_len = m->desc.output_width;

  float regularization = 1 - m->desc.learning_rate * m->desc.weight_decay;

  int j = 0;
  for (; j < w_len; j++)
    m->params[j] =
        regularization * m->params[j] - m->desc.learning_rate * m->grads[j];

  for (; j < w_len + b_len; j++)
    m->params[j] -= m->desc.learning_rate * m->grads[j];
}
