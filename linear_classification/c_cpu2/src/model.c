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
      .data = malloc(2 * sizeof(float_t) * (w_len + b_len)),
      .desc = desc,
  };

  int j = 0;
  for (; j < w_len; j++)
    m.data[j] = (float_t)randnf();

  for (; j < w_len + b_len; j++)
    m.data[j] = 0.001;

  int w_sizes[2] = {desc.input_width, desc.output_width};
  int b_sizes[1] = {desc.output_width};
  m.w = ftensor_init(m.data, w_sizes, 2);
  m.b = ftensor_init(m.w.data + m.w.desc.data_len, b_sizes, 1);
  m.g_w = ftensor_init(m.b.data + m.b.desc.data_len, w_sizes, 2);
  m.g_b = ftensor_init(m.g_w.data + m.g_w.desc.data_len, b_sizes, 1);

  return m;
}

void model_deinit(const Model *m) { free(m->data); }

FTensor model_forward(Model *m, UTensor x, float_t *out_data, int n) {
  float_t *tmp = m->data + (m->desc.output_width * (m->desc.input_width + 1));
  return ftensor_softmax(utensor_matmuladd(out_data, x, m->w, m->b), tmp);
}

float_t model_evaluate(const Model *m, UTensor y, FTensor y_hat, int n) {
  return -ftensor_crossentropysum(y, y_hat) / n;
}

void model_backward(Model *m, UTensor x, UTensor y, FTensor y_hat, int n) {
  float_t tmp = n;
  FTensor len = ftensor_init(&size_data, (int[]){1}, 1);

  ftensor_oneshotdiff(y_hat, y);

  ftensor_assign(m->g_w, 0);
  ftensor_addmatmul(m->g_w, ftensor_transpose(x), y_hat);
  ftensor_div(m->g_w, len);

  ftensor_assign(m->g_b, 0);
  ftensor_add(m->g_b, y_hat);
  ftensor_div(m->g_b, len);
}

void model_update(Model *m) {
  int w_len = m->desc.input_width * m->desc.output_width;
  int b_len = m->desc.output_width;

  float_t *params_data = m->data;
  const float_t *grads_data = m->data + w_len + b_len;

  float regularization = 1 - m->desc.learning_rate * m->desc.weight_decay;

  int j = 0;
  for (; j < w_len; j++)
    params_data[j] =
        regularization * params_data[j] - m->desc.learning_rate * grads_data[j];

  for (; j < w_len + b_len; j++)
    params_data[j] -= m->desc.learning_rate * grads_data[j];
}
