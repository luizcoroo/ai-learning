#include "tensor.h"

typedef struct {
  int input_width;
  int output_width;
  float learning_rate;
  float weight_decay;
} ModelDesc;

typedef struct Model {
  float_t *data;
  FTensor w, b;
  FTensor g_w, g_b;
  ModelDesc desc;
} Model;

Model model_init(ModelDesc desc);
void model_deinit(const Model *model);

FTensor model_forward(Model *m, UTensor x, float_t *out_data, int n);
float model_evaluate(const Model *model, UTensor y, FTensor y_hat, int n);
void model_backward(Model *m, UTensor x, UTensor y, FTensor y_hat, int n);
void model_update(Model *model);
