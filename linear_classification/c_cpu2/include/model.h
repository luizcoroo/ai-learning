#include "tensor.h"

typedef unsigned char ubyte;

typedef struct {
  int input_width;
  int output_width;
  float learning_rate;
  float weight_decay;
} ModelDesc;

typedef struct Model {
  float *params;
  float *grads;
  TensorViewF32 wv, bv, gradsv;
  ModelDesc desc;
} Model;

Model model_init(ModelDesc desc);

void model_deinit(const Model *model);

void model_forward(Model *m, ubyte *x, float *log_y_hat, int n);

float model_evaluate(const Model *model, const float *log_y_hat, const ubyte *y,
                     int n);

void model_backward(Model *m, const ubyte *x, const float *log_y_hat,
                    const ubyte *y, int n);

void model_update(Model *model);
