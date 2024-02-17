typedef struct {
  float *w;
  int width;
  float weight_decay;
  float learning_rate;
  float noise;
} ModelDesc;

typedef struct {
  const float *x;
  const float *y;
  int size;
} DataDesc;

typedef struct Model {
  float *w;
  int width;
  float weight_decay;
  float learning_rate;
} Model;

Model model_init(ModelDesc desc);

void model_deinit(const Model *model);

void model_forward(const Model *model, const DataDesc *ddesc, float *y_hat);

float model_evaluate(const Model *model, const DataDesc *desc,
                     const float *y_hat);

void model_compute_grad(const Model *model, const DataDesc *ddesc,
                        const float *y_hat, float *grad);

void model_backward(const Model *model, const float *grad);
