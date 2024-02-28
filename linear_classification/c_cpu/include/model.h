typedef struct {
  int input_width;
  int output_width;
  float learning_rate;
  float weight_decay;
  float noise;
} ModelDesc;

typedef struct Model {
  float *w, *b, *o;
  float *grads;
  int input_width;
  int output_width;
  float learning_rate;
  float weight_decay;
} Model;

Model model_init(ModelDesc desc);

void model_deinit(const Model *model);

void model_forward(Model *m, const float *x, float *o, float *log_y_hat, int n);

float model_evaluate(const Model *model, const float *log_y_hat, const float *y,
                     int n);

void model_backward(Model *m, const float *x, const float *o,
                    const float *log_y_hat, const float *y, int n);

void model_update(Model *model);
