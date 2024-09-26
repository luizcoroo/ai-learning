typedef struct {
  int width;
  float learning_rate;
  float weight_decay;
  float noise;
} ModelDesc;

typedef struct Model {
  float *cpu_data;
  float *gpu_data;
  float *w, *b;
  float *grads;
  int width;
  float learning_rate;
  float weight_decay;
} Model;

Model model_init(ModelDesc desc);

void model_deinit(const Model *model);

void model_forward(const Model *m, const float *x, float *y, int n);

float model_evaluate(const Model *model, const float *y_hat, const float *y,
                     int n);

void model_backward(Model *m, const float *x, const float *y_hat,
                    const float *y, int n);

void model_update(Model *model);


void model_to_cpu(Model *model);
