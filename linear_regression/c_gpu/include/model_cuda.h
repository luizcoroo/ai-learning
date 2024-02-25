void model_cuda_forward(const float *x, const float *w, const float *b, float *y_hat,
                        int n_rows, int n_columns);

float model_cuda_evaluate(const float *y_hat, const float *y, int n_rows);

void model_cuda_backward(const float *x, const float *y_hat, const float *y,
                         float *grads, int n_rows, int n_columns);

void model_cuda_update(const float *grads, float *w, float *b, int n_columns, float learning_rate, float weight_decay);
