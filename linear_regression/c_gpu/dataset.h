typedef struct {
  const float *w;
  int width;
  int num_train;
  int num_val;
  float noise;
} DatasetDesc;

typedef struct Dataset {
  float *x;
  float *y;
  int num_train, size, width;
} Dataset;

Dataset dataset_init(DatasetDesc desc);
void dataset_deinit(const Dataset *dataset);

void dataset_shuffle_train(Dataset *dataset);
