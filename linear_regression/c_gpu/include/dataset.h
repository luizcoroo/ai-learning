typedef struct {
  const float *source_w;
  float source_b;
  int width;
  int size;
  float noise;
} DatasetDesc;

typedef struct Dataset {
  float *x;
  float *y;
  int size, width;
} Dataset;

Dataset dataset_init(DatasetDesc desc);
void dataset_deinit(const Dataset *dataset);

void dataset_get_item(const Dataset *dataset, int i, float *x, float *y);
