#include "config.h"

typedef struct {
  const char *root_dir;
} DatasetDesc;

typedef struct Dataset {
  uint_t *x;
  uint_t *y;
  int size, width, classes;
} Dataset;

Dataset dataset_init(DatasetDesc desc);
void dataset_deinit(const Dataset *dataset);

void dataset_get_item(const Dataset *dataset, int i, uint_t *x, uint_t *y);
