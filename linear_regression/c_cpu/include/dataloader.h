typedef struct Dataset Dataset;

typedef struct {
  const Dataset *dataset;
  int batch_size;
} DataLoaderDesc;

typedef struct {
  float *data;
  int *permutation;
  int batch_size;
  int n_batches;
  const Dataset *dataset;
} DataLoader;

typedef struct {
  float *x;
  float *y;
  float *y_hat;
  int size;
  int i;
  const DataLoader *ref;
} DataLoaderIterator;

DataLoader dataloader_init(DataLoaderDesc desc);
void dataloader_deinit(const DataLoader *dataloader);

DataLoaderIterator dataloader_iterator(DataLoader *dataloader);

int dataloader_iterator_next(DataLoaderIterator *iterator);
