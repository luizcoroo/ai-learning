#include "tensor.h"

typedef struct Dataset Dataset;

typedef struct {
  const Dataset *dataset;
  int batch_size;
} DataLoaderDesc;

typedef struct {
  uint_t *data;
  int *permutation;
  DataLoaderDesc desc;
} DataLoader;

typedef struct {
  UTensor x;
  UTensor y;
  int i;
  const DataLoader *ref;
} DataLoaderIterator;

DataLoader dataloader_init(DataLoaderDesc desc);
void dataloader_deinit(const DataLoader *dataloader);

DataLoaderIterator dataloader_iterator(DataLoader *dataloader);

int dataloader_iterator_next(DataLoaderIterator *iterator);
