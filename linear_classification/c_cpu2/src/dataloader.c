#include <stdlib.h>

#include "dataloader.h"
#include "dataset.h"
#include "util.h"

DataLoader dataloader_init(DataLoaderDesc desc) {
  int data_len = desc.batch_size * (desc.dataset->width + 1);
  DataLoader dl = {
      .data = malloc(sizeof(uint_t) * data_len),
      .permutation = malloc(sizeof(int) * desc.dataset->size),
      .desc = desc,
  };

  for (int i = 0; i < desc.dataset->size; i++)
    dl.permutation[i] = i;

  return dl;
}

void dataloader_deinit(const DataLoader *dl) {
  free(dl->data);
  free(dl->permutation);
}

DataLoaderIterator dataloader_iterator(DataLoader *dl) {
  shuffleiarr(dl->permutation, dl->desc.dataset->size);
  return (DataLoaderIterator){.i = 0, .ref = dl};
}

int dataloader_iterator_next(DataLoaderIterator *it) {
  const DataLoader *dl = it->ref;
  DataLoaderDesc desc = dl->desc;
  const Dataset *dataset = desc.dataset;

  if (it->i >= dataset->size / desc.batch_size)
    return 0;

  int x_end = desc.batch_size * dataset->width;
  uint_t *x = dl->data + 0;
  uint_t *y = dl->data + x_end;

  int batch_start = it->i * desc.batch_size;
  int width = dataset->width;
  for (int i = 0; i < desc.batch_size; i++) {
    int idx = it->ref->permutation[batch_start + i];
    dataset_get_item(dataset, idx, x + i * width, y + i);
  }

  int classes = dataset->classes;
  it->x = utensor_init(x, (int[]){desc.batch_size, width}, 2);
  it->y = utensor_init(y, (int[]){desc.batch_size, classes}, 2);
  it->i += 1;

  return 1;
}
