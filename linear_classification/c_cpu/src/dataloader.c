#include <stdlib.h>

#include "dataloader.h"
#include "dataset.h"
#include "util.h"

DataLoader dataloader_init(DataLoaderDesc desc) {
  int number_of_floats =
      desc.batch_size * desc.dataset->width + 2 * desc.batch_size;

  DataLoader dl = {
      .data = malloc(sizeof(float) * number_of_floats),
      .permutation = malloc(sizeof(int) * desc.dataset->size),
      .batch_size = desc.batch_size,
      .n_batches = desc.dataset->size / desc.batch_size,
      .dataset = desc.dataset,
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
  int x_end = dl->batch_size * dl->dataset->width;
  int y_end = x_end + dl->batch_size;
  shuffleiarr(dl->permutation, dl->dataset->size);

  return (DataLoaderIterator){
      .x = dl->data + 0,
      .y = dl->data + x_end,
      .y_hat = dl->data + y_end,
      .size = dl->batch_size,
      .i = 0,
      .ref = dl,
  };
}

int dataloader_iterator_next(DataLoaderIterator *it) {
  if (it->i >= it->ref->n_batches)
    return 0;

  int batch_start = it->i * it->size;
  int width = it->ref->dataset->width;
  for (int i = 0; i < it->size; i++) {
    int idx = it->ref->permutation[batch_start + i];
    dataset_get_item(it->ref->dataset, idx, it->x + i * width, it->y + i);
  }

  it->i += 1;
  return 1;
}
