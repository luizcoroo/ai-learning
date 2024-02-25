#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "dataloader.h"
#include "dataset.h"
#include "util.h"

DataLoader dataloader_init(DataLoaderDesc desc) {

  DataLoader dl = {
      .permutation = malloc(sizeof(int) * desc.dataset->size),
      .batch_size = desc.batch_size,
      .n_batches = desc.dataset->size / desc.batch_size,
      .dataset = desc.dataset,
  };

  int bytes = sizeof(float) * desc.batch_size * (desc.dataset->width + 2);
  cudaMallocHost((void **)&dl.cpu_data, bytes);
  cudaMalloc((void **)&dl.gpu_data, bytes);

  for (int i = 0; i < desc.dataset->size; i++)
    dl.permutation[i] = i;

  return dl;
}

void dataloader_deinit(const DataLoader *dl) {
  free(dl->permutation);
  cudaFreeHost(dl->cpu_data);
  cudaFree(dl->gpu_data);
}

DataLoaderIterator dataloader_iterator(DataLoader *dl) {
  shuffleiarr(dl->permutation, dl->dataset->size);

  DataLoaderIterator it = {.size = dl->batch_size, .i = 0, .ref = dl};
  it.x = dl->gpu_data;
  it.y = it.x + dl->batch_size * dl->dataset->width;
  it.y_hat = it.y + dl->batch_size;

  return it;
}

int dataloader_iterator_next(DataLoaderIterator *it) {
  if (it->i >= it->ref->n_batches)
    return 0;

  DataLoader *dl = it->ref;
  const Dataset *d = dl->dataset;

  float *x = dl->cpu_data;
  float *y = x + it->size * d->width;
  const int *perm = dl->permutation;

  int start = it->i * it->size;
  for (int i = 0; i < it->size; i++)
    dataset_get_item(d, perm[start + i], x + i * d->width, y + i);

  int bytes = sizeof(float) * it->size * (d->width + 1);
  cudaMemcpy(dl->gpu_data, dl->cpu_data, bytes, cudaMemcpyHostToDevice);

  it->i += 1;
  return 1;
}
