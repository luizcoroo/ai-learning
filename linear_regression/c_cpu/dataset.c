#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "util.h"

Dataset dataset_init(DatasetDesc desc) {
  int size = desc.num_train + desc.num_val;
  Dataset d = {
      .x = malloc(sizeof(float) * size * desc.width),
      .y = malloc(sizeof(float) * size),
      .num_train = desc.num_train,
      .size = size,
      .width = desc.width,
  };

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < d.width - 1; j++)
      d.x[i * d.width + j] = rand_normal();

    d.x[i * d.width + d.width - 1] = 1;
  }

  for (int i = 0; i < d.size; i++) {
    d.y[i] = rand_normal() * desc.noise;
    for (int j = 0; j < d.width; j++)
      d.y[i] += d.x[i * d.width + j] * desc.w[j];
  }

  return d;
}

void dataset_deinit(const Dataset *d) {
  free(d->x);
  free(d->y);
}

void dataset_shuffle_train(Dataset *d) {
  float xk_tmp[d->width];

  for (int i = 0; i < d->num_train - 1; i++) {
    size_t k = i + rand() / (RAND_MAX / (d->num_train - i) + 1);

    float *xi = d->x + (i * d->width);
    float *xk = d->x + (k * d->width);
    int bytes = sizeof(float) * d->width;

    memcpy(xk_tmp, xk, bytes);
    memcpy(xk, xi, bytes);
    memcpy(xi, xk_tmp, bytes);

    float yk_tmp = d->y[k];
    d->y[k] = d->y[i];
    d->y[i] = yk_tmp;
  }
}
