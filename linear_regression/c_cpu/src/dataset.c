#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "util.h"

Dataset dataset_init(DatasetDesc desc) {
  Dataset d = {
      .x = malloc(sizeof(float) * desc.size * desc.width),
      .y = malloc(sizeof(float) * desc.size),
      .size = desc.size,
      .width = desc.width,
  };

  for (int i = 0; i < d.size; i++)
    for (int j = 0; j < d.width; j++)
      d.x[i * d.width + j] = randnf();

  for (int i = 0; i < d.size; i++) {
    d.y[i] = desc.source_b + randnf() * desc.noise;
    for (int j = 0; j < d.width; j++)
      d.y[i] += d.x[i * d.width + j] * desc.source_w[j];
  }

  return d;
}

void dataset_deinit(const Dataset *d) {
  free(d->x);
  free(d->y);
}

void dataset_get_item(const Dataset *d, int i, float *x_out, float *y_out) {
  memcpy(x_out, d->x + i * d->width, sizeof(float) * d->width);
  *y_out = d->y[i];
}
