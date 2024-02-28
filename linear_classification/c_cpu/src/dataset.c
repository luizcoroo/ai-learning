#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "fashion_mnist_reader.h"
#include "util.h"

Dataset dataset_init(DatasetDesc desc) {
  FashionMnistReader reader = fashionmnist_reader_init(desc.root_dir);
  int image_size = reader.n_rows * reader.n_cols;

  Dataset d = {
      .x = malloc(sizeof(ubyte) * reader.number_of_images * image_size),
      .y = malloc(sizeof(ubyte) * reader.number_of_images),
      .size = reader.number_of_images,
      .width = image_size,
      .classes = 10,
  };

  fashionmnist_read_images_to(&reader, d.x);
  fashionmnist_read_labels_to(&reader, d.y);
  fashionmnist_reader_deinit(&reader);

  return d;
}

void dataset_deinit(const Dataset *d) {
  free(d->x);
  free(d->y);
}

void dataset_get_item(const Dataset *d, int i, ubyte *x_out, ubyte *y_out) {
  memcpy(x_out, d->x + i * d->width, sizeof(ubyte) * d->width);
  *y_out = d->y[i];
}
