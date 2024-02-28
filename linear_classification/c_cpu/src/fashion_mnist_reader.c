#include <stdlib.h>

#include "fashion_mnist_reader.h"

void write_filename_to(const char *base_dir, const char *filename, char *out);
int reverse_int(int i);

FashionMnistReader fashionmnist_reader_init(const char *root_dir) {
  FashionMnistReader reader = {0};

  char x_filename[300];
  char y_filename[300];
  write_filename_to(root_dir, "train-images-idx3-ubyte", x_filename);
  write_filename_to(root_dir, "train-labels-idx1-ubyte", y_filename);

  int magic_number = 0;

  reader.images_fp = fopen(x_filename, "rb");
  fread(&magic_number, sizeof(int), 1, reader.images_fp);
  fread(&reader.number_of_images, sizeof(int), 1, reader.images_fp);
  reader.number_of_images = reverse_int(reader.number_of_images);
  fread(&reader.n_rows, sizeof(int), 1, reader.images_fp);
  reader.n_rows = reverse_int(reader.n_rows);
  fread(&reader.n_cols, sizeof(int), 1, reader.images_fp);
  reader.n_cols = reverse_int(reader.n_cols);

  reader.labels_fp = fopen(y_filename, "rb");
  fread(&magic_number, sizeof(int), 1, reader.labels_fp);

  int number_of_labels = 0;
  fread(&number_of_labels, sizeof(int), 1, reader.labels_fp);
  number_of_labels = reverse_int(number_of_labels);

  if (number_of_labels != reader.number_of_images)
    exit(1);

  return reader;
}

void fashionmnist_reader_deinit(FashionMnistReader *reader) {
  fclose(reader->images_fp);
  fclose(reader->labels_fp);
}

void fashionmnist_read_images_to(FashionMnistReader *reader, ubyte *out) {
  int bytes = reader->number_of_images * reader->n_rows * reader->n_cols;
  fread(out, sizeof(ubyte), bytes, reader->images_fp);
}

void fashionmnist_read_labels_to(FashionMnistReader *reader, ubyte *out) {
  int bytes = reader->number_of_images;
  fread(out, sizeof(ubyte), bytes, reader->labels_fp);
}

void write_filename_to(const char *base_dir, const char *filename, char *out) {
  int i = 0;
  int j = 0;
  while (base_dir[j] != '\0')
    out[i++] = base_dir[j++];

  j = 0;
  while (filename[j] != '\0')
    out[i++] = filename[j++];

  out[i] = '\0';
}

int reverse_int(int i) {
  ubyte b1, b2, b3, b4;

  b1 = i & 255;
  b2 = (i >> 8) & 255;
  b3 = (i >> 16) & 255;
  b4 = (i >> 24) & 255;

  return ((int)b1 << 24) + ((int)b2 << 16) + ((int)b3 << 8) + b4;
}
