#include <stdio.h>

typedef unsigned char ubyte;

typedef struct {
  FILE *images_fp;
  FILE *labels_fp;
  int number_of_images;
  int n_rows;
  int n_cols;
} FashionMnistReader;

FashionMnistReader fashionmnist_reader_init(const char *root_dir);
void fashionmnist_reader_deinit(FashionMnistReader *reader);

void fashionmnist_read_images_to(FashionMnistReader *reader, ubyte *out);
void fashionmnist_read_labels_to(FashionMnistReader *reader, ubyte *out);
