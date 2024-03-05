typedef unsigned char ubyte;

typedef struct {
  const char *root_dir;
} DatasetDesc;

typedef struct Dataset {
  ubyte *x;
  ubyte *y;
  int size, width, classes;
} Dataset;

Dataset dataset_init(DatasetDesc desc);
void dataset_deinit(const Dataset *dataset);

void dataset_get_item(const Dataset *dataset, int i, ubyte *x, ubyte *y);
