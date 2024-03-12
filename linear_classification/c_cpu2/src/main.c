#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"
#include "dataloader.h"
#include "dataset.h"
#include "model.h"
#include "util.h"

int main() {
  srand(1);

  int batch_size = 64;
  float learning_rate = 0.001;
  float weight_decay = 0.1;
  int max_epochs = 30;

  Dataset dataset =
      dataset_init((DatasetDesc){.root_dir = "../../data/FashionMNIST/raw/"});

  Model model = model_init((ModelDesc){
      .input_width = dataset.width,
      .output_width = dataset.classes,
      .learning_rate = learning_rate,
      .weight_decay = weight_decay,
  });

  DataLoader dataloader = dataloader_init((DataLoaderDesc){
      .dataset = &dataset,
      .batch_size = batch_size,
  });

  float_t *out_data =
      malloc(2 * sizeof(float_t) * batch_size * dataset.classes);

  clock_t t0 = clock();

  float_t loss = -1;
  for (int e = 0; e < max_epochs; e++) {
    DataLoaderIterator it = dataloader_iterator(&dataloader);

    loss = 0;
    while (dataloader_iterator_next(&it)) {
      FTensor y_hat = model_forward(&model, it.x, out_data, batch_size);
      loss += model_evaluate(&model, it.y, y_hat, batch_size);
      model_backward(&model, it.x, it.y, y_hat, batch_size);
      model_update(&model);
    }
    loss /= dataset.size / (float_t)batch_size;
  }

  double time_taken = ((double)clock() - t0) / CLOCKS_PER_SEC;
  printf("%lf, %f\n", time_taken, loss);

  dataset_deinit(&dataset);
  model_deinit(&model);
  dataloader_deinit(&dataloader);

  return 0;
}
