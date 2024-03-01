#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataloader.h"
#include "dataset.h"
#include "model.h"
#include "util.h"

int main() {
  srand(1);

  int batch_size = 64;
  float learning_rate = 0.001;
  float weight_decay = 0.1;
  float noise = 0.01;
  int max_epochs = 30;

  Dataset dataset =
      dataset_init((DatasetDesc){.root_dir = "../../data/FashionMNIST/raw/"});

  Model model = model_init((ModelDesc){
      .input_width = dataset.width,
      .output_width = dataset.classes,
      .learning_rate = learning_rate,
      .weight_decay = weight_decay,
      .noise = noise,
  });

  DataLoader dataloader = dataloader_init((DataLoaderDesc){
      .dataset = &dataset,
      .batch_size = batch_size,
  });

  clock_t t0 = clock();

  float loss = -1;
  for (int e = 0; e < max_epochs; e++) {
    DataLoaderIterator it = dataloader_iterator(&dataloader);

    loss = 0;
    while (dataloader_iterator_next(&it)) {
      model_forward(&model, it.x, it.y_hat, it.size);
      loss += model_evaluate(&model, it.y_hat, it.y, it.size);
      model_backward(&model, it.x, it.y_hat, it.y, it.size);
      model_update(&model);
    }
    loss /= dataloader.n_batches;
    printf("%f\n", loss);
  }

  double time_taken = ((double)clock() - t0) / CLOCKS_PER_SEC;
  printf("%lf, %f\n", time_taken, loss);

  {
    DataLoaderIterator it = dataloader_iterator(&dataloader);
    dataloader_iterator_next(&it);
    model_forward(&model, it.x, it.y_hat, it.size);

    for(int i = 0; i < it.size; ++i) {
      for(int j = 0; j < model.output_width; ++j)
        printf("%.2f ", expf(it.y_hat[i * model.output_width + j]));

      printf(" -- %d\n", it.y[i]);
    }

  }

  dataset_deinit(&dataset);
  model_deinit(&model);
  dataloader_deinit(&dataloader);

  return 0;
}
