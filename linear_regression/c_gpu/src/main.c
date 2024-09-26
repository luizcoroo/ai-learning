#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataloader.h"
#include "dataset.h"
#include "model.h"
#include "util.h"

#define WEIGHT_LEN 4

int main() {
  srand(1);

  float w[WEIGHT_LEN] = { 3.0, -2.0, 1.5, -0.5 };
  float b = 2;
  // randnfarr(w, WEIGHT_LEN);
  // float b = randnf()

  // 3.0, -2.0, 1.5, -0.5

  int batch_size = 100;
  int train_size = 100;
  float learning_rate = 0.01;
  float weight_decay = 0;
  float noise = 0.01;
  int max_epochs = 1000;

  Dataset dataset = dataset_init((DatasetDesc){
      .source_w = w,
      .source_b = b,
      .width = WEIGHT_LEN,
      .size = train_size,
      .noise = noise,
  });

  Model model = model_init((ModelDesc){
      .width = WEIGHT_LEN,
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
    printf("%lf\n", loss);
  }

  double time_taken = ((double)clock() - t0) / CLOCKS_PER_SEC;
  printf("%lf, %f\n", time_taken, loss);

  model_to_cpu(&model);

  printf("w = ");
  for(int i = 0; i < WEIGHT_LEN; i++)
    printf("%f, ", model.w[i]);

    
  printf("b = %f", *model.b);

  dataset_deinit(&dataset);
  model_deinit(&model);
  dataloader_deinit(&dataloader);

  return 0;
}
