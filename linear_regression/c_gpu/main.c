#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "model.h"
#include "trainer.h"

float distance(const float *arr1, const float *arr2, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    float diff = arr1[i] - arr2[i];
    sum += diff * diff;
  }
  return sqrt(sum / n);
}

int main() {
  float w[6] = {2, -3.4, 4, 0.1, -5.2, 4.2};
  int batch_size = 64;

  srand(time(NULL));

  Dataset dataset = dataset_init((DatasetDesc){
      .w = w,
      .width = 6,
      .num_train = batch_size * 100,
      .num_val = batch_size * 100,
      .noise = 0.01,
  });

  Model model = model_init((ModelDesc){
      .w = (float[6]){0},
      .width = 6,
      .weight_decay = 2,
      .learning_rate = 0.01,
      .noise = 0.01,
  });

  Trainer trainer = trainer_init((TrainerDesc){
      .batch_size = batch_size,
      .max_epochs = 100,
      .acceptable_loss = 0,
  });

  clock_t t0 = clock();

  trainer_fit(&trainer, &model, &dataset);

  double time_taken = ((double)clock() - t0) / CLOCKS_PER_SEC;
  printf("%f, %f, %f\n", time_taken, distance(w, model.w, 5),
         distance(w + 5, model.w + 5, 1));

  dataset_deinit(&dataset);
  model_deinit(&model);
  trainer_deinit(&trainer);

  return 0;
}
