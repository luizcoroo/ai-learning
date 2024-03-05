#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataloader.h"
#include "dataset.h"
#include "model.h"
#include "util.h"

// int main() {
//   float data[16];
//   for (int i = 0; i < 16; i++)
//     data[i] = i;
//
//   TensorView view = tensor_view(data, (int[]){2, 2, 4}, 3);
//   tensor_describe(view);
//   printf("\n");
//
//   float data2[12];
//   for (int i = 0; i < 12; i++)
//     data2[i] = i;
//
//   TensorView view2 = tensor_view(data2, (int[]){4, 3}, 2);
//   tensor_describe(view2);
//   printf("\n");
//
//   float data3[1000];
//   TensorView view3 = tensor_matmul(data3, view, view2);
//   tensor_describe(view3);
//   printf("\n");
// }

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
  }

  double time_taken = ((double)clock() - t0) / CLOCKS_PER_SEC;
  printf("%lf, %f\n", time_taken, loss);

  dataset_deinit(&dataset);
  model_deinit(&model);
  dataloader_deinit(&dataloader);

  return 0;
}
