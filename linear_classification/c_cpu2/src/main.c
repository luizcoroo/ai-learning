#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataloader.h"
#include "dataset.h"
#include "model.h"
#include "tensor.h"
#include "util.h"

// int main() {
//   uint_t x_data[36];
//   for (int i = 0; i < 3; i++)
//     for (uint_t j = 0; j < 12; ++j)
//       x_data[i * 12 + j] = j;
//
//   UTensor x = utensor_init(x_data, (int[]){3, 4, 3}, 3);
//
//   printf("x: \n");
//   utensor_describe(x);
//   printf("\n");
//
//   float_t w_data[15];
//   for (uint_t i = 0; i < 15; ++i)
//     w_data[i] = (float_t)i;
//
//   FTensor w = ftensor_init(w_data, (int[]){3, 5}, 2);
//   printf("w: \n");
//   ftensor_describe(w);
//   printf("\n");
//
//   float_t b_data[5];
//   for (uint_t i = 0; i < 5; ++i)
//     b_data[i] = 1.0;
//
//   FTensor b = ftensor_init(b_data, (int[]){1}, 1);
//   printf("b: \n");
//   ftensor_describe(b);
//   printf("\n");
//
//   float_t res_data[60];
//   FTensor res = ftensor_umatmuladd(res_data, x, w, b);
//   printf("res: \n");
//   ftensor_describe(res);
//   printf("\n");
//
//   // float_t tmp_data[20];
//   // FTensor res_soft = ftensor_logsoftmax(res, tmp_data);
//   // printf("res: \n");
//   // ftensor_describe(res_soft);
//   // printf("\n");
//
//   return 0;
// }

int main() {
  srand(1);

  int batch_size = 64;
  float learning_rate = 0.001;
  float weight_decay = 0.1;
  int max_epochs = 30;

  Dataset dataset =
      dataset_init((DatasetDesc){.root_dir =
      "../../data/FashionMNIST/raw/"});

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
      model_update(&model, batch_size);
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
