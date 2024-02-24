#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "model.h"
#include "trainer.h"

Trainer trainer_init(TrainerDesc d) {
  return (Trainer){
      .y_hat = malloc(sizeof(float) * d.batch_size),
      .batch_size = d.batch_size,
      .max_epochs = d.max_epochs,
      .acceptable_loss = d.acceptable_loss,
  };
}

void trainer_deinit(const Trainer *t) { free(t->y_hat); }

void trainer_fit(Trainer *t, Model *m, Dataset *d) {
  float grad_tmp[m->width];
  float loss = FLT_MAX;

  for (int e = 0; loss > t->acceptable_loss && e < t->max_epochs; e++) {
    dataset_shuffle_train(d);

    for (int i = 0; i < d->num_train; i += t->batch_size) {
      DataDesc batch = {
          .x = d->x + (i * m->width),
          .y = d->y + i,
          .size = t->batch_size,
      };

      model_forward(m, &batch, t->y_hat);
      model_compute_grad(m, &batch, t->y_hat, grad_tmp);
      model_backward(m, grad_tmp);
    }

    loss = 0;
    for (int i = d->num_train; i < d->size; i += t->batch_size) {
      DataDesc batch = {
          .x = d->x + (i * m->width),
          .y = d->y + i,
          .size = t->batch_size,
      };

      model_forward(m, &batch, t->y_hat);
      loss += model_evaluate(m, &batch, t->y_hat);
    }
    loss /= (d->size - d->num_train) / (float)t->batch_size;
  }
}
