// forward declarations
typedef struct Model Model;
typedef struct Dataset Dataset;

typedef struct {
  int width;
  int batch_size;
  int max_epochs;
  float acceptable_loss;
} TrainerDesc;

typedef struct Trainer {
  float *y_hat;
  int width;
  int batch_size;
  int max_epochs;
  float acceptable_loss;
} Trainer;

Trainer trainer_init(TrainerDesc desc);
void trainer_deinit(const Trainer *trainer);

void trainer_fit(Trainer *trainer, Model *model, Dataset *dataset);
