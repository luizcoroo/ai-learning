from trainer import Trainer
from data import SyntheticRegressionData
from model import WeightDecayLinearRegression

import time
import math
import torch


def distance(a, b):
    return math.sqrt(sum((a - b) ** 2))


batch_size = 64
data = SyntheticRegressionData(
    w=torch.tensor([2, -3.4, 4, 0.1, -5.2]),
    b=4.2,
    noise=0.01,
    num_train=batch_size * 100,
    num_val=batch_size * 100,
    batch_size=batch_size,
)


trainer = Trainer(max_epochs=100)
model = WeightDecayLinearRegression(wd=2, lr=0.01)

t0 = time.time()
trainer.fit(model, data)
t1 = time.time()

with torch.no_grad():
    model_w, model_b = model.get_w_b()
    print(
        f"{t1 - t0:.6f}, {distance(data.w, model_w.reshape(data.w.shape)):.6f}, {distance(data.b, model_b):.6f}"
    )
