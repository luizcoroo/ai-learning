from util import HyperParameters

import torch


class WeightDecayLinearRegression(torch.nn.Module, HyperParameters):
    def __init__(self, wd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def loss(self, y_hat, y):
        fn = torch.nn.MSELoss()
        return fn(y_hat, y)

    def forward(self, X):
        return self.net(X)

    def training_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {"params": self.net.weight, "weight_decay": self.wd},
                {"params": self.net.bias},
            ],
            lr=self.lr,
        )

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)
