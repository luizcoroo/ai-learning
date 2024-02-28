import time
import torch
from torchvision import transforms, datasets


class LinearClassificationModel(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super(LinearClassificationModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=in_features, out_features=out_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(1)

input_len = 28 * 28
output_len = 10
batch_size = 64
learning_rate = 0.001
weight_decay = 0.1
max_epochs = 10
device = torch.device("cuda")

dataset = datasets.FashionMNIST(
    root="../../data", train=True, download=False, transform=transforms.ToTensor()
)
model = LinearClassificationModel(in_features=input_len, out_classes=output_len)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

t0 = time.time()
last_loss = None

model.to(device)
for _ in range(max_epochs):
    last_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        last_loss += loss
    last_loss /= len(dataloader)

t1 = time.time()

print(f"{t1 - t0:.6f}, {last_loss:.6f}")
