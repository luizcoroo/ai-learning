import torch
import time


class LinearRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, w, b, n, noise=0.01):
        noise = torch.randn(n, 1) * noise
        self.x = torch.randn(n, len(w))
        self.y = torch.matmul(self.x, torch.reshape(w, (-1, 1))) + b + noise

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(2024)

weight_len = 4098
batch_size = 64
train_size = batch_size * 100
max_epochs = 100
device = torch.device("cuda")

dataset = LinearRegressionDataset(
    w=torch.randn(weight_len),
    b=torch.randn(1),
    noise=0.01,
    n=train_size,
)
model = LinearRegressionModel(in_features=weight_len)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, weight_decay=0.1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

t0 = time.time()
loss = None

model.to(device)
for _ in range(max_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()

t1 = time.time()

print(f"{t1 - t0:.6f}, {loss:.6f}")
