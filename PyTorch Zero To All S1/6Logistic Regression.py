import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))  # 시그모이드 함수(F.sigmoid is deprecated)
        return y_pred


model = Model()

criterion = torch.nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print('epoch: {} | Loss: {:.4f}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = torch.tensor([[1.0]])
y_pred = model(x)
print('Predict (after): {} -> {}'.format(x.item(), y_pred.item() > 0.5))
x = torch.tensor([[7.0]])
print('Predict (after): {} -> {}'.format(x.item(), model(x).item() > 0.5))
