import torch
import numpy as np

xy = np.loadtxt('../data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])
print('X shape: {} | Y shape: {}'.format(x_data.shape, y_data.shape))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 1. 계층 수를 늘려보기
        self.linear1 = torch.nn.Linear(8, 20)
        self.linear2 = torch.nn.Linear(20, 18)
        self.linear3 = torch.nn.Linear(18, 16)
        self.linear4 = torch.nn.Linear(16, 14)
        self.linear5 = torch.nn.Linear(14, 12)
        self.linear6 = torch.nn.Linear(12, 10)
        self.linear7 = torch.nn.Linear(10, 8)
        self.linear8 = torch.nn.Linear(8, 6)
        self.linear9 = torch.nn.Linear(6, 4)
        self.linear10 = torch.nn.Linear(4, 1)

        # 2. 다른 활성화 함수를 사용해보기
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.relu(self.linear4(out))
        out = self.relu(self.linear5(out))
        out = self.relu(self.linear6(out))
        out = self.relu(self.linear7(out))
        out = self.relu(self.linear8(out))
        out = self.relu(self.linear9(out))
        y_pred = self.sigmoid(self.linear10(out))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)    # 3. Adam optimizer 사용해보기

for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print('epoch: {} | Loss: {:.4f}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
