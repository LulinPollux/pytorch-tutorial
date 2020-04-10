import torch
import torch.utils.data
import numpy as np


# 데이터셋 초기화, 다운로드 등등
class DiabetesDataset(torch.utils.data.Dataset):
    def __init__(self):
        xy = np.loadtxt('../data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# Data Loader로 데이터 로딩
dataset = DiabetesDataset()
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.linear1(x))
        out2 = self.sigmoid(self.linear2(out1))
        y_pred = self.sigmoid(self.linear3(out2))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(2):
    for batch_idx, xy_data in enumerate(train_loader):
        x_data, y_data = xy_data   # 하나의 배치로 들어온 데이터를 x_data, y_data로 언패킹
        x_data, y_data = torch.tensor(x_data), torch.tensor(y_data)  # 텐서로 자료형 변환(생략 가능)

        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)
        print('epoch: {} | batch: {} | Loss: {:.4f}'.format(epoch, batch_idx, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
