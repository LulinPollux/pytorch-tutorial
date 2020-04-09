import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


# 1. 모델 클래스 설계
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # (input size, output size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()

# 2. 손실 함수, 최적화기 구성 (직접 만들거나 API에서 선택)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. 훈련 (forward, loss, backward, step)
for epoch in range(500):
    # 순전파(forward): 모델에 x를 넣어서 예측 y를 계산
    y_pred = model(x_data)

    # 손실 계산
    loss = criterion(y_pred, y_data)
    print('epoch: {} | Loss: {:.3f}'.format(epoch, loss.item()))

    # 기울기를 0으로 초기화, 역전파 수행, 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 모델 테스트
x = torch.tensor([[4.0]])
y_pred = model(x)
print('Predict (after): {} -> {:.3f}'.format(x.item(), y_pred.item()))
