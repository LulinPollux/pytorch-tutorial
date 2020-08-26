import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channel, 3x3 사각형 컨볼루션 커널
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # affine 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)   # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 창으로 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 정사각형인 경우 단일 숫자만으로도 지정할 수 있다. (필수는 아님)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_feature(self, x):
        size = x.size()[1:]     # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s   # 1 x 행 x 열 (ex: 1 x 32 x 32 = 1024)
        return num_features


net = Net()
print(net)

# 모델의 학습 가능한 매개변수들은 net.parameters() 에 의해 반환됩니다.
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 임의의 32x32 입력값을 넣어보겠습니다.
input = torch.randn(1, 1, 32, 32)   # 배치, 채널, 행, 열
out = net(input)
print(out)

# 모든 매개변수의 변화도 버퍼(gradient buffer)를 0으로 설정하고, 무작위 값으로 역전파를 합니다.
net.zero_grad()
out.backward(torch.randn(1, 10))

'''
    손실 함수 (Loss Function)
'''
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()    # 평균 제곱 오차

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)     # MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU

'''
    역전파 (Backprop)
'''
net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듭니다.

print('backward 전의 conv1.bias.grad')
print(net.conv1.bias.grad)

loss.backward()

print('backward 후의 conv1.bias.grad')
print(net.conv1.bias.grad)

'''
    가중치 갱신
'''
import torch.optim as optim

# Optimizer를 생성한다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
