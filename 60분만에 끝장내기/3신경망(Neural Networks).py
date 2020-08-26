# 튜토리얼 설명 먼저 읽기

"""
1. 신경망 정의하기
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 정사각형 convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # affine 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)   # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) window로 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 만약 크기가 정사각형인 경우 단일 숫자만으로도 지정할 수 있다. (필수는 아님)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s   # 1 x 행 x 열 (ex: 1 x 32 x 32 = 1024)
        return num_features


net = Net()
print(net)

# forward 함수만 정의하고 나면, (변화도를 계산하는) backward 함수는 autograd 를 사용하여 자동으로 정의됩니다.
# forward 함수에서는 어떠한 Tensor 연산을 사용해도 됩니다.
# 모델의 학습 가능한 매개변수들은 net.parameters()에 의해 반환됩니다.
params = list(net.parameters())
print(len(params))
print(params[0].size())     # conv1의 가중치

# 임의의 32x32 입력값을 넣어보겠습니다.
# Note: 이 신경망(LeNet)의 예상되는 입력 크기는 32x32입니다.
# 이 신경망에 MNIST 데이터셋을 사용하기 위해서는, 데이터셋의 이미지 크기를 32x32로 변경해야 합니다.
input = torch.randn(1, 1, 32, 32)   # 배치, 채널, 행, 열
out = net(input)
print(out)

# 모든 매개변수의 변화도 버퍼(gradient buffer)를 0으로 설정하고, 무작위 값으로 역전파를 합니다.
net.zero_grad()
out.backward(torch.randn(1, 10))

'''
2. 손실 함수 (Loss Function)
'''
# 손실 함수는 (output, target)을 한 쌍(pair)의 입력으로 받아,
# 출력(output)이 정답(target)으로부터 얼마나 멀리 떨어져있는지 추정하는 값을 계산합니다.
# 간단한 손실 함수로는 출력과 대상간의 평균제곱오차(mean-squared error)를 계산하는 nn.MSEloss가 있습니다.
output = net(input)
target = torch.randn(10)        # 예시로 dummy target을 사용합니다.
target = target.view(1, -1)     # 출력과 같은 모양으로 만듭니다.
criterion = nn.MSELoss()        # 평균 제곱 오차

loss = criterion(output, target)
print(loss)

# 이제 .grad_fn속성을 사용하여 loss를 역방향에서 따라가다보면, 이러한 모습의 연산 그래프를 볼 수 있습니다.
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# 따라서 loss.backward()를 실행할 때, 전체 그래프는 손실(loss)에 대하여 미분되며,
# 그래프 내의 requires_grad=True인 모든 Tensor는 변화도(gradient)가 누적된 .grad Tensor를 갖게 됩니다.
# 설명을 위해, 역전파의 몇 단계를 따라가보겠습니다.
print(loss.grad_fn)     # MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU

'''
3. 역전파 (Backprop)
'''
# 오차(error)를 역전파하기 위해서는 loss.backward() 만 해주면 됩니다.
# 기존 변화도를 없애는 작업이 필요한데, 그렇지 않으면 변화도가 기존의 것에 누적되기 때문입니다.
# 이제 loss.backward() 를 호출하여 역전파 전과 후에 conv1의 bias gradient를 살펴보겠습니다.
net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듭니다.
print('backward 전의 conv1.bias.grad')
print(net.conv1.bias.grad)

loss.backward()
print('backward 후의 conv1.bias.grad')
print(net.conv1.bias.grad)

'''
4. 가중치 갱신
'''
# 실제로 많이 사용되는 가장 단순한 갱신 규칙은 확률적 경사하강법(SGD; Stochastic Gradient Descent)입니다.
# 가중치(wiehgt) = 가중치(weight) - 학습율(learning rate) * 변화도(gradient)
# 신경망을 구성할 때 SGD, Nesterov-SGD, Adam, RMSProp 등과 같은 다양한 갱신 규칙을 사용하고 싶을 수 있습니다.
# 이를 위해서 torch.optim 라는 작은 패키지에 이러한 방법들을 모두 구현해두었습니다. 사용법은 매우 간단합니다.
import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다.
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

# Note
# optimizer.zero_grad()를 사용하여 수동으로 변화도 버퍼를 0으로 설정하는 것에 유의하세요.
# 이는 역전파(Backprop) 섹션에서 설명한 것처럼 변화도가 누적되기 때문입니다.
