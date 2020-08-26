# 튜토리얼 설명 먼저 읽기 (특히 "데이터는 어떻게 하나요?" 절은 반드시 읽을 것!)

"""
1. CIFAR10를 불러오고 정규화하기
"""
# torchvision을 사용하여 매우 쉽게 CIFAR10을 불러올 수 있습니다.
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage 이미지입니다.
# 이를 [-1, 1]의 범위로 정규화된 Tensor로 변환합니다.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 재미삼아 학습용 이미지 몇 개를 보겠습니다.
import matplotlib.pyplot as plt
import numpy as np


# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize (정규화 복원)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

'''
2. 합성곱 신경망(Convolution Neural Network) 정의하기
'''
# 이전의 신경망 섹션에서 신경망을 복사한 후,
# 기존에 1채널 이미지만 처리하도록 정의된 것을 3채널 이미지를 처리할 수 있도록 수정합니다.
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

'''
3. 손실 함수와 Optimizer 정의하기
'''
# 교차 엔트로피 손실(Cross-Entropy loss)과 모멘텀(momentum) 값을 갖는 SGD를 사용합니다.
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
4. 신경망 학습하기
'''
# 이제 재미있는 부분이 시작됩니다.
# 단순히 데이터를 반복해서 신경망에 입력으로 제공하고, 최적화(Optimize)만 하면 됩니다.
for epoch in range(2):  # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후
        inputs, labels = data[0].to(device), data[1].to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:  # 2000개의 미니배치마다 출력
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 학습한 모델을 저장해보겠습니다.
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

'''
5. 시험용 데이터로 신경망 검사하기
'''
# 지금까지 학습용 데이터셋을 2회 반복하며 신경망을 학습시켰습니다.
# 신경망이 전혀 배운게 없을지도 모르니 확인해봅니다.
# 신경망이 예측한 출력과 진짜 정답(Ground-truth)을 비교하는 방식으로 확인합니다.
# 만약 예측이 맞다면 샘플을 ‘맞은 예측값(correct predictions)’ 목록에 넣겠습니다.
# 첫번째로 시험용 데이터를 좀 보겠습니다.
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 이제, 저장했던 모델을 불러오도록 하겠습니다.
# 여기에서는 모델을 저장하고 다시 불러오는 작업이 불필요하지만, 어떻게 하는지 설명을 위해 해보겠습니다.
net = Net()
net.load_state_dict(torch.load(PATH))

# 좋습니다, 이제 이 예제들을 신경망이 어떻게 예측했는지를 보겠습니다.
outputs = net(images)

# 출력은 10개 분류 각각에 대한 값으로 나타납니다.
# 어떤 분류에 대해서 더 높은 값이 나타난다는 것은, 신경망이 그 이미지가 해당 분류에 더 가깝다고 생각한다는 것입니다.
# 따라서 가장 높은 값을 갖는 인덱스(index)를 뽑아보겠습니다:
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 결과가 괜찮아보이네요. 그럼 전체 데이터셋에 대해서는 어떻게 동작하는지 보겠습니다.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# (10가지 분류 중에 하나를 무작위로) 찍었을 때의 정확도인 10% 보다는 나아보입니다.
# 신경망이 뭔가 배우긴 한 것 같네요.
# 그럼 어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지 알아보겠습니다.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

'''
6. GPU에서 학습하기 (본 코드에 이미 적용되어 있음)
'''
# Tensor를 GPU로 이동했던 것처럼, 신경망 또한 GPU로 옮길 수 있습니다.
# 1. (CUDA를 사용할 수 있다면) 첫번째 CUDA 장치를 사용하도록 설정합니다.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 2. 이 함수는 재귀적으로 모든 모듈의 매개변수와 버퍼를 CUDA tensor로 변경합니다.
# net.to(device)
# 3. 각 단계에서 입력(input)과 정답(target)도 GPU로 보내야 한다는 것도 기억해야 합니다.
# inputs, labels = data[0].to(device), data[1].to(device)

# CPU와 비교했을 때 어마어마한 속도 차이가 나지 않는 것은 왜 그럴까요?
# 그 이유는 바로 신경망이 너무 작기 때문입니다.
# 연습: 신경망의 크기를 키워보고, 얼마나 빨라지는지 확인해보세요.
# (첫번째 nn.Conv2d 의 2번째 인자와 두번째 nn.Conv2d 의 1번째 인자는 같은 숫자여야 합니다.)
