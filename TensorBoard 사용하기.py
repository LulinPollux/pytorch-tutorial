import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epoch = 10
batch_size = 256

# transforms
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))])

# 데이터셋 준비
trainset = torchvision.datasets.FashionMNIST(root='./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', download=True, train=False, transform=transform)

# 데이터셋 로딩
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

# 클래스(카테고리) 목록
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# 모델 클래스
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model().to(device)

# 손실 함수, 최적화기 구성
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

'''
    1. TensorBoard 설정
'''
# 기본 log_dir은 'runs'이다.
writer = torch.utils.tensorboard.SummaryWriter('runs/fashion_mnist')

'''
    2. TensorBoard에 기록하기
'''
# 임의의 학습 이미지를 가져옵니다.
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 그리드를 만듭니다.
img_grid = torchvision.utils.make_grid(images)

# tensorboard에 기록합니다.
writer.add_image('Fashion MNIST images', img_grid)

# 모델을 시각화합니다.
writer.add_graph(model, images.to(device))
writer.close()

'''
    3. TensorBoard로 모델 학습 추적하기
'''
def train(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)

        output = model(x_data)

        loss = criterion(output, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(trainloader.dataset)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_data, y_data in test_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)

            output = model(x_data)

            test_loss += criterion(output, y_data, reduction='sum').item()

            pred = torch.max(output, 1)[1]
            correct += pred.eq(y_data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy


if __name__ == '__main__':
    for epoch in range(epoch):
        train_loss = train(model, trainloader, criterion, optimizer)
        writer.add_scalar('Training loss', train_loss, epoch)

        test_loss, test_accuracy = test(model, testloader, criterion)
        writer.add_scalar('Test loss', test_loss, epoch)
