import torch
import torch.utils.data
import torchvision.datasets
import time

# Training settings
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training MNIST Model on', device)
print("=" * 60)

# MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 520)
        self.l2 = torch.nn.Linear(520, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        return self.l5(x)


model = Net()
model.to(device)    # GPU 사용이 가능하면 모델을 GPU로 옮김

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        # x_data, y_data를 GPU로 옮김 (GPU 사용 가능시)
        x_data, y_data = x_data.to(device), y_data.to(device)

        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 훈련 결과 출력
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(x_data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    loss = 0
    correct = 0
    for x_data, y_data in test_loader:
        x_data, y_data = x_data.to(device), y_data.to(device)

        y_pred = model(x_data)

        loss += criterion(y_pred, y_data).item()

        pred = torch.max(y_pred, 1)[1]
        correct += pred.eq(y_data.view_as(pred)).cpu().sum()

    loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()

        train(epoch)
        min, sec = divmod(time.time() - epoch_start, 60)
        print('Training time: {:.0f}m {:.0f}s'.format(min, sec))

        test()
        min, sec = divmod(time.time() - epoch_start, 60)
        print('Testing time: {:.0f}m {:.0f}s'.format(min, sec))
        print("=" * 60)

    min, sec = divmod(time.time() - since, 60)
    print('Total time: {:.0f}m {:.0f}s'.format(min, sec))
