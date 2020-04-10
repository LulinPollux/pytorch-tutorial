import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import time

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training MNIST Model on', device)
print("=" * 60)

train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # NLLLoss에는 log softmax를 사용


model = Net()
model.to(device)

criterion = torch.nn.NLLLoss()  # 손실 함수를 NLLLoss로 변경
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)

        y_pred = model(x_data)

        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
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
