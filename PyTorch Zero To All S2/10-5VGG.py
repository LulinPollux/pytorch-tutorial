import torch
import torch.utils.data
import torchvision
import visdom

vis = visdom.Visdom()
vis.close(env='main')


def loss_graph(loss_plot, loss_value, num):
    vis.line(X=num, Y=loss_value, win=loss_plot, update='append')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''
    1. CIFAR10를 불러오고 정규화하기
'''
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
    2. CNN 정의하기
'''
layers = torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['D'], batch_norm=False)
vgg16 = torchvision.models.VGG(layers, num_classes=10, init_weights=True)
vgg16.to(device)

'''
    3. 손실 함수와 Optimizer 정의하기
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.01, momentum=0.9)
lr_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

'''
    4. 신경망 학습하기
'''
epoch_plt = vis.line(Y=torch.tensor([0]), opts={'title': 'epoch_loss',
                                                'legend': ['loss'],
                                                'showlegend': 'True'})
running_plt = vis.line(Y=torch.tensor([0]), opts={'title': 'running_loss',
                                                  'legend': ['loss'],
                                                  'showlegend': 'True'})

print(len(trainloader))
for epoch in range(100):
    epoch_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        running_loss = loss.item()
        loss_graph(running_plt, torch.tensor([running_loss]), torch.tensor([i + epoch * len(trainloader)]))

    loss_graph(epoch_plt, torch.tensor([epoch_loss / len(trainloader)]), torch.tensor([epoch + 1]))
    lr_sche.step()

print('Finished Training')

# 학습한 모델을 저장한다.
PATH = './vgg16_cifar10.pth'
torch.save(vgg16.state_dict(), PATH)

'''
    5. 신경망 테스트하기
'''
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지 알아보기
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
