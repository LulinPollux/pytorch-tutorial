import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# Before training
print('predict (before): {} -> {:.3f}'.format(4, forward(4).item()))

# Training loop
for epoch in range(30):
    lr = 0.01
    loss_val = 0

    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print('\tgrad: {} {} {:.3f}'.format(x, y, w.grad.item()))
        w.data = w.data - lr * w.grad.item()
        w.grad.data.zero_()

    print('epoch= {}, w= {}, loss= {:.3f}'.format(epoch, w, loss_val.item()))

# After training
print('predict (after): {} -> {:.3f}'.format(4, forward(4).item()))
