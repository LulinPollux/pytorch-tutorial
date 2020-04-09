x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0     # 랜덤값


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 기울기(gradient) 계산 2x(xw - y)
def gradient(x, y):
    return 2 * x * (x * w - y)


# Before training
print('predict (before): {} -> {:.3f}'.format(4, forward(4)))

# Training loop
for epoch in range(30):
    lr = 0.01
    loss_val = 0

    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - lr * grad
        print('\tgrad: {} {} {:.3f}'.format(x, y, grad))
        loss_val = loss(x, y)

    print('epoch= {}, w= {:.3f}, loss= {:.3f}'.format(epoch, w, loss_val))

# After training
print('predict (after): {} -> {:.3f}'.format(4, forward(4)))
