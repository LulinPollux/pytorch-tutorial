import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0     # 랜덤값, weight


# 해당 모델의 순전파(forward pass)
def forward(x):
    return x * w


# 손실 함수 (Loss function)
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 각 입력에서의 가중치와 평균 제곱 오차(Mean Square Error) 리스트
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w = %.1f' % w)
    loss_sum = 0

    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        loss_val = loss(x, y)
        loss_sum += loss_val
        print('\t%.1f %.1f %.1f %.1f' % (x, y, y_pred, loss_val))

    mse = loss_sum / 3
    print('MSE = %.2f' % mse)
    w_list.append(w)
    mse_list.append(mse)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
