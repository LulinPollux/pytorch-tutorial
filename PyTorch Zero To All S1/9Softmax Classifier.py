import torch
import numpy as np

# Cross Entropy example using Numpy
Y = np.array([1, 0, 0])
Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print('np Loss1: {:.4}'.format(np.sum(-Y * np.log(Y_pred1))))  # 0.3567
print('np Loss2: {:.4}'.format(np.sum(-Y * np.log(Y_pred2))))  # 2.303

# Softmax + CrossEntropy
loss = torch.nn.CrossEntropyLoss()

# 정답 데이터는 one-hot이 아님! 클래스(카테고리)로 이루어짐
# 0 <= value < 클래스 개수 (0, 1, 2, ...)
Y = torch.tensor([0])

# 예측 데이터는 softmax를 거치기 전의 값임! (logit)
Y_pred1 = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred2 = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
print('Torch Loss1: {:.4f} | Torch Loss2: {:.4f}'.format(l1.item(), l2.item()))
print('Y_pred1:', torch.max(Y_pred1, 1)[1].item())
print('Y_pred2:', torch.max(Y_pred2, 1)[1].item())


# 배치로 처리할 경우
Y = torch.tensor([2, 0, 1])

Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])

Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
print('Batch Loss1: {:.4f} | Batch Loss2: {:.4f}'.format(l1.item(), l2.item()))
