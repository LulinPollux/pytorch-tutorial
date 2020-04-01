import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
    Tensors
'''
# 초기화되지 않은 5x3 행렬 생성
x = torch.empty(5, 3).to(device)
print(x)

# 무작위로 초기화된 행렬 생성
x = torch.rand(5, 3).to(device)
print(x)

# dtype이 long이고 0으로 채워진 행렬을 생성
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 데이터로부터 tensor를 직접 생성
x = torch.tensor([5.5, 3])
print(x)

# 존재하는 tensor를 바탕으로 tensor를 만듬
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 행렬의 크기를 구함
print(x.size())

'''
    연산 (Operations)
'''
# 덧셈 연산
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 덧셈: In-place 방식
y.add_(x)
print(y)

# tensor의 크기(size)나 모양(shape)을 변경
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # -1은 다른 차원들을 사용하여 유추합니다.
print(x.size(), y.size(), z.size())

# .item()을 사용하면 숫자 값을 얻을 수 있음.
x = torch.randn(1)
print(x)
print(x.item())

'''
    Numpy 변환
'''
# Torch Tensor를 Numpy 배열로 변환하기
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# CPU Torch Tensor와 NumPy 배열은 저장 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됨.
a.add_(1)
print(a)
print(b)

# NumPy 배열을 Torch Tensor로 변환하기
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

'''
    CUDA Tensors
'''
# .to()를 사용하여 Tensor를 어떤 장치로도 이동시킬 수 있습니다. (move)
y = torch.ones_like(x, device=device)   # GPU 상에 직접적으로 tensor를 생성
x = x.to(device)    # .to()로 CPU에서 생성한 tensor를 GPU로 옮김
z = x + y
print(x, y)
print(z)
print(z.to("cpu", torch.double))    # .to()는 dtype도 함께 변경 가능
