import torch

'''
1. Tensors
'''
# 초기화되지 않은 5x3 행렬을 생성합니다.
x = torch.empty(5, 3)
print(x)

# 무작위로 초기화된 행렬을 생성합니다.
x = torch.rand(5, 3)
print(x)

# dtype이 long이고 0으로 채워진 행렬을 생성합니다.
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 데이터로부터 tensor를 직접 생성합니다.
x = torch.tensor([5.5, 3])
print(x)

# 기존 tensor를 바탕으로 새로운 tensor를 만듭니다.
# 함수는 사용자로부터 새로운 값을 제공받지 않은 한, 입력 tensor의 속성들(예. dtype)을 재사용합니다.
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 행렬의 크기를 구합니다.
print(x.size())

'''
2. 연산 (Operations)
'''
# 덧셈: 문법1
y = torch.rand(5, 3)
print(x + y)

# 덧셈: 문법2
print(torch.add(x, y))

# 덧셈: 결과 tensor를 인자로 제공
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 덧셈: In-place 방식
y.add_(x)
print(y)

# NumPy스러운 인덱싱 표기 방법을 사용
print(x[:, 1])

# 크기 변경: tensor의 크기(size)나 모양(shape)을 변경하고 싶다면 torch.view 를 사용합니다.
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # -1은 다른 차원에서 유추합니다.
print(x.size(), y.size(), z.size())

# .item()을 사용하면 숫자 값을 얻을 수 있음.
x = torch.randn(1)
print(x)
print(x.item())

'''
3. Numpy 변환
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
# NumPy 배열을 변경하면 Torch Tensor의 값도 자동 변경됨.
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# CharTensor를 제외한 CPU 상의 모든 Tensor는 NumPy로 변환할 수 있고, (NumPy에서 Tensor로의) 반대 변환도 가능합니다.

'''
4. CUDA Tensors
'''
# .to() 함수를 사용하여 Tensor를 어떠한 장치로도 옮길 수 있습니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CUDA 장치 객체(device object)로
y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
x = x.to(device)  # .to()로 CPU에서 생성한 tensor를 GPU로 옮김
z = x + y
print(x, y)
print(z)
print(z.to("cpu", torch.double))  # .to()는 dtype도 함께 변경 가능
