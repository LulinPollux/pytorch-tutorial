import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
