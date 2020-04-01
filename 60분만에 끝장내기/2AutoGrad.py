import torch

'''
    Tensor
'''
# tensor를 생성하고 requires_grad=True를 설정하여 연산을 기록
x = torch.ones(2, 2, requires_grad=True)
print(x)

# tensor에 연산을 수행
y = x + 2
print(y)

# y는 연산의 결과로 생성된 것이므로 grad_fn을 갖는다.
print(y.grad_fn)

# y에 다른 연산을 수행
z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_( ... )는 기존 Tensor의 requires_grad 값을 바꿔치기 (in-place)하여 변경합니다.
# 입력값이 지정되지 않으면 기본값은 False 입니다.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
print(a.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

'''
    변화도 (Gradient)
'''
out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
