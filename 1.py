import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.empty(5, 3).to(device)
y = torch.rand(5, 3).to(device)
print(x)
print(y)
