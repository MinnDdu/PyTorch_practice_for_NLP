import torch
import numpy as np

# 흔히 사용하는 Loss Functions

# 1. 평균 제곱 오차 (MSE - Mean Squared Error): 출력(y hat)과 입력(y)이 연속값인 회귀문제에서 널리 사용
# sum(y - y_hat)**2 / n
mse_loss = torch.nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)

# 2. 범주형 크로스 엔트로피 (CE - Categorical Cross-Entropy): 출력을 클래스에 대한 확률 예측으로 이해할 수 있는 다중 분류 문제에서 널리 사용
ce_loss = torch.nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)

# 3. 이진 크로스 엔트로피 (BCE - Binary Cross-Entropy): 구분하는 클래스가 2개일때 사용
bce_loss = torch.nn.BCELoss()
sigmoid = torch.nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4,1)
loss = bce_loss(probabilities, targets)
print(loss)

