import torch
import numpy as np
import matplotlib.pyplot as plt

# 퍼셉트론 (Perceptron) - 가장 간단한 신경망 y = f(wx + b)
#                       선형함수(wx+b)와 비선형함수(f)의 조합
class Perceptron(torch.nn.Module):
    """Perceptron is one of linear layer"""
    def __int__(self, input_dim):
        """
        Parameter:
            input_dim (int): size of input feature
        """
        super(Perceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x_in):
        """
        Perceptron's forward propagation
        Parameter: 
            x_in (torch.Tensor): input data Tensor
            x_in.shape == (batch, num_features)
        return:
            output Tensor -> shape == (batch,)
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()
    
# 활성함수 (Activation Functions)
x = torch.range(-5., 5, 0.1)
# 1. 시그모이드 (Sigmoid) - 미분 가능한 부드러운 형태의 압축함수 / 출력범위 (0, 1)
# 입력범위 대부분에서 기울기가 꽤 큼 -> 입력범위 대부분에서 매우 빠르게 포화 (극단적인 출력)
# 이로 인해서 그레디언트가 0이 되는 그레디어언트 소실 문제 (vanishing gradient problem) 혹은 
#          그레디언트가 발산하는 그레디언트 폭주 문제 (exploding gradient problem) 발생
# 부동소수 오버플로우가 되는 문제 발생가능 -> 보통 마지막 레이어인 확률을 구하는 출력층 (output layer)에서 주로 쓰임 -> 0~1
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# 2. 하어퍼볼릭 탄젠트 (hyperbolic tangent - tanh) - sigmoid 함수의 아핀 변환 (선형 변환) 버전
# tanh = 2 * sig(2x) - 1
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# 3. 렐루 (Rectified Linear Unit - ReLU) - 매우 중요한 활성화 함수
# relu = max(0, x) - 음수를 0으로 만듦 -> 그레디언트 소실 문제(vanishing gradient problem)해결에 도움이 됨
# 문제점 - 죽은 렐루 -> 신경망의 특정 출력이 0이 되면 다시 돌아오지 못한다는 문제점 -> 해결책? Leaky ReLU / Parametric ReLU
y = torch.relu(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

prelu = torch.nn.PReLU(num_parameters=1)
y = prelu(x)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.show()

# 4. 소프트맥스 (Softmax) - 여러 클래스 분류 문제의 출력층에 자주 사용됨 - sigmoid와 비슷하게 0~1로 압축
# 모든 출력의 합으로 각 출력을 나누고 k개 클래스에 대한 이산 확률 분포를 만듦 (출력의 합은 1)
# 범주형 크로스 엔트로피 (categorical cross entropy)와 자주 같이 사용됨
softmax = torch.nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y = softmax(x_input)
print(x_input)
print(y)
print(y.sum())
