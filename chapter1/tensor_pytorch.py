import torch
import numpy as np

def describe(tensor):
    print("Type : {}".format(tensor.type()))
    print("Shape: {}".format(tensor.shape)) # we can check the dimension of the tensor
    print("Value: {}\n".format(tensor))

describe(torch.tensor([[2], [3]]))
describe(torch.rand(2, 3)) # 균등 분포 [0, 1)
describe(torch.randn(2, 3)) # 표준 정규 (Normal) 분포 -> 평균 0, 분산 1
print("------------------------------------")

describe(torch.zeros((3, 4)))
x = torch.ones((4, 3))
describe(x)
x = x.fill_(10) # 모든 값을 10으로 채움 ** 
#'_' (underscore)가 쳐져있는 메소드는 In-Place 메소드로 새로운 객체를 만들지 않고 텐서갑을 바꿔주는 연산을 진행해줌
describe(x)

x = x.normal_() # 표준 정규 분포로 초기화
describe(x)

x = x.uniform_() # 균등 분포로 초기화
describe(x)
print("------------------------------------")

# Tensor와 Numpy Array는 언제나 상호 변환 가능
npy = np.random.randn(2, 3)
describe(torch.from_numpy(npy)) # Type: torch.FloatTensor가 아닌 torch.DoubleTensor! -> 랜덤 넘파이 배열 기본이 float64이기 때문
# about TYPE -> 1. 특정 텐서 타입의 생성자(constructor) 직접 호출
#               2. torch.tensor()의 dtype parameter 사용
x = torch.FloatTensor([[1, 2, 3], 
                       [4, 5, 6]])
describe(x)

x = x.long()
describe(x)

print("------------------------------------")

x = torch.arange(6).view(2, 3) # 동일한 데이터를 공유하는 새로운 Tensor
describe(x)

npx = np.array(np.arange(6)) 
npx2 = npx.view() # Numpy의 view()는 Tensor의 view()와는 약간 다름
npx[0] = 10
print(npx2) # 같은 배열(행렬)을 참조하기때문에 원본을 바꿔도 같이 바뀜!

describe(torch.sum(x, dim=0)) # 행끼리 더함
describe(torch.sum(x, dim=1)) # 열끼리 더함
describe(torch.transpose(x, 0, 1))

print("------------------------------------")

# Indexing, Sliciing, Linking - Tensor / Numpy Array
x = torch.arange(12).view(3,4)
describe(x[1:, 2:]) # 1행 부터 & 2열 부터 
describe(x[:2, 2:]) # 2행 이전 (1 행까지) & 2열 부터 

describe(torch.index_select(x, dim=0, index=torch.LongTensor([0,2]))) # dim=0 행기준 0행, 2행 # index는 LongTensor이어야함
describe(torch.index_select(x, dim=1, index=torch.LongTensor([1,2]))) # dim=1 열기준 1열, 2열

describe(torch.cat([x, x], dim=0)) # 행 기준 다음 행(아래)에 붙임
describe(torch.cat([x, x], dim=1)) # 열 기준 다음 열(옆)에 붙임

describe(torch.stack([x, x])) # 행렬 2개를 쌓음 Ex) (3,4), (3,4) -> (2,3,4)
print(torch.stack([x, x]).shape)
print("------------------------------------")
# matrix multiplication
x1 = torch.ones((2, 3))
x2 = torch.zeros((3, 2))
x2[:, 1] += 1
describe(x2)
describe(torch.mm(x1, x2)) # matrix multiplication
print("------------------------------------")

# Gradient computation을 할 수 있는 Tensor 만들기 -> Just set True the 'requires_grad' boolean parameter!
# requires_grad=True -> loss (cost) function / gradients of tensor 를 기록하는 부가적인 연산 활성화
# 1. 파이토치가 Forward propagation의 계산값을 기록
# 2. 파이토치가 스칼라 값 하나를 이용해 Backward propagation 수행 / Tensor의 gradients 값 기록
# Gradient - 함수 입력에 대한 함수 출력의 기울기
#          - 계산 그래프에서 gradient는 model의 parameter 마다 존재하고 오류 신호에 대한 parameter의 기여로 생각 가능!
x_grad = torch.ones((2, 3), requires_grad=True)
describe(x_grad)
print(x_grad.grad is None)

y = torch.mm((x_grad + 2), torch.transpose((x_grad + 1), 0, 1))
describe(y)
print(x_grad.grad is None)

z = y.mean()
describe(z)
z.backward()
print(x_grad.grad is None) # it gives False!
print("------------------------------------")
# GPU / CPU devices
# 두 Tensor 객체로 연산을 수행할때는 반드시 같은 장치에 있는지 확인!
print(torch.cuda.is_available()) # CUDA version PyTorch 확인법

# 바람직한 방법! -> 장치에 무관환 텐서 초기화 (M1 mackbook version)
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

print(device)
print(f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.

x_gpu = torch.randn((3, 3)).to(device)
describe(x_gpu) # device='mps:0' -> GPU 사용중!

x_not_gpu = torch.randn((3, 3))
# describe(x_gpu + x_not_gpu) # -> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
# ** GPU로 데이터를 넣고 꺼내는 작업은 비용이 많이듦. 
# 병렬 계산 (parallel computation)은 일반적으로 GPU에서 수행 후, 최종결과만 CPU로 전송하는 방식이 효율적!

print("------------------------------------")
# Exercises
# 1. 2D 텐서를 만들고 차원 0 위치에 크기가 1인 차원을 추가하세요.
one = torch.rand(3, 3)
one = one.unsqueeze(dim=1) # 차원이 1인 차원 추가 dim=n 설정시 n차원의 차원(차원 1인)만 제거

print("1: ", one)

# 2. 이전 텐서에 추가한 자원을 삭제하세요.
one = one.squeeze(1) # 차원이 1인 차원 제거해줌 (말그대로 짜버려서 차원 1인 애들 없앰) ** batch 차원(첫 차원)이 1인경우 batch 차원도 사라지므로 주의!
print("2: ", one)

# 3. 범위가 [3, 7)이고 크기가 5x3인 랜덤한 텐서를 만드세요
three = 3 + torch.rand((5, 3)) * (7 - 3)
print("3: ", three)

# 4. 정규 분포를 사용해 텐서를 만드세요.
four = torch.randn(3,3)
print("4: ", four)

# 5. 텐서 torch.tensor([1, 1, 1, 0, 1])에서 0이 아닌 원소의 인덱스를 추출하세요.
five = torch.tensor([1, 1, 1, 0, 1])
print("5: ", torch.nonzero(five)) # 0이 아닌 요소의 인덱스를 2차원 행렬로 반환

# 6. 크기가 (3, 1)인 랜덤한 텐서를 만들고 네 벌을 복사해 쌓으세요.
six = torch.rand(3, 1)
six = six.expand(3, 4)
print("6: ", six)

# 7. 3차원 행렬 두 개 (a=torch.rand(3,4,5), b=torch.rand(3,5,4))의 배치 행렬 곱셈(Batch matix-matrix product)를 계산하세요.
seven_a = torch.rand(3, 4, 5)
seven_b = torch.rand(3, 5, 4)
print("7: ", torch.bmm(seven_a, seven_b))

# 8. 3차원 행렬 (a.torch.rand(3,4,5))과 2차원 행렬(b=torch.rand(5,4))의 배치 행렬 곱셈을 계산하세요. 
eight_b = torch.rand(5, 4)
eight_b = eight_b.unsqueeze(0).expand(seven_a.shape[0], *eight_b.shape)
print("8: ", torch.bmm(seven_a, eight_b))
