import torch

# 1. 예제 데이터 - 이진 클래스 구분해보기 - 서로다른 클래스를 두 개의 Gaussian Distribution (Normal Distribution)에서 sampling함
# 2. 모댈 선택 - 간단하게 기초적인 Perceptron
# 3. 확률을 클래스로 변환? -> 결정 경계 (threshold)를 적용해 출력 확률을 두 개의 클래스로 바꾸어야함 -> 결과는 1 아니면 0
# 4. 손실 함수 선택 - 모델의 출력이 확률이다? -> Cross-Entropy 기반 함수
#                  이진 출력을 만들 예정 -> Binary Cross-Entropy    
# 5. 옵티마이저 선택 - 모델이 예측을 하고 손실 함수가 예측과 타깃 사이의 오차를 측정함
#                - 그 후, 옵티마이저가 오차 신호를 이용하여 모델의 가중치를 업데이트
#                - 가장 간단한 하나의 하이퍼파리미터는 Learning-Rate (alpha) - 학습률
#                - 확률적 경사하갈 Stochastic Gradient Descent (SGD) - 매우 고전적인 알고리즘 -> 기초적이지만 수렴 문제에 있어 나쁜 모델 생성가능
#                - 현재는 Adam, Adagrad 같은 적응형 옵티마이저 선호
#                - 항상 여러 종류의 옵티마이저를 시도해보아야함을 명심!!
# 6. 그레디언트 알고리즘 작동 - 1. 모델 객체안의 gradients와 부가정보를 grad_zero()로 초기화 (Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문)
#                       - 2. 모델이 입력 데이터(x)에 대한 출력(y_hat / y_pred)을 계산
#                       - 3. 모델 출력(y_hat)과 기대하는 타깃(y_target)을 비교해 loss 계산
#                       ---------Supervised Learning의 지도 (Supervising)에 해당-------
#                       - 4. 파이토치 손실(loss) 객체에는 backward() 메소드 존재 -> 역전파 계산
#                       - 5. 각 파라미터에 대한 그레디언트를 계산
#                       - 6. 옵티마이저(opt)의 step() 함수로 파라미터에 대한 그레디언트 업데이트 방법 정함 -> 그 최적화 알고리즘으로 파라미터 업데이트

# Batch 배치 - 그레디언트 각 단계는 배치 하나에서 수행됨 (전체 데이터셋은 배치로 분할됨)
# Mini-Batch 미니배치 - 각 배치가 훈련 데이터 크기보다 훨씬 작음을 강조할때
# Epoch 에포크 - 완전한 훈현 반복 한 번 / 모델은 여허 에포크 동안 훈련 됨
# 내부 반복문은 데이터셋/배치 개수 에 대해 순회 / 외부 반복문은 에포크 횟수나 다른 종료 조건에 의해 순회
epochs = 100
batches = 32
for epoch_i in range(epochs):
    for batch_i in range(batches):
        pass

# 데이터 분할
# 1. Training 훈련 / Validation 검증 / Test 테스트 set
# -> 비율 때마다 다름 보통 7 / 2 / 1 에서 6 / 2 / 2 등 다양
# 2. K-fold cross validation K겹 교차 검증
# -> 데이터셋을 같은 크기의 K 폴드로 나눔 K-1를 훈련에 사용 / 1개를 마지막 평가에 사용 -> 계산 비용 높지만 작은 데이터셋에선 유용

# 훈련 중지 시점
# 1. 에포크 횟수만큼 -> 가장 기초적이지만 임의적이고 필수도아님
# 2. 조기 종료 Early Stopping -> 에포크 마다 valid set으로 성능 기록, 성능이 안좋아지는 시기 감지 -> 계속 안좋아지면 종료 (종료 전 기다리는 에포크 횟수를 인내 Patience라고 함)
# 모델이 개선되지 않는 지점을 모델이 수혐한 곳 이라고 함 -> 실전에서 모델이 끝까지 수렴하는 경우는 드뭄 -> 시간 오래 걸림 / overfitting 가능성

# 하이퍼파라미터 Hyperparameters
# 하이퍼파라미터는 모델의 파라미터 개수와 값에 영향을 미치는 모든 모델 설정임
# cost function, optimizer, learning rate, layer size, patience size, regulaton 등등

# 규제 Regulation
# 수치 최적화 이론에서 유래 / 오캄의 면도날 이론에 따르면 간단한 설명이 복잡한 설명보다 나음
# L2 Regulation - 부드럽게 만드는 제약 (ex-여러 점을 지나는 곡선을 삐뚤빼뚤하고 극단적인 곡선보단 직관적이고 합리적으로 만드는 과정)
# L1 Regulation - 주로 희소한 솔루션 만드는데 사용 -> 대부분의 모델 파라미터가 0에 가까움
# Dropout Regulation - 확률적으로 몇개를 0으로 만듦
