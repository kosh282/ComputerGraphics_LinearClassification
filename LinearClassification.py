import numpy as np
import os
import pickle

# 1. 로컬 CIFAR-10 데이터 로드 함수
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
        Y = np.array(Y)
        return X, Y

def load_cifar10(root_path):
    xs = []
    ys = []
    # 5개의 학습 배치 로드
    for b in range(1, 6):
        f = os.path.join(root_path, f'data_batch_{b}')
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    # 테스트 배치 로드
    Xte, Yte = load_cifar_batch(os.path.join(root_path, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

# 2. 선형 분류기 클래스 구현 (NumPy)
class LinearClassifier:
    def __init__(self):
        self.W = None
        self.b = None

    def train(self, X, y, learning_rate=1e-6, reg=1e-4, num_iters=200, batch_size=200):
        """
        경사하강법(Gradient Descent)을 이용해 가중치 W와 b를 최적화
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        
        # 가중치 초기화 (작은 무작위 값)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
            self.b = np.zeros((1, num_classes))

        for it in range(num_iters):
            # 미니배치 생성
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # 손실(Loss)과 그레디언트(Gradient) 계산
            loss, dW, db = self.loss_function(X_batch, y_batch, reg)

            # 가중치 업데이트 (수식: W = W - lr * dW)
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if it % 20 == 0:
                print(f"반복 횟수 {it:4d}/{num_iters}: 손실값(Loss) = {loss:.4f}")

    def loss_function(self, X, y, reg):
        """
        멀티클래스 SVM (Hinge Loss) 및 max(0, loss) 수식 구현
        """
        num_train = X.shape[0]
        
        # 1. 순방향 전파: 스코어 계산 (f = XW + b)
        scores = X.dot(self.W) + self.b  # 크기: (배치사이즈, 10)
        
        # 정답 클래스의 점수들만 쏙 뽑아내기
        correct_class_scores = scores[range(num_train), y].reshape(-1, 1) # 크기: (배치사이즈, 1)
        
        # 공식 적용 -> max(0, 오답점수 - 정답점수 + 1)
        margins = np.maximum(0, scores - correct_class_scores + 1.0)
        
        # 정답 클래스 자기 자신과의 비교(정답 - 정답 + 1 = 1)는 벌점에서 제외해야 하므로 0으로 만듭니다.
        margins[range(num_train), y] = 0
        
        # 이번 배치 데이터들의 평균 데이터 손실(Loss) 계산
        data_loss = np.sum(margins) / num_train
        reg_loss = 0.5 * reg * np.sum(self.W * self.W) 
        loss = data_loss + reg_loss

        # 2. 역방향 전파: max(0, x) 힌지 손실의 미분(그레디언트) 계산
        # max(0, x)에서 x가 0보다 컸던(벌점을 유발한) 곳들만 미분값이 1이 됩니다.
        binary = margins
        binary[margins > 0] = 1
        
        # 정답 클래스 위치에는 오답들이 정답을 넘어선 횟수만큼 마이너스 그레디언트를 누적합니다.
        row_sum = np.sum(binary, axis=1)
        binary[range(num_train), y] = -row_sum
        binary /= num_train

        # 최종 가중치 미분값 도출
        dW = X.T.dot(binary) + reg * self.W
        db = np.sum(binary, axis=0, keepdims=True)

        return loss, dW, db

    def predict(self, X):
        """
        학습된 W와 b를 이용해 가장 높은 점수의 클래스를 예측
        """
        scores = X.dot(self.W) + self.b
        return np.argmax(scores, axis=1)


if __name__ == "__main__":

    cifar10_dir = 'cifar-10-batches-py' 
    
    print("1. CIFAR-10 로컬 데이터를 불러오는 중...")
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    # 2. 이미지 데이터 전처리 (Flatten 및 정규화)
    # (50000, 32, 32, 3) -> (50000, 3072) 형태로 1차원 펼치기
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 픽셀 값 정규화 (0~255 -> 0~1) 및 데이터 중심화 (Zero-centering) - bias
    mean_image = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean_image
    X_test_flat -= mean_image

    print(f"학습 데이터 형태: {X_train_flat.shape}")
    print(f"테스트 데이터 형태: {X_test_flat.shape}")

    # 3. 모델 선언 및 학습
    print("\n2. 선형 분류기 학습 시작...")
    classifier = LinearClassifier()

    classifier.train(X_train_flat, y_train, learning_rate=1e-6, reg=1e-3, num_iters=200, batch_size=200)

    # 4. 정확도 평가
    print("\n3. 정확도 검증 중...")
    train_pred = classifier.predict(X_train_flat)
    test_pred = classifier.predict(X_test_flat)
    
    train_acc = np.mean(train_pred == y_train) * 100
    test_acc = np.mean(test_pred == y_test) * 100

    print(f"학습 데이터 정확도 (Train Accuracy): {train_acc:.2f}%")
    print(f"테스트 데이터 정확도 (Test Accuracy): {test_acc:.2f}%")