import pennylane as qml
from pennylane import numpy as np

# --- 장치 설정 ---
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# --- 변분 회로 정의 ---
def variational_circuit(x, weights):
    # 입력 인코딩
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
#wires 큐비트 뭐쓸지
    # 레이어 1
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=1)

    # 레이어 2 (깊이 추가)
    qml.RY(weights[4], wires=0)
    qml.RY(weights[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[6], wires=0)
    qml.RY(weights[7], wires=1)

    # 첫 번째 큐비트 측정
    return qml.expval(qml.PauliZ(0))

# --- QNode ---
@qml.qnode(dev)
def quantum_model(x, weights):
    return variational_circuit(x, weights)

# --- 손실 함수 (MSE) ---
def loss(weights, X, Y):
    preds = np.array([(quantum_model(x, weights) + 1) / 2 for x in X])  # 0~1 범위
    return np.mean((preds - Y) ** 2)

# --- 학습 데이터 (AND 연산) ---
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = np.array([0, 0, 0, 1])

# --- 초기 가중치 ---
weights = np.random.randn(8)  # 2레이어 x 4파라미터

# --- 옵티마이저 ---
opt = qml.GradientDescentOptimizer(stepsize=0.3)
epochs = 100

# --- 학습 ---
for epoch in range(epochs):
    weights = opt.step(lambda w: loss(w, X_train, Y_train), weights)
    if (epoch + 1) % 10 == 0:
        current_loss = loss(weights, X_train, Y_train)
        print(f"Epoch {epoch + 1}: loss = {current_loss:.4f}")

# --- 예측 ---
print("\n--- Predictions ---")

for x in X_train:
    pred = (quantum_model(x, weights) + 1) / 2
    label = 1 if pred > 0.5 else 0
    print(f"Input: {x}, Prediction: {pred:.3f}, Label: {label}")
print("=" * 30)
