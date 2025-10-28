# 에러 체크리스트 및 해결 방안

## ✅ 해결된 부분

### 1. predict_proba에서 모델 체크 추가
```python
if self.model is None:
    raise RuntimeError("Model has not been fitted yet. Call fit() first.")
```

### 2. 차원 검증 추가
```python
if X_tensor.shape[1] % self.T != 0:
    raise ValueError(f"Input dimension {X_tensor.shape[1]} is not divisible by T={self.T}")
```

### 3. 모델 로드 부분 개선
```python
# Handle both Pipeline and TemporalGATWrapper
model = pack["model"]
```

## ⚠️ 여전히 잠재적 문제

### 1. torch_geometric 의존성
**에러**: `ModuleNotFoundError: No module named 'torch_geometric'`
**해결**: 
```bash
pip install torch-geometric
```

### 2. GAT 그래프 구조 부재
**현재**: dummy_edge_index 사용
**문제**: 실제 그래프 구조가 없어 GAT의 이점을 못 살림
**해결 방안**:
- **옵션 A**: 단순 LSTM/GRU로 교체
- **옵션 B**: 종목 상관관계 기반 그래프 구성

### 3. 모델 저장/로드
**문제**: TemporalGATWrapper 객체 전체 저장
**해결**: state_dict만 저장하거나 별도 로드 로직 필요

### 4. 입력 차원 불일치
**에러**: T=20인데 입력이 80 차원이 아닌 경우
**예방**: build_window_matrix에서 검증

## 테스트 필수 항목

### A. 라이브러리 설치 확인
```python
try:
    import torch
    import torch_geometric
    print("✓ Required libraries installed")
except ImportError as e:
    print(f"✗ Missing library: {e}")
```

### B. 입력 차원 검증
```python
def check_input_dimensions(X, T):
    """Validate input dimensions"""
    if X.shape[1] % T != 0:
        raise ValueError(f"Input shape {X.shape} incompatible with T={T}")
    F = X.shape[1] // T
    print(f"✓ Valid: Input [batch={X.shape[0]}, T*F={X.shape[1]}] = [batch, T={T}, F={F}]")
```

### C. 모델 학습 가능 여부
```python
def test_model_trainable():
    """Test if model can be trained"""
    try:
        model = TemporalGATWrapper(T=20)
        X = torch.randn(10, 80)  # 10 samples, T=20, F=4
        y = torch.randint(0, 3, (10,))
        model.fit(X.numpy(), y.numpy())
        print("✓ Model training successful")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
```

## 권장 사항

### 즉시 개선 가능
1. ✅ 차원 검증 추가 (완료)
2. ✅ 모델 체크 추가 (완료)
3. ⚠️ 예외 처리 강화

### 중장기 개선
1. 실제 그래프 구조 구현
2. 더 나은 모델 아키텍처
3. 하이퍼파라미터 튜닝

## 빠른 대안

현재 구현이 복잡하거나 에러가 많이 발생하면:

### Option 1: 단순 LSTM만 사용
```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        out, _ = self.lstm(x)  # x: [batch, T, F]
        return self.fc(out[:, -1, :])  # Last timestep
```

### Option 2: sklearn Pipeline 유지
- 현재 LogisticRegression 사용
- 추가적인 복잡도 없음
- 먼저 동작 확인 후 모델 개선

