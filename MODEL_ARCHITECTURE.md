# Temporal-GAT + GRU 모델 구조 (수정)

## ⚠️ 현재 구조의 문제점

현재 코드는 **단일 샘플(종목 1개)만 처리**하는 구조입니다!

```
입력: [1개 샘플, T*F] 예: [1, 80]
  ↓
Reshape: [1, T, F] 예: [1, 20, 4]
  ↓
상관관계 계산 → [N종목, N종목] 행렬? ❌ (샘플 1개라 불가능)
  ↓
실제로는: 샘플 1개로 그래프 구성이 의미 없음
```

## 올바른 구조 (의도된 방식)

### 다중 종목 동시 처리 구조

```
입력: [N종목, T*F] 예: [50개종목, 80]
  ↓
Reshape: [N종목, T, F] 예: [50, 20, 4]
         │     │  │
         └─ 각 종목이 하나의 노드
  ↓
종목 간 상관관계 계산:
  corr[i,j] = correlation(종목i_시계열, 종목j_시계열)
  → [N×N] 상관관계 행렬
  ↓
상관관계 > threshold → edge 생성
  edge_index: [2, E] (E는 edge 수)
  ↓
GAT 처리:
  - 각 종목 = 노드 (node feature = [F])
  - 상관관계 > 0.3 → edge 연결
  - GAT로 종목 간 정보 전파
  ↓
Global Mean Pooling: [50종목, 64] → [1, 64]
  ↓
GRU + FC → 예측
```

## 현재 코드의 실제 동작

```
학습 샘플: [총샘플수, T*F]

for 각 샘플:
    X_sample = [1, T*F]  ← 샘플 1개만
      ↓
    Reshape [1, T, F]
      ↓
    상관관계 계산하려면 N×N 행렬이 필요
    하지만 샘플은 1개뿐 → 상관관계 계산 불가능!
      ↓
    node_features = mean(시간 축)  ← 단순 평균!
      ↓
    의미 없는 GAT 처리
```

## 해결 방안

### 방안 1: 배치 단위로 종목 묶어 처리 (권장)

```python
# 입력을 재구성
# 현재: [샘플수=월말수×종목수, T*F]
# 변경: [월말수, 종목수, T*F]

for 각 월말:
    X_month = [N종목, T*F]  ← 종목들을 함께
      ↓
    Reshape [N종목, T, F]
      ↓
    종목 간 상관관계 계산 (실제로 의미 있음!)
      ↓
    GAT 처리 (종목 간 정보 전파)
      ↓
    Pooling → GRU → 예측
```

### 방안 2: 각 종목 독립 처리 (현재와 유사하지만 구조 명확화)

```python
# 샘플: [1, T*F]
  ↓
종목 단위로만 처리
  ↓
시계열 특성 평균 → node feature [F]
  ↓
GAT 불필요 → 단순 MLP/GRU 사용
```

## 권장 방안: 구조 개선

### 옵션 1: 배치 단위 종목 묶음

```python
def build_windows_for_month(
    df, symbols, month_end, T
) -> Tuple[np.ndarray, np.ndarray]:
    """월말 기준으로 모든 종목의 윈도우를 함께 반환"""
    X_list = []
    returns_list = []
    
    for sym in symbols:
        X_sym, returns_sym, valid = extract_window(...)
        X_list.append(X_sym)      # [20, 4]
        returns_list.append(returns_sym)  # [20]
    
    X = np.stack(X_list, axis=0)  # [N종목, 20, 4]
    returns = np.stack(returns_list, axis=0)  # [N종목, 20]
    
    return X, returns
```

### 옵션 2: GAT 제거하고 단순 시계열 모델 사용

```python
# 현재 GAT-GRU는 종목 간 관계를 모델링하려 했지만
# 실제로는 동작하지 않는 구조
# 대신 GRU만 사용하는 것이 명확함
```

## 결론

**현재 코드는 의도와 다른 구조입니다.**
- 원래 의도: 종목 간 그래프 + GAT
- 실제 동작: 샘플 단위로 처리 (GAT 무의미)

**수정 필요:**
1. 배치 단위로 종목 묶어 처리 (권장)
2. 또는 GAT 제거하고 단순 시계열 모델 사용
