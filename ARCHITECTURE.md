# Temporal-GAT + GRU 아키텍처

## 개요

GAT(Graph Attention Network)와 GRU를 결합하여 시계열 예측을 수행하는 모델입니다.

## 아키텍처 구조

```
입력: [batch, T*F] (예: [N, 20*4] = [N, 80])
    ↓
Reshape: [batch, T, F] (예: [N, 20, 4])
    ↓
GAT Layers
    ├─ GAT1: Graph Attention (F → hidden_dim*4, 4 heads)
    └─ GAT2: Graph Attention (hidden_dim*4 → hidden_dim, 1 head)
    ↓
Global Mean Pooling (graph-level representation)
    ↓
GRU Layers
    ├─ Time dimension: [batch, 1, hidden_dim]
    └─ GRU: (hidden_dim → hidden_dim, 2 layers)
    ↓
Output Layer
    └─ Linear: (hidden_dim → 3 classes)
    ↓
출력: [batch, 3] (확률 값)
```

## 주요 컴포넌트

### 1. GAT (Graph Attention Network)
- **목적**: 종목 간 관계를 모델링
- **입력**: 종목의 T영업일 특성 데이터
- **출력**: graph-level representation

### 2. GRU (Gated Recurrent Unit)
- **목적**: 시간적 패턴 학습
- **입력**: GAT의 출력 (graph embedding)
- **출력**: 시계열 처리된 hidden state

### 3. 최종 분류기
- **입력**: GRU의 마지막 출력
- **출력**: 3개 클래스 (0, 1, 2) 확률

## 데이터 흐름

### 학습 데이터 구성
```
입력 X: 과거 20/40/60 영업일 특성
    - log_return (로그 수익률)
    - vol_z (거래량 z-score)
    - volatility20 (20일 변동성)
    - mom20 (20일 모멘텀)

출력 y: 이후 20영업일 수익률 버킷
    - 0: 수익률 < 5%
    - 1: 5% ≤ 수익률 < 10%
    - 2: 수익률 ≥ 10%
```

### 예시 데이터 흐름
```
asof = 2021-04-30 (마지막 영업일)
T = 20

입력 X:
  기간: [2021-04-06 ~ 2021-04-30] 20영업일
  형태: [N개종목, 20영업일, 4특성] → [N, 80]

출력 y:
  기준: 2021-04-30 가격
  목표: 2021-05-28 가격 (20영업일 후)
  레이블: 수익률 버킷 (0/1/2)
```

## 모델 파라미터

```python
TemporalGATWrapper(
    T=20,              # 입력 윈도우 크기 (20/40/60)
    hidden_dim=64,     # 은닉층 차원
    num_layers=2,      # GRU 레이어 수
    num_heads=4,       # GAT 어텐션 헤드 수
    lr=0.001,         # 학습률
    epochs=50         # 에포크 수
)
```

## GAT의 역할

### 이론적 목적
- **종목 간 관계 모델링**: 상관관계가 높은 종목들의 관계를 그래프로 표현
- **어텐션 메커니즘**: 중요한 종목에 더 많은 가중치 부여

### 현재 구현
- **간소화 버전**: 각 샘플을 독립적으로 처리
- **향후 개선**: 실제 종목 간 상관관계 행렬을 그래프로 구성

## GRU의 역할

### 목적
- **시간적 패턴 학습**: 20영업일 데이터의 시간적 의존성 포착
- **시계열 압축**: T개의 시점 데이터를 하나의 hidden representation으로 압축

### 구조
```python
nn.GRU(
    input_size=hidden_dim,    # GAT 출력
    hidden_size=hidden_dim,   # 은닉 상태 크기
    num_layers=2,             # 2개 레이어
    batch_first=True          # [batch, time, features] 순서
)
```

## 학습 과정

### 1. 데이터 준비
```python
X: [batch, T*F] → [batch, T, F]로 reshape
y: [batch] 클래스 레이블
```

### 2. GAT 처리
```python
# 각 종목의 T영업일 특성을 graph로 처리
features = torch.mean(X_reshaped, dim=1)  # [batch, F]
graph_embedding = GAT(features, edge_index)  # [batch, hidden_dim]
```

### 3. GRU 처리
```python
# 시간적 시퀀스 처리
hidden = GRU(graph_embedding)  # [batch, 1, hidden_dim]
output = hidden[:, -1, :]  # 마지막 시점만 사용
```

### 4. 분류
```python
# 최종 예측
logits = Linear(output)  # [batch, 3]
probs = Softmax(logits)
```

## 앙상블 전략

```
T=20 모델 → 20영업일 패턴 학습
T=40 모델 → 40영업일 패턴 학습
T=60 모델 → 60영업일 패턴 학습
     ↓
평균 확률로 앙상블 → 최종 예측
```

## 특징

### 장점
✅ **그래프 구조**: 종목 간 관계 모델링 가능
✅ **어텐션 메커니즘**: 중요한 정보에 집중
✅ **시간적 처리**: GRU로 시계열 패턴 학습
✅ **앙상블**: 다중 윈도우로 강건성 확보

### 현재 제한사항
⚠️ **간소화된 GAT**: 실제 그래프 구조 미구현
⚠️ **단일 그래프**: 종목 간 관계 정보 필요

