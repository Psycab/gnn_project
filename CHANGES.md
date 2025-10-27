# 주요 변경사항

## 2025-10-27

### 변경 내용
1. **NaN 값 처리 추가**: 60영업일 윈도우 계산으로 인한 빈 셀 제거
2. **데이터 시작일 변경**: 2021-01-26 이후 데이터만 사용
3. **종목코드 위치 자동 인식**: 엑셀 구조를 파악하여 종목코드와 종목명 자동 처리

### 수정된 파일

#### `temporal_gat_monthly_ensemble_pipeline.py`
```python
# 추가된 코드 (169-180행)
# NaN 값 처리 및 2021-01-26 이후 데이터만 사용
print(f"   [INFO] NaN 값 처리 전: {len(df)} 행")
# 특성 컬럼(파생 지표)에 NaN이 있는 행 제거
df = df.dropna(subset=FEATURES)
print(f"   [INFO] NaN 제거 후: {len(df)} 행")

# 2021-01-26 이후 데이터만 사용
cutoff_date = pd.Timestamp("2021-01-26")
df = df[df["date"] >= cutoff_date].copy()
print(f"   [INFO] 2021-01-26 이후: {len(df)} 행")
print(f"   [INFO] 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
```

#### `preprocess_excel_data.py`
```python
# 추가된 코드 (92-102행)
# NaN 값 처리 및 2021-01-26 이후 데이터만 사용
print(f"   [INFO] NaN 값 처리 전: {len(df)} 행")
# 특성 컬럼(파생 지표)에 NaN이 있는 행 제거
FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]
df = df.dropna(subset=FEATURES)
print(f"   [INFO] NaN 제거 후: {len(df)} 행")

# 2021-01-26 이후 데이터만 사용
cutoff_date = pd.Timestamp("2021-01-26")
df = df[df["date"] >= cutoff_date].copy()
print(f"   [INFO] 2021-01-26 이후: {len(df)} 행")
```

### 변경 이유

1. **60영업일 윈도우**: vol_z 계산 시 60영업일 역사적 데이터가 필요한데, 초기 데이터에는 이 충분한 이력이 없어 NaN이 생성됩니다.
2. **2021-01-26**: 일관된 시작 시점을 제공하고, 초기 데이터의 누락을 피합니다.

### 영향
- 모델 학습/예측: 2021-01-26부터의 데이터만 사용
- 빈 셀 제거로 인한 데이터 손실 최소화
- 일관된 특성 데이터 보장

