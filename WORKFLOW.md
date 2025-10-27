# 데이터 전처리 및 예측 파이프라인 워크플로우

## 📋 전체 프로세스

### Step 1: 원본 엑셀 구조 확인

```bash
python inspect_excel.py
```

엑셀 파일에 `price`와 `volume` 시트가 있는지 확인합니다.

**예상 구조:**
- **price 시트**: 첫 번째 컬럼=날짜, 나머지=종목코드, 값=종가
- **volume 시트**: 첫 번째 컬럼=날짜, 나머지=종목코드, 값=거래량

```
price 시트 예시:
date       | 005930 | 000660 | 035420
2024-01-02 | 70000  | 200000 | 55000
2024-01-03 | 71000  | 205000 | 56000
...

volume 시트 예시:
date       | 005930  | 000660  | 035420
2024-01-02 | 1000000 | 500000  | 800000
2024-01-03 | 1200000 | 550000  | 850000
...
```

---

### Step 2: 데이터 전처리 및 엑셀 저장 ⭐

```bash
python preprocess_excel_data.py
```

이 스크립트가 수행하는 작업:

1. **엑셀 읽기**
   - `price` 시트: 가격 데이터
   - `volume` 시트: 거래량 데이터

2. **형태 변환**
   - [날짜 × 종목코드] → [date, symbol, close/volume]
   - `melt()` 함수 사용

3. **데이터 병합**
   - `(date, symbol, close)` + `(date, symbol, volume)` 
   - → `(date, symbol, close, volume)`

4. **특성 엔지니어링**
   - `log_return`: 로그 수익률
   - `vol_z`: 거래량 z-score (60일 기준)
   - `volatility20`: 20일 이동평균 변동성
   - `mom20`: 20일 누적 수익률

5. **엑셀 저장**
   - `preprocessed_data.xlsx` 생성
   - **시트 구성:**
     - `data`: 메인 데이터 (전처리된 모든 필드)
     - `전체_통계`: 기본 통계 정보
     - `결측치_통계`: 결측치 분석
     - `종목별_통계`: 종목별 상세 통계

**출력 파일:** `preprocessed_data.xlsx`

---

### Step 3: 모델 학습

```bash
python temporal_gat_monthly_ensemble_pipeline.py --mode train --asof 2024-09-30
```

**작업 내용:**
1. `preprocessed_data.xlsx` 자동 로드
2. 전처리된 특성 사용
3. 월말 기준 모델 학습 (T=20, 40, 60)
4. 모델 저장: `models/model_T{T}_{YYYYMMDD}.joblib`

**출력:**
- `models/model_T20_20240930.joblib`
- `models/model_T40_20240930.joblib`
- `models/model_T60_20240930.joblib`

---

### Step 4: 예측 실행

```bash
python temporal_gat_monthly_ensemble_pipeline.py --mode predict --asof 2024-09-30
```

**작업 내용:**
1. 학습된 모델 3개 로드
2. 각 모델로 예측 수행
3. 앙상블 (평균 확률)
4. 결과를 엑셀로 저장

**출력 파일:** `output/pred_matrix_20240930.xlsx`

**시트 내용:**
- `proba_ens_0`, `proba_ens_1`, `proba_ens_2`: 앙상블 예측 확률
- `pred_ensemble`: 앙상블 예측 클래스
- `proba_T20/40/60_0/1/2`: 각 모델별 확률
- `pred_T20/40/60`: 각 모델별 예측

---

## 🔄 전체 흐름도

```
┌─────────────────────────┐
│ price_volume_timeseries.xlsx │
│  (원본 행렬 형태)          │
└────────────┬────────────┘
             │
             ↓ preprocess_excel_data.py
             │
┌────────────▼─────────────────────────┐
│  preprocessed_data.xlsx              │
│  - data 시트 (긴 형태 + 특성)         │
│  - 통계 시트                          │
└────────────┬─────────────────────────┘
             │
             ↓ temporal_gat_monthly_ensemble_pipeline.py
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
─train─          ─predict─
    │                 │
    ↓                 ↓
┌──────┐      ┌───────────────┐
│models│      │ pred_matrix   │
│.joblib│      │ .xlsx         │
└──────┘      └───────────────┘
```

---

## 📝 주요 함수 설명

### `load_and_engineer()`

두 가지 형식을 자동 감지하여 처리:

1. **전처리된 파일**: `data` 시트가 있으면 바로 로드
2. **원본 파일**: `price`, `volume` 시트를 읽어 변환

```python
# 전처리된 파일 우선 사용
if "data" in sheet_names:
    return pd.read_excel(data_path, sheet_name="data")
else:
    # 원본 파일 처리
    price = pd.read_excel(data_path, sheet_name="price").melt(...)
    volume = pd.read_excel(data_path, sheet_name="volume").melt(...)
    return pd.merge(price, volume, ...)
```

### 전처리 특성

- `log_return`: `log(pct_change() + 1)` (로그 수익률)
- `vol_z`: `(volume - mean_60d) / std_60d` (거래량 표준화)
- `volatility20`: `std(log_return, 20)` (20일 변동성)
- `mom20`: `sum(log_return, 20)` (20일 모멘텀)

---

## 🎯 사용 예시

```bash
# 1. 엑셀 구조 확인
python inspect_excel.py

# 2. 전처리 (엑셀로 저장)
python preprocess_excel_data.py

# 3. 모델 학습
python temporal_gat_monthly_ensemble_pipeline.py --mode train --asof 2024-09-30

# 4. 예측
python temporal_gat_monthly_ensemble_pipeline.py --mode predict --asof 2024-09-30
```

모두 완료하면:
- ✅ `preprocessed_data.xlsx` (전처리된 데이터)
- ✅ `models/` (학습된 모델들)
- ✅ `output/pred_matrix_YYYYMMDD.xlsx` (예측 결과)

