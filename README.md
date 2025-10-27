# GNN Project - Monthly Ensemble Pipeline

## 개요

주식 가격 및 거래량 데이터를 사용하여 월별 예측 모델을 학습하고 앙상블 예측을 수행하는 파이프라인입니다.

## 전체 워크플로우

```
원본 엑셀 (행렬 형태)
    ↓
[전처리 스크립트 실행]
    ↓
전처리된 엑셀 (긴 형태 + 특성)
    ↓
[모델 학습 및 예측]
    ↓
예측 결과 엑셀
```

## 주요 변경사항

### 1. 엑셀 데이터 형식 지원

기존 코드는 긴 형태(long format)의 데이터를 기대했지만, 이제 **행렬 형태([날짜 X 종목코드])**의 엑셀 파일을 지원합니다.

- `price` 시트: 날짜가 첫 번째 컬럼, 나머지 컬럼이 종목코드, 값은 종가(close)
- `volume` 시트: 날짜가 첫 번째 컬럼, 나머지 컬럼이 종목코드, 값은 거래량(volume)

```
예시:
price 시트:
date       | 005930 | 000660 | 035420 | ...
2023-01-02 | 60000  | 200000 | 50000  | ...
2023-01-03 | 61000  | 205000 | 51000  | ...

volume 시트:
date       | 005930 | 000660 | 035420 | ...
2023-01-02 | 1000000| 500000 | 800000 | ...
2023-01-03 | 1200000| 550000 | 850000 | ...
```

### 2. 데이터 자동 변환

`load_and_engineer()` 함수가 자동으로:
1. 두 시트를 각각 읽어옴
2. `melt()` 함수로 긴 형태로 변환
3. 두 데이터프레임을 병합하여 `(date, symbol, close, volume)` 형태로 생성

## 파일 구조

```
gnn_project/
├── price_volume_timeseries.xlsx       # 원본 입력 데이터 (price, volume 시트)
├── preprocessed_data.xlsx             # 전처리된 데이터 (생성됨)
├── temporal_gat_monthly_ensemble_pipeline.py  # 메인 파이프라인
├── preprocess_excel_data.py          # 데이터 전처리 스크립트
├── test_data_loader.py               # 데이터 로더 테스트
├── inspect_excel.py                  # 엑셀 구조 확인 도구
└── README.md                         # 이 파일
```

## 사용 방법

### 1. 데이터 구조 확인

먼저 엑셀 파일 구조를 확인합니다:

```bash
python inspect_excel.py
```

### 2. 데이터 전처리 (중요!)

원본 행렬 형태 데이터를 전처리하여 엑셀로 저장합니다:

```bash
python preprocess_excel_data.py
```

이 스크립트는:
- `price_volume_timeseries.xlsx`의 price, volume 시트를 읽음
- [날짜 × 종목코드] 행렬 → 긴 형태 변환
- 특성 엔지니어링 (log_return, vol_z, volatility20, mom20)
- `preprocessed_data.xlsx`로 저장 (data 시트 + 통계 시트들)

### 3. 데이터 로더 테스트

데이터가 제대로 로드되는지 확인합니다:

```bash
python test_data_loader.py
```

### 4. 모델 학습

특정 월말 데이터로 모델을 학습합니다:

```bash
python temporal_gat_monthly_ensemble_pipeline.py --mode train --asof 2025-09-30
```

### 5. 예측 수행

학습된 모델을 사용하여 다음 달 예측을 수행합니다:

```bash
python temporal_gat_monthly_ensemble_pipeline.py --mode predict --asof 2025-09-30
```

또는 전처리된 파일을 명시적으로 지정:

```bash
python temporal_gat_monthly_ensemble_pipeline.py --mode train --asof 2025-09-30 --data preprocessed_data.xlsx
```

## 출력 파일

- **모델**: `models/model_T{T}_{YYYYMMDD}.joblib` (T=20,40,60)
- **예측 결과**: `output/pred_matrix_{YYYYMMDD}.xlsx`

## 주요 함수 수정 내용

### `load_and_engineer(data_path: str)`

```python
def load_and_engineer(data_path: str) -> pd.DataFrame:
    """엑셀 파일에서 'price'와 'volume' 시트를 읽어 [날짜 X 종목코드] 형태에서 
    긴 형태(long format)로 변환"""
    # 시트 읽기
    df_price = pd.read_excel(data_path, sheet_name="price")
    df_volume = pd.read_excel(data_path, sheet_name="volume")
    
    # Melt: [날짜 X 종목코드] → [date, symbol, value]
    df_price_long = df_price.melt(
        id_vars=["date"],
        var_name="symbol",
        value_name="close"
    )
    
    df_volume_long = df_volume.melt(
        id_vars=["date"],
        var_name="symbol",
        value_name="volume"
    )
    
    # 병합 및 특성 엔지니어링
    # ...
```

## 필수 라이브러리

```bash
pip install pandas numpy scikit-learn joblib openpyxl
```

## 주의사항

1. 엑셀 파일은 반드시 `price`와 `volume` 두 개의 시트가 있어야 합니다
2. 각 시트의 첫 번째 컬럼은 날짜여야 합니다
3. 나머지 컬럼들은 종목코드(문자열)여야 합니다
4. 날짜는 pandas가 파싱 가능한 형식이어야 합니다

