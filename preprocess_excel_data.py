"""
엑셀 데이터 전처리 및 변환 스크립트
- 입력: price, volume 시트 (행렬 형태)
- 출력: 전처리된 긴 형태 데이터를 엑셀로 저장
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_and_preprocess_excel(input_path: str) -> pd.DataFrame:
    """
    엑셀 파일에서 price, volume 시트를 읽어서 전처리 후 반환
    
    Args:
        input_path: 입력 엑셀 파일 경로
        
    Returns:
        전처리된 DataFrame (date, symbol, close, volume, log_return, vol_z, volatility20, mom20)
    """
    print(f"[INFO] 엑셀 파일 로딩 중: {input_path}")
    
    # 시트 읽기 (헤더는 13번째 행, 종목코드는 7번째 행에 있음)
    print("   [INFO] price 시트 읽는 중...")
    df_price = pd.read_excel(input_path, sheet_name="price", header=13, engine='openpyxl')
    print(f"   - Price 시트: {df_price.shape}")
    
    # 종목코드를 컬럼명으로 설정
    print("   [INFO] 종목코드를 컬럼명으로 설정 중...")
    df_temp = pd.read_excel(input_path, sheet_name="price", header=None, engine='openpyxl')
    symbols = df_temp.iloc[7, 1:].tolist()  # 행 7(인덱스 7), 1번 컬럼부터
    df_price.columns = ['date'] + symbols
    print(f"   - 종목 수: {len(symbols)}")
    print(f"   - 처음 5개 종목: {symbols[:5]}")
    
    print("   [INFO] volume 시트 읽는 중...")
    df_volume = pd.read_excel(input_path, sheet_name="volume", header=13, engine='openpyxl')
    df_volume.columns = ['date'] + symbols
    
    # 날짜 컬럼 변환
    print("   [INFO] 날짜 컬럼 변환 중...")
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
    df_volume["date"] = pd.to_datetime(df_volume["date"]).dt.normalize()
    
    # Melt: [날짜 X 종목코드] → [date, symbol, value]
    print("   [INFO] 행렬 형태 -> 긴 형태 변환 중...")
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
    
    # 두 데이터프레임 합치기
    df = pd.merge(
        df_price_long,
        df_volume_long,
        on=["date", "symbol"],
        how="inner"
    )
    
    print(f"   [OK] 병합 완료: {df.shape}")
    
    # 정렬
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # 특성 엔지니어링
    print("   [INFO] 특성 엔지니어링 수행 중...")
    
    # log return
    df["log_return"] = np.log(df.groupby("symbol")["close"].pct_change().add(1.0))
    
    # log(volume) and z-score over 60-day window by symbol
    df["log_vol"] = np.log(df["volume"].fillna(0) + 1)
    win = 60
    grp = df.groupby("symbol")
    df["vol_z"] = grp["log_vol"].transform(
        lambda s: (s - s.rolling(win).mean()) / (s.rolling(win).std() + 1e-8)
    )
    
    # rolling volatility (20d std of returns)
    df["volatility20"] = grp["log_return"].transform(lambda s: s.rolling(20).std())
    
    # mom20 (cumulative log return over 20d)
    df["mom20"] = grp["log_return"].transform(lambda s: s.rolling(20).sum())
    
    # NaN 값 처리 및 2021-04-30 이후 데이터만 사용
    print(f"   [INFO] NaN 값 처리 전: {len(df)} 행")
    # 특성 컬럼(파생 지표)에 NaN이 있는 행 제거
    FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]
    df = df.dropna(subset=FEATURES)
    print(f"   [INFO] NaN 제거 후: {len(df)} 행")
    
    # 2021-04-30 이후 데이터만 사용
    cutoff_date = pd.Timestamp("2021-04-30")
    df = df[df["date"] >= cutoff_date].copy()
    print(f"   [INFO] 2021-04-30 이후: {len(df)} 행")
    
    print("   [OK] 특성 엔지니어링 완료")
    print(f"\n[INFO] 최종 데이터:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {df.columns.tolist()}")
    print(f"   - Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   - Symbols: {df['symbol'].nunique()} 개")
    print(f"\n샘플 데이터 (처음 5행):")
    print(df.head())
    
    return df


def save_to_excel(df: pd.DataFrame, output_path: str, include_stats: bool = True):
    """
    전처리된 데이터를 엑셀로 저장
    
    Args:
        df: 전처리된 DataFrame
        output_path: 출력 파일 경로
        include_stats: 통계 정보 시트 포함 여부
    """
    print(f"\n[INFO] 엑셀 파일 저장 중: {output_path}")
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 메인 데이터 시트
        df.to_excel(writer, sheet_name="data", index=False)
        print("   [OK] 'data' 시트 저장 완료")
        
        if include_stats:
            # 통계 정보 시트
            stats_dict = {}
            
            # 전체 통계
            stats_dict["전체_통계"] = [
                ["항목", "값"],
                ["총 행 수", len(df)],
                ["종목 수", df["symbol"].nunique()],
                ["날짜 수", df["date"].nunique()],
                ["날짜 범위", f"{df['date'].min()} ~ {df['date'].max()}"],
            ]
            
            # 결측치 통계
            missing_stats = df.isna().sum()
            stats_dict["결측치_통계"] = pd.DataFrame({
                "컬럼": missing_stats.index,
                "결측치 수": missing_stats.values,
                "결측비율(%)": (missing_stats.values / len(df) * 100).round(2)
            })
            
            # 종목별 통계
            symbol_stats = df.groupby("symbol").agg({
                "close": ["mean", "std", "min", "max"],
                "volume": ["mean", "std", "min", "max"],
                "log_return": ["mean", "std"],
            }).round(2)
            symbol_stats.columns = ["_".join(col).strip() for col in symbol_stats.columns]
            
            # 각 통계를 시트로 저장
            for sheet_name, data in stats_dict.items():
                if isinstance(data, list):
                    pd.DataFrame(data[1:], columns=data[0]).to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )
                else:
                    data.to_excel(writer, sheet_name=sheet_name)
            
            symbol_stats.to_excel(writer, sheet_name="종목별_통계")
            print("   [OK] 통계 시트 저장 완료")
    
    print(f"\n[OK] 저장 완료! 총 파일 크기: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"   위치: {os.path.abspath(output_path)}")


def main():
    """메인 실행 함수"""
    # 입출력 경로 설정
    input_file = "price_volume_timeseries.xlsx"
    output_file = "preprocessed_data.xlsx"
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_file}")
        print(f"   현재 디렉토리: {os.getcwd()}")
        return
    
    try:
        # 데이터 로드 및 전처리
        df = load_and_preprocess_excel(input_file)
        
        # 엑셀로 저장
        save_to_excel(df, output_file, include_stats=True)
        
        print("\n" + "="*60)
        print("[SUCCESS] 전처리 완료!")
        print("="*60)
        print(f"입력: {input_file}")
        print(f"출력: {output_file}")
        print(f"\n다음 단계:")
        print("1. preprocessed_data.xlsx 파일 확인")
        print("2. temporal_gat_monthly_ensemble_pipeline.py로 모델 학습 및 예측")
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

