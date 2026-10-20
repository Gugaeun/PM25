"""장기 데이터셋 만들기: 2018~2021, 2022~2023, 2024~2025(기존), 2026-01~09 의 전처리 결과를 시간순으로 이어 붙인다.

각 조각은 같은 전처리(01_preprocess.py)로 만들었다. 이어 붙인 뒤 시간이 빠짐없이 1시간 간격인지, 결측이 없는지 확인하고
연도별 PM2.5 분포(평균, 95분위, 나쁨/매우 나쁨 시간 수)를 출력한다.

입력: data/processed/seoul_hourly_2018_2021.csv, seoul_hourly_2022_2023.csv, seoul_hourly.csv, seoul_hourly_2026.csv
출력: data/processed/seoul_hourly_long.csv, results/long_dataset_summary.csv
실행: python src/14_build_long_dataset.py
"""
import pandas as pd

from data_split import ROOT

P = ROOT / "data" / "processed"
PARTS = ["seoul_hourly_2018_2021.csv", "seoul_hourly_2022_2023.csv", "seoul_hourly.csv", "seoul_hourly_2026.csv"]


def main():
    df = pd.concat([pd.read_csv(P / f, encoding="utf-8-sig", parse_dates=["datetime"]) for f in PARTS]).sort_values("datetime").reset_index(drop=True)
    gaps = df["datetime"].diff().dropna()
    assert df["datetime"].is_unique, "시각이 겹칩니다"
    assert (gaps == pd.Timedelta(hours=1)).all(), f"시간 간격이 1시간이 아닌 곳이 {int((gaps != pd.Timedelta(hours=1)).sum())}곳 있습니다"
    assert df.isna().sum().sum() == 0, "결측이 있습니다"
    df.to_csv(P / "seoul_hourly_long.csv", index=False, encoding="utf-8-sig")
    print(f"장기 데이터: {df.datetime.min()} ~ {df.datetime.max()} | {len(df):,}시간 | 간격·결측 검사 통과")
    y = df.groupby(df["datetime"].dt.year)
    s = pd.DataFrame({"시간 수": y.size(), "PM2.5 평균": y["PM25"].mean(), "PM2.5 95분위": y["PM25"].quantile(0.95),
                      "나쁨 이상 시간(>35)": y["PM25"].apply(lambda v: int((v > 35).sum())), "매우 나쁨 시간(>75)": y["PM25"].apply(lambda v: int((v > 75).sum())),
                      "평균 측정소 수": y["n_station_pm25"].mean()}).round(2)
    s.to_csv(ROOT / "results" / "long_dataset_summary.csv", encoding="utf-8-sig")
    print(s.to_string())


if __name__ == "__main__":
    main()
