"""서울시 대기오염물질 측정정보(long format) -> 서울 평균 시간별 표.

사용법:
    python 01_preprocess.py <csv가 들어있는 폴더>... --out data/processed/seoul_hourly.csv

처리 규칙
- 25개 자치구 측정소(코드 101~125)만 사용 (나머지 코드는 성격 불분명)
- 측정기 상태가 0(정상)인 행만 사용, 나머지는 결측 처리
- 풍향은 산술평균이 불가능하므로 풍속과 합쳐 동서(u)/남북(v) 벡터로 변환한 뒤 평균
- 서울 평균 = 정상 측정된 구 측정소 값의 평균
- 짧은 결측(최대 6시간)만 선형 보간, 그 이상은 NaN으로 남김
"""
import argparse, glob, os
import numpy as np
import pandas as pd

ITEMS = {  # 측정항목 코드표(서울시 측정항목 정보)로 확정된 것 + 값 검증으로 확인한 기상 항목
    "1": "SO2", "3": "NO2", "5": "CO", "6": "O3", "8": "PM10", "9": "PM25",
    "11": "wd", "12": "ws", "13": "temp", "14": "humid",
}
GU_RANGE = (101, 125)
MAX_GAP = 6
# 물리적으로 불가능한 값(센서 오류)을 걸러내는 유효 범위. 범위 밖이면 결측 처리한다.
VALID = {
    "SO2": (0, 1), "NO2": (0, 2), "CO": (0, 50), "O3": (0, 0.5),
    "PM10": (0, 1000), "PM25": (0, 500),
    "wd": (0, 360), "ws": (0, 50), "temp": (-40, 50), "humid": (0, 100),
}


def load(folders):
    parts = []
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            d = pd.read_csv(f, encoding="cp949", dtype=str, usecols=[0, 1, 2, 3, 4])
            d.columns = ["dt", "station", "item", "val", "status"]
            d = d[d.item.isin(ITEMS)]
            d = d[d.station.astype(int).between(*GU_RANGE)]
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["dt"] = pd.to_datetime(d.dt.str[:19])
    d["v"] = pd.to_numeric(d.val)
    d.loc[d.status != "0", "v"] = np.nan
    d["name"] = d.item.map(ITEMS)
    lo = d.name.map(lambda n: VALID[n][0])
    hi = d.name.map(lambda n: VALID[n][1])
    bad = (d.v < lo) | (d.v > hi)
    print("범위 밖 값 제거:", int(bad.sum()), "행")
    print(d[bad].groupby("name").v.agg(["count", "min", "max"]).to_string())
    d.loc[bad, "v"] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = load(a.folders)
    # 측정소 x 시간 표로 피벗 (측정소별로 풍향을 벡터로 바꾸기 위해)
    w = d.pivot_table(index=["dt", "station"], columns="name", values="v", aggfunc="mean")
    rad = np.deg2rad(w["wd"])
    w["wind_u"] = -w["ws"] * np.sin(rad)
    w["wind_v"] = -w["ws"] * np.cos(rad)
    w = w.drop(columns="wd")

    seoul = w.groupby(level="dt").mean()           # 측정소 평균 (NaN 제외)
    n_st = w["PM25"].groupby(level="dt").count()   # 그 시간에 PM2.5가 정상인 측정소 수
    seoul["n_station_pm25"] = n_st
    seoul = seoul.asfreq("h")                      # 빠진 시간이 있으면 NaN 행으로 채움

    raw_nan = seoul.isna().sum()
    cols = [c for c in seoul.columns if c != "n_station_pm25"]
    seoul[cols] = seoul[cols].interpolate(limit=MAX_GAP, limit_area="inside")
    seoul["n_station_pm25"] = seoul["n_station_pm25"].fillna(0).astype(int)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    seoul.round(5).to_csv(a.out, encoding="utf-8-sig", index_label="datetime")

    print("기간:", seoul.index.min(), "~", seoul.index.max(), "| 행:", len(seoul))
    print("\n[보간 전 결측 시간 수 -> 보간 후]")
    print(pd.DataFrame({"before": raw_nan, "after": seoul.isna().sum()}).to_string())
    print("\n[요약 통계]")
    print(seoul[cols].describe().round(4).T.to_string())


if __name__ == "__main__":
    main()
