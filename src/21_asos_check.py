"""기상청 ASOS 서울(108) 시간 자료를 실시간 모델 입력에 쓸 수 있는지 확인한다(결과: 쓰지 않기로 함).

실시간 화면은 에어코리아에 기상 변수가 없어 PM2.5 전용 모델을 쓴다. 기상 변수 모델(cnn_weather_L48_y5)을 쓰려면
기온·습도·풍속·풍향을 따로 받아야 하므로, 공공데이터포털의 "기상청_지상(종관, ASOS) 시간자료 조회서비스"로 같은 값을 받아 본다.
인증키는 에어코리아와 같은 공공데이터포털 키(환경변수 AIRKOREA_KEY)를 쓴다.

1) 학습 데이터(서울 25개 구 평균)와 ASOS 서울 관측소 값의 일치도 (2025-12, 2026-09)
2) 2026-01~09 테스트에서 입력 기상 값을 바꿨을 때 24시간 평균 MAE
결과는 results/asos_check.md 에 저장한다.
"""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import airkorea  # noqa: E402  (인증키 읽기)

URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
WX = ["temp", "humid", "ws", "wind_u", "wind_v"]
MONTHS = [("20260101", "20260131"), ("20260201", "20260228"), ("20260301", "20260331"), ("20260401", "20260430"), ("20260501", "20260531"),
          ("20260601", "20260630"), ("20260701", "20260731"), ("20260801", "20260831"), ("20260901", "20260930")]


def asos(start: str, end: str) -> pd.DataFrame:
    """ASOS 서울(108) 시간 자료. 전처리와 같은 방식으로 풍향을 동서(u)/남북(v) 벡터로 바꾼다."""
    q = urllib.parse.urlencode({"serviceKey": airkorea.get_key(), "pageNo": 1, "numOfRows": 999, "dataType": "JSON", "dataCd": "ASOS",
                                "dateCd": "HR", "startDt": start, "startHh": "00", "endDt": end, "endHh": "23", "stnIds": "108"})
    last = None
    for k in range(6):  # 공공 API 는 가끔 느려서 재시도
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (k + 1))
    else:
        raise last
    df = pd.DataFrame(d["response"]["body"]["items"]["item"])
    df["datetime"] = pd.to_datetime(df["tm"])
    for c in ("ta", "hm", "ws", "wd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rad = np.deg2rad(df["wd"])
    df["wind_u"], df["wind_v"] = -df["ws"] * np.sin(rad), -df["ws"] * np.cos(rad)
    return df.set_index("datetime")[["ta", "hm", "ws", "wind_u", "wind_v"]].rename(columns={"ta": "temp", "hm": "humid"})


def asos_months(months) -> pd.DataFrame:
    a = pd.concat([asos(s, e) for s, e in months])
    return a[~a.index.duplicated()]


def agreement(train: pd.DataFrame, a: pd.DataFrame, label: str) -> list[str]:
    j = train[WX].join(a, how="inner", lsuffix="_train", rsuffix="_asos").dropna()
    rows = [f"| {label} | {c} | {j[c + '_train'].corr(j[c + '_asos']):.2f} | {j[c + '_train'].mean():.2f} | {j[c + '_asos'].mean():.2f} |" for c in WX]
    return rows


def mae(name: str, train: pd.DataFrame, wx: pd.DataFrame, stride: int = 3):
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text(encoding="utf-8"))
    L = meta["L"]
    sess = ort.InferenceSession(str(ROOT / "models" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    mu, sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
    df = train.copy()
    for c in WX:
        df[c] = wx[c]
    X, y = df[meta["cols"]].to_numpy(np.float32), df["PM25"].to_numpy(np.float32)
    ok = ~np.isnan(X).any(1)
    errs, base = [], []
    for i in range(L - 1, len(df) - 24, stride):
        if not ok[i - L + 1:i + 1].all() or np.isnan(y[i + 1:i + 25]).any():
            continue
        h = (df.index[i].hour + np.arange(-L + 1, 1)) % 24
        tt = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], 1).astype(np.float32)
        p = np.clip(sess.run(None, {"x": ((np.concatenate([X[i - L + 1:i + 1], tt], 1) - mu) / sd)[None]})[0][0] * sd[0] + mu[0], 0, None)
        errs.append(np.abs(p - y[i + 1:i + 25]))
        base.append(np.abs(y[i] - y[i + 1:i + 25]))
    return float(np.mean(errs)), float(np.mean(base)), len(errs)


def main():
    P = ROOT / "data" / "processed"
    t26 = pd.read_csv(P / "seoul_hourly_2026.csv", parse_dates=["datetime"], index_col="datetime")
    t25 = pd.read_csv(P / "seoul_hourly.csv", parse_dates=["datetime"], index_col="datetime")
    a = asos_months(MONTHS)
    agree = agreement(t25, asos_months([("20251201", "20251231")]), "2025-12") + agreement(t26, a[a.index.month == 9], "2026-09")

    a = a.reindex(t26.index).interpolate(limit=6)
    tw = t26[WX]
    k = tw["ws"].mean() / a["ws"].mean()
    scaled = a.copy()
    scaled[["ws", "wind_u", "wind_v"]] *= k
    flat = a.copy()
    for c in ("ws", "wind_u", "wind_v"):
        flat[c] = tw[c].mean()
    cases = [("학습에 쓴 기상 값(25개 구 평균, 이상적인 경우)", "cnn_weather_L48_y5", tw),
             ("ASOS 값 그대로", "cnn_weather_L48_y5", a),
             (f"ASOS, 바람은 평균비(k={k:.2f})로 보정", "cnn_weather_L48_y5", scaled),
             ("ASOS 기온·습도, 바람은 평균 상수", "cnn_weather_L48_y5", flat),
             ("PM2.5 전용 모델(현재 실시간 모델)", "cnn_base_L48_y5", tw)]
    res = [(label,) + mae(name, t26, wx) for label, name, wx in cases]

    out = ["# 기상청 ASOS를 실시간 모델에 쓸 수 있는지 확인 (적용하지 않음)", "",
           "## 1. 학습 데이터와 ASOS 서울(108) 값의 일치도", "",
           "| 기간 | 변수 | 상관 | 학습 데이터 평균 | ASOS 평균 |", "|---|---|---|---|---|", *agree, "",
           "기온·습도는 잘 맞고, 풍속은 ASOS가 약 0.8m/s 크며(관측소 한 곳 vs 25개 구 평균), 풍향 성분(wind_v)은 거의 맞지 않는다.", "",
           f"## 2. 입력 기상 값을 바꿨을 때 24시간 평균 MAE (2026-01~09, 시점 {res[0][3]}개, persistence {res[0][2]:.2f})", "",
           "| 입력 | MAE |", "|---|---|", *[f"| {label} | {m:.2f} |" for label, m, _, _ in res], "",
           "결론: 이상적인 기상 값이 있어도 PM2.5 전용 모델 대비 개선은 약 2.5%이고, ASOS 값은 어느 방식으로 넣어도 PM2.5 전용 모델보다 나쁘다. "
           "그래서 실시간 화면은 PM2.5 전용 모델을 유지한다. (ASOS 한 관측소와 학습 데이터의 25개 구 평균은 정의가 달라 입력 분포가 어긋남. "
           "ASOS로 다시 학습하는 것은 하지 않았다.)", ""]
    (ROOT / "results" / "asos_check.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
