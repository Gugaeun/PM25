"""장기 데이터 실험: (1) 학습 때 한 번도 못 본 2026-01~09 로 평가, (2) 학습 기간을 3년으로 늘리면 좋아지는가.

두 설정을 같은 모델·같은 평가로 비교한다.
  A (기존): 학습 2024 / 검증 2025-01~06            → 평가 T1=2025-07~12(기존 테스트), T2=2026-01~09(새 테스트)
  B (확대): 학습 2022~2024 / 검증 2025-01~06        → 같은 T1, T2
A 의 T1 결과는 이전 결과(예: 1D-CNN+기상 4.55)와 같아야 한다(같은 시드·같은 학습 데이터). 이것이 재현 검증이다.
모델: persistence, LightGBM(PM2.5만/+기상), 1D-CNN(PM2.5만/+기상), LSTM+기상 (NN 은 시드 3개 앙상블).
지표: MAE, persistence 대비 감소율(하루 단위 부트스트랩 95% 구간), 다음날 평균 MAE, 고농도(>35) 6h/24h 재현율·정밀도, 24h 나쁨 구간 편향.

실행: python src/15_longer_data.py   →  results/longer_data.csv, results/longer_data.md, results/long/*.npy (이어 실행 가능)
"""
import importlib.util
import time

import numpy as np
import pandas as pd

from data_split import HORIZONS, MIN_HISTORY, ROOT, load

DATA_LONG = ROOT / "data" / "processed" / "seoul_hourly_long.csv"
RES = ROOT / "results"
CACHE = RES / "long"
CACHE.mkdir(exist_ok=True)
T = 35.0


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bl = _load("bl", "02_baselines.py")
pt = _load("pt", "04_pytorch.py")
ev = _load("ev", "11_event_metrics.py")

SPLITS = {  # (시작, 끝)
    "A": {"train": ("2024-01-01 00:00", "2024-12-31 23:00"), "val": ("2025-01-01 00:00", "2025-06-30 23:00"),
          "T1": ("2025-07-01 00:00", "2025-12-31 23:00"), "T2": ("2026-01-01 00:00", "2026-09-30 23:00")},
    "B": {"train": ("2022-01-01 00:00", "2024-12-31 23:00"), "val": ("2025-01-01 00:00", "2025-06-30 23:00"),
          "T1": ("2025-07-01 00:00", "2025-12-31 23:00"), "T2": ("2026-01-01 00:00", "2026-09-30 23:00")},
}


def origins(df, a, b):
    idx = df.index
    pos = np.where((idx >= a) & (idx <= b))[0]
    pos = pos[pos + 24 <= np.where(idx <= b)[0].max()]  # 정답이 구간을 넘지 않게
    return pos[pos >= MIN_HISTORY]


def boot_ci(err_m, err_b, day, n=2000, seed=0):
    """하루 단위로 다시 뽑아(부트스트랩) persistence 대비 MAE 감소율(%)의 95% 구간."""
    rng = np.random.RandomState(seed)
    d = pd.DataFrame({"m": err_m, "b": err_b, "day": day}).groupby("day").sum()
    cnt = pd.Series(day).value_counts().sort_index().values
    sm, sb = d["m"].values, d["b"].values
    k = len(sm)
    r = []
    for _ in range(n):
        i = rng.randint(0, k, k)
        r.append(100 * (1 - sm[i].sum() / sb[i].sum()))
    return np.percentile(r, [2.5, 97.5])


def evaluate(df, y, S, preds, tname):
    pos = S[tname]
    Y = np.stack([y[pos + h] for h in HORIZONS], 1)
    day = df.index[pos].floor("D").values
    d23 = df.index[pos].hour == 23
    rows = []
    pers = np.abs(preds["persistence"][tname] - Y).mean(1)
    for m, P in preds.items():
        p = P[tname]
        e = np.abs(p - Y).mean(1)
        row = {"모델": m, "MAE": float(e.mean()), "다음날 평균 MAE": float(np.abs(p[d23].mean(1) - Y[d23].mean(1)).mean())}
        if m != "persistence":
            lo, hi = boot_ci(e, pers, day)
            row.update({"persistence 대비 %": 100 * (1 - e.mean() / pers.mean()), "구간 하한": lo, "구간 상한": hi})
        for h in (6, 24):
            a = Y[:, h - 1]
            rec, pre, f1, _ = ev.prf(a > T, p[:, h - 1] > T)
            row.update({f"{h}h 재현율": rec, f"{h}h 정밀도": pre})
        a24, p24 = Y[:, 23], p[:, 23]
        for lab, lo_, hi_ in (("나쁨", 35, 75), ("매우나쁨", 75, 1e9)):
            mm = (a24 > lo_) & (a24 <= hi_)
            row[f"24h 편향({lab})"] = float((p24[mm] - a24[mm]).mean()) if mm.sum() else np.nan
        row["고농도 시간(>35)"] = int((Y[:, 23] > T).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    full = load(DATA_LONG)
    results = []
    for setting in ("A", "B"):
        df = full.loc["2024-01-01":] if setting == "A" else full.loc["2022-01-01":]  # 학습 시작일부터 자른다(장기 데이터가 더 길어져도 같은 결과)
        y = df["PM25"].values
        S = {k: origins(df, *v) for k, v in SPLITS[setting].items()}
        S3 = {"train": S["train"], "val": S["val"], "test": S["T1"], "test2": S["T2"]}  # run_one/lgbm 이 쓰는 이름
        print(f"[{setting}] 학습 샘플 {len(S['train']):,} / 검증 {len(S['val']):,} / T1 {len(S['T1']):,} / T2 {len(S['T2']):,}", flush=True)
        preds = {"persistence": {t: bl.persistence_pred(y, S[t]) for t in ("T1", "T2")},
                 "naive": {t: bl.naive_pred(y, S[t]) for t in ("T1", "T2")}}

        def cached(name, fn):
            files = {t: CACHE / f"{setting}_{name}_{t}.npy" for t in ("T1", "T2")}
            if not all(f.exists() for f in files.values()):
                t1 = time.time()
                out = fn()
                for t, f in files.items():
                    np.save(f, out[t])
                print(f"  {setting} {name}: {time.time() - t1:.0f}s", flush=True)
            return {t: np.load(f) for t, f in files.items()}

        for g, label in (("base", "LightGBM(PM2.5만)"), ("weather", "LightGBM+기상")):
            wc = [c for c in pt.feature_cols(g) if c != "PM25"]
            preds[label] = cached(f"lgbm_{g}", lambda wc=wc: (lambda P: {"T1": P["test"], "T2": P["test2"]})(bl.lgbm_pred(df, S3, wc)))
        for model, g, label in (("cnn", "base", "1D-CNN(PM2.5만)"), ("cnn", "weather", "1D-CNN+기상"), ("lstm", "weather", "LSTM+기상")):
            def fn(model=model, g=g):
                runs = [pt.run_one(model, g, 48, sd, df, S3)[0] for sd in range(3)]
                return {"T1": np.mean([r["test"] for r in runs], 0), "T2": np.mean([r["test2"] for r in runs], 0)}
            preds[label] = cached(f"{model}_{g}", fn)
        for tname in ("T1", "T2"):
            r = evaluate(df, y, S, preds, tname)
            r.insert(0, "평가 구간", {"T1": "2025-07~12", "T2": "2026-01~09"}[tname])
            r.insert(0, "학습", {"A": "2024 (1년)", "B": "2022~2024 (3년)"}[setting])
            results.append(r)
            pd.set_option("display.width", 250)
            print(r.round(2).to_string(index=False), flush=True)
    out = pd.concat(results)
    out.to_csv(RES / "longer_data.csv", index=False, encoding="utf-8-sig")
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
