"""학습 곡선: 학습 기간을 1·3·5·7년으로 늘리면 얼마나 좋아지는가, 코로나 시기를 빼면 달라지는가.

검증 2025-01~06, 평가 T1=2025-07~12 와 T2=2026-01~09 는 모든 설정에서 같다. 학습만 바꾼다(항상 2024-12-31 까지).
  1년  2024
  3년  2022~2024         (15_longer_data 의 설정 B 와 같다 → 캐시 재사용)
  5년  2020~2024
  7년  2018~2024
  5년(코로나 제외)  2018~2019 + 2022~2024  (2020~2021 배출 감소기를 학습에서 뺀 비교)
모델: LightGBM(PM2.5만/+기상), 1D-CNN(PM2.5만/+기상), LSTM+기상. 신경망은 GPU 로 시드 5개 앙상블.
주의: GPU 와 CPU 는 같은 시드라도 결과가 조금 달라서, 이 실험은 1년·3년 설정도 GPU 로 새로 학습한다(15_longer_data 의 CPU 결과와 소수점이 다를 수 있다).
평가는 15_longer_data 와 같다.
이어 실행 가능(results/long/*.npy 캐시). 실행: python src/16_learning_curve.py → results/learning_curve.csv
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("PM25_DEVICE", "cuda")  # 신경망은 GPU 로 학습(04_pytorch 를 불러오기 전에 정해야 한다)
from data_split import ROOT, load


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ld = _load("ld", "15_longer_data.py")
bl, pt = ld.bl, ld.pt
E = "2024-12-31 23:00"
VAL, T1, T2 = ("2025-01-01 00:00", "2025-06-30 23:00"), ("2025-07-01 00:00", "2025-12-31 23:00"), ("2026-01-01 00:00", "2026-09-30 23:00")
SETS = {  # 캐시 키 → (표시 이름, 학습 구간들)
    "A": ("1년 (2024)", [("2024-01-01 00:00", E)]),
    "B": ("3년 (2022~2024)", [("2022-01-01 00:00", E)]),
    "S5": ("5년 (2020~2024)", [("2020-01-01 00:00", E)]),
    "S7": ("7년 (2018~2024)", [("2018-01-01 00:00", E)]),
    "S5nc": ("5년 코로나 제외 (2018~19+2022~24)", [("2018-01-01 00:00", "2019-12-31 23:00"), ("2022-01-01 00:00", E)]),
}


def main():
    t0 = time.time()
    full = load(ld.DATA_LONG)
    out = []
    for key, (label, ranges) in SETS.items():
        start = ranges[0][0]
        df = full.loc[start:]
        y = df["PM25"].values
        tr = np.unique(np.concatenate([ld.origins(df, a, b) for a, b in ranges]))
        S = {"train": tr, "val": ld.origins(df, *VAL), "T1": ld.origins(df, *T1), "T2": ld.origins(df, *T2)}
        S3 = {"train": S["train"], "val": S["val"], "test": S["T1"], "test2": S["T2"]}
        print(f"[{label}] 학습 샘플 {len(tr):,}", flush=True)
        preds = {"persistence": {t: bl.persistence_pred(y, S[t]) for t in ("T1", "T2")}}

        def cached(name, fn):
            files = {t: ld.CACHE / f"{key}_{name}_{t}.npy" for t in ("T1", "T2")}
            if not all(f.exists() for f in files.values()):
                t1 = time.time()
                res = fn()
                for t, f in files.items():
                    np.save(f, res[t])
                print(f"  {key} {name}: {time.time() - t1:.0f}s", flush=True)
            return {t: np.load(f) for t, f in files.items()}

        for g, name in (("base", "LightGBM(PM2.5만)"), ("weather", "LightGBM+기상")):
            wc = [c for c in pt.feature_cols(g) if c != "PM25"]
            preds[name] = cached(f"lgbm_{g}", lambda wc=wc: (lambda P: {"T1": P["test"], "T2": P["test2"]})(bl.lgbm_pred(df, S3, wc)))

        for model, g, mlabel in (("cnn", "base", "1D-CNN(PM2.5만)"), ("cnn", "weather", "1D-CNN+기상"), ("lstm", "weather", "LSTM+기상")):
            def fn(model=model, g=g):
                runs = [pt.run_one(model, g, 48, sd, df, S3)[0] for sd in range(5)]
                return {"T1": np.mean([r["test"] for r in runs], 0), "T2": np.mean([r["test2"] for r in runs], 0)}
            preds[mlabel] = cached(f"{model}_{g}_gpu5", fn)
        for tname in ("T1", "T2"):
            r = ld.evaluate(df, y, S, preds, tname)
            r.insert(0, "평가 구간", {"T1": "2025-07~12", "T2": "2026-01~09"}[tname])
            r.insert(0, "학습", label)
            r.insert(1, "학습 샘플", len(tr))
            out.append(r)
        pd.set_option("display.width", 250)
        print(pd.concat(out[-2:])[["학습", "평가 구간", "모델", "MAE", "persistence 대비 %", "6h 재현율", "24h 편향(나쁨)", "24h 편향(매우나쁨)"]].round(2).to_string(index=False), flush=True)
    pd.concat(out).to_csv(ld.RES / "learning_curve.csv", index=False, encoding="utf-8-sig")
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
