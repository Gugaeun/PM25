"""GPU 와 CPU 학습 차이 확인: 1D-CNN+기상을 CPU 로 시드 5개 앙상블로 다시 학습해 16_learning_curve 의 GPU 결과(시드 5개)와 비교한다.

설정은 학습 1년(A)과 5년(S5), 평가는 T1=2025-07~12, T2=2026-01~09 로 16_learning_curve 와 같다.
이어 실행 가능(results/long/*_cpu5_*.npy 캐시). 실행: python src/17_cpu_check.py → results/cpu_check.csv
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

os.environ["PM25_DEVICE"] = "cpu"  # 04_pytorch 를 불러오기 전에 정해야 한다
from data_split import ROOT, load


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


lc = _load("lc", "16_learning_curve.py")
ld, pt, bl = lc.ld, lc.pt, lc.bl
assert pt.DEVICE == "cpu"


def main():
    t0 = time.time()
    full = load(ld.DATA_LONG)
    out = []
    for key in ("A", "S5"):
        label, ranges = lc.SETS[key]
        df = full.loc[ranges[0][0]:]
        y = df["PM25"].values
        tr = np.unique(np.concatenate([ld.origins(df, a, b) for a, b in ranges]))
        S = {"train": tr, "val": ld.origins(df, *lc.VAL), "T1": ld.origins(df, *lc.T1), "T2": ld.origins(df, *lc.T2)}
        S3 = {"train": S["train"], "val": S["val"], "test": S["T1"], "test2": S["T2"]}
        files = {t: ld.CACHE / f"{key}_cnn_weather_cpu5_{t}.npy" for t in ("T1", "T2")}
        if not all(f.exists() for f in files.values()):
            t1 = time.time()
            runs = [pt.run_one("cnn", "weather", 48, sd, df, S3)[0] for sd in range(5)]
            for t, k in (("T1", "test"), ("T2", "test2")):
                np.save(files[t], np.mean([r[k] for r in runs], 0))
            print(f"  {key} CPU 5시드: {time.time() - t1:.0f}s", flush=True)
        preds = {"persistence": {t: bl.persistence_pred(y, S[t]) for t in ("T1", "T2")},
                 "1D-CNN+기상 (CPU 시드5)": {t: np.load(files[t]) for t in ("T1", "T2")},
                 "1D-CNN+기상 (GPU 시드5)": {t: np.load(ld.CACHE / f"{key}_cnn_weather_gpu5_{t}.npy") for t in ("T1", "T2")}}
        for tname in ("T1", "T2"):
            r = ld.evaluate(df, y, S, preds, tname)
            r.insert(0, "평가 구간", {"T1": "2025-07~12", "T2": "2026-01~09"}[tname])
            r.insert(0, "학습", label)
            out.append(r)
        pd.set_option("display.width", 250)
        print(pd.concat(out[-2:])[["학습", "평가 구간", "모델", "MAE", "persistence 대비 %", "6h 재현율", "24h 편향(나쁨)"]].round(2).to_string(index=False), flush=True)
    pd.concat(out).to_csv(ld.RES / "cpu_check.csv", index=False, encoding="utf-8-sig")
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
