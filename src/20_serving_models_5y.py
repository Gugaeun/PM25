"""서빙용 모델을 학습 5년(2020~2024)으로 다시 학습해 ONNX 로 내보낸다.

왜: 16_learning_curve 에서 학습 기간을 1년 → 5년으로 늘리면 persistence 대비 감소율이 좋아졌다(2026 평가 1D-CNN+기상 10.9% → 15.1%).
검증 2025-01~06, 테스트 2025-07~12 는 그대로이고 학습은 2024-12-31 까지다(데모가 재현하는 2025-07 이후는 학습에 쓰지 않는다).
모델 3개(시드 0, 윈도우 48h): 1D-CNN+기상, LSTM+기상, 1D-CNN(PM2.5만, 실시간용). 파일 이름 뒤에 _y5 를 붙이고 기존 1년 모델은 그대로 둔다.
산출: models/*_L48_y5.{pt,onnx,json}, results/serving_y5.csv, results/serving_y5.md
실행: PM25_DEVICE=cuda python src/20_serving_models_5y.py (학습은 GPU, 없으면 CPU)
"""
import importlib.util
import json
import os

os.environ.setdefault("PM25_DEVICE", "cuda")
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from data_split import HORIZONS, ROOT, load

MODELS = ROOT / "models"
L = 48
E = "2024-12-31 23:00"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ld = _load("ld", "15_longer_data.py")
ob = _load("ob", "06_onnx_benchmark.py")
pt = ob.pt


def onnx_mae(stem, df, S):
    """저장된 ONNX + JSON(정규화) 으로 test(T1)·T2 의 1~24h 평균 MAE 와 persistence 대비 감소율."""
    meta = json.loads((MODELS / f"{stem}.json").read_text())
    sess = ort.InferenceSession(str(MODELS / f"{stem}.onnx"), providers=["CPUExecutionProvider"])
    y = df["PM25"].values
    mu, sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
    Xn = ((pt.build_inputs(df, meta["cols"], meta["L"]) - mu) / sd).astype(np.float32)
    yn = ((y - mu[0]) / sd[0]).astype(np.float32)
    out = {}
    for t in ("T1", "T2"):
        X, _ = pt.make_set(Xn, yn, S[t], meta["L"])
        p = np.clip(sess.run(None, {"x": X.numpy()})[0] * sd[0] + mu[0], 0, None)
        Y = np.stack([y[S[t] + h] for h in HORIZONS], 1)
        pers = np.stack([np.repeat(y[S[t]][:, None], 24, 1)], 0)[0]
        out[t] = (float(np.abs(p - Y).mean()), float(np.abs(pers - Y).mean()))
    return out


def main():
    full = load(ld.DATA_LONG)
    df5 = full.loc["2020-01-01":]
    y5 = df5["PM25"].values
    S5 = {"train": ld.origins(df5, "2020-01-01 00:00", E), "val": ld.origins(df5, *ld.VAL) if hasattr(ld, "VAL") else None}
    VAL, T1, T2 = ("2025-01-01 00:00", "2025-06-30 23:00"), ("2025-07-01 00:00", "2025-12-31 23:00"), ("2026-01-01 00:00", "2026-09-30 23:00")
    S5 = {"train": ld.origins(df5, "2020-01-01 00:00", E), "val": ld.origins(df5, *VAL), "T1": ld.origins(df5, *T1), "T2": ld.origins(df5, *T2)}
    S3 = {"train": S5["train"], "val": S5["val"], "test": S5["T1"], "test2": S5["T2"]}
    # 기존 1년 모델을 같은 평가 구간에서 다시 재려고 학습 1년 데이터프레임도 만든다(평가 시점은 같은 날짜)
    df1 = full.loc["2024-01-01":]
    S1 = {"T1": ld.origins(df1, *T1), "T2": ld.origins(df1, *T2)}
    print(f"학습 샘플 {len(S5['train']):,} (장치 {pt.DEVICE})", flush=True)

    rows = []
    for name, group in (("cnn", "weather"), ("lstm", "weather"), ("cnn", "base")):
        stem, old = f"{name}_{group}_L{L}_y5", f"{name}_{group}_L{L}"
        pth, onx = MODELS / f"{stem}.pt", MODELS / f"{stem}.onnx"
        pt.run_one(name, group, L, 0, df5, S3, save_to=pth)
        net, ck = ob.load_net(pth)
        dyn = {"x": {0: "batch", 1: "seq"}, "y": {0: "batch"}} if name == "lstm" else {"x": {0: "batch"}, "y": {0: "batch"}}
        torch.onnx.export(net, torch.zeros(1, ck["L"], ck["n_in"]), str(onx), input_names=["x"], output_names=["y"], dynamic_axes=dyn, opset_version=17, dynamo=False)
        (MODELS / f"{stem}.json").write_text(json.dumps({"model": name, "cols": ck["cols"], "L": ck["L"], "mu": [float(v) for v in ck["mu"]], "sd": [float(v) for v in ck["sd"]]}, indent=1), encoding="utf-8")
        # PyTorch 와 ONNX 출력 차이
        Xall = pt.build_inputs(df5, ck["cols"], ck["L"])
        Xn = ((Xall - ck["mu"]) / ck["sd"]).astype(np.float32)
        X, _ = pt.make_set(Xn, ((y5 - ck["mu"][0]) / ck["sd"][0]).astype(np.float32), S5["T1"], ck["L"])
        with torch.no_grad():
            p_t = net(X).numpy()
        sess = ort.InferenceSession(str(onx), providers=["CPUExecutionProvider"])
        diff = float(np.abs(p_t - sess.run(None, {"x": X.numpy()})[0]).max() * ck["sd"][0])
        new_m, old_m = onnx_mae(stem, df5, S5), onnx_mae(old, df1, S1)
        for t, lab in (("T1", "2025-07~12"), ("T2", "2026-01~09")):
            rows.append({"모델": f"{name}_{group}", "평가 구간": lab, "1년 학습 MAE": round(old_m[t][0], 3), "5년 학습 MAE": round(new_m[t][0], 3),
                         "persistence MAE": round(new_m[t][1], 3), "1년 감소율%": round((1 - old_m[t][0] / old_m[t][1]) * 100, 1),
                         "5년 감소율%": round((1 - new_m[t][0] / new_m[t][1]) * 100, 1), "torch-onnx 최대차(µg/m³)": round(diff, 6) if t == "T1" else None})
        print(pd.DataFrame(rows[-2:]).to_string(index=False), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "serving_y5.csv", index=False, encoding="utf-8-sig")
    (ROOT / "results" / "serving_y5.md").write_text("# 서빙 모델: 학습 1년 대 5년 (시드 0 단일 모델, 1~24h 평균 MAE)\n\n" + r.to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
