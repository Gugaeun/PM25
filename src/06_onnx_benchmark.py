"""PyTorch 모델 저장 → ONNX 변환 → 출력 차이(max diff) / 추론 속도 비교.

대상: 기상 변수(weather)까지 넣은 1D-CNN, LSTM (시드 0, 윈도우 48h)
산출: models/*.pt, models/*.onnx, results/onnx_benchmark.csv
실행: python src/06_onnx_benchmark.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from data_split import HORIZONS, ROOT, load, metrics, split_origins

spec = importlib.util.spec_from_file_location("pt", ROOT / "src" / "04_pytorch.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
GROUP, L = "weather", 48
SUFFIX = os.environ.get("MODEL_SUFFIX", "")  # "_y5" 이면 학습 5년 모델(20_serving_models_5y.py)을 잰다


def load_net(path):
    ck = torch.load(path, weights_only=False)
    net = pt.LSTMNet(ck["n_in"]) if ck["model"] == "lstm" else pt.CNNNet(ck["n_in"])
    net(torch.zeros(1, ck["L"], ck["n_in"]))  # LazyLinear 초기화
    net.load_state_dict(ck["state"])
    return net.eval(), ck


def bench(fn, x, n=300, warm=30):
    for _ in range(warm):
        fn(x)
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1000)
    return np.percentile(t, 50), np.percentile(t, 95)


def main():
    torch.set_num_threads(1)  # 서빙 환경(요청당 소수 코어)을 가정한 공정 비교
    df = load()
    S = split_origins(df)
    y = df["PM25"].values
    rows = []
    for name in ("cnn", "lstm"):
        pth, onx = MODELS / f"{name}_{GROUP}_L{L}{SUFFIX}.pt", MODELS / f"{name}_{GROUP}_L{L}{SUFFIX}.onnx"
        if not pth.exists():
            torch.set_num_threads(8)
            pt.run_one(name, GROUP, L, 0, df, S, save_to=pth)
            torch.set_num_threads(1)
        net, ck = load_net(pth)

        # 입력은 시퀀스 길이(L)와 배치를 동적 축으로 (CNN은 flatten 크기 때문에 L 고정)
        dyn = {"x": {0: "batch", 1: "seq"}, "y": {0: "batch"}} if name == "lstm" else {"x": {0: "batch"}, "y": {0: "batch"}}
        dummy = torch.zeros(1, ck["L"], ck["n_in"])
        torch.onnx.export(net, dummy, str(onx), input_names=["x"], output_names=["y"],
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        # API 가 torch 없이 쓰도록 정규화 통계/입력 사양을 JSON 으로 저장
        (MODELS / f"{name}_{GROUP}_L{L}{SUFFIX}.json").write_text(json.dumps(
            {"model": name, "cols": ck["cols"], "L": ck["L"], "mu": [float(v) for v in ck["mu"]],
             "sd": [float(v) for v in ck["sd"]]}, indent=1), encoding="utf-8")
        sess = ort.InferenceSession(str(onx), providers=["CPUExecutionProvider"])

        # 테스트 입력 (정규화 적용)
        Xall = pt.build_inputs(df, ck["cols"], ck["L"])
        Xn = ((Xall - ck["mu"]) / ck["sd"]).astype(np.float32)
        X, Y = pt.make_set(Xn, ((y - ck["mu"][0]) / ck["sd"][0]).astype(np.float32), S["test"], ck["L"])
        with torch.no_grad():
            p_torch = net(X).numpy()
        p_onnx = sess.run(None, {"x": X.numpy()})[0]
        maxdiff = float(np.abs(p_torch - p_onnx).max() * ck["sd"][0])  # µg/m³ 단위
        Yt = np.stack([y[S["test"] + h] for h in HORIZONS], 1)
        un = lambda p: np.clip(p * ck["sd"][0] + ck["mu"][0], 0, None)

        x1, x64 = X[:1], X[:64]
        with torch.no_grad():
            t1 = bench(lambda a: net(a), x1)
            t64 = bench(lambda a: net(a), x64, n=100, warm=10)
        o1 = bench(lambda a: sess.run(None, {"x": a}), x1.numpy())
        o64 = bench(lambda a: sess.run(None, {"x": a}), x64.numpy(), n=100, warm=10)

        row = {"model": name, "onnx_MB": onx.stat().st_size / 1e6, "max_abs_diff(µg/m³)": maxdiff,
               "test_MAE_torch": metrics(Yt, un(p_torch))["MAE"], "test_MAE_onnx": metrics(Yt, un(p_onnx))["MAE"],
               "torch_b1_p50_ms": t1[0], "torch_b1_p95_ms": t1[1], "onnx_b1_p50_ms": o1[0], "onnx_b1_p95_ms": o1[1],
               "torch_b64_p50_ms": t64[0], "onnx_b64_p50_ms": o64[0], "speedup_b1_p50": t1[0] / o1[0]}
        if name == "lstm":  # 동적 시퀀스 길이 확인: L=24 입력도 실행되는가
            out = sess.run(None, {"x": X[:4, -24:].numpy()})[0]
            row["dynamic_seq_ok"] = bool(out.shape == (4, 24))
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / f"onnx_benchmark{SUFFIX}.csv" if SUFFIX else ROOT / "results" / "onnx_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
