"""실시간 데모용 PM2.5 전용 모델: 학습 → ONNX 변환 → 정규화 JSON 저장.

에어코리아 API 에는 기온·습도·풍속이 없어서, 입력이 PM2.5(과 시각)뿐인 1D-CNN 을 쓴다.
산출: models/cnn_base_L48.{pt,onnx,json}  (api/main.py 의 /live/predict 가 사용)
실행: python src/08_live_model.py
"""
import importlib.util
import json

import torch

from data_split import ROOT, load, split_origins

spec = importlib.util.spec_from_file_location("pt", ROOT / "src" / "04_pytorch.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

NAME, GROUP, L = "cnn", "base", 48
MODELS = ROOT / "models"


def main():
    df = load()
    S = split_origins(df)
    stem = f"{NAME}_{GROUP}_L{L}"
    pth, onx = MODELS / f"{stem}.pt", MODELS / f"{stem}.onnx"
    if not pth.exists():
        pt.run_one(NAME, GROUP, L, 0, df, S, save_to=pth)
    ck = torch.load(pth, weights_only=False)
    net = pt.CNNNet(ck["n_in"])
    net(torch.zeros(1, ck["L"], ck["n_in"]))  # LazyLinear 초기화
    net.load_state_dict(ck["state"])
    net.eval()
    torch.onnx.export(net, torch.zeros(1, ck["L"], ck["n_in"]), str(onx), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)
    (MODELS / f"{stem}.json").write_text(json.dumps(
        {"model": NAME, "cols": ck["cols"], "L": ck["L"], "mu": [float(v) for v in ck["mu"]],
         "sd": [float(v) for v in ck["sd"]]}, indent=1), encoding="utf-8")
    print("saved", onx)


if __name__ == "__main__":
    main()
