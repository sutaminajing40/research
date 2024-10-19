import argparse
import os
import sys

import torch

from .consts import INPUT
from .model import XOR_PNN

MODEL_DIR = "experiments/trained_models/xor_pnn"


def parse_arguments():
    parser = argparse.ArgumentParser(description="PNNモデルの推論を実行します。")
    parser.add_argument("model_path", type=str, help="ロードするモデルファイルのパス")
    return parser.parse_args()


args = parse_arguments()

if not args.model_path:
    print("エラー: モデルファイルのパスを指定してください。", file=sys.stderr)
    sys.exit(1)

xor_pnn = XOR_PNN.load_model(os.path.join(MODEL_DIR, args.model_path))

print(xor_pnn(torch.tensor(INPUT, dtype=torch.float32)))
