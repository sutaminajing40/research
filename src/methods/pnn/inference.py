import argparse
import os
import sys

from .model import PNN

MODEL_DIR = "experiments/trained_models/pnn"


def parse_arguments():
    parser = argparse.ArgumentParser(description="PNNモデルの推論を実行します。")
    parser.add_argument("model_path", type=str, help="ロードするモデルファイルのパス")
    return parser.parse_args()


args = parse_arguments()

if not args.model_path:
    print("エラー: モデルファイルのパスを指定してください。", file=sys.stderr)
    sys.exit(1)

pnn = PNN.load_model(os.path.join(MODEL_DIR, args.model_path))

print(pnn([1, 1]))
