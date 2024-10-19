import glob
import json
import os
from datetime import datetime

import torch
import torch.nn as nn


class PNN(nn.Module):
    def __init__(self):
        super(PNN, self).__init__()
        self.centroids = nn.Parameter(torch.Tensor(), requires_grad=False)
        self.output_weights = nn.Parameter(torch.Tensor(), requires_grad=False)

    def gaussian(self, x: torch.Tensor, centroid_vec: torch.Tensor, radius: float = 1):
        return torch.exp((-1 * torch.norm(x - centroid_vec, p=2, dim=-1)) / (radius**2))

    def forward(self, x: torch.Tensor):
        # 入力 -> 隠れ層
        hidden = self.gaussian(
            x=x.unsqueeze(1), centroid_vec=self.centroids.unsqueeze(0), radius=0.5
        )

        # 隠れ層 -> 出力
        output = torch.matmul(hidden, self.output_weights)

        return output

    def train(self, data_dir: str):
        # JSONファイルの読み込み
        data = self.load_json_files(data_dir)

        # セントロイドと出力重みの設定
        self.centroids = nn.Parameter(
            torch.tensor([d["input"] for d in data], dtype=torch.float32),
            requires_grad=False,
        )
        self.output_weights = nn.Parameter(
            torch.tensor([d["output"][0] for d in data], dtype=torch.float32).unsqueeze(
                1
            ),
            requires_grad=False,
        )

    def load_json_files(self, data_dir: str):
        data = []
        for file_path in glob.glob(os.path.join(data_dir, "*.json")):
            with open(file_path, "r") as f:
                data.append(json.load(f))
        return data

    def save_model(self, save_dir: str):
        current_date = datetime.now().strftime("%Y%m%d")
        model_name = f"pnn_model_{current_date}.pth"
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load_model(cls, file_path: str):
        model = cls()
        model.load_state_dict(torch.load(file_path))
        return model
