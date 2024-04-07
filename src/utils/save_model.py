import os
from datetime import datetime

import torch
import torch.nn as nn


def save_model(
        model: nn.Module,
        epoch: int,
        experiment: str,
        model_path='./experiments/trained_models/',
        ):
    class_name = model.__class__.__name__
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        model_path,
        experiment,
        f"{class_name}_epoch_{epoch}_{current_time}.pth")

    # モデルの保存
    torch.save(model, model_path)
    print(f'Model saved at {model_path}')
