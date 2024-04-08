import os

import matplotlib.pyplot as plt
import torch

from src.methods.GAN.models.generator import generator


def generate_image(model_name, save_path, num_images=1):
    # モデルの読み込み
    model = generator()
    model_path = f"experiments/trained_models/GAN/{model_name}"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 乱数を生成器の入力として使用
    for _ in range(num_images):
        noise = torch.randn(1, 128)
        with torch.no_grad():
            generated_image = model(noise)

        # 画像を保存
        plt.imshow(generated_image.squeeze().cpu().numpy(), cmap='gray')
        image_save_path = f"{save_path}/{model_name.split('.')[0]}"
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        plt.savefig(f"{image_save_path}/generated_image_{_}.png")

    print(f"successfully generated images for {model_name}!")


if __name__ == '__main__':
    generate_image(
        'generator_epoch_29_20240408_052953.pth',
        'experiments/results/GAN/generated_images',
        num_images=5)
