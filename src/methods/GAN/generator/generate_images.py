import os

from src.methods.GAN.generator.generate_image import generate_image


def generate_images_for_new_models(models_path, save_path, num_images=1):
    # 保存されているモデルのリストを取得
    model_files = [f for f in os.listdir(models_path) if f.endswith('.pth')]

    for model_file in model_files:
        model_name = model_file.split('.')[0]
        image_dir_path = f"{save_path}/{model_name}"

        # 既に画像が生成されているかチェック
        if not os.path.exists(image_dir_path):
            generate_image(model_file, save_path, num_images)
        else:
            print(f"Images for {model_name} already generated. Skipping.")


if __name__ == '__main__':
    generate_images_for_new_models(
        'experiments/trained_models/GAN',
        'experiments/results/GAN/generated_images',
        num_images=5)
