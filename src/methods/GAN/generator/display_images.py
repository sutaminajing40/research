import os

import streamlit as st
from PIL import Image


def load_images(images_path):
    images = [
        os.path.join(images_path, img) for img in os.listdir(images_path)
        if img.endswith('.png')
    ]
    return images


def display_images(images):
    for img_path in images:
        img = Image.open(img_path)
        st.image(
            img,
            caption=os.path.basename(img_path),
            use_column_width=True
        )


def main():
    st.sidebar.title("GAN 画像ビューア")
    models_path = 'experiments/results/GAN/generated_images'
    model_names = [f for f in os.listdir(models_path) if not f.startswith('.')]

    selected_model = st.sidebar.selectbox("モデルを選択", model_names)
    images_path = os.path.join(models_path, selected_model)
    images = load_images(images_path)

    display_images(images)

    if st.sidebar.button("モデル比較"):
        st.title("モデル比較")
        for model_name in model_names:
            st.header(model_name)
            images_path = os.path.join(models_path, model_name)
            images = load_images(images_path)
            display_images(images)


if __name__ == '__main__':
    main()
