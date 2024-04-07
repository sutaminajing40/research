import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.save_model import save_model

from ..models.discriminator import discriminator
from ..models.generator import generator
from .dataloader import get_dataloader


def train():
    # GPU利用可否確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ハイパーパラメタ設定
    epochs = 30
    lr = 2e-4
    batch_size = 256
    loss = nn.BCELoss()

    # Model
    G_model = generator().to(device)
    D_model = discriminator().to(device)

    G_optimizer = optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))

    train_loader = get_dataloader(batch_size)

    for epoch in range(epochs):
        with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}") as t:  # noqa
            for idx, (imgs, _) in t:
                idx += 1  # tqdmのenumerateは0から始まるため、1を足す

                # 識別器の学習
                # 本物画像を識別器へ入力
                real_inputs = imgs.to(device)
                real_outputs = D_model(real_inputs)

                # 本物画像に対する正解ラベル(絶対に1)
                real_label = torch.ones(real_inputs.shape[0], 1).to(device)

                noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
                noise = noise.to(device)
                fake_inputs = G_model(noise)
                fake_outputs = D_model(fake_inputs)
                fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

                outputs = torch.cat((real_outputs, fake_outputs), 0)
                targets = torch.cat((real_label, fake_label), 0)

                D_loss = loss(outputs, targets)
                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                # Generatorのトレーニング
                noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
                noise = noise.to(device)

                fake_inputs = G_model(noise)
                fake_outputs = D_model(fake_inputs)
                fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
                G_loss = loss(fake_outputs, fake_targets)
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                if idx % 100 == 0 or idx == len(train_loader):
                    t.write(f"Iteration {idx}: discriminator_loss {D_loss.item():.3f} generator_loss {G_loss.item():.3f}")  # noqa

        if epoch == epochs-1:
            # モデルの保存
            save_model(G_model, epoch, 'GAN')


if __name__ == '__main__':
    train()
