import torch
import torch.nn as nn
import torch.optim as optim
from models.discriminator import discriminator
from models.generator import generator

from src.utils.save_model import save_model

from .dataloader import get_dataloader


def train():
    # GPU利用可否確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ハイパーパラメタ設定
    epochs = 30
    lr = 2e-4
    batch_size = 64
    loss = nn.BCELoss()

    # Model
    G_model = generator().to(device)
    D_model = discriminator().to(device)

    G_optimizer = optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))

    train_loader = get_dataloader(batch_size)

    for epoch in range(epochs):
        for idx, (imgs, _) in enumerate(train_loader):
            idx += 1

            # 識別器の学習
            # 本物の入力は，MNISTデータセットの実際の画像
            # 偽の入力はジェネレータから
            # 本物の入力は1に、偽物は0に分類されるべきである
            real_inputs = imgs.to(device)
            real_outputs = D_model(real_inputs)
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
            # ジェネレータにとっての目標は 識別者に全てが1であると信じさせること
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
                print("""
                    Epoch {} Iteration {}: discriminator_loss {:.3f}
                    generator_loss {:.3f}
                    """.format(
                        epoch, idx, D_loss.item(), G_loss.item()))

        if epoch == epochs-1:
            # モデルの保存
            save_model(G_model, epoch, 'GAN')
