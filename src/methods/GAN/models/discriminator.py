import torch.nn as nn


# 識別器
# 784 -> 512 -> 1の3層構造
# 784 -> 512の間にLeakyReLUを
# 512 -> 1でsigmoidを活性化関数でかけてるモデル
# LeakyReLU: 0以下の入力に対しても非常に小さな正の勾配を出力するReLUの変種
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x
