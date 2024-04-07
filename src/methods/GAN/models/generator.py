import torch.nn as nn


# 生成器
# 128 -> 1024 -> 2048 -> 784の4層構造
# 128 -> 1024 -> 2048 の間にReLUを
# 2048 -> 784 でtanhを活性化関数でかけてるモデル
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 28*28)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        # tanh: -1~1に抑える
        return nn.Tanh()(x)
