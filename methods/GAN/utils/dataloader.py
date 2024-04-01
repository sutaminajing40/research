from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def get_dataloader(batch_size: int = 64):
    # Transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load data
    train_set = datasets.MNIST(
        'mnist/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
