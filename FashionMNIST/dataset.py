from torchvision import datasets
from torchvision.transforms import ToTensor


# torch.utils.data.Dataset - Dataset stores the samples and their corresponding labels
# torch.utils.data.DataLoader - DataLoader wraps an iterable around the Dataset

# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets.

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)