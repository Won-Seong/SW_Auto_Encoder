from torchvision.datasets import CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, RandomHorizontalFlip, Resize
import torch

class DataGenerator():
    def __init__(self, ):
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)
            ])
        
    def fashion_mnist(self, path = './datasets', batch_size : int = 128, train : bool = True):
        train_data = FashionMNIST(path, download = True, train = train, transform = self.transform)
        dl = DataLoader(train_data, batch_size, shuffle = True)
        return dl
        
    def noise(self, size, batch_size : int = 128):
        train_data = torch.randn(size = size)
        dl = DataLoader(train_data, batch_size, shuffle = True)
        return dl
    
    