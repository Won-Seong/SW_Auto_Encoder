from torchvision.datasets import CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, RandomHorizontalFlip, Resize
import torch
from tqdm import tqdm

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
    
    def make_encoded_data(self, path : str, dl : DataLoader, auto_encoder : torch.nn.Module):
        encoded_data = []
        device = next(auto_encoder.parameters()).device
        with torch.no_grad():
            for batch in tqdm(dl):
                if type(batch) == list:
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                encoded = auto_encoder.encode(x)
                encoded_data.append(encoded.cpu())
        
        encoded_data = torch.cat(encoded_data, dim = 0)
        torch.save({"encoded_data": encoded_data}, f = path)
        print(f"Encoded data save to {path}")