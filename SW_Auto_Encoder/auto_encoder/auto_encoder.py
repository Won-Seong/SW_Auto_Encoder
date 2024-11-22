import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from auto_encoder.Decoder import Decoder

from auto_encoder.encoder import Encoder
import yaml


class AutoEncoder(nn.Module):
    def __init__(self, config_path : str, device = None):
        super().__init__()
        self.device = device
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            
        self.encoder = Encoder(**config["encoder"])
        self.decoder = Decoder(**config["decoder"])
        
    def encode(self, x):
        h = self.encoder(x)
        return h
        
    def decode(self, z):
        z = self.decoder(z)
        return z
    
    def reconstruct(self, x):
        return self.decode(self.encode(x))
    
    def loss(self, x, x_hat):
        return F.mse_loss(x, x_hat)
        
    def load(self, file_name : str):
        self.load_state_dict(torch.load('models/' + file_name + '.pt', map_location = self.device))
        self.eval()
        print("=====Model loaded!=====")
        
    def forward(self, x):
        return self.reconstruct(x)
        
    