import torch
import torch.nn as nn
import numpy as np

class ResnetBlock(nn.module):
    def __init__(self, *, in_channels, out_channels = None):
        super().__init__()
        self.