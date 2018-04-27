import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_features=64, out_features=64):
        super().__init__()
                    
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )
        
    def forward(self, x):
        return x + self.block(x)

def weights_initialization(layer):
    name = layer.__class__.__name__
    if name.find('Linear') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif name.find('BatchNorm') != -1:
        layer.weight.data.normal_(0.0, 0.02)
        layer.bias.data.fill_(0)