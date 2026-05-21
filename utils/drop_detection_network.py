import torch
from torch import nn

class SimpleCNN3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 5, 3), nn.ReLU(),
            nn.Conv2d(5, 10, 3), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(10),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(160, 10), nn.ReLU(),
            nn.Linear(10, num_classes), nn.Sigmoid(), nn.Flatten(start_dim=0)
        )
    def forward(self,x):
        x = self.layers.forward(x)
        return x
    
class AudioPipeline(nn.Module):
    def __init__(self, transform_list:list, device:torch._C.device):
        super().__init__()
        self.pipline = nn.ModuleList()
        self.device = device
        for module in transform_list:
            module:nn.Module
            self.pipline.append(module.to(device=device))
            
    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.to(device=self.device)
        for module in self.pipline:
            x = module(x)
        return x