from typing import Any
import torch
from torch import nn
import math
import numpy as np

class SqueezeAndExite(nn.Module):
    def __init__(self, C):
        super(SqueezeAndExite, self).__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(C, C),
                                    nn.ReLU(),
                                    nn.Linear(C, C),
                                    nn.Sigmoid())

    def forward(self, x):
        scale = self.layers.forward(x)
        return x * scale.view(-1, x.size(1), 1, 1)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=True, activation:nn.Module=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        if self.padding:
            padding = kernel_size // 2
        else:
            padding = 0

        if in_channels != out_channels:
            self.unequal_channels = True
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) # used implementation from https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L185
            mid_channels = abs(out_channels - in_channels) // 2
        else:
            self.unequal_channels = False
            mid_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride=stride, padding=padding),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        out = self.layers(x)
        
        if self.stride > 1 or self.padding == False:
            skip = nn.functional.adaptive_avg_pool2d(x, out.shape[-2:])
        else:
            skip = x
        
        if self.unequal_channels:
            skip = self.downsample(skip)
        return skip + out

class MelExtractor(nn.Module):
    def __init__(self, output_size:int, activation:torch.nn.Module = torch.nn.ReLU(), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3), activation,
            ResidualBlock(8, 16, 5, padding=False, activation=nn.ReLU()), activation,
            SqueezeAndExite(16),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            ResidualBlock(16, 32, 3, activation=nn.ReLU()), activation,
            nn.Conv2d(32, 32, 1), activation,
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32,64, 3, activation=nn.ReLU()), activation,
            SqueezeAndExite(64),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            ResidualBlock(64, output_size, 3, activation=nn.ReLU()), activation,
            nn.Conv2d(output_size, output_size, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

class NoveltyExtractor(nn.Module):
    def __init__(self, output_size:int, activation:torch.nn.Module = torch.nn.ReLU(), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 3), activation,
            nn.Conv1d(8, 16, 3), activation,
            nn.Conv1d(16, 32, 3), activation,
            nn.Conv1d(32, 64, 3), activation,
            nn.Conv1d(64, output_size, 3), activation,
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
        
class IntensityExtractor(nn.Module):
    def __init__(self, output_size:int, activation:torch.nn.Module = torch.nn.ReLU(), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, 3), activation,
            nn.Conv1d(8, 16, 3), activation,
            nn.MaxPool1d(4), nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 5, dilation=3), activation,
            nn.Conv1d(32, 64, 3), activation,
            nn.MaxPool1d(4), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3), activation,
            nn.AdaptiveAvgPool1d(50), nn.Flatten()
        )
        self.ffnn = nn.Sequential(
            nn.Linear(3201, 256), activation,
            nn.Linear(256, output_size), activation
        )
        
    def forward(self, x:torch.Tensor, ti:torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = torch.hstack((x, ti.unsqueeze(1))).to(torch.float32)
        return self.ffnn(x)
        
        
class FeedForwadNeuralNetwork(nn.Module):
    def __init__(self, input_size:int, output_size:int, activation:torch.nn.Module = torch.nn.ReLU(), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), activation,
            nn.Linear(512, 512), activation,
            nn.Linear(512, output_size)
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.net(x)
        

class SegmentationNetwork(nn.Module):
    def __init__(self, mel_extractor:MelExtractor, novelty_extractor:NoveltyExtractor, intensity_extractor:IntensityExtractor, ffnn:FeedForwadNeuralNetwork, device:torch.device|None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.mel_extractor = mel_extractor.to(device)
        self.novelty_extractor = novelty_extractor.to(device)
        self.intensity_extractor = intensity_extractor.to(device)
        self.ffnn = ffnn.to(device)
        
    def forward(self, x:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mel, novelty, intensity, ti = x
        x_mel = self.mel_extractor(mel)
        x_nov = self.novelty_extractor(novelty)
        x_intensity = self.intensity_extractor(intensity, ti)
        
        x = torch.cat([x_mel, x_nov, x_intensity], dim=1)
        return self.ffnn(x)
        