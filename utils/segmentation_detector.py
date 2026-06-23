from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as t
import numpy as np
import bisect

import sys
sys.path.append("./.")
import utils.helpers as hlp

class SegmentaionDetector(nn.Module):
    def __init__(self, model:nn.Module, window_size:int, output_shape:tuple[int,int], sr:int=22050, hop_length:int=512, threshold_high = 0.7, threshold = 0.5, 
                 intensity_filter_length:float = 2.0, batch_size:int = 1000, every_nth_window:int = 1, num_classes:int = 8,
                 device:torch.device|None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.window_size = window_size
        self.sr = sr
        self.hop_length = hop_length
        self.intensity_filter_length = intensity_filter_length
        self.batch_size = batch_size
        self.every_nth_window = every_nth_window
        self.num_classes = num_classes
        self.threshold = threshold
        self.threshold_high = threshold_high
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.merge = hlp.MergeChannels()
        self.mel_pipeline = hlp.AudioPipeline([t.MelSpectrogram(self.sr, n_fft=2048, hop_length=self.hop_length), hlp.ToDB()], device=self.device)
        self.pool = nn.AdaptiveAvgPool2d(output_shape)
        
    def forward(self, file:str) -> tuple[list[dict], torch.Tensor]:
        self.model.eval()
        # Get Data from file
        data_gen = self.data_generator(file)
        # Process model
        logits_list = []
        for x in data_gen:
            x_device = []
            for t in x:
                x_device.append(t.to(self.device))
            x = tuple(x_device)
            logits = self.model(x)
            # Softmax
            logits = F.softmax(logits, dim=1)
            logits_list.append(logits.detach())
        # combine all batches
        logits = torch.cat(logits_list).reshape(1, 1, -1, self.num_classes)

        # Get prob function for every output
        prob = torch.conv2d(logits.transpose(2, 3), torch.ones((1, 1, 1, self.window_size//self.every_nth_window), device=self.device), padding=(0,self.window_size//self.every_nth_window-1)).squeeze().cpu()
        prob = prob/self.window_size*self.every_nth_window
        Y_hat = []
        for i, label_prob in enumerate(prob[1:]): # Index 0 is no segment
            indices = torch.arange(label_prob.shape[0])
            offset=0
            # Get segmentations for each prob function
            while 1:
                try:
                    start = label_prob[indices].where(label_prob[indices]>self.threshold, 0).nonzero().min().item() + offset
                    indices = indices[indices>start]
                    stop = label_prob[indices].where(label_prob[indices]<self.threshold, 0).nonzero().min().item() + start
                    indices = indices[indices>stop]
                    if label_prob[start:stop].max() > self.threshold_high:
                        # Order segmentations in time
                        bisect.insort(Y_hat, {"time": label_prob[start:stop].argmax()+start, "label": i+1}, key=lambda x: x["time"])
                    offset = stop
                except:
                    break
        
        # Retrun tuple (segmentation_label, start_in_s)
        return Y_hat, prob
        
    
    def data_generator(self, file:str):
        # Read file
        y, sr_ = torchaudio.load(file)
        # merge channels and resample
        y = t.Resample(sr_, self.sr)(y)
        y:torch.Tensor = self.merge(y)
        # Compute mel spectrum
        mel = self.mel_pipeline(y)
        # Compute intensity
        intenstiy:torch.Tensor = hlp.compute_intensity(y, self.sr, self.hop_length, self.intensity_filter_length)
        intenstiy = F.interpolate(intenstiy.unsqueeze(0).unsqueeze(0), size=1000, mode="linear", align_corners=False)
        # Compute novelty
        novelty:torch.Tensor = hlp.compute_novelty(y, self.sr)
        novelty = F.interpolate(novelty.unsqueeze(0).unsqueeze(0), mel.shape[-1], mode="linear", align_corners=False)
        # yield windows
        mel_gen = hlp.window_batch_generator(mel.reshape(1, 1, 128, -1), self.window_size, batch_size=self.batch_size, every_nth_window=self.every_nth_window)
        nov_gen = hlp.window_batch_generator(novelty.reshape(1, 1, 1, -1), self.window_size, batch_size=self.batch_size, every_nth_window=self.every_nth_window)
        num_windows = num_windows = int((mel.shape[-1]-self.window_size)/self.every_nth_window+1)
        for i, (x_mel, x_nov) in enumerate(zip(mel_gen, nov_gen)):
            ti = torch.arange(i*self.batch_size, hlp.clamp((i+1)*self.batch_size, n_min= 0, n_max=i*self.batch_size+x_mel.shape[0]))/num_windows
            yield self.pool(x_mel.unsqueeze(1)), x_nov, intenstiy.expand(x_mel.shape[0], -1, -1), ti

if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    from cnn import SegmentationNetwork, MelExtractor, IntensityExtractor, NoveltyExtractor, FeedForwadNeuralNetwork
    file = "D:/Datasets/raveform/structures/0002.kfJQCu-Jbec.m4a"
    hidden_size = 128
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SegmentationNetwork(MelExtractor(hidden_size), NoveltyExtractor(hidden_size), IntensityExtractor(hidden_size), FeedForwadNeuralNetwork(3*hidden_size, 8), device)
    model.load_state_dict(torch.load('./network_segmentation/backup/segmentation_mel_nov_int/best_eval.pth', weights_only=True, map_location=device))
    detector = SegmentaionDetector(model=model, window_size=400, output_shape=(64, 200), every_nth_window=4, device=device)
    y_hat, prob = detector(file)
    print(y_hat)
    