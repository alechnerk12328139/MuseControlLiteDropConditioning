"""\
This file contains functions which might be useful in several scripts.

Usage: If script is on the same level as this helpers file: import helpers as hlp
       Else this script can be imported as follwed:\n
       import sys \n
       sys.path.append("./") # Working directory needs to be top-level of git repository \n
       import helpers as hlp
"""

import os
import torch
from torch import nn
import numpy as np
import librosa
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import matplotlib.figure
import matplotlib.axes
from matplotlib.ticker import FormatStrFormatter

class AudioPipeline(nn.Module):
    def __init__(self, transform_list:list, device:torch._C.device):
        super().__init__()
        self.pipline = nn.ModuleList()
        self.device = device
        for module in transform_list:
            module:nn.Module
            self.pipline.append(module.to(device=device))
            
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(device=self.device)
        for module in self.pipline:
            x = module(x)
        return x

def clamp(n, n_min, n_max): return max(n_min, min(n, n_max))

def normalize(X:torch.Tensor):
    X-=X.min()
    X /= X.max()
    return X

import math
def window_batch_generator(X:torch.Tensor, window_size:int, batch_size:int, num_windows:int|None = None, every_nth_window:int = 1):
        """Sliding window generator for BxCxFxN Tensor over Dimension N to create batch_sizexFxwindow_size Tensors

        Args:
            X (torch.Tensor): Input FxN tensor 
            window_size (int): Size of window
            batch_size (int): Batch size
            num_windows (int|None): Number of windows to yield. Default None expands to number of possible windows without padding
            every_nth_window (int): Step size for window selection. Default is 1 (every window).
        Yields:
            torch.Tensor: Window
        """
        if num_windows is None:
            num_windows = int((X.shape[-1]-window_size)/every_nth_window+1)

        net = nn.Unfold((X.shape[-2],window_size), 1, 0, every_nth_window)
        if batch_size == -1 or batch_size >= num_windows:
            yield net(X).transpose(1,2).reshape(-1, X.shape[-2], window_size)[:num_windows]
        else:
            for i in range(math.ceil(num_windows/batch_size)):
                x = X[:, :, :, i*batch_size*every_nth_window:(i+1)*batch_size*every_nth_window+window_size-1]
                x = net(x).transpose(1,2).reshape(-1, X.shape[-2], window_size)
                yield x

import torchaudio
from torchaudio import transforms as t
def plot_mel_spectrum(file:str, sr:int=22050, fig:matplotlib.figure.Figure|None = None, ax:matplotlib.axes.Axes|None = None):
    y, sr_ = torchaudio.load(file)
    y = t.Resample(sr_, sr)(y)
    y = MergeChannels()(y).numpy()

    ##Create MelSpectogram
    librosa.feature.melspectrogram(y=y, sr=sr)

    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr, hop_length=512)

    ##Plot Spectogram
    if fig is None or ax is None:
        #Create Default Plot
        fig, ax = plt.subplots(figsize=(15,5))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='s', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f'Mel-frequency spectrogram for {file}')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def compute_intensity(y:np.ndarray|torch.Tensor, sr:int = 22050, hop_length:int = 512, filter_context:float = 2.0) -> np.ndarray|torch.Tensor:
    if isinstance(y, torch.Tensor):
        Torch = True
        device = y.device
        y = y.cpu().numpy()
    else:
        Torch = False
    rms = librosa.feature.rms(y=y, hop_length=hop_length).reshape(-1)
    rms_dB = 20*np.log10(rms+10e-6)
    intensity = savgol_filter(rms_dB, int(filter_context*sr/hop_length), 3)
    if Torch:
        intensity = torch.from_numpy(intensity).to(device=device, dtype=torch.float32)
    return intensity

from scipy.spatial.distance import pdist,squareform
from scipy.signal.windows import gaussian

def autocorr(x):
    """From https://github.com/mixerzeyu/edm-segmentation"""
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def adaptive_mean(x, N):
    """From https://github.com/mixerzeyu/edm-segmentation"""
    return np.convolve(x, [1.0]*int(N), mode='same')/N

def compute_novelty(y:np.ndarray|torch.Tensor, sr:int, minBPM:float = 60.0, maxBPM:float = 180.0, stepBPM:float = 0.01) -> np.ndarray|torch.Tensor:
    """Compute novelty based on https://github.com/mixerzeyu/edm-segmentation

    Args:
        y (np.ndarray|torch.Tensor): Audio data
        sr (sample rate): sample rate of given audio data
        
    Returns:
        novelty (np.ndarray|torch.Tensor)
    """
    if isinstance(y, torch.Tensor):
        Torch = True
        device = y.device
        y = y.cpu().numpy()
    else:
        Torch = False
     # Get downbeats
    valid_bpms = np.arange(minBPM, maxBPM, stepBPM)
    hop_size = sr//10
    # Step 1: compute onset curve
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_size, fmax=400, n_mels=8)
    
    # Step 2: half-wave rectify the result
    novelty_mean = adaptive_mean(onset_strength, 16.0)
    novelty_hwr = (onset_strength - novelty_mean).clip(min=0)
    
    # Step 3: then calculate the autocorrelation of this signal
    novelty_autocorr = autocorr(novelty_hwr)  
    
    # Step 4: Sum over constant intervals to detect most likely BPM
    bpm_collection = []
    for bpm in valid_bpms:
        fpb = (60.0 * sr)/(hop_size * bpm)
        frames = (np.round(np.arange(0,np.size(novelty_autocorr), fpb)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
        bpm_collection.append(np.sum(novelty_autocorr[frames])/np.size(frames))
    
    bpm_max_index = np.argmax(bpm_collection)
    bpm = valid_bpms[bpm_max_index]
    bpm_energy = max(bpm_collection)
    
    # in uncertainty (more than one peak), take the bpm in 2/4 rhythm, not in 3
    bpm_collection[bpm_max_index] = 0
    bpm2 = valid_bpms[np.argmax(bpm_collection)]
    bpm_energy2 = max(bpm_collection)
    if bpm_energy2 > bpm_energy * 0.8:
        if abs(bpm * 3 / 4 - bpm2) < 1 or abs(bpm * 3 / 2 - bpm2) < 1:
            bpm = bpm2
            
    bpm_collection[bpm_max_index] = bpm_energy
    
    fpb = (60.0 * sr)/(hop_size * bpm)
    
    # Step 5: locate (the delay of) first beat
    delay = 0.0
    
    valid_delays = np.arange(0.0, 60.0/bpm, 0.001) # Valid delays in SECONDS
    delay_collection = []
    delay_count = []
    for p in valid_delays:
        # Convert delay from seconds to frames
        delay_frames = (p * sr) / hop_size
        frames = (np.round(np.arange(delay_frames,np.size(novelty_hwr), fpb)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)

        delay_collection.append(np.sum(novelty_hwr[frames])/np.size(frames))
        delay_count.append(np.count_nonzero(novelty_hwr[frames]))
    
    delay = valid_delays[np.argmax(delay_collection)]
    delay_energy = max(delay_collection)
    
    # in uncertainty (more than one prominent peak), take first peak
    delay2 = valid_delays[np.argmax(delay_collection[:len(delay_collection)//4])]
    delay_energy2 = max(delay_collection[:len(delay_collection)//4])
    if delay_energy2 > delay_energy * 0.8:
        delay = delay2

    spb = 60./bpm #seconds per beat
    beat_size = spb * sr
    attack = delay
    
    beat_times = (np.arange(attack, float(len(y))/sr, spb).astype('single'))[:-1]
    beat_count = len(beat_times)
    
    # align the audio to the attack of the first beat (retaining 20 ms prior to the attack)
    y_nov = y[int(max(0, attack - 0.02) * sr):]
    # retrieve the location of every beat
    beat_locs = np.round(np.arange(0, len(y_nov), beat_size)).astype('int')[:-1]
    beat_size_int = int(beat_size)
    
    # Calculate Novelty function
    window = 16 # novelty curve of 16 frames
    beat_in_window = range(window, beat_count - window)
    
    kernel = np.kron(np.eye(2), np.ones((window,window))) - np.kron([[0, 1], [1, 0]], np.ones((window,window)))
    g = gaussian(2*window, window)
    kernel = np.multiply(kernel, np.multiply.outer(g.T, g))
    
    mfcc = [np.transpose(librosa.feature.mfcc(y=y_nov[i:i+beat_size_int-1], hop_length=beat_size_int)[:,0]) for i in beat_locs]
    ssm_mfcc = squareform(1 - pdist(mfcc, 'cosine'))
    novelty_mfcc = [window*2 + sum(sum(ssm_mfcc[i-window:i+window, i-window:i+window] * kernel)) for i in beat_in_window]
    novelty = np.concatenate([np.full(window, novelty_mfcc[0]), novelty_mfcc, np.full(window, novelty_mfcc[-1])]) - window / 2
    if Torch:
        # Convert back to tensor
        novelty = torch.from_numpy(novelty).to(device=device, dtype=torch.float32)
     
    return novelty

def compute_novelty_ssm(S:np.ndarray, kernel:np.ndarray|None=None, L:int=10, var:float=0.5, exclude:bool=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

def compute_kernel_checkerboard_gaussian(L:int, var:float=1, normalize:bool=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def create_mask(size:int, start:int, stop:int) -> torch.Tensor:
    """Creates a mask tenso of 1s and 0s. 1s Start ad index start and stop at index stop

    Args:
        size (int): size of output
        start (int): start index
        stop (int): sotp index
        
    Returns:
        torch.Tensor: Mask
    """
    mask = torch.sum(nn.functional.one_hot(torch.arange(start=start, end=stop), num_classes=size), dim=0)
    return mask

class ToDB(nn.Module):
    def __init__(self):
        """
        Converts amplitude into DB scale to use as torchaudio.transforms
        """
        super().__init__()
        
    def forward(self, x):
        return torch.log1p(100*x)
    
class MergeChannels(nn.Module):
    """
    Merges Channels by calculating the mean
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x:torch.Tensor):
        if x.dim() > 1:
            x = torch.mean(x, dim=0)
        return x

def get_files(dir:str, ftype:str=".mp3"):
    """Searches for files with 'ftype' ending in directory and all subdirectories and returns list

    Args:
        dir (str): Path of the root directory
        ftype (str): Ending of file

    Returns:
        List: List of all ftype files
    """
    files = []
    # Create list of all files
    for root, dirs, filenames in os.walk(dir):
        for i, name in enumerate(filenames):
            if name.endswith(ftype):
                        files.append(os.path.join(root, name))
    return files

def create_directory(path:str, debug:bool = False):
    """
    Checks if directory exists and creates the whole chain if it doesnt.
    
    Parameters:
    -----------
    path:str
        Relative path, including directory name
    """
    if not os.path.exists(path):
        head = path
        tail = []
        while(1):
            head, tail_it = os.path.split(head)
            tail.append(tail_it)
            if not os.path.exists(get_absolute_path(head)):
                continue
            else:
                head_it = head
                for name in reversed(tail):
                    head_it = os.path.join(head_it, name)
                    os.mkdir(head_it)
                if debug:
                    print(f"Creating direcory {get_absolute_path(path)}")
                break

def get_absolute_path(path:str) -> str:
    """
    Given an relative path from current working directory this function returns the absolut path.

    Parameters:
    -----------
    path:str
        Relative path

    Returns:
    --------
    out:str
        Absolute path
    """
    return os.path.join(os.getcwd(), path)

import re
NUMBERS = re.compile(r'(\d+)')
def numericalSort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


import numpy as np
import torch
from PIL import Image

def img_to_greyscale(path:str, normalize:bool = False)-> torch.Tensor:
    """
    Transforms .png file to greyscale Pytorch Tensor.

    Parameters:
    -----------
    path:str
        Relative path, including directory name
    normalize:bool
        Normalizes pixel values from [0,255] to [0,1]

    Returns:
    ----------
    out:torch Tensor
        converted image to torch tensor
    """
    img = Image.open(path)
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    if normalize:
        img_tensor = img_tensor/256
    return img_tensor

def get_source_path(path:str, use_mock:bool = False) -> str:
    """
    Prepends data_mock to given path if use_mock is true and retruns path
    """
    if (use_mock):
        path = os.path.join('data_mock', path)
    return path