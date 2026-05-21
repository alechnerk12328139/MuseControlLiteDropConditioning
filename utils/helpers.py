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

def pad_or_truncate(input_list: list, target_len: int) -> list:
    """Fixes length of input list to fixed target length by either adding 0 values or truncating

    Args:
        input_list (list): input list to be fixed
        target_len (int): desired length of output list

    Returns:
        list: output list with fixed length
    """
    return input_list[:target_len] + [0]*(target_len - len(input_list))

def compute_intensity(S:np.ndarray, sr:int = 22050, hop_length:int = 512, filter_context:float = 2.0):
    rms = librosa.feature.rms(S=S, hop_length=hop_length).reshape(-1)
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    rms_dB = 20*np.log10(rms)
    intensity = savgol_filter(rms_dB, int(filter_context*sr/hop_length), 3)
    return intensity

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

import math
def mel_window_batch_generator(X:torch.Tensor, window_size:int, batch_size:int, num_windows:int|None = None):
    """Sliding window generator for BxCxFxN Tensor over Dimension N to create batch_sizexFxwindow_size Tensors

    Args:
        X (torch.Tensor): Input FxN tensor 
        window_size (int): Size of window
        batch_size (int): Batch size
        num_windows (int|None): Number of windows to yield. Default None expands to number of possible windows without padding

    Yields:
        torch.Tensor: Window
    """
    if num_windows is None:
        num_windows = X.shape[1]-window_size+1
   
    net = nn.Unfold((128,window_size), 1, 0, 1)
    if batch_size == -1 or batch_size >= num_windows:
        yield net(X).transpose(1,2).reshape(-1, 128, window_size)
    else:
        for i in range(math.ceil(num_windows/batch_size)):
            x = X[:, :, :, i*batch_size:(i+1)*batch_size+window_size-1]
            x = net(x).transpose(1,2).reshape(-1, 128, window_size)
            yield x

def mel_window_generator(X:torch.Tensor, window_size:int, num_windows:int|None = None):
    """Sliding window generator for FxN Tensor over Dimension N to create Fxwindow_size Tensors

    Args:
        X (torch.Tensor): Input FxN tensor 
        window_size (int): Size of window
        num_windows (int|None): Number of windows to yield. Default None expands to number of possible windows without padding

    Yields:
        torch.Tensor: Window
    """
    if num_windows is None:
        num_windows = X.shape[1]-window_size+1
    F = X.shape[0]
    for i in range(num_windows):
        yield X[:, i:i+window_size]

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
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x:torch.Tensor):
        if x.dim() > 1:
            x = torch.mean(x, dim=1)
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

from typing import List

def process_sample(sample:List[str], destination:str, gt_path:str):
    """
    Given a list of paths of one sample this function checks if all 13 files exist. If all of them exist, it moves the ground truth pictures
    to "./gt_path/<batch name>/gt/*" and the other files, including the parameter.txt file, to "./destination/<sample number>/*". If not all
    13 samples exist this function does nothing.

    Parameters:
    -----------
    smaple:List[str]
        List of relative paths/strings that correspond to the same sample.
    destination:str
        Relative path to where to move the parameters.txt and <batch>_<sample>_pose_<sampleNb>_thermal.png files.
    gt_path:str
        Relative path to where the ground truth pictures are moved.
    """
    if (len(sample) == 13):
        #good sample -> move
        for img in sample:
            name = os.path.basename(img)
            name_split = name.split('_')
            if (name_split[2]=="GT"):
                _, batch = os.path.split(destination)
                destination_it = os.path.join(gt_path, batch, 'gt') 
            else:
                destination_it = os.path.join(destination, name_split[1])

            create_directory(destination_it)
            os.rename(img, os.path.join(destination_it, name)) #improve by checking if file already exists at destination first

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