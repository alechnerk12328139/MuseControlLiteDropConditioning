import os
import re
import torchaudio
import numpy as np
from scipy.signal import savgol_filter
import librosa
import torch
import torchaudio
import scipy.signal as signal
from torchaudio import transforms as T
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import json

import utils.helpers as hlp
from utils.drop_detection_network import SimpleCNN3, AudioPipeline



def compute_melody_v2(stereo_audio:str) -> np.ndarray:
    """
    Args:
            stereo_audio (str): Path to a stereo audio file (e.g., WAV, MP3).
    Returns:
        c: np.ndarray of shape (8, T_frames)，
           每一列代表： [L1, R1, L2, R2, L3, R3, L4, R4]（按 frame 交錯），
           且每個值都 ∈ {1, 2, …, 128}，對應 CQT 的頻率 bin。
    """
    audio, sr = torchaudio.load(stereo_audio)
    # 1. 先針對左、右聲道分別計算 CQT (128 bins)，回傳 cqt_db 形狀都是 (128, T_frames)
    cqt_left  = compute_music_represent(audio[0], sr)  # shape: (128, T_frames)
    cqt_right = compute_music_represent(audio[1], sr)  # shape: (128, T_frames)

    # 2. 取得時框 (frame) 數量
    #    注意：librosa.cqt 的輸出 cqt_db 對應的「時框數」就是第二維度
    T_frames = cqt_left.shape[1]

    # 3. 預先配置輸出矩陣 c，dtype 用 int，shape = (8, T_frames)
    c = np.zeros((8, T_frames), dtype=np.int32)

    # 4. 逐一 frame 處理：對每個 frame 的 128 維度做 top-4
    for j in range(T_frames):
        # 4.1 取出當前時框的左、右聲道 CQT 能量（分貝值）
        col_L = cqt_left[:, j]   # shape: (128,)
        col_R = cqt_right[:, j]  # shape: (128,)

        # 4.2 用 numpy.argsort 找到「前 4 大」的索引
        #     np.argsort 預設是從小到大排序，所以取最後 4 個，再反轉取大到小
        idx4_L = np.argsort(col_L)[-4:][::-1]  # 0-based, 長度=4
        idx4_R = np.argsort(col_R)[-4:][::-1]  # 0-based, 長度=4

        # 4.3 轉成 1-based（因為題意寫 pixel ∈ {1,2,…,128}）
        idx4_L = idx4_L + 1  # 現在範圍是 1..128
        idx4_R = idx4_R + 1

        # 4.4 交錯填入 c 的第 j 欄
        #     我們希望 c[:, j] = [L1, R1, L2, R2, L3, R3, L4, R4]
        for k in range(4):
            c[2 * k    , j] = idx4_L[k]
            c[2 * k + 1, j] = idx4_R[k]

    return c[:,:4097]


def compute_music_represent(audio, sr):
    filter_y = torchaudio.functional.highpass_biquad(audio, sr, 261.6)
    fmin = librosa.midi_to_hz(0)
    cqt_spec = librosa.cqt(y=filter_y.numpy(), fmin=fmin, sr=sr, n_bins=128, bins_per_octave=12, hop_length=512)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    return cqt_db

def keep_top4_pitches_per_channel(cqt_db):
    """
    cqt_db is assumed to have shape: (2, 128, time_frames).
    We return a combined 2D array of shape (128, time_frames)
    where only the top 4 pitch bins in each channel are kept
    (for a total of up to 8 bins per time frame).
    """
    # Parse shapes
    num_channels, num_bins, num_frames = cqt_db.shape
    
    # Initialize an output array that combines both channels
    # and has zeros everywhere initially
    combined = np.zeros((num_bins, num_frames), dtype=cqt_db.dtype)
    
    for ch in range(num_channels):
        for t in range(num_frames):
            # Find the top 4 pitch bins for this channel at frame t
            # argsort sorts ascending; we take the last 4 indices for top 4
            top4_indices = np.argsort(cqt_db[ch, :, t])[-4:]
            
            # Copy their values into the combined array
            # We add to it in case there's overlap between channels
            combined[top4_indices, t] = 1
    return combined
def compute_melody(input_audio:str) -> np.ndarray:
    # Initialize parameters
    sample_rate = 44100

    # Load audio file
    wav, sr = torchaudio.load(input_audio)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = resample(wav)
    # Truncate or pad the audio to 2097152 samples
    target_length = 2097152
    if wav.size(1) > target_length:
        # Truncate the audio if it is longer than the target length
        wav = wav[:, :target_length]
    elif wav.size(1) < target_length:
        # Pad the audio with zeros if it is shorter than the target length
        padding = target_length - wav.size(1)
        wav = torch.cat([wav, torch.zeros(wav.size(0), padding)], dim=1)
    melody = compute_music_represent(wav, 44100)
    melody = keep_top4_pitches_per_channel(melody)    
    return melody

def compute_drops(audio_file:str, target_file:str, n_fft:int=1024, hop_length:int=160, target_sample_rate:int=44100, cut:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the drop curve for a given audio file

    Args:
        audio_file (str): Path to the audio file.
        target_file (str): Path to the target file.
        nfft (int, optional): Length of the FFT. Defaults to 1024.
        hop_length (int, optional): Length of the hop between STFT frames. Defaults to 160.
        target_sample_rate (int, optional): Target sample rate for the audio. Defaults to 44100.
        cut (bool, optional): Whether to cut the audio to a fixed length. Defaults to True.
        
    Returns:
        drop_curve (numpy.ndarray): The computed drop curve.
        target_times (numpy.ndarray): The target drop timestamps in seconds.
    """
    # Load audio file
    waveform, original_sample_rate = torchaudio.load(audio_file)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if cut:
        waveform = waveform[:, :2097152]
        
    # Calculate length of feature curve
    feature_length = waveform.size(1) // hop_length
    
    target_df = pd.read_csv(target_file)
    audio_chunk_name = os.path.basename(audio_file)
    
    regex = re.search(r"(\d+)_chunk(\d+)\.wav", audio_chunk_name) #Chunk files are .wav
    assert regex is not None, f"Filename {audio_chunk_name} does not match expected pattern."
    chunk_nr = regex.group(2)
    start_index = int(chunk_nr) * 2097152
    filename = regex.group(1) + ".mp3" # Original files are .mp3
    
    df_index = target_df['file'].str.contains(filename).idxmax()
    target_seconds = np.trim_zeros(target_df.iloc[df_index][1:].to_numpy(dtype=np.float32).reshape(-1))
    target_indices = (target_seconds * target_sample_rate).astype(int) - start_index
    #remvoe indices smaller than 0 or larger than 2097152
    target_indices = target_indices[(target_indices >= 0) & (target_indices < 2097152)]
    target_indices = target_indices // hop_length
    drop_curve = np.zeros(feature_length)
    drop_curve[target_indices] = 1
    return drop_curve, target_indices/target_sample_rate*hop_length
    
def drop_detection(audio_file:str, device, window_size:int = 600, batch_size:int=1000, THRESHOLD:float=0.5, THRESHOLD_HIGH:float=0.7, output_shape:tuple=(64,200)):
    """
    Runs the drop detection network over the given audio file

    Args:
        audio_file (str): path to the audio file
        device (torch.Device): the device to run the model on (e.g., "cuda" or "cpu")
        window_size (int, optional): Window size for the sliding window. Defaults to 600.
        batch_size (int, optional): Batch size for window generating. Defaults to 1000.
        THRESHOLD (float, optional): Threshold for drop splitting. If the probability drops below this value, it is considered a differnt drop from the maximum before. Defaults to 0.5.
        THRESHOLD_HIGH (float, optional): Threshold for the local maxima to be considered as drops. Defaults to 0.7.
        output_shape (tuple, optional): Output shape for the adaptive average pooling before input into the network. Defaults to (64,200).

    Returns:
        Y_hat_seconds (numpy.ndarray): Detected drop timestamps in seconds.
        prob_mapping (numpy.ndarray): A 2D array where the first row contains time in seconds and the second row contains the corresponding drop probabilities.
    """
    net = SimpleCNN3(1, 1)
    net.load_state_dict(torch.load("./utils/current_model.pth", map_location=device, weights_only=True))
    net = net.to(device)
    net.eval()
    transform = [T.MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512), hlp.ToDB()]
    pipeline = AudioPipeline(transform, device)
    
    y, sr = librosa.load(audio_file, sr=22050)
    S_dB = pipeline(y)
    num_win = S_dB.shape[1]-window_size+1
    assert num_win > 0, f"Window size {window_size} is too large for the number of frames {S_dB.shape[1]}."
    X = hlp.mel_window_batch_generator(S_dB.reshape(1, 1, S_dB.shape[0], S_dB.shape[1]), window_size, batch_size, num_win)
    logits_list = []
    for i, x in enumerate(X):
        x = x.view(-1, 1, 128, window_size)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_shape)
        x = x.to(device)
        logits:torch.Tensor = net(x)
        logits_list.append(logits.detach())
        
            
    logits = torch.cat(logits_list).reshape(1, 1, -1)
    prob = torch.conv1d(logits, torch.ones((1, 1, window_size), device=device), padding=window_size-1).squeeze().cpu()
    prob = prob/window_size
    indices = torch.arange(prob.shape[0])
    Y_hat = []
    offset=0
    while 1:
        try:
            start = prob[indices].where(prob[indices]>THRESHOLD, 0).nonzero().min().item() + offset
            indices = indices[indices>start]
            stop = prob[indices].where(prob[indices]<THRESHOLD, 0).nonzero().min().item() + start
            indices = indices[indices>stop]
            if prob[start:stop].max() > THRESHOLD_HIGH:
                Y_hat.append(prob[start:stop].argmax()+start)
            offset = stop
        except:
            break
        
    # convert to seconds:
    indices_to_seconds = 512 / 22050
    Y_hat_seconds = np.array(Y_hat)*indices_to_seconds
    
    prob_indices = np.arange(prob.shape[0])
    prob_seconds = prob_indices*indices_to_seconds
    prob_mapping = np.vstack((prob_seconds, prob.cpu().numpy()))
    return Y_hat_seconds, prob_mapping

def compute_segments(audio_file:str, label_file:str, n_fft:int=1024, hop_length:int=160, target_sample_rate:int=44100, cut:bool=True, 
                     target_encoding:dict = {"buildup": 1, "drop": 2, "breakdown": 3, "cooldown": 4, "bridge": 5, "outro": 6, "end": 7}):
    # Load audio file
    waveform, original_sample_rate = torchaudio.load(audio_file)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if cut:
        waveform = waveform[:, :2097152]
        
    # Calculate length of feature curve
    feature_length = waveform.size(1) // hop_length
    
    with open(label_file, mode="r+") as f:
        struct_dict_list = json.load(f)
    audio_chunk_name = os.path.basename(audio_file)
    
    regex = re.search(r"(\d+)_chunk(\d+)\.wav", audio_chunk_name) #Chunk files are .wav
    assert regex is not None, f"Filename {audio_chunk_name} does not match expected pattern."
    chunk_nr = regex.group(2)
    start_index = int(chunk_nr) * 2097152 // hop_length
    filename = regex.group(1) + ".mp3" # Original files are .mp3
    
    struct_dict = next(struct_dict for struct_dict in struct_dict_list if filename in struct_dict["file"])
    if (len(struct_dict['sections']) < 2):
        raise RuntimeError(f"Number of segments too low: {struct_dict}")
    target_label = []
    target_seconds = []
    for sections in struct_dict['sections']:
        target_seconds.append(sections['time'])
        target_label.append(target_encoding[sections['label']])
    target_seconds = np.array(target_seconds)
    #create condition for whole file, then slice chunk
    target_indices = (target_seconds * target_sample_rate / hop_length).astype(int)
    #get sizes for condition
    y, sr_ = torchaudio.load(struct_dict["file"])
    if sr_ != target_sample_rate:
        resampler = T.Resample(sr_, target_sample_rate)
        y = resampler(y)
    length = y.shape[1] //hop_length
    dims = len(target_encoding.keys())+1
    segment_curve = np.zeros((dims, length), dtype=np.float32)
    #Add intro
    segment_curve[0, 0:target_indices[0]] = 1.0
    for i in range(len(target_indices)-1):
        segment_curve[target_label[i], target_indices[i]:target_indices[i+1]] = 1.0
    #Add last segment
    segment_curve[target_label[-1], target_indices[-1]:] = 1.0
    segment_curve = segment_curve[:, start_index:start_index+(2097152//hop_length)]
    return segment_curve

def compute_dynamics(audio_file:str, hop_length:int=160, target_sample_rate:int=44100, cut:bool=True) -> np.ndarray:
    """
    Compute the dynamics curve for a given audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        hop_length (int, optional): Length of the hop between STFT frames. Defaults to 160.
        target_sample_rate (int, optional): Target sample rate for the audio. Defaults to 44100.
        cut (bool, optional): Whether to cut the audio to a fixed length. Defaults to True.

    Returns:
        dynamics_curve (numpy.ndarray): The computed dynamic values in dB.
    """
    # Load audio file
    waveform, original_sample_rate = torchaudio.load(audio_file)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if cut:
        waveform = waveform[:, :2097152]
    # Ensure waveform has a single channel (e.g., select the first channel if multi-channel)
    waveform = waveform.mean(dim=0, keepdim=True)  # Mix all channels into one
    waveform = waveform.clamp(-1, 1).numpy()
    
    S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=hop_length))
    mel_filter_bank = librosa.filters.mel(sr=target_sample_rate, n_fft=1024, n_mels=64, fmin=0, fmax=8000)
    S = np.dot(mel_filter_bank, S)
    energy = np.sum(S**2, axis=0)
    dynamics_db = np.clip(energy, 1e-6, None)
    dynamics_db = librosa.amplitude_to_db(energy, ref=np.max).squeeze(0)
    smoothed_dynamics:np.ndarray = savgol_filter(dynamics_db, window_length=279, polyorder=1)
    # print(smoothed_dynamics.shape)
    return smoothed_dynamics


def extract_melody_one_hot(audio_path:str,
                           sr:int=44100,
                           cutoff:float=261.2, 
                           win_length:int=2048,
                           hop_length:int=256) -> np.ndarray:
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma
def evaluate_f1_rhythm(input_timestamps:np.ndarray, generated_timestamps:np.ndarray, tolerance:float=0.07) -> tuple[float, float, float]:
    """
    Evaluates precision, recall, and F1-score for beat/downbeat timestamp alignment.
    
    Args:
        input_timestamps (ndarray): 2D array of shape [n, 2], where column 0 contains timestamps.
        generated_timestamps (ndarray): 2D array of shape [m, 2], where column 0 contains timestamps.
        tolerance (float): Alignment tolerance in seconds (default: 70ms).
    
    Returns:
        tuple: (precision, recall, f1)
    """
    # Extract and sort timestamps
    input_timestamps = np.asarray(input_timestamps)
    generated_timestamps = np.asarray(generated_timestamps)
    
    # If you only need the first column
    if input_timestamps.size > 0:  
        input_timestamps = input_timestamps[:, 0]
        input_timestamps.sort()
    else:
        input_timestamps = np.array([])
        
    if generated_timestamps.size > 0:
        generated_timestamps = generated_timestamps[:, 0]
        generated_timestamps.sort()
    else:
        generated_timestamps = np.array([])

    # Handle empty cases
    # Case 1: Both are empty
    if len(input_timestamps) == 0 and len(generated_timestamps) == 0:
        # You could argue everything is correct since there's nothing to detect,
        # but returning all zeros is a common convention.
        return 0.0, 0.0, 0.0

    # Case 2: No ground-truth timestamps, but predictions exist
    if len(input_timestamps) == 0 and len(generated_timestamps) > 0:
        # All predictions are false positives => tp=0, fp = len(generated_timestamps)
        # => precision=0, recall is undefined (tp+fn=0), typically we treat recall=0
        return 0.0, 0.0, 0.0

    # Case 3: Ground-truth timestamps exist, but no predictions
    if len(input_timestamps) > 0 and len(generated_timestamps) == 0:
        # Everything in input_timestamps is a false negative => tp=0, fn = len(input_timestamps)
        # => recall=0, precision is undefined (tp+fp=0), typically we treat precision=0
        return 0.0, 0.0, 0.0

    # If we get here, both arrays are non-empty
    tp = 0
    fp = 0
    
    # Track matched ground-truth timestamps
    matched_inputs = np.zeros(len(input_timestamps), dtype=bool)
    
    for gen_ts in generated_timestamps:
        # Calculate absolute differences to each reference timestamp
        diffs = np.abs(input_timestamps - gen_ts)
        # Find index of the closest input timestamp
        min_diff_idx = np.argmin(diffs)
        
        # Check if that difference is within tolerance and unmatched
        if diffs[min_diff_idx] < tolerance and not matched_inputs[min_diff_idx]:
            tp += 1
            matched_inputs[min_diff_idx] = True
        else:
            fp += 1  # no suitable match found or closest was already matched
    
    # Remaining unmatched input timestamps are false negatives
    fn = np.sum(~matched_inputs)
    
    # Compute precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
