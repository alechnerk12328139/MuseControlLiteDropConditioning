import os
import json
import torchaudio
from utils.extract_conditions import compute_melody, compute_dynamics, compute_drops
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from madmom.features.downbeats import RNNDownBeatProcessor
import argparse
import time

# This file is for preparing the musical attribute condition
# Function to process each item
def process_item(index, audio_path, dynamics_dir, melody_dir, rhythm_dir, drop_dir, drop_targets_path):
    try:
        dynamics_path = os.path.join(dynamics_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        melody_path = os.path.join(melody_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        rhythm_path = os.path.join(rhythm_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        drop_path = os.path.join(drop_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        audio, sample_rate = torchaudio.load(audio_path)
        num_frames = audio.size(1)
        assert num_frames == 2097152, f"Expected 2097152 frames, but got {num_frames} frames in {audio_path}"
        assert sample_rate == 44100, f"Expected sample rate of 44100 Hz, but got {sample_rate} Hz in {audio_path}"
        if not os.path.exists(melody_path):
            melody_curve = compute_melody(audio_path)
            assert melody_curve.shape == (128, 4097), f"Expected melody curve shape of (128, 4097), but got {melody_curve.shape} in {audio_path}"
            np.save(melody_path, melody_curve)
        if not os.path.exists(rhythm_path):
            rnn_processor = RNNDownBeatProcessor()
            rhythm_curve = rnn_processor(audio_path)
            assert rhythm_curve.shape == (4756, 2), f"Expected rhythm curve shape of (4756, 2), but got {rhythm_curve.shape} in {audio_path}"
            np.save(rhythm_path, rhythm_curve)
        if not os.path.exists(dynamics_path):
            dynamics_curve = compute_dynamics(audio_path)
            assert dynamics_curve.shape == (13108,), f"Expected dynamics curve shape of (13108,), but got {dynamics_curve.shape} in {audio_path}"
            np.save(dynamics_path, dynamics_curve)
        if not os.path.exists(drop_path):
            drop_curve = compute_drops(audio_path, target_file=drop_targets_path)
            np.save(drop_path, drop_curve)
        return index, dynamics_path, melody_path, rhythm_path, drop_path
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return index, None, None, None, None

# Multi-processing
if __name__ == "__main__":
    # Paths
    parser = argparse.ArgumentParser(description="Stable-audio VAE encode")
    parser.add_argument("--meta_path", type=str, default="D:/Datasets/drops-47s/filtered_vocal_all_caption.json", help="A list with dictionaries save in a json file")
    parser.add_argument("--drop_targets_path", type=str, default="D:/Datasets/jamendo_labels.csv", help="Path to the drop targets file")
    parser.add_argument("--new_json", type=str, default="./test_condition.json", help="json file with conditions")
    args = parser.parse_args()  # Parse the arguments

    meta_path = args.meta_path # This is a json file that contains a list of dictionaries, there are two keys in the dictionary: "path" and "Qwen_caption"
    new_json = args.new_json # The json file same as meta_path, but added the condition paths
    drop_targets_path = args.drop_targets_path # Path to the drop targets file
    dynamics_dir = "./mtg_full_47s_conditions/dynamics_condition_dir"
    melody_dir = "./mtg_full_47s_conditions/melody_condition_dir"
    rhythm_dir = "./mtg_full_47s_conditions/rhythm_condition_dir"
    drop_dir = "./mtg_full_47s_conditions/drop_condition_dir"

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    invalid_audio = []
    os.makedirs(dynamics_dir, exist_ok=True)
    os.makedirs(melody_dir, exist_ok=True)
    os.makedirs(rhythm_dir, exist_ok=True)
    os.makedirs(drop_dir, exist_ok=True)
    num_processes = min(cpu_count(), 12)  # Use up to 12 processes or the number of available CPUs
    print(f"Using {num_processes} processes for parallel processing.")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_item, i, meta[i]['path'], dynamics_dir, melody_dir, rhythm_dir, drop_dir, drop_targets_path)
            for i in range(len(meta))
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            i, dynamics_path, melody_path, rhythm_path, drop_path = future.result()
            if dynamics_path:
                meta[i]['dynamics_path'] = dynamics_path
            if melody_path:
                meta[i]['melody_path'] = melody_path
            if rhythm_path:
                meta[i]['rhythm_path'] = rhythm_path
            if drop_path:
                meta[i]['drop_path'] = drop_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            # Optionally handle results or exceptions
            future.result() 
    # Update metadata
    # Save updated metadata
    with open(new_json, "w") as json_file:
        json.dump(meta, json_file, indent=4)
    print(f"Updated metadata saved to {new_json}")
    print("invalid_audio", invalid_audio)