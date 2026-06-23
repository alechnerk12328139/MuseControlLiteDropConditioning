import os
import json
import torchaudio
from utils.extract_conditions import compute_segments
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from madmom.features.downbeats import RNNDownBeatProcessor
import argparse
import time

# This file is for preparing the musical attribute condition
# Function to process each item
def process_item(index, audio_path, segments_dir, segments_labels_path):
    try:
        segments_path = os.path.join(segments_dir, os.path.basename(audio_path).replace('.wav', '.npy'))
        audio, sample_rate = torchaudio.load(audio_path)
        num_frames = audio.size(1)
        assert num_frames == 2097152, f"Expected 2097152 frames, but got {num_frames} frames in {audio_path}"
        assert sample_rate == 44100, f"Expected sample rate of 44100 Hz, but got {sample_rate} Hz in {audio_path}"
        if not os.path.exists(segments_path):
            segment_curve = compute_segments(audio_path, label_file=segments_labels_path)
            assert (segment_curve.shape == (8, 2097152//160)), f"segemnent curve has wrong shape: {segment_curve.shape}"
            np.save(segments_path, segment_curve)
        return index, segments_path
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return index, None

# Multi-processing
if __name__ == "__main__":
    # Paths
    parser = argparse.ArgumentParser(description="Stable-audio VAE encode")
    parser.add_argument("--meta_path", type=str, default="D:/Datasets/segments-47s/filtered_vocal_all_caption.json", help="A list with dictionaries save in a json file")
    parser.add_argument("--segments_targets_path", type=str, default="D:/Datasets/segments.json", help="Path to the segments targets file")
    parser.add_argument("--new_json", type=str, default="./segments_conditions.json", help="json file with conditions")
    args = parser.parse_args()  # Parse the arguments

    meta_path = args.meta_path # This is a json file that contains a list of dictionaries, there are two keys in the dictionary: "path" and "Qwen_caption"
    new_json = args.new_json # The json file same as meta_path, but added the condition paths
    segments_targets_path = args.segments_targets_path # Path to the segments targets file
    segments_dir = "./47s_conditions/segments_condition_dir"

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    invalid_audio = []
    os.makedirs(segments_dir, exist_ok=True)
    num_processes = min(cpu_count(), 8)  # Use up to 12 processes or the number of available CPUs
    print(f"Using {num_processes} processes for parallel processing.")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_item, i, meta[i]['path'], segments_dir, segments_targets_path)
            for i in range(len(meta))
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            i, segments_path = future.result()
            if segments_path:
                meta[i]['segments_path'] = segments_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            # Optionally handle results or exceptions
            future.result() 
    # Update metadata
    # Save updated metadata
    with open(new_json, "w") as json_file:
        json.dump(meta, json_file, indent=4)
    print(f"Updated metadata saved to {new_json}")
    print("invalid_audio", invalid_audio)