import numpy as np
import os

# --- YOU NEED TO SET THIS TO YOUR ACTUAL TRAINING DoLP DIRECTORY ---
dolp_train_dir = 'data/train/dolp_images/' 

if os.path.exists(dolp_train_dir):
    print(f"--- Checking DoLP .npy files in: {dolp_train_dir} ---")
    for filename in os.listdir(dolp_train_dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(dolp_train_dir, filename)
            try:
                dolp_array = np.load(file_path)
                print(f"\nFile: {filename}")
                print(f"  Shape: {dolp_array.shape}")
                print(f"  Min value: {dolp_array.min()}")
                print(f"  Max value: {dolp_array.max()}")
                print(f"  Mean value: {dolp_array.mean()}")
                unique_vals, counts = np.unique(dolp_array, return_counts=True)
                print(f"  Unique values & counts: {dict(zip(unique_vals, counts))}")
                if np.all(dolp_array == 0):
                    print("  INFO: This array is ALL ZEROS.")
                elif np.any(dolp_array > 0):
                    print("  INFO: This array contains non-zero values.")
            except Exception as e:
                print(f"Could not load or process {filename}: {e}")
else:
    print(f"ERROR: DoLP training directory not found at {dolp_train_dir}")