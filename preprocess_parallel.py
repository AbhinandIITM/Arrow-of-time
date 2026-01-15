# preprocess_flow_multiprocess.py
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import itertools # Used for flattening the task list if needed, but not strictly required here

# --- Configuration ---
root = "./stab_11/stab_10"
out_root = "./flow_11/stab_10"
# Get the number of CPU cores to use (e.g., all but one for safety)
NUM_CORES = max(1, multiprocessing.cpu_count() - 10) 
# --- Configuration ---


def process_single_clip(clip_task):
    """
    Calculates and saves the optical flow for a single video clip.
    This function is executed by one of the multiprocessing workers.
    
    Args:
        clip_task (tuple): (movie_dir, clip_dir, clip_path)
    """
    movie_dir, clip_dir, clip_path = clip_task
    
    # Re-initialize the OptFlow object inside the worker process
    # This prevents serialization issues and ensures each process has its own object
    opt = cv2.optflow.createOptFlow_DualTVL1() 
    
    # 1. Gather Frames
    frames = sorted(glob.glob(os.path.join(clip_path, "image_*.png")))
    if len(frames) < 2:
        # print(f"Skipping {movie_dir}/{clip_dir}: Less than 2 frames.")
        return 
        
    flows = []
    
    # 2. Process Frames Sequentially
    try:
        # Read and convert the first frame
        prev = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2GRAY)
        
        for fp in frames[1:]:
            curr = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY)
            
            # Calculate DualTVL1 Optical Flow
            flow = opt.calc(prev, curr, None)  # (H, W, 2), float32
            flows.append(flow)
            prev = curr
            
    except Exception as e:
        print(f"Error processing {movie_dir}/{clip_dir}: {e}")
        return # Exit this worker function on error

    # 3. Save Flow Data
    if flows:
        flows_array = np.stack(flows, axis=0)  # (T-1, H, W, 2)
        
        out_dir = os.path.join(out_root, movie_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        output_filepath = os.path.join(out_dir, f"{clip_dir}.npy")
        np.save(output_filepath, flows_array)
        # print(f"Saved flow for {movie_dir}/{clip_dir} to {output_filepath}")

    
if __name__ == '__main__':
    
    # Ensure the main output directory exists
    os.makedirs(out_root, exist_ok=True)
    
    print(f"--- Optical Flow Preprocessing ---")
    print(f"Input Root: {root}")
    print(f"Output Root: {out_root}")
    print(f"Using {NUM_CORES} CPU cores for parallel processing.\n")

    # 1. Gather all tasks (clip paths)
    all_clips_to_process = []
    
    # Traverse the directory structure to find all clips
    for movie_dir in sorted(os.listdir(root)):
        movie_path = os.path.join(root, movie_dir)
        if not os.path.isdir(movie_path):
            continue
            
        for clip_dir in sorted(os.listdir(movie_path)):
            clip_path = os.path.join(movie_path, clip_dir)
            if not os.path.isdir(clip_path):
                continue
            
            # The worker function needs all three pieces of information
            all_clips_to_process.append((movie_dir, clip_dir, clip_path))

    
    total_tasks = len(all_clips_to_process)
    if total_tasks == 0:
        print("No video clips found to process. Exiting.")
    else:
        print(f"Found {total_tasks} clips to process.")
        
        # 2. Execute tasks in parallel using a Pool
        try:
            with multiprocessing.Pool(processes=NUM_CORES) as pool:
                
                # Use imap_unordered to process results as they finish (better for tqdm)
                # list() consumes the generator and forces execution
                list(tqdm(pool.imap_unordered(process_single_clip, all_clips_to_process),
                          total=total_tasks,
                          desc="Processing Clips"))
            
            print("\n✅ All clips processed successfully.")
            
        except Exception as e:
            print(f"\nAn error occurred during parallel processing: {e}")