# preprocess_flow.py
import os, glob, cv2, numpy as np
from tqdm import tqdm

root = "./stab_11/stab_10"
out_root = "./flow_11/stab_10"
os.makedirs(out_root, exist_ok=True)

opt = cv2.optflow.createOptFlow_DualTVL1()

for movie_dir in tqdm(sorted(os.listdir(root))):
    movie_path = os.path.join(root, movie_dir)
    if not os.path.isdir(movie_path):
        continue
    for clip_dir in sorted(os.listdir(movie_path)):
        clip_path = os.path.join(movie_path, clip_dir)
        if not os.path.isdir(clip_path):
            continue

        frames = sorted(glob.glob(os.path.join(clip_path, "image_*.png")))
        if len(frames) < 2:
            continue

        flows = []
        prev = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2GRAY)
        for fp in frames[1:]:
            curr = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY)
            flow = opt.calc(prev, curr, None)  # (H, W, 2), float32
            flows.append(flow)
            prev = curr

        flows = np.stack(flows, axis=0)  # (T-1, H, W, 2)
        out_dir = os.path.join(out_root, movie_dir)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{clip_dir}.npy"), flows)
