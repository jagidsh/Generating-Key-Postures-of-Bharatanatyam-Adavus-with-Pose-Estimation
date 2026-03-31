
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import os
import numpy as np
import cv2
import mediapipe as mp
from torchvision.datasets import ImageFolder



kp_cache_dir = "./kp_cache"
data_path = "./keyposture_dataset"

dataset = ImageFolder(root=data_path)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(image, num_keypoints=33):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            return np.array(keypoints)
        else:
            return np.zeros((num_keypoints, 2))
    except:
        return np.zeros((num_keypoints, 2))


def get_or_compute_keypoints(img_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(img_path)
    kp_path = os.path.join(cache_dir, filename + ".npy")

    # ✅ If exists → load
    if os.path.exists(kp_path):
        return np.load(kp_path)

    # ❗ Else compute and save
    image = cv2.imread(img_path)
    keypoints = extract_keypoints(image)

    np.save(kp_path, keypoints)
    return keypoints
    
for i, (path, _) in enumerate(dataset.samples):
    print(f"[{i}/{len(dataset)}] Processing {path}")
    get_or_compute_keypoints(path, kp_cache_dir)

    
