import numpy as np
import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VideoPose3D'))

import torch
from common.model import TemporalModelOptimized1f
from common.camera import normalize_screen_coordinates

keypoints_2d = np.load("h36m_keypoints.npy")
N = keypoints_2d.shape[0]
print(f"Loaded {N} frames of H36M keypoints")

cap = cv2.VideoCapture("test_vid.mp4")
video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video size: {video_width}x{video_height}")

keypoints_norm = normalize_screen_coordinates(
    keypoints_2d, w=video_width, h=video_height
)

print("Loading model...")
checkpoint = torch.load(
    'VideoPose3D/pretrained_h36m_detectron_coco.bin',
    map_location='cpu'
)

model = TemporalModelOptimized1f(
    num_joints_in=17,
    in_features=2,
    num_joints_out=17,
    filter_widths=[3, 3, 3, 3, 3],
    dropout=0.25,
    channels=1024
)

model.load_state_dict(checkpoint['model_pos'])
model.eval()
print("Model loaded!")

# The receptive field is 243 for filter_widths=[3,3,3,3,3]
# For TemporalModelOptimized1f, input must be exactly 243 frames
# and it outputs exactly 1 frame (the center frame)
receptive_field = 243
pad = receptive_field // 2  # 121 frames on each side
print(f"Receptive field: {receptive_field}, pad: {pad}")

# Pad sequence by repeating edge frames
left_pad  = np.tile(keypoints_norm[0:1],  (pad, 1, 1))
right_pad = np.tile(keypoints_norm[-1:],  (pad, 1, 1))
keypoints_padded = np.concatenate([left_pad, keypoints_norm, right_pad], axis=0)
print(f"Padded length: {keypoints_padded.shape[0]}")

# Slide a window of exactly 243 frames, one step at a time
# Each window produces exactly 1 output frame
print(f"Running sliding window inference over {N} frames...")
all_predictions = []

for i in range(N):
    # Extract window of 243 frames centered on frame i
    window = keypoints_padded[i : i + receptive_field]  # (243, 17, 2)

    # Shape must be (1, 243, 17, 2)
    input_tensor = torch.from_numpy(
        window.astype(np.float32)
    ).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_tensor)  # (1, 1, 17, 3)

    all_predictions.append(pred.squeeze().numpy())  # (17, 3)

    if (i + 1) % 30 == 0:
        print(f"  Processed {i+1}/{N} frames...")

predicted_3d = np.stack(all_predictions, axis=0)  # (N, 17, 3)
print(f"Final output shape: {predicted_3d.shape}")

np.save("smooth_3d_poses.npy", predicted_3d)
print("Saved smooth_3d_poses.npy — done!")