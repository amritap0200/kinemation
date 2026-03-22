import numpy as np
from adapter import convert_sequence

kps = np.load("yolo_keypoints.npy")
print(f"Loaded YOLO keypoints: {kps.shape}")  # (N, 17, 2)

h36m = convert_sequence(kps)
print(f"Converted to H36M:     {h36m.shape}")  # (N, 17, 2)

np.save("h36m_keypoints.npy", h36m)
print("Saved h36m_keypoints.npy")