import cv2
import numpy as np

kps = np.load("yolo_keypoints.npy")
if kps.ndim == 3:
    kps = kps[:, np.newaxis, :, :]

N = kps.shape[0]
max_people = kps.shape[1]

# Print keypoint ranges so we know what coordinate space they're in
print(f"Keypoints shape: {kps.shape}")
print(f"X range: {kps[:,:,:,0][kps[:,:,:,0]>0].min():.1f} to {kps[:,:,:,0][kps[:,:,:,0]>0].max():.1f}")
print(f"Y range: {kps[:,:,:,1][kps[:,:,:,1]>0].min():.1f} to {kps[:,:,:,1][kps[:,:,:,1]>0].max():.1f}")

# Open video and check its resolution
cap = cv2.VideoCapture("test_vid.mp4")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Video resolution: {W}x{H}")
print(f"Video frame count: {total}")
print(f"Keypoints frame count: {N}")
print(f"Expected keypoint X max: 1440, Y max: 810")