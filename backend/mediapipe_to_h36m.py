"""MediaPipe to COCO/H36M keypoint conversion"""

import numpy as np

MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# Convert MediaPipe 33 landmarks to COCO 17 keypoints
def mediapipe_to_coco(mp_landmarks):
    mp_kps = np.array(mp_landmarks, dtype=np.float32)
    if mp_kps.shape[0] < 33:
        padded = np.zeros((33, 2), dtype=np.float32)
        padded[:mp_kps.shape[0]] = mp_kps
        mp_kps = padded
    coco = np.zeros((17, 2), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
        coco[coco_idx] = mp_kps[mp_idx]
    return coco


# Alias for compatibility
def mediapipe_to_h36m(mp_landmarks):
    return mediapipe_to_coco(mp_landmarks)


# Convert full video sequence
def convert_sequence(mp_sequence):
    N = mp_sequence.shape[0]
    coco_sequence = np.zeros((N, 17, 2), dtype=np.float32)
    for i in range(N):
        coco_sequence[i] = mediapipe_to_coco(mp_sequence[i])
    return coco_sequence


# Convert multi-person sequence
def convert_multiperson_sequence(mp_sequence):
    N, P = mp_sequence.shape[0], mp_sequence.shape[1]
    coco_sequence = np.zeros((N, P, 17, 2), dtype=np.float32)
    for i in range(N):
        for p in range(P):
            if mp_sequence[i, p].max() > 0:
                coco_sequence[i, p] = mediapipe_to_coco(mp_sequence[i, p])
    return coco_sequence


COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

H36M_CONNECTIONS = [
    (0, 2), (2, 3),
    (0, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

H36M_JOINT_NAMES = [
    'hip', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle', 'spine', 'neck',
    'nose', 'head_top', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]
