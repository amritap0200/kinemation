"""
MediaPipe BlazePose 33 Landmarks → COCO 17 Keypoints Adapter

IMPORTANT: VideoPose3D pretrained_h36m_detectron_coco.bin expects:
  - INPUT:  COCO 17-keypoint format (2D)
  - OUTPUT: H36M 17-joint format (3D)

MediaPipe BlazePose landmark indices:
    0  = nose
    1  = left_eye_inner
    2  = left_eye
    3  = left_eye_outer
    4  = right_eye_inner
    5  = right_eye
    6  = right_eye_outer
    7  = left_ear
    8  = right_ear
    9  = mouth_left
    10 = mouth_right
    11 = left_shoulder
    12 = right_shoulder
    13 = left_elbow
    14 = right_elbow
    15 = left_wrist
    16 = right_wrist
    17-22 = hands (not used)
    23 = left_hip
    24 = right_hip
    25 = left_knee
    26 = right_knee
    27 = left_ankle
    28 = right_ankle
    29-32 = feet (not used)

COCO 17 keypoint format (VideoPose3D INPUT):
    0  = nose
    1  = left_eye
    2  = right_eye
    3  = left_ear
    4  = right_ear
    5  = left_shoulder
    6  = right_shoulder
    7  = left_elbow
    8  = right_elbow
    9  = left_wrist
    10 = right_wrist
    11 = left_hip
    12 = right_hip
    13 = left_knee
    14 = right_knee
    15 = left_ankle
    16 = right_ankle

H36M 17 joint format (VideoPose3D OUTPUT):
    0  = Hip (pelvis center)
    1  = RHip
    2  = RKnee
    3  = RAnkle
    4  = LHip
    5  = LKnee
    6  = LAnkle
    7  = Spine (torso)
    8  = Neck
    9  = Nose/Head
    10 = Head top
    11 = LShoulder
    12 = LElbow
    13 = LWrist
    14 = RShoulder
    15 = RElbow
    16 = RWrist
"""

import numpy as np

# MediaPipe index for each COCO keypoint
# COCO format is what VideoPose3D expects as 2D INPUT
MEDIAPIPE_TO_COCO = [
    0,   # 0  nose           → MediaPipe 0
    2,   # 1  left_eye       → MediaPipe 2
    5,   # 2  right_eye      → MediaPipe 5
    7,   # 3  left_ear       → MediaPipe 7
    8,   # 4  right_ear      → MediaPipe 8
    11,  # 5  left_shoulder  → MediaPipe 11
    12,  # 6  right_shoulder → MediaPipe 12
    13,  # 7  left_elbow     → MediaPipe 13
    14,  # 8  right_elbow    → MediaPipe 14
    15,  # 9  left_wrist     → MediaPipe 15
    16,  # 10 right_wrist    → MediaPipe 16
    23,  # 11 left_hip       → MediaPipe 23
    24,  # 12 right_hip      → MediaPipe 24
    25,  # 13 left_knee      → MediaPipe 25
    26,  # 14 right_knee     → MediaPipe 26
    27,  # 15 left_ankle     → MediaPipe 27
    28,  # 16 right_ankle    → MediaPipe 28
]


def mediapipe_to_coco(mp_landmarks):
    """
    Convert MediaPipe 33 landmarks to COCO 17 keypoints for VideoPose3D input.
    
    Args:
        mp_landmarks: numpy array of shape (33, 2) - MediaPipe landmarks
        
    Returns:
        coco_kps: numpy array of shape (17, 2) - COCO format keypoints
    """
    mp_kps = np.array(mp_landmarks, dtype=np.float32)
    
    # Ensure we have at least 33 landmarks
    if mp_kps.shape[0] < 33:
        padded = np.zeros((33, 2), dtype=np.float32)
        padded[:mp_kps.shape[0]] = mp_kps
        mp_kps = padded
    
    coco = np.zeros((17, 2), dtype=np.float32)
    
    for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
        coco[coco_idx] = mp_kps[mp_idx]
    
    return coco


# Keep old function name for compatibility
def mediapipe_to_h36m(mp_landmarks):
    """
    Alias for mediapipe_to_coco - converts to VideoPose3D input format.
    Note: Despite the name, this outputs COCO format which is what 
    VideoPose3D's pretrained model expects as input.
    """
    return mediapipe_to_coco(mp_landmarks)


def convert_sequence(mp_sequence):
    """
    Convert a full video sequence of MediaPipe landmarks to COCO keypoints.
    
    Args:
        mp_sequence: numpy array of shape (N, 33, 2)
        
    Returns:
        coco_sequence: numpy array of shape (N, 17, 2)
    """
    N = mp_sequence.shape[0]
    coco_sequence = np.zeros((N, 17, 2), dtype=np.float32)
    
    for i in range(N):
        coco_sequence[i] = mediapipe_to_coco(mp_sequence[i])
    
    return coco_sequence


def convert_multiperson_sequence(mp_sequence):
    """
    Convert multi-person sequence of MediaPipe landmarks to COCO keypoints.
    
    Args:
        mp_sequence: numpy array of shape (N, P, 33, 2)
                     N = frames, P = max people per frame
        
    Returns:
        coco_sequence: numpy array of shape (N, P, 17, 2)
    """
    N, P = mp_sequence.shape[0], mp_sequence.shape[1]
    coco_sequence = np.zeros((N, P, 17, 2), dtype=np.float32)
    
    for i in range(N):
        for p in range(P):
            if mp_sequence[i, p].max() > 0:
                coco_sequence[i, p] = mediapipe_to_coco(mp_sequence[i, p])
    
    return coco_sequence


# COCO skeleton connections for 2D visualization (input to VideoPose3D)
COCO_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


# H36M skeleton connections for 3D visualization (output from VideoPose3D)
# Modified for traditional stick figure (hip directly to knee, skip intermediate hip joints)
H36M_CONNECTIONS = [
    # Right leg - traditional stick figure: hip → knee → ankle
    (0, 2), (2, 3),  # Skip joint 1 (right_hip socket) for cleaner look
    # Left leg - traditional stick figure: hip → knee → ankle
    (0, 5), (5, 6),  # Skip joint 4 (left_hip socket) for cleaner look
    # Spine
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left arm (from spine/thorax area)
    (8, 11), (11, 12), (12, 13),
    # Right arm
    (8, 14), (14, 15), (15, 16),
]

H36M_JOINT_NAMES = [
    'hip', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle', 'spine', 'neck',
    'nose', 'head_top', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]
