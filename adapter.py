import numpy as np

COCO_TO_H36M = [
    None, 12, 14, 16, 11, 13, 15, None, None, 0, 0, 5, 7, 9, 6, 8, 10
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


def coco_to_h36m(coco_kps):
    """Convert COCO 17 keypoints to H36M 17 keypoints."""
    h36m = np.zeros((17, 2), dtype=np.float32)
    
    for h36m_idx, coco_idx in enumerate(COCO_TO_H36M):
        if coco_idx is not None:
            h36m[h36m_idx] = coco_kps[coco_idx]
    
    l_hip = coco_kps[11]
    r_hip = coco_kps[12]
    l_shoulder = coco_kps[5]
    r_shoulder = coco_kps[6]
    
    if l_hip.max() > 0 and r_hip.max() > 0:
        h36m[0] = (l_hip + r_hip) / 2.0
    elif l_hip.max() > 0:
        h36m[0] = l_hip
    elif r_hip.max() > 0:
        h36m[0] = r_hip
    
    if l_shoulder.max() > 0 and r_shoulder.max() > 0:
        h36m[8] = (l_shoulder + r_shoulder) / 2.0
    elif l_shoulder.max() > 0:
        h36m[8] = l_shoulder
    elif r_shoulder.max() > 0:
        h36m[8] = r_shoulder
    
    if h36m[0].max() > 0 and h36m[8].max() > 0:
        h36m[7] = (h36m[0] + h36m[8]) / 2.0
    
    return h36m


def convert_sequence(coco_sequence):
    """Convert full video sequence from COCO to H36M."""
    N = coco_sequence.shape[0]
    out = np.zeros((N, 17, 2), dtype=np.float32)
    for i in range(N):
        out[i] = coco_to_h36m(coco_sequence[i])
    return out


def convert_multiperson_sequence(coco_sequence):
    """Convert multi-person sequence from COCO to H36M."""
    N, P = coco_sequence.shape[0], coco_sequence.shape[1]
    out = np.zeros((N, P, 17, 2), dtype=np.float32)
    for i in range(N):
        for p in range(P):
            if coco_sequence[i, p].max() > 0:
                out[i, p] = coco_to_h36m(coco_sequence[i, p])
    return out