import numpy as np

# YOLO/COCO 17 keypoints → H36M 17 keypoints remapping
# COCO: 0=nose,1=l_eye,2=r_eye,3=l_ear,4=r_ear,
#       5=l_shoulder,6=r_shoulder,7=l_elbow,8=r_elbow,
#       9=l_wrist,10=r_wrist,11=l_hip,12=r_hip,
#       13=l_knee,14=r_knee,15=l_ankle,16=r_ankle
#
# H36M:  0=hip_center,1=r_hip,2=r_knee,3=r_ankle,
#        4=l_hip,5=l_knee,6=l_ankle,7=spine,
#        8=thorax,9=nose,10=head,
#        11=l_shoulder,12=l_elbow,13=l_wrist,
#        14=r_shoulder,15=r_elbow,16=r_wrist

COCO_TO_H36M = [
    None,   # 0  hip_center    → computed as avg of hips
    12,     # 1  r_hip         → COCO 12
    14,     # 2  r_knee        → COCO 14
    16,     # 3  r_ankle       → COCO 16
    11,     # 4  l_hip         → COCO 11
    13,     # 5  l_knee        → COCO 13
    15,     # 6  l_ankle       → COCO 15
    None,   # 7  spine         → computed
    None,   # 8  thorax        → computed as avg of shoulders
    0,      # 9  nose          → COCO 0
    0,      # 10 head          → COCO 0 (approximate)
    5,      # 11 l_shoulder    → COCO 5
    7,      # 12 l_elbow       → COCO 7
    9,      # 13 l_wrist       → COCO 9
    6,      # 14 r_shoulder    → COCO 6
    8,      # 15 r_elbow       → COCO 8
    10,     # 16 r_wrist       → COCO 10
]

def coco_to_h36m(coco_kps):
    """
    Convert one frame of 17 COCO keypoints to 17 H36M keypoints.
    Input:  coco_kps  — shape (17, 2)
    Output: h36m_kps  — shape (17, 2)
    """
    h36m = np.zeros((17, 2), dtype=np.float32)

    for h36m_idx, coco_idx in enumerate(COCO_TO_H36M):
        if coco_idx is not None:
            h36m[h36m_idx] = coco_kps[coco_idx]

    # Joint 0: hip center = average of left and right hip
    h36m[0] = (coco_kps[11] + coco_kps[12]) / 2.0

    # Joint 8: thorax = average of shoulders
    h36m[8] = (coco_kps[5] + coco_kps[6]) / 2.0

    # Joint 7: spine = midpoint between hip center and thorax
    h36m[7] = (h36m[0] + h36m[8]) / 2.0

    return h36m

def convert_sequence(coco_sequence):
    """
    Convert full video sequence.
    Input:  (N, 17, 2) COCO
    Output: (N, 17, 2) H36M
    """
    N = coco_sequence.shape[0]
    out = np.zeros((N, 17, 2), dtype=np.float32)
    for i in range(N):
        out[i] = coco_to_h36m(coco_sequence[i])
    return out