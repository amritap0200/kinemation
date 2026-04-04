"""3D Pose Estimation Pipeline"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

stderr_fd = sys.stderr.fileno()
devnull_fd = os.open(os.devnull, os.O_WRONLY)
saved_stderr = os.dup(stderr_fd)
os.dup2(devnull_fd, stderr_fd)

import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import numpy as np
import argparse
import mediapipe as mp
from ultralytics import YOLO

os.dup2(saved_stderr, stderr_fd)
os.close(devnull_fd)
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'VideoPose3D'))

import torch
from common.model import TemporalModelOptimized1f

from mediapipe_to_h36m import mediapipe_to_h36m, H36M_CONNECTIONS, COCO_CONNECTIONS


MAX_PEOPLE = 6  # Reduced for better tracking stability
RECEPTIVE_FIELD = 243  # VideoPose3D receptive field for filter_widths=[3,3,3,3,3]

PERSON_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue (BGR)
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Yellow
]


def apply_clahe(frame):
    """Apply CLAHE in LAB color space for better contrast."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame, max_dim=800, apply_blur=True, apply_clahe_enhancement=True):
    """Preprocess frame with optional CLAHE and Gaussian blur."""
    h, w = frame.shape[:2]
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if apply_clahe_enhancement:
        frame = apply_clahe(frame)
    
    if apply_blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame


def detect_persons(yolo_model, frame, confidence=0.5):
    """Detect persons in frame using YOLOv8."""
    results = yolo_model(frame, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
    return boxes


def estimate_pose_mediapipe(landmarker, frame, box):
    """Estimate 2D pose for a person using MediaPipe."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    
    pad_w = int((x2 - x1) * 0.15)
    pad_h = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
    
    stderr_fd = sys.stderr.fileno()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(stderr_fd)
    os.dup2(devnull_fd, stderr_fd)
    result = landmarker.detect(mp_image)
    os.dup2(saved, stderr_fd)
    os.close(devnull_fd)
    
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None
    
    crop_h, crop_w = crop.shape[:2]
    landmarks = np.zeros((33, 2), dtype=np.float32)
    
    for i, lm in enumerate(result.pose_landmarks[0]):
        landmarks[i, 0] = x1 + lm.x * crop_w
        landmarks[i, 1] = y1 + lm.y * crop_h
    
    return landmarks


def get_bbox_from_keypoints(keypoints):
    """Compute bounding box from keypoints."""
    valid = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return np.array([valid[:, 0].min(), valid[:, 1].min(),
                     valid[:, 0].max(), valid[:, 1].max()])


def compute_iou(bbox_a, bbox_b):
    """Compute IoU between two bounding boxes."""
    xi1 = max(bbox_a[0], bbox_b[0])
    yi1 = max(bbox_a[1], bbox_b[1])
    xi2 = min(bbox_a[2], bbox_b[2])
    yi2 = min(bbox_a[3], bbox_b[3])
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter)


def batch_track_people(raw_keypoints, iou_threshold=0.3):
    """
    Track people across all frames using IoU-based Hungarian algorithm.
    This is a BATCH process - processes entire video at once for consistent IDs.
    
    NEW: IoU threshold prevents bad assignments when people don't overlap.
    Assignments with IoU < threshold are rejected (detection becomes new track).
    
    Args:
        raw_keypoints: array of shape (N, max_people, n_joints, 2)
        iou_threshold: minimum IoU for valid assignment (default 0.3)
        
    Returns:
        tracked: array of same shape with consistent person IDs
    """
    N, max_people, n_joints, _ = raw_keypoints.shape
    tracked = np.zeros_like(raw_keypoints)
    
    tracked[0] = raw_keypoints[0]
    prev_boxes = [get_bbox_from_keypoints(raw_keypoints[0, p]) for p in range(max_people)]
    
    for frame_idx in range(1, N):
        frame_kps = raw_keypoints[frame_idx]
        
        det_boxes = []
        det_indices = []
        for d in range(max_people):
            bbox = get_bbox_from_keypoints(frame_kps[d])
            if bbox is not None:
                det_boxes.append(bbox)
                det_indices.append(d)
        
        if not det_boxes:
            continue
        
        cost = np.ones((max_people, len(det_boxes)))
        iou_matrix = np.zeros((max_people, len(det_boxes)))
        
        for t in range(max_people):
            if prev_boxes[t] is None:
                continue
            for di, db in enumerate(det_boxes):
                iou = compute_iou(prev_boxes[t], db)
                iou_matrix[t, di] = iou
                cost[t, di] = 1.0 - iou
        
        t_ids, d_ids = linear_sum_assignment(cost)
        assigned_t, assigned_d = set(), set()
        
        for t_id, d_id in zip(t_ids, d_ids):
            iou = iou_matrix[t_id, d_id]
            
            if iou < iou_threshold and prev_boxes[t_id] is not None:
                continue
            
            orig = det_indices[d_id]
            tracked[frame_idx, t_id] = frame_kps[orig]
            prev_boxes[t_id] = det_boxes[d_id]
            assigned_t.add(t_id)
            assigned_d.add(d_id)
        
        leftover = [det_indices[d] for d in range(len(det_boxes)) if d not in assigned_d]
        empty = [t for t in range(max_people) if t not in assigned_t]
        for slot, orig in zip(empty, leftover):
            tracked[frame_idx, slot] = frame_kps[orig]
            prev_boxes[slot] = get_bbox_from_keypoints(frame_kps[orig])
    
    return tracked


def filter_short_tracks(tracked, min_frames=10):
    """Remove short detection runs that are likely false positives."""
    N, max_people = tracked.shape[:2]
    filtered = tracked.copy()
    
    for p in range(max_people):
        has_det = np.array([tracked[i, p].max() > 0 for i in range(N)])
        
        in_run = False
        run_start = 0
        for i in range(N + 1):
            active = i < N and has_det[i]
            if active and not in_run:
                run_start = i
                in_run = True
            elif not active and in_run:
                run_len = i - run_start
                if run_len < min_frames:
                    filtered[run_start:i, p] = 0.0
                in_run = False
    
    return filtered


def smooth_track(kps_track, sigma=2):
    """Smooth a single person's keypoint trajectory."""
    N, n_joints, n_coords = kps_track.shape
    smoothed = kps_track.copy()
    
    has_detection = np.array([kps_track[i].max() > 0 for i in range(N)])
    
    if has_detection.sum() < 3:
        return smoothed
    
    runs = []
    in_run = False
    run_start = 0
    
    for i in range(N):
        if has_detection[i] and not in_run:
            run_start = i
            in_run = True
        elif not has_detection[i] and in_run:
            runs.append((run_start, i))
            in_run = False
    if in_run:
        runs.append((run_start, N))
    
    for start, end in runs:
        if end - start < 3:
            continue
        
        for joint_idx in range(n_joints):
            for coord in range(n_coords):
                segment = kps_track[start:end, joint_idx, coord].astype(np.float64)
                valid = segment > 0
                
                if valid.sum() < 2:
                    continue
                
                indices = np.arange(len(segment))
                filled = np.interp(indices, indices[valid], segment[valid])
                smooth = gaussian_filter1d(filled, sigma=sigma)
                
                result = smoothed[start:end, joint_idx, coord]
                result[valid] = smooth[valid]
                smoothed[start:end, joint_idx, coord] = result
    
    return smoothed.astype(np.float32)


def smooth_all_tracks(tracked_keypoints, sigma=2):
    """Smooth all person tracks in the sequence."""
    N, max_people = tracked_keypoints.shape[:2]
    smoothed = np.zeros_like(tracked_keypoints)
    
    for person_id in range(max_people):
        smoothed[:, person_id] = smooth_track(tracked_keypoints[:, person_id], sigma=sigma)
    
    return smoothed



JOINT_SIGMA_3D = {
    0: 2.2,   # hip (root) - anchor point, should be stable
    7: 2.2,   # spine - core stability
    8: 2.0,   # neck - upper body stability
    
    1: 1.9,   # r_hip - hip angle
    4: 1.9,   # l_hip - hip angle
    2: 1.7,   # r_knee - leg joint
    5: 1.7,   # l_knee - leg joint
    3: 1.5,   # r_ankle - foot
    6: 1.5,   # l_ankle - foot
    
    9: 1.7,   # nose - follows body
    10: 1.7,  # head_top - follows body
    
    11: 1.4,  # l_shoulder - arm root
    14: 1.4,  # r_shoulder - arm root
    12: 1.2,  # l_elbow - mid-arm
    15: 1.2,  # r_elbow - mid-arm
    13: 1.0,  # l_wrist - hand (most expressive)
    16: 1.0,  # r_wrist - hand (most expressive)
}

VELOCITY_LIMITS_3D = {
    0: 0.025, 7: 0.025, 8: 0.035,
    1: 0.050, 4: 0.050,
    2: 0.060, 5: 0.060, 3: 0.080, 6: 0.080,
    11: 0.080, 14: 0.080, 12: 0.120, 15: 0.120, 13: 0.150, 16: 0.150,
    9: 0.050, 10: 0.050,
}


def smooth_3d_trajectory(poses_3d, sigma=1.5, hip_sigma_multiplier=1.5, 
                         use_adaptive_sigma=True, apply_velocity_limit=True):
    """
    Enhanced 3D pose smoothing with joint-adaptive parameters (planv5).
    
    Two-pass smoothing algorithm:
    1. Joint-specific Gaussian smoothing (removes high-frequency noise)
    2. Velocity limiting (removes physically impossible movement spikes)
    
    The smoothing is adaptive based on joint role:
    - Core joints (spine, hip root): High sigma for stability
    - Locomotion joints (hip angles, knees): Medium sigma
    - Manipulation joints (arms, wrists): Low sigma to preserve expressiveness
    
    Args:
        poses_3d: (N, 17, 3) array - 3D poses for one person
        sigma: float - Base Gaussian kernel sigma (default 1.5, used if adaptive disabled)
        hip_sigma_multiplier: float - Legacy parameter (ignored if adaptive enabled)
        use_adaptive_sigma: bool - Use per-joint sigma values from JOINT_SIGMA_3D
        apply_velocity_limit: bool - Apply velocity limiting pass to remove spikes
    
    Returns:
        smoothed: (N, 17, 3) array - smoothed 3D poses
    
    H36M joint indices:
        0=hip, 1=r_hip, 2=r_knee, 3=r_ankle, 4=l_hip, 5=l_knee, 6=l_ankle,
        7=spine, 8=neck, 9=nose, 10=head_top, 11=l_shoulder, 12=l_elbow,
        13=l_wrist, 14=r_shoulder, 15=r_elbow, 16=r_wrist
    """
    N = poses_3d.shape[0]
    if N < 3:
        return poses_3d
    
    smoothed = poses_3d.copy()
    
    valid_mask = np.abs(poses_3d).max(axis=(1, 2)) > 0.001
    valid_frames = np.where(valid_mask)[0]
    
    if len(valid_frames) < 3:
        return poses_3d


    for joint in range(17):
        if use_adaptive_sigma:
            joint_sigma = JOINT_SIGMA_3D.get(joint, sigma)
        else:
            hip_joints = {0, 1, 4}
            joint_sigma = sigma * hip_sigma_multiplier if joint in hip_joints else sigma
        
        for coord in range(3):
            valid_values = poses_3d[valid_mask, joint, coord]
            if len(valid_values) >= 3:
                smoothed_values = gaussian_filter1d(valid_values, sigma=joint_sigma)
                smoothed[valid_mask, joint, coord] = smoothed_values


    if apply_velocity_limit and len(valid_frames) > 1:
        for i in range(1, len(valid_frames)):
            prev_f = valid_frames[i - 1]
            curr_f = valid_frames[i]
            
            for joint in range(17):
                max_vel = VELOCITY_LIMITS_3D.get(joint, 0.05)
                delta = smoothed[curr_f, joint] - smoothed[prev_f, joint]
                velocity = np.linalg.norm(delta)
                
                if velocity > max_vel:
                    scale = max_vel / velocity
                    smoothed[curr_f, joint] = smoothed[prev_f, joint] + delta * scale
    
    return smoothed.astype(np.float32)


def smooth_all_3d_tracks(keypoints_3d, sigma=1.5, use_adaptive_sigma=True, apply_velocity_limit=True):
    """
    Smooth all person 3D trajectories with enhanced algorithm (planv5).
    
    Args:
        keypoints_3d: (N, max_people, 17, 3) array
        sigma: float - Base Gaussian kernel sigma (used if adaptive disabled)
        use_adaptive_sigma: bool - Use per-joint sigma values from JOINT_SIGMA_3D
        apply_velocity_limit: bool - Apply velocity limiting pass
    
    Returns:
        smoothed: (N, max_people, 17, 3) array
    """
    N, max_people = keypoints_3d.shape[:2]
    smoothed = np.zeros_like(keypoints_3d)
    
    for person_id in range(max_people):
        person_poses = keypoints_3d[:, person_id]
        smoothed[:, person_id] = smooth_3d_trajectory(
            person_poses, 
            sigma=sigma,
            use_adaptive_sigma=use_adaptive_sigma,
            apply_velocity_limit=apply_velocity_limit
        )
    
    return smoothed


H36M_SYMMETRIC_BONES = [
    (4, 5, 1, 2),     # hip-to-knee (L: 4→5, R: 1→2) - thigh
    (5, 6, 2, 3),     # knee-to-ankle (L: 5→6, R: 2→3) - shin
    (11, 12, 14, 15), # shoulder-to-elbow (L: 11→12, R: 14→15) - upper arm  
    (12, 13, 15, 16), # elbow-to-wrist (L: 12→13, R: 15→16) - forearm
]


def compute_bone_lengths(pose_3d):
    """
    Compute bone lengths for a single 3D pose.
    
    Args:
        pose_3d: (17, 3) array - single H36M pose
        
    Returns:
        dict mapping bone tuple to length
    """
    from mediapipe_to_h36m import H36M_CONNECTIONS
    
    lengths = {}
    for idx_a, idx_b in H36M_CONNECTIONS:
        bone_vec = pose_3d[idx_b] - pose_3d[idx_a]
        length = np.linalg.norm(bone_vec)
        if length > 0.001:  # Skip invalid bones
            lengths[(idx_a, idx_b)] = length
    return lengths


def enforce_bone_constraints(poses_3d, symmetry_weight=0.5):
    """
    Enforce bone length constraints on 3D pose sequence.
    
    This addresses two issues:
    1. Bone length variance across frames (uses median length per bone)
    2. Left-right asymmetry (enforces symmetric limb pairs)
    
    Args:
        poses_3d: (N, 17, 3) array - 3D poses for one person
        symmetry_weight: float - how much to enforce symmetry (0=none, 1=full)
    
    Returns:
        constrained: (N, 17, 3) array - poses with more consistent bone lengths
    """
    from mediapipe_to_h36m import H36M_CONNECTIONS
    
    N = poses_3d.shape[0]
    if N < 5:
        return poses_3d
    
    constrained = poses_3d.copy()
    
    valid_mask = np.abs(poses_3d).max(axis=(1, 2)) > 0.001
    if valid_mask.sum() < 5:
        return poses_3d
    
    all_lengths = {}  # bone → list of lengths
    
    for frame_idx in np.where(valid_mask)[0]:
        lengths = compute_bone_lengths(poses_3d[frame_idx])
        for bone, length in lengths.items():
            if bone not in all_lengths:
                all_lengths[bone] = []
            all_lengths[bone].append(length)
    
    median_lengths = {bone: np.median(lengths) for bone, lengths in all_lengths.items()}
    
    for left_a, left_b, right_a, right_b in H36M_SYMMETRIC_BONES:
        left_bone = (left_a, left_b)
        right_bone = (right_a, right_b)
        
        if left_bone in median_lengths and right_bone in median_lengths:
            avg_length = (median_lengths[left_bone] + median_lengths[right_bone]) / 2
            median_lengths[left_bone] = (1 - symmetry_weight) * median_lengths[left_bone] + symmetry_weight * avg_length
            median_lengths[right_bone] = (1 - symmetry_weight) * median_lengths[right_bone] + symmetry_weight * avg_length
    
    for frame_idx in np.where(valid_mask)[0]:
        pose = constrained[frame_idx].copy()
        
        for idx_a, idx_b in H36M_CONNECTIONS:
            bone = (idx_a, idx_b)
            if bone not in median_lengths:
                continue
            
            target_length = median_lengths[bone]
            current_vec = pose[idx_b] - pose[idx_a]
            current_length = np.linalg.norm(current_vec)
            
            if current_length < 0.001:
                continue
            
            scale = target_length / current_length
            
            direction = current_vec / current_length
            pose[idx_b] = pose[idx_a] + direction * target_length
        
        constrained[frame_idx] = pose
    
    return constrained


def enforce_all_bone_constraints(keypoints_3d, symmetry_weight=0.5):
    """
    Apply bone length constraints to all person tracks.
    
    Args:
        keypoints_3d: (N, max_people, 17, 3) array
        symmetry_weight: float - symmetry enforcement strength
    
    Returns:
        constrained: (N, max_people, 17, 3) array
    """
    N, max_people = keypoints_3d.shape[:2]
    constrained = np.zeros_like(keypoints_3d)
    
    for person_id in range(max_people):
        person_poses = keypoints_3d[:, person_id]
        constrained[:, person_id] = enforce_bone_constraints(person_poses, symmetry_weight)
    
    return constrained


def normalize_screen_coordinates(X, w, h):
    """Normalize 2D keypoints to [-1, 1] range, preserving aspect ratio."""
    return X / w * 2 - np.array([1, h / w])


class VideoPose3DLifter:
    """Wraps VideoPose3D model for 2D to 3D lifting."""
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.receptive_field = RECEPTIVE_FIELD
        self.pad = self.receptive_field // 2
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.model = TemporalModelOptimized1f(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3],
            dropout=0.25,
            channels=1024
        )
        self.model.load_state_dict(checkpoint['model_pos'])
        self.model.eval()
        self.model.to(device)
    
    def lift_sequence(self, keypoints_2d_sequence, video_width, video_height):
        """Lift a sequence of 2D keypoints to 3D."""
        N = keypoints_2d_sequence.shape[0]
        
        keypoints_norm = normalize_screen_coordinates(
            keypoints_2d_sequence, w=video_width, h=video_height
        )
        
        left_pad = np.tile(keypoints_norm[0:1], (self.pad, 1, 1))
        right_pad = np.tile(keypoints_norm[-1:], (self.pad, 1, 1))
        keypoints_padded = np.concatenate([left_pad, keypoints_norm, right_pad], axis=0)
        
        all_predictions = []
        
        with torch.no_grad():
            for i in range(N):
                window = keypoints_padded[i:i + self.receptive_field]
                input_tensor = torch.from_numpy(
                    window.astype(np.float32)
                ).unsqueeze(0).to(self.device)
                
                pred = self.model(input_tensor)
                all_predictions.append(pred.squeeze().cpu().numpy())
        
        return np.stack(all_predictions, axis=0)
    
    def lift_multiperson_sequence(self, keypoints_2d_all, video_width, video_height, tracked_mask=None):
        """Lift 2D keypoints for multiple people."""
        N, max_people = keypoints_2d_all.shape[:2]
        keypoints_3d_all = np.zeros((N, max_people, 17, 3), dtype=np.float32)
        
        for person_id in range(max_people):
            person_kps = keypoints_2d_all[:, person_id]
            
            has_detection = np.array([person_kps[i].max() > 0 for i in range(N)])
            if has_detection.sum() < 3:
                continue
            
            print(f"  Lifting person {person_id + 1} ({has_detection.sum()} frames)...")
            keypoints_3d = self.lift_sequence(person_kps, video_width, video_height)
            
            for i in range(N):
                if not has_detection[i]:
                    keypoints_3d[i] = 0
            
            keypoints_3d_all[:, person_id] = keypoints_3d
        
        return keypoints_3d_all


def draw_skeleton_2d(canvas, keypoints_2d, color, tracked_mask=None):
    """
    Draw 2D skeleton with clean visualization:
    - Face circle (instead of wavy lines connecting eyes/ears)
    - Single spine line (instead of torso rectangle)
    - Shoulder and hip bars
    
    The input keypoints_2d is in COCO 17-joint format.
    
    COCO indices:
        0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear
        5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow, 9=l_wrist, 10=r_wrist
        11=l_hip, 12=r_hip, 13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle
    """
    if keypoints_2d is None or keypoints_2d.max() == 0:
        return canvas
    
    # COCO indices
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    def pt(idx):
        """Get point as tuple if valid, else None"""
        p = keypoints_2d[idx]
        if p[0] > 0 and p[1] > 0:
            return (int(p[0]), int(p[1]))
        return None


    nose = pt(NOSE)
    l_ear = pt(L_EAR)
    r_ear = pt(R_EAR)
    
    if nose and l_ear and r_ear:
        ear_dist_l = np.sqrt((nose[0] - l_ear[0])**2 + (nose[1] - l_ear[1])**2)
        ear_dist_r = np.sqrt((nose[0] - r_ear[0])**2 + (nose[1] - r_ear[1])**2)
        radius = int(max((ear_dist_l + ear_dist_r) / 2, 10))
        
        cv2.circle(canvas, nose, radius, color, 2)
        
        neck_bottom = (nose[0], nose[1] + radius)
        l_sh = pt(L_SHOULDER)
        r_sh = pt(R_SHOULDER)
        if l_sh and r_sh:
            mid_shoulder = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
            cv2.line(canvas, neck_bottom, mid_shoulder, color, 2)
    elif nose:
        cv2.circle(canvas, nose, 15, color, 2)


    l_sh = pt(L_SHOULDER)
    r_sh = pt(R_SHOULDER)
    if l_sh and r_sh:
        cv2.line(canvas, l_sh, r_sh, color, 2)


    l_hip = pt(L_HIP)
    r_hip = pt(R_HIP)
    if l_sh and r_sh and l_hip and r_hip:
        mid_shoulder = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
        mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
        cv2.line(canvas, mid_shoulder, mid_hip, color, 2)


    if l_hip and r_hip:
        cv2.line(canvas, l_hip, r_hip, color, 2)


    arm_connections = [
        (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    ]
    for idx_a, idx_b in arm_connections:
        pa = pt(idx_a)
        pb = pt(idx_b)
        if pa and pb:
            cv2.line(canvas, pa, pb, color, 2)


    
    l_hip = pt(L_HIP)
    l_knee = pt(L_KNEE)
    l_ankle = pt(L_ANKLE)
    
    if l_hip and l_knee:
        cv2.line(canvas, l_hip, l_knee, color, 2)
    if l_knee and l_ankle:
        cv2.line(canvas, l_knee, l_ankle, color, 2)
    
    r_hip = pt(R_HIP)
    r_knee = pt(R_KNEE)
    r_ankle = pt(R_ANKLE)
    
    if r_hip and r_knee:
        cv2.line(canvas, r_hip, r_knee, color, 2)
    if r_knee and r_ankle:
        cv2.line(canvas, r_knee, r_ankle, color, 2)


    for i in range(5, 17):  # Skip face landmarks
        point = keypoints_2d[i]
        if point[0] > 0 and point[1] > 0:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, color, -1)
    
    return canvas


def project_3d_to_2d_anchored(keypoints_3d, keypoints_2d):
    """
    Project 3D skeleton to 2D screen coordinates using bbox-anchored approach.
    
    NOTE: This is a PRACTICAL DISPLAY HEURISTIC, not true geometric camera projection.
    True reprojection would require camera intrinsics/extrinsics which we don't have.
    
    This approach:
    1. Computes bounding box of 2D keypoints (screen anchor)
    2. Computes bounding box of 3D keypoints in XY plane
    3. Scales and translates 3D XY to match 2D bbox
    4. Uses Z only for depth coloring (orthographic projection)
    
    Args:
        keypoints_3d: (17, 3) array - VideoPose3D output (root-relative, metric-like)
        keypoints_2d: (17, 2) array - H36M 2D keypoints in pixel coordinates
        
    Returns:
        points_2d: list of (x, y) tuples or None for invalid joints
        z_values: list of z values for depth coloring
    """
    points_2d = []
    z_values = []
    
    valid_2d_mask = (keypoints_2d[:, 0] > 0) & (keypoints_2d[:, 1] > 0)
    valid_3d_mask = np.abs(keypoints_3d).max(axis=1) > 0.001
    
    if valid_2d_mask.sum() < 2 or valid_3d_mask.sum() < 2:
        return [None] * 17, [0] * 17
    
    valid_2d = keypoints_2d[valid_2d_mask]
    bbox_2d_min = valid_2d.min(axis=0)
    bbox_2d_max = valid_2d.max(axis=0)
    bbox_2d_center = (bbox_2d_min + bbox_2d_max) / 2
    bbox_2d_size = bbox_2d_max - bbox_2d_min
    bbox_2d_size = np.maximum(bbox_2d_size, 1)  # Avoid division by zero
    
    valid_3d = keypoints_3d[valid_3d_mask]
    bbox_3d_min = valid_3d[:, :2].min(axis=0)  # Only X,Y
    bbox_3d_max = valid_3d[:, :2].max(axis=0)
    bbox_3d_center = (bbox_3d_min + bbox_3d_max) / 2
    bbox_3d_size = bbox_3d_max - bbox_3d_min
    bbox_3d_size = np.maximum(bbox_3d_size, 0.001)  # Avoid division by zero
    
    scale = min(bbox_2d_size[0] / bbox_3d_size[0], 
                bbox_2d_size[1] / bbox_3d_size[1])
    
    # H36M: index 11 = left_hip (from COCO 11), index 12 = right_hip (from COCO 12)
    hip_center_2d = None
    if keypoints_2d[11].max() > 0 and keypoints_2d[12].max() > 0:
        hip_center_2d = (keypoints_2d[11] + keypoints_2d[12]) / 2
    
    for i in range(17):
        x, y, z = keypoints_3d[i]
        
        # CRITICAL FIX: Root joint (index 0) is ALWAYS at origin (0,0,0)
        if i == 0:
            if hip_center_2d is not None:
                points_2d.append((int(hip_center_2d[0]), int(hip_center_2d[1])))
                z_values.append(0)
            else:
                points_2d.append((int(bbox_2d_center[0]), int(bbox_2d_center[1])))
                z_values.append(0)
            continue
        
        if abs(x) < 0.001 and abs(y) < 0.001 and abs(z) < 0.001:
            points_2d.append(None)
            z_values.append(0)
            continue
        
        px = (x - bbox_3d_center[0]) * scale + bbox_2d_center[0]
        py = (y - bbox_3d_center[1]) * scale + bbox_2d_center[1]
        
        points_2d.append((int(px), int(py)))
        z_values.append(z)
    
    return points_2d, z_values


def draw_skeleton_3d(canvas, keypoints_3d, keypoints_2d, color):
    """
    Draw 3D skeleton projected to 2D using bbox-anchored projection.
    
    Now includes:
    - Face circle (instead of drawing nose/head_top connections)
    - Proper joint filtering (no dots for head joints)
    
    H36M indices:
        0=hip, 1=r_hip, 2=r_knee, 3=r_ankle
        4=l_hip, 5=l_knee, 6=l_ankle
        7=spine, 8=neck, 9=nose, 10=head_top
        11=l_shoulder, 12=l_elbow, 13=l_wrist
        14=r_shoulder, 15=r_elbow, 16=r_wrist
    
    Args:
        canvas: output image to draw on
        keypoints_3d: (17, 3) - VideoPose3D output (root-relative coordinates)
        keypoints_2d: (17, 2) - H36M 2D keypoints in pixel coordinates (for anchoring)
        color: base color for skeleton
    """
    if keypoints_3d is None or np.abs(keypoints_3d).max() == 0:
        return canvas
    
    points_2d, z_values = project_3d_to_2d_anchored(keypoints_3d, keypoints_2d)
    
    valid_z = [z for z in z_values if z != 0]
    if valid_z:
        z_min, z_max = min(valid_z), max(valid_z)
        z_range = max(z_max - z_min, 0.1)
    else:
        z_min, z_range = 0, 1.0
    
    h, w = canvas.shape[:2]
    
    # H36M head-related indices (for face circle, skip these connections)
    H36M_NOSE = 9
    H36M_HEAD_TOP = 10
    H36M_NECK = 8
    
    face_connections = {(8, 9), (9, 10)}  # neck→nose, nose→head_top


    for idx_a, idx_b in H36M_CONNECTIONS:
        if (idx_a, idx_b) in face_connections:
            continue
            
        if points_2d[idx_a] is not None and points_2d[idx_b] is not None:
            pa = points_2d[idx_a]
            pb = points_2d[idx_b]
            
            pa = (np.clip(pa[0], 0, w-1), np.clip(pa[1], 0, h-1))
            pb = (np.clip(pb[0], 0, w-1), np.clip(pb[1], 0, h-1))
            
            avg_z = (z_values[idx_a] + z_values[idx_b]) / 2
            brightness = 0.5 + 0.5 * (1 - (avg_z - z_min) / z_range)
            brightness = np.clip(brightness, 0.3, 1.0)
            line_color = tuple(int(c * brightness) for c in color)
            
            cv2.line(canvas, pa, pb, line_color, 2)


    nose_2d = points_2d[H36M_NOSE]
    head_2d = points_2d[H36M_HEAD_TOP]
    neck_2d = points_2d[H36M_NECK]
    
    if nose_2d and head_2d:
        radius = int(np.sqrt((nose_2d[0] - head_2d[0])**2 + 
                              (nose_2d[1] - head_2d[1])**2))
        radius = max(radius, 10)
        
        center_x = (nose_2d[0] + head_2d[0]) // 2
        center_y = (nose_2d[1] + head_2d[1]) // 2
        center = (np.clip(center_x, 0, w-1), np.clip(center_y, 0, h-1))
        
        avg_z = (z_values[H36M_NOSE] + z_values[H36M_HEAD_TOP]) / 2
        brightness = 0.5 + 0.5 * (1 - (avg_z - z_min) / z_range)
        brightness = np.clip(brightness, 0.3, 1.0)
        circle_color = tuple(int(c * brightness) for c in color)
        
        cv2.circle(canvas, center, radius, circle_color, 2)
        
        if neck_2d:
            neck_pt = (np.clip(neck_2d[0], 0, w-1), np.clip(neck_2d[1], 0, h-1))
            circle_bottom = (center[0], min(center[1] + radius, h-1))
            cv2.line(canvas, circle_bottom, neck_pt, circle_color, 2)
    
    elif nose_2d:
        nose_pt = (np.clip(nose_2d[0], 0, w-1), np.clip(nose_2d[1], 0, h-1))
        cv2.circle(canvas, nose_pt, 15, color, 2)


    # H36M joint indices to skip:
    skip_joints = {H36M_NOSE, H36M_HEAD_TOP, 1, 4}
    
    for i, point in enumerate(points_2d):
        if i in skip_joints:
            continue  # Skip head and intermediate hip joints
            
        if point is not None:
            px = np.clip(point[0], 0, w-1)
            py = np.clip(point[1], 0, h-1)
            
            brightness = 0.5 + 0.5 * (1 - (z_values[i] - z_min) / z_range)
            brightness = np.clip(brightness, 0.3, 1.0)
            joint_color = tuple(int(c * brightness) for c in color)
            cv2.circle(canvas, (px, py), 4, joint_color, -1)
    
    return canvas


def render_frame(keypoints_2d_all, keypoints_3d_all, original_frame, 
                 video_width, video_height, mode='side_by_side', tracked_mask=None):
    """
    Render a single frame with 2D and/or 3D poses.
    
    The 3D visualization uses bbox-anchored orthographic projection:
    - 3D skeletons are scaled and positioned to match 2D detection bboxes
    - This is a DISPLAY HEURISTIC, not true camera projection
    - Z depth is used only for brightness effects
    
    Args:
        keypoints_2d_all: (max_people, 17, 2) - H36M 2D keypoints in pixels
        keypoints_3d_all: (max_people, 17, 3) - VideoPose3D output (root-relative)
        original_frame: the video frame to overlay on
        video_width, video_height: original video dimensions (unused in new projection)
        mode: 'side_by_side' or 'skeleton'
        tracked_mask: optional per-person tracking mask
    """
    h, w = original_frame.shape[:2]
    max_people = keypoints_2d_all.shape[0]
    
    if mode == 'side_by_side':
        left = original_frame.copy()
        right = np.zeros((h, w, 3), dtype=np.uint8)
        
        for p in range(max_people):
            has_2d = keypoints_2d_all[p].max() > 0
            has_3d = np.abs(keypoints_3d_all[p]).max() > 0
            
            if has_2d:
                color = PERSON_COLORS[p % len(PERSON_COLORS)]
                left = draw_skeleton_2d(left, keypoints_2d_all[p], color)
            
            if has_3d and has_2d:
                color = PERSON_COLORS[p % len(PERSON_COLORS)]
                right = draw_skeleton_3d(right, keypoints_3d_all[p], keypoints_2d_all[p], color)
            elif has_3d:
                pass
        
        cv2.putText(left, "2D Pose", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right, "3D Pose (bbox-anchored)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return np.hstack([left, right])
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        for p in range(max_people):
            has_2d = keypoints_2d_all[p].max() > 0
            has_3d = np.abs(keypoints_3d_all[p]).max() > 0
            
            if has_3d and has_2d:
                color = PERSON_COLORS[p % len(PERSON_COLORS)]
                canvas = draw_skeleton_3d(canvas, keypoints_3d_all[p], keypoints_2d_all[p], color)
        
        return canvas


class PoseEstimationPipeline:
    """Complete 3D pose estimation pipeline."""
    
    def __init__(self, models_dir=None, device='cpu'):
        if models_dir is None:
            models_dir = os.path.join(script_dir, 'models')
        
        self.device = device
        
        yolo_path = os.path.join(models_dir, 'yolov8n.pt')
        print("Loading models...")
        self.yolo_model = YOLO(yolo_path, verbose=False)
        
        mp_model_path = os.path.join(models_dir, 'pose_landmarker_lite.task')
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        stderr_fd = sys.stderr.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(stderr_fd)
        os.dup2(devnull_fd, stderr_fd)
        self.landmarker = PoseLandmarker.create_from_options(options)
        os.dup2(saved, stderr_fd)
        os.close(devnull_fd)
        
        videopose_path = os.path.join(script_dir, 'VideoPose3D', 'pretrained_h36m_detectron_coco.bin')
        self.lifter = VideoPose3DLifter(videopose_path, device=device)
        print("Models loaded!")
    
    def process_video(self, video_path, output_path, 
                      smoothing_sigma=2,
                      render_mode='side_by_side',
                      export_npy=False,
                      show_progress=True):
        """Process a video through the full 3D pose estimation pipeline."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {video_path}")
        print(f"Resolution: {video_width}x{video_height}, FPS: {fps}, Frames: {total_frames}")
        
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            cap.release()
            return
        
        processed_frame = preprocess_frame(first_frame)
        out_h, out_w = processed_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


        print("\n[Phase 1/4] Extracting 2D poses...")
        
        all_raw_keypoints = []  # Raw detections per frame
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            processed = preprocess_frame(frame)
            
            boxes = detect_persons(self.yolo_model, processed)
            
            frame_kps = np.zeros((MAX_PEOPLE, 33, 2), dtype=np.float32)
            
            for i, box in enumerate(boxes[:MAX_PEOPLE]):
                landmarks = estimate_pose_mediapipe(self.landmarker, processed, box)
                if landmarks is not None:
                    frame_kps[i] = landmarks
            
            all_raw_keypoints.append(frame_kps)
            
            if show_progress and frame_num % 30 == 0:
                print(f"  Frame {frame_num}/{total_frames} - {len(boxes)} person(s)")
        
        cap.release()
        
        all_raw_keypoints = np.stack(all_raw_keypoints, axis=0)
        N = all_raw_keypoints.shape[0]
        print(f"  Extracted: {all_raw_keypoints.shape}")


        print("\n[Phase 2/4] Batch tracking for consistent IDs...")
        
        all_tracked = batch_track_people(all_raw_keypoints)
        all_tracked = filter_short_tracks(all_tracked, min_frames=10)
        
        active_tracks = sum(1 for p in range(MAX_PEOPLE) 
                          if any(all_tracked[i, p].max() > 0 for i in range(N)))
        print(f"  Active tracks: {active_tracks}")


        print("\n[Phase 3/4] Converting and lifting to 3D...")
        
        all_h36m_2d = np.zeros((N, MAX_PEOPLE, 17, 2), dtype=np.float32)
        
        for i in range(N):
            for p in range(MAX_PEOPLE):
                if all_tracked[i, p].max() > 0:
                    all_h36m_2d[i, p] = mediapipe_to_h36m(all_tracked[i, p])
        
        all_h36m_2d_smooth = smooth_all_tracks(all_h36m_2d, sigma=smoothing_sigma)
        
        all_3d_keypoints = self.lifter.lift_multiperson_sequence(
            all_h36m_2d_smooth, out_w, out_h
        )
        print(f"  3D keypoints: {all_3d_keypoints.shape}")
        
        all_3d_keypoints = smooth_all_3d_tracks(
            all_3d_keypoints, 
            sigma=smoothing_sigma * 0.75,  # Base sigma (used if adaptive disabled)
            use_adaptive_sigma=True,        # Use per-joint sigma from JOINT_SIGMA_3D
            apply_velocity_limit=True       # Remove impossible velocity spikes
        )
        print(f"  Applied enhanced 3D smoothing (adaptive sigma + velocity limiting)")
        
        all_3d_keypoints = enforce_all_bone_constraints(all_3d_keypoints, symmetry_weight=0.7)
        print(f"  Applied bone length constraints (symmetry=0.7)")


        print("\n[Phase 4/4] Rendering output video...")
        
        if render_mode == 'side_by_side':
            output_width = out_w * 2
        else:
            output_width = out_w
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, out_h))
        
        cap = cv2.VideoCapture(video_path)
        
        for frame_idx in range(N):
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = preprocess_frame(frame)
            
            output_frame = render_frame(
                all_h36m_2d_smooth[frame_idx],
                all_3d_keypoints[frame_idx],
                processed,
                out_w, out_h,
                mode=render_mode
            )
            
            out.write(output_frame)
            
            if show_progress and (frame_idx + 1) % 30 == 0:
                print(f"  Rendered {frame_idx + 1}/{N} frames")
        
        cap.release()
        out.release()
        
        if export_npy:
            npy_path = output_path.rsplit('.', 1)[0] + '_3d_keypoints.npy'
            np.save(npy_path, all_3d_keypoints)
            print(f"  Saved 3D keypoints to: {npy_path}")
        
        print(f"\n✓ Output saved to: {output_path}")
    
    def close(self):
        """Release resources."""
        self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline_3d.py --input video.mp4 --output output_3d.mp4
    python pipeline_3d.py --input video.mp4 --output output.mp4 --mode side_by_side
    python pipeline_3d.py --input video.mp4 --output output.mp4 --export-npy
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--mode', '-m', choices=['skeleton', 'side_by_side'], 
                        default='side_by_side', help='Render mode (default: side_by_side)')
    parser.add_argument('--smoothing', '-s', type=float, default=2.0,
                        help='Temporal smoothing sigma (default: 2.0)')
    parser.add_argument('--export-npy', action='store_true',
                        help='Export 3D keypoints to NPY file')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device for VideoPose3D (default: cpu)')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    pipeline = PoseEstimationPipeline(device=args.device)
    
    try:
        pipeline.process_video(
            video_path=args.input,
            output_path=args.output,
            smoothing_sigma=args.smoothing,
            render_mode=args.mode,
            export_npy=args.export_npy
        )
    finally:
        pipeline.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

