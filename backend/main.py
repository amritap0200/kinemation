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
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'VideoPose3D'))

import torch
from common.model import TemporalModelOptimized1f
from mediapipe_to_h36m import mediapipe_to_h36m, H36M_CONNECTIONS, COCO_CONNECTIONS

MAX_PEOPLE = 6
RECEPTIVE_FIELD = 243

PERSON_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (0, 165, 255), (128, 0, 128), (0, 255, 255),
]

JOINT_SIGMA_3D = {
    0: 2.2, 7: 2.2, 8: 2.0,
    1: 1.9, 4: 1.9, 2: 1.7, 5: 1.7, 3: 1.5, 6: 1.5,
    9: 1.7, 10: 1.7,
    11: 1.4, 14: 1.4, 12: 1.2, 15: 1.2, 13: 1.0, 16: 1.0,
}

VELOCITY_LIMITS_3D = {
    0: 0.025, 7: 0.025, 8: 0.035,
    1: 0.050, 4: 0.050,
    2: 0.060, 5: 0.060, 3: 0.080, 6: 0.080,
    11: 0.080, 14: 0.080, 12: 0.120, 15: 0.120, 13: 0.150, 16: 0.150,
    9: 0.050, 10: 0.050,
}

H36M_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8),
    (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

H36M_SYMMETRIC_BONES = [
    (4, 5, 1, 2), (5, 6, 2, 3),
    (11, 12, 14, 15), (12, 13, 15, 16),
]


# Apply CLAHE in LAB color space
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Preprocess frame with CLAHE and blur
def preprocess_frame(frame, max_dim=800, apply_blur=True, apply_clahe_enhancement=True):
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if apply_clahe_enhancement:
        frame = apply_clahe(frame)
    if apply_blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


# Detect persons using YOLOv8
def detect_persons(yolo_model, frame, confidence=0.5):
    results = yolo_model(frame, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
    return boxes


# Estimate 2D pose using MediaPipe
def estimate_pose_mediapipe(landmarker, frame, box):
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


# Get bounding box from keypoints
def get_bbox_from_keypoints(keypoints):
    valid_points = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_points) == 0:
        return None
    x_min, y_min = valid_points.min(axis=0)
    x_max, y_max = valid_points.max(axis=0)
    return np.array([x_min, y_min, x_max, y_max])


# Calculate IoU between two boxes
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


# Batch track people across frames using Hungarian algorithm
def batch_track_people(all_raw_keypoints, iou_threshold=0.3):
    N, max_people = all_raw_keypoints.shape[:2]
    tracked = np.zeros_like(all_raw_keypoints)
    tracked[0] = all_raw_keypoints[0]
    prev_bboxes = [get_bbox_from_keypoints(all_raw_keypoints[0, p]) for p in range(max_people)]
    
    for frame_idx in range(1, N):
        frame_keypoints = all_raw_keypoints[frame_idx]
        det_bboxes = []
        det_indices = []
        
        for d in range(max_people):
            bbox = get_bbox_from_keypoints(frame_keypoints[d])
            if bbox is not None:
                det_bboxes.append(bbox)
                det_indices.append(d)
        
        if len(det_bboxes) == 0:
            continue
        
        cost_matrix = np.ones((max_people, len(det_bboxes)))
        iou_matrix = np.zeros((max_people, len(det_bboxes)))
        
        for track_id in range(max_people):
            if prev_bboxes[track_id] is None:
                continue
            for det_id, det_bbox in enumerate(det_bboxes):
                iou_val = iou(prev_bboxes[track_id], det_bbox)
                iou_matrix[track_id, det_id] = iou_val
                cost_matrix[track_id, det_id] = 1.0 - iou_val
        
        track_ids, det_ids = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_dets = set()
        
        for track_id, det_id in zip(track_ids, det_ids):
            if iou_matrix[track_id, det_id] < iou_threshold and prev_bboxes[track_id] is not None:
                continue
            original_idx = det_indices[det_id]
            tracked[frame_idx, track_id] = frame_keypoints[original_idx]
            prev_bboxes[track_id] = det_bboxes[det_id]
            assigned_tracks.add(track_id)
            assigned_dets.add(det_id)
        
        unassigned_dets = [det_indices[d] for d in range(len(det_bboxes)) if d not in assigned_dets]
        empty_slots = [t for t in range(max_people) if t not in assigned_tracks]
        
        for slot, orig_idx in zip(empty_slots, unassigned_dets):
            tracked[frame_idx, slot] = frame_keypoints[orig_idx]
            prev_bboxes[slot] = get_bbox_from_keypoints(frame_keypoints[orig_idx])
    
    return tracked


# Filter out short tracks
def filter_short_tracks(tracked, min_frames=10):
    N, max_people = tracked.shape[:2]
    filtered = tracked.copy()
    
    for p in range(max_people):
        has_detection = np.array([tracked[i, p].max() > 0 for i in range(N)])
        in_run = False
        run_start = 0
        
        for i in range(N + 1):
            is_active = i < N and has_detection[i]
            if is_active and not in_run:
                run_start = i
                in_run = True
            elif not is_active and in_run:
                run_length = i - run_start
                if run_length < min_frames:
                    filtered[run_start:i, p] = 0.0
                in_run = False
    
    return filtered


# Smooth a single track
def smooth_single_track(keypoints_track, sigma=2):
    N = keypoints_track.shape[0]
    smoothed = keypoints_track.copy()
    has_detection = np.array([keypoints_track[i].max() > 0 for i in range(N)])
    
    if has_detection.sum() < 3:
        return smoothed
    
    runs = []
    in_run = False
    for i in range(N):
        if has_detection[i] and not in_run:
            run_start = i
            in_run = True
        elif not has_detection[i] and in_run:
            runs.append((run_start, i))
            in_run = False
    if in_run:
        runs.append((run_start, N))
    
    n_joints = keypoints_track.shape[1]
    n_coords = keypoints_track.shape[2]
    
    for start, end in runs:
        if end - start < 3:
            continue
        for j in range(n_joints):
            for c in range(n_coords):
                segment = keypoints_track[start:end, j, c].astype(np.float64)
                valid_mask = segment > 0
                if valid_mask.sum() < 2:
                    continue
                indices = np.arange(len(segment))
                filled = np.interp(indices, indices[valid_mask], segment[valid_mask])
                smooth = gaussian_filter1d(filled, sigma=sigma)
                smoothed[start:end, j, c][valid_mask] = smooth[valid_mask]
    
    return smoothed.astype(np.float32)


# Smooth all tracks
def smooth_all_tracks(tracked, sigma=2):
    N, max_people = tracked.shape[:2]
    smoothed = np.zeros_like(tracked)
    for p in range(max_people):
        smoothed[:, p] = smooth_single_track(tracked[:, p], sigma=sigma)
    return smoothed


# Smooth 3D trajectory with joint-adaptive sigma
def smooth_3d_trajectory(poses_3d, base_sigma=1.5, use_adaptive_sigma=True, apply_velocity_limit=True):
    N = poses_3d.shape[0]
    if N < 3:
        return poses_3d
    
    smoothed = poses_3d.copy()
    valid_mask = np.abs(poses_3d).max(axis=(1, 2)) > 0.001
    valid_frames = np.where(valid_mask)[0]
    
    if len(valid_frames) < 3:
        return poses_3d
    
    for joint_idx in range(17):
        if use_adaptive_sigma:
            sigma = JOINT_SIGMA_3D.get(joint_idx, base_sigma)
        else:
            sigma = base_sigma
        
        for coord in range(3):
            values = poses_3d[valid_mask, joint_idx, coord]
            if len(values) >= 3:
                smooth_values = gaussian_filter1d(values, sigma=sigma)
                smoothed[valid_mask, joint_idx, coord] = smooth_values
    
    if apply_velocity_limit and len(valid_frames) > 1:
        for i in range(1, len(valid_frames)):
            prev_frame = valid_frames[i - 1]
            curr_frame = valid_frames[i]
            for joint_idx in range(17):
                max_velocity = VELOCITY_LIMITS_3D.get(joint_idx, 0.05)
                delta = smoothed[curr_frame, joint_idx] - smoothed[prev_frame, joint_idx]
                velocity = np.linalg.norm(delta)
                if velocity > max_velocity:
                    smoothed[curr_frame, joint_idx] = smoothed[prev_frame, joint_idx] + delta * (max_velocity / velocity)
    
    return smoothed.astype(np.float32)


# Smooth all 3D tracks
def smooth_all_3d_tracks(keypoints_3d, sigma=1.5, use_adaptive_sigma=True, apply_velocity_limit=True):
    N, max_people = keypoints_3d.shape[:2]
    smoothed = np.zeros_like(keypoints_3d)
    for p in range(max_people):
        smoothed[:, p] = smooth_3d_trajectory(
            keypoints_3d[:, p], sigma, use_adaptive_sigma, apply_velocity_limit
        )
    return smoothed


# Compute bone lengths
def compute_bone_lengths(pose_3d):
    lengths = {}
    for a, b in H36M_BONE_PAIRS:
        vec = pose_3d[b] - pose_3d[a]
        length = np.linalg.norm(vec)
        if length > 0.001:
            lengths[(a, b)] = length
    return lengths


# Enforce bone length constraints
def enforce_bone_constraints(poses_3d, symmetry_weight=0.5):
    N = poses_3d.shape[0]
    if N < 5:
        return poses_3d
    
    constrained = poses_3d.copy()
    valid_mask = np.abs(poses_3d).max(axis=(1, 2)) > 0.001
    
    if valid_mask.sum() < 5:
        return poses_3d
    
    all_lengths = {}
    for frame_idx in np.where(valid_mask)[0]:
        lengths = compute_bone_lengths(poses_3d[frame_idx])
        for bone, length in lengths.items():
            if bone not in all_lengths:
                all_lengths[bone] = []
            all_lengths[bone].append(length)
    
    median_lengths = {}
    for bone, lengths in all_lengths.items():
        median_lengths[bone] = np.median(lengths)
    
    for la, lb, ra, rb in H36M_SYMMETRIC_BONES:
        left_bone = (la, lb)
        right_bone = (ra, rb)
        if left_bone in median_lengths and right_bone in median_lengths:
            avg = (median_lengths[left_bone] + median_lengths[right_bone]) / 2
            median_lengths[left_bone] = (1 - symmetry_weight) * median_lengths[left_bone] + symmetry_weight * avg
            median_lengths[right_bone] = (1 - symmetry_weight) * median_lengths[right_bone] + symmetry_weight * avg
    
    for frame_idx in np.where(valid_mask)[0]:
        pose = constrained[frame_idx].copy()
        for a, b in H36M_BONE_PAIRS:
            if (a, b) not in median_lengths:
                continue
            target_length = median_lengths[(a, b)]
            vec = pose[b] - pose[a]
            current_length = np.linalg.norm(vec)
            if current_length < 0.001:
                continue
            pose[b] = pose[a] + (vec / current_length) * target_length
        constrained[frame_idx] = pose
    
    return constrained


# Enforce bone constraints for all people
def enforce_all_bone_constraints(keypoints_3d, symmetry_weight=0.5):
    N, max_people = keypoints_3d.shape[:2]
    constrained = np.zeros_like(keypoints_3d)
    for p in range(max_people):
        constrained[:, p] = enforce_bone_constraints(keypoints_3d[:, p], symmetry_weight)
    return constrained


# Normalize screen coordinates for VideoPose3D
def normalize_screen_coordinates(X, w, h):
    return X / w * 2 - np.array([1, h / w])


# VideoPose3D wrapper
class VideoPose3DLifter:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.receptive_field = RECEPTIVE_FIELD
        self.pad = self.receptive_field // 2
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = TemporalModelOptimized1f(
            num_joints_in=17, in_features=2, num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3], dropout=0.25, channels=1024
        )
        self.model.load_state_dict(checkpoint['model_pos'])
        self.model.eval()
        self.model.to(device)
    
    def lift_sequence(self, keypoints_2d_sequence, video_width, video_height):
        N = keypoints_2d_sequence.shape[0]
        kps_normalized = normalize_screen_coordinates(keypoints_2d_sequence.copy(), w=video_width, h=video_height)
        
        pad_left = np.tile(kps_normalized[0:1], (self.pad, 1, 1))
        pad_right = np.tile(kps_normalized[-1:], (self.pad, 1, 1))
        kps_padded = np.concatenate([pad_left, kps_normalized, pad_right], axis=0)
        
        predictions = []
        with torch.no_grad():
            for i in range(N):
                window = kps_padded[i:i + self.receptive_field]
                input_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                if self.device == 'cuda':
                    input_tensor = input_tensor.cuda()
                output = self.model(input_tensor)
                predictions.append(output.squeeze().cpu().numpy())
        
        return np.stack(predictions, axis=0)
    
    def lift_multiperson_sequence(self, keypoints_2d_all, video_width, video_height):
        N, max_people = keypoints_2d_all.shape[:2]
        keypoints_3d_all = np.zeros((N, max_people, 17, 3), dtype=np.float32)
        
        for person_idx in range(max_people):
            person_keypoints = keypoints_2d_all[:, person_idx]
            has_detection = np.array([person_keypoints[i].max() > 0 for i in range(N)])
            
            if has_detection.sum() < 3:
                continue
            
            print(f"  Lifting person {person_idx + 1} ({has_detection.sum()} frames)...")
            keypoints_3d = self.lift_sequence(person_keypoints, video_width, video_height)
            
            for i in range(N):
                if not has_detection[i]:
                    keypoints_3d[i] = 0
            
            keypoints_3d_all[:, person_idx] = keypoints_3d
        
        return keypoints_3d_all


# Draw face circle for 2D skeleton
def draw_face_circle_2d(canvas, keypoints, color):
    def get_point(idx):
        p = keypoints[idx]
        if p[0] > 0 and p[1] > 0:
            return (int(p[0]), int(p[1]))
        return None
    
    nose = get_point(0)
    left_ear = get_point(3)
    right_ear = get_point(4)
    left_shoulder = get_point(5)
    right_shoulder = get_point(6)
    
    if nose is None:
        return canvas
    
    if left_ear and right_ear:
        dist_left = np.sqrt((nose[0] - left_ear[0])**2 + (nose[1] - left_ear[1])**2)
        dist_right = np.sqrt((nose[0] - right_ear[0])**2 + (nose[1] - right_ear[1])**2)
        radius = int(max((dist_left + dist_right) / 2, 10))
    else:
        radius = 15
    
    cv2.circle(canvas, nose, radius, color, 2)
    
    if left_shoulder and right_shoulder:
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2)
        neck_bottom = (nose[0], nose[1] + radius)
        cv2.line(canvas, neck_bottom, mid_shoulder, color, 2)
    
    return canvas


# Draw 2D skeleton
def draw_skeleton_2d(canvas, keypoints, color):
    if keypoints is None or keypoints.max() == 0:
        return canvas
    
    canvas = draw_face_circle_2d(canvas, keypoints, color)
    
    def get_point(idx):
        p = keypoints[idx]
        if p[0] > 0 and p[1] > 0:
            return (int(p[0]), int(p[1]))
        return None
    
    left_shoulder = get_point(5)
    right_shoulder = get_point(6)
    left_hip = get_point(11)
    right_hip = get_point(12)
    
    if left_shoulder and right_shoulder:
        cv2.line(canvas, left_shoulder, right_shoulder, color, 2)
    
    if left_shoulder and right_shoulder and left_hip and right_hip:
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2)
        mid_hip = ((left_hip[0] + right_hip[0]) // 2,
                   (left_hip[1] + right_hip[1]) // 2)
        cv2.line(canvas, mid_shoulder, mid_hip, color, 2)
    
    if left_hip and right_hip:
        cv2.line(canvas, left_hip, right_hip, color, 2)
    
    arm_connections = [(5, 7), (7, 9), (6, 8), (8, 10)]
    for idx_a, idx_b in arm_connections:
        pt_a = get_point(idx_a)
        pt_b = get_point(idx_b)
        if pt_a and pt_b:
            cv2.line(canvas, pt_a, pt_b, color, 2)
    
    left_knee = get_point(13)
    left_ankle = get_point(15)
    right_knee = get_point(14)
    right_ankle = get_point(16)
    
    if left_hip and left_knee:
        cv2.line(canvas, left_hip, left_knee, color, 2)
    if left_knee and left_ankle:
        cv2.line(canvas, left_knee, left_ankle, color, 2)
    if right_hip and right_knee:
        cv2.line(canvas, right_hip, right_knee, color, 2)
    if right_knee and right_ankle:
        cv2.line(canvas, right_knee, right_ankle, color, 2)
    
    for i in range(5, 17):
        pt = keypoints[i]
        if pt[0] > 0 and pt[1] > 0:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, color, -1)
    
    return canvas


# Project 3D to 2D using bbox anchoring
def project_3d_to_2d_anchored(keypoints_3d, keypoints_2d):
    points_2d = []
    z_values = []
    
    valid_2d_mask = (keypoints_2d[:, 0] > 0) & (keypoints_2d[:, 1] > 0)
    valid_3d_mask = np.abs(keypoints_3d).max(axis=1) > 0.001
    
    if valid_2d_mask.sum() < 2 or valid_3d_mask.sum() < 2:
        return [None] * 17, [0] * 17
    
    valid_2d_points = keypoints_2d[valid_2d_mask]
    bbox_2d_min = valid_2d_points.min(axis=0)
    bbox_2d_max = valid_2d_points.max(axis=0)
    bbox_2d_center = (bbox_2d_min + bbox_2d_max) / 2
    bbox_2d_size = np.maximum(bbox_2d_max - bbox_2d_min, 1)
    
    valid_3d_points = keypoints_3d[valid_3d_mask]
    bbox_3d_min = valid_3d_points[:, :2].min(axis=0)
    bbox_3d_max = valid_3d_points[:, :2].max(axis=0)
    bbox_3d_center = (bbox_3d_min + bbox_3d_max) / 2
    bbox_3d_size = np.maximum(bbox_3d_max - bbox_3d_min, 0.001)
    
    scale = min(bbox_2d_size[0] / bbox_3d_size[0], bbox_2d_size[1] / bbox_3d_size[1])
    
    hip_center_2d = None
    if keypoints_2d[11].max() > 0 and keypoints_2d[12].max() > 0:
        hip_center_2d = (keypoints_2d[11] + keypoints_2d[12]) / 2
    
    for i in range(17):
        x, y, z = keypoints_3d[i]
        
        if i == 0:
            if hip_center_2d is not None:
                points_2d.append((int(hip_center_2d[0]), int(hip_center_2d[1])))
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


# Draw 3D skeleton
def draw_skeleton_3d(canvas, keypoints_3d, keypoints_2d, color):
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
    face_connections = {(8, 9), (9, 10)}
    skip_joints = {9, 10, 1, 4}
    
    for idx_a, idx_b in H36M_CONNECTIONS:
        if (idx_a, idx_b) in face_connections:
            continue
        if points_2d[idx_a] is not None and points_2d[idx_b] is not None:
            pa = (np.clip(points_2d[idx_a][0], 0, w-1), np.clip(points_2d[idx_a][1], 0, h-1))
            pb = (np.clip(points_2d[idx_b][0], 0, w-1), np.clip(points_2d[idx_b][1], 0, h-1))
            avg_z = (z_values[idx_a] + z_values[idx_b]) / 2
            brightness = 0.5 + 0.5 * (1 - (avg_z - z_min) / z_range)
            brightness = np.clip(brightness, 0.3, 1.0)
            line_color = tuple(int(c * brightness) for c in color)
            cv2.line(canvas, pa, pb, line_color, 2)
    
    nose_2d = points_2d[9]
    head_2d = points_2d[10]
    neck_2d = points_2d[8]
    
    if nose_2d and head_2d:
        radius = int(np.sqrt((nose_2d[0] - head_2d[0])**2 + (nose_2d[1] - head_2d[1])**2))
        radius = max(radius, 10)
        center_x = (nose_2d[0] + head_2d[0]) // 2
        center_y = (nose_2d[1] + head_2d[1]) // 2
        center = (np.clip(center_x, 0, w-1), np.clip(center_y, 0, h-1))
        avg_z = (z_values[9] + z_values[10]) / 2
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
    
    for i, point in enumerate(points_2d):
        if i in skip_joints:
            continue
        if point is not None:
            px = np.clip(point[0], 0, w-1)
            py = np.clip(point[1], 0, h-1)
            brightness = 0.5 + 0.5 * (1 - (z_values[i] - z_min) / z_range)
            brightness = np.clip(brightness, 0.3, 1.0)
            joint_color = tuple(int(c * brightness) for c in color)
            cv2.circle(canvas, (px, py), 4, joint_color, -1)
    
    return canvas


# Render a frame with 2D and/or 3D poses
def render_frame(keypoints_2d_all, keypoints_3d_all, original_frame, video_width, video_height, mode='side_by_side', tracked_mask=None):
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
        
        cv2.putText(left, "2D Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right, "3D Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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


# Main pipeline class
class PoseEstimationPipeline:
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
    
    def process_video(self, video_path, output_path, smoothing_sigma=2, render_mode='side_by_side', export_npy=False, show_progress=True, progress_callback=None):
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
        if progress_callback:
            progress_callback(0.05, 'Extracting 2D poses...', 'Phase 1/4: 2D Pose Extraction')
        all_raw_keypoints = []
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
        if progress_callback:
            progress_callback(0.35, 'Tracking people across frames...', 'Phase 2/4: Multi-Person Tracking')
        all_tracked = batch_track_people(all_raw_keypoints)
        all_tracked = filter_short_tracks(all_tracked, min_frames=10)
        
        active_tracks = sum(1 for p in range(MAX_PEOPLE) if any(all_tracked[i, p].max() > 0 for i in range(N)))
        print(f"  Active tracks: {active_tracks}")
        
        print("\n[Phase 3/4] Converting and lifting to 3D...")
        if progress_callback:
            progress_callback(0.45, 'Lifting 2D poses to 3D...', 'Phase 3/4: 3D Pose Lifting')
        all_h36m_2d = np.zeros((N, MAX_PEOPLE, 17, 2), dtype=np.float32)
        
        for i in range(N):
            for p in range(MAX_PEOPLE):
                if all_tracked[i, p].max() > 0:
                    all_h36m_2d[i, p] = mediapipe_to_h36m(all_tracked[i, p])
        
        all_h36m_2d_smooth = smooth_all_tracks(all_h36m_2d, sigma=smoothing_sigma)
        
        all_3d_keypoints = self.lifter.lift_multiperson_sequence(all_h36m_2d_smooth, out_w, out_h)
        print(f"  3D keypoints: {all_3d_keypoints.shape}")
        
        all_3d_keypoints = smooth_all_3d_tracks(
            all_3d_keypoints, sigma=smoothing_sigma * 0.75,
            use_adaptive_sigma=True, apply_velocity_limit=True
        )
        print(f"  Applied enhanced 3D smoothing (adaptive sigma + velocity limiting)")
        
        all_3d_keypoints = enforce_all_bone_constraints(all_3d_keypoints, symmetry_weight=0.7)
        print(f"  Applied bone length constraints (symmetry=0.7)")
        
        print("\n[Phase 4/4] Rendering output video...")
        if progress_callback:
            progress_callback(0.70, 'Rendering output video...', 'Phase 4/4: Video Rendering')
        
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
                processed, out_w, out_h, mode=render_mode
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
        if progress_callback:
            progress_callback(1.0, 'Processing complete!', 'Done')
    
    def close(self):
        self.landmarker.close()


# Process video file and return results (API function)
def process_video_file(input_path, output_path, mode='side_by_side', smoothing=2.0, export_npy=False, device='cpu'):
    if not os.path.isfile(input_path):
        return {"success": False, "error": f"Input file not found: {input_path}"}
    
    pipeline = PoseEstimationPipeline(device=device)
    try:
        pipeline.process_video(
            video_path=input_path,
            output_path=output_path,
            smoothing_sigma=smoothing,
            render_mode=mode,
            export_npy=export_npy
        )
        return {"success": True, "output_path": output_path}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        pipeline.close()


# Process webcam stream (live mode)
def process_webcam(output_path=None, duration=None, mode='side_by_side', device='cpu'):
    pipeline = PoseEstimationPipeline(device=device)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        pipeline.close()
        return {"success": False, "error": "Cannot open webcam"}
    
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if output_path:
        out_w = width * 2 if mode == 'side_by_side' else width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, height))
    
    frame_buffer = []
    buffer_size = RECEPTIVE_FIELD
    
    print("Starting webcam... Press 'q' to stop")
    start_time = cv2.getTickCount()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = preprocess_frame(frame)
            out_h, out_w = processed.shape[:2]
            
            boxes = detect_persons(pipeline.yolo_model, processed)
            frame_kps = np.zeros((MAX_PEOPLE, 33, 2), dtype=np.float32)
            for i, box in enumerate(boxes[:MAX_PEOPLE]):
                landmarks = estimate_pose_mediapipe(pipeline.landmarker, processed, box)
                if landmarks is not None:
                    frame_kps[i] = landmarks
            
            frame_buffer.append(frame_kps)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            h36m_2d = np.zeros((MAX_PEOPLE, 17, 2), dtype=np.float32)
            for p in range(MAX_PEOPLE):
                if frame_kps[p].max() > 0:
                    h36m_2d[p] = mediapipe_to_h36m(frame_kps[p])
            
            if len(frame_buffer) >= 3:
                buffer_arr = np.stack(frame_buffer, axis=0)
                h36m_buffer = np.zeros((len(frame_buffer), MAX_PEOPLE, 17, 2), dtype=np.float32)
                for i in range(len(frame_buffer)):
                    for p in range(MAX_PEOPLE):
                        if buffer_arr[i, p].max() > 0:
                            h36m_buffer[i, p] = mediapipe_to_h36m(buffer_arr[i, p])
                
                all_3d = pipeline.lifter.lift_multiperson_sequence(h36m_buffer, out_w, out_h)
                keypoints_3d = all_3d[-1]
            else:
                keypoints_3d = np.zeros((MAX_PEOPLE, 17, 3), dtype=np.float32)
            
            output_frame = render_frame(h36m_2d, keypoints_3d, processed, out_w, out_h, mode=mode)
            
            if out:
                out.write(output_frame)
            
            cv2.imshow('Kinemation - 3D Pose Estimation', output_frame)
            frame_count += 1
            
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if duration and elapsed >= duration:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        pipeline.close()
    
    return {"success": True, "frames_processed": frame_count, "output_path": output_path}


def main():
    parser = argparse.ArgumentParser(description='3D Pose Estimation Pipeline')
    parser.add_argument('--input', '-i', help='Input video path (use "webcam" for live feed)')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--mode', '-m', choices=['skeleton', 'side_by_side'], default='side_by_side', help='Render mode')
    parser.add_argument('--smoothing', '-s', type=float, default=2.0, help='Temporal smoothing sigma')
    parser.add_argument('--export-npy', action='store_true', help='Export 3D keypoints to NPY file')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device for VideoPose3D')
    parser.add_argument('--duration', '-d', type=float, help='Webcam recording duration in seconds')
    
    args = parser.parse_args()
    
    if args.input and args.input.lower() == 'webcam':
        result = process_webcam(args.output, args.duration, args.mode, args.device)
    elif args.input:
        result = process_video_file(args.input, args.output, args.mode, args.smoothing, args.export_npy, args.device)
    else:
        print("Error: --input is required (use video path or 'webcam')")
        sys.exit(1)
    
    if result["success"]:
        print("\nDone!")
    else:
        print(f"\nError: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
