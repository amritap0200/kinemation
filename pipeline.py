#!/usr/bin/env python3
"""Unified 3D Pose Estimation Pipeline"""

import cv2
import numpy as np
import sys
import os
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VideoPose3D'))

import torch
from common.model import TemporalModelOptimized1f
from common.camera import normalize_screen_coordinates
from adapter import coco_to_h36m


def apply_clahe(frame):
    """Apply CLAHE in LAB color space for contrast enhancement."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame, max_dim=800, apply_blur=True, apply_clahe_enhancement=True):
    """Preprocess frame with aspect-ratio preservation."""
    h, w = frame.shape[:2]
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if apply_clahe_enhancement:
        frame = apply_clahe(frame)
    
    if apply_blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame


def extract_2d_poses(video_path, model, max_people=6, progress_interval=30):
    """Extract 2D keypoints using YOLO11n-pose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"  Original: {width_orig}x{height_orig}, FPS: {fps}, Frames: {total_frames}")
    print("\n[Phase 1/4] Extracting 2D poses...")
    
    all_frames = []
    frame_count = 0
    processed_width = None
    processed_height = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_preprocessed = preprocess_frame(frame, max_dim=800, apply_blur=True, apply_clahe_enhancement=True)
        
        if processed_width is None:
            processed_height, processed_width = frame_preprocessed.shape[:2]
            print(f"  Preprocessed: {processed_width}x{processed_height}")
        
        results = model(frame_preprocessed, verbose=False)[0]
        frame_data = np.zeros((max_people, 17, 2), dtype=np.float32)
        
        if results.keypoints is not None:
            kps = results.keypoints.xy.cpu().numpy()
            for p in range(min(len(kps), max_people)):
                frame_data[p] = kps[p]
        
        all_frames.append(frame_data)
        frame_count += 1
        
        if frame_count % progress_interval == 0:
            print(f"  Frame {frame_count}/{total_frames}...")
    
    cap.release()
    
    keypoints_array = np.array(all_frames, dtype=np.float32)
    print(f"  Extracted: {keypoints_array.shape}")
    
    video_info = {
        'width': processed_width,
        'height': processed_height,
        'fps': fps,
        'total_frames': frame_count,
        'width_orig': width_orig,
        'height_orig': height_orig
    }
    
    return keypoints_array, video_info


def convert_to_h36m(coco_keypoints):
    """Convert COCO keypoints to H36M format."""
    if coco_keypoints.ndim == 4:
        N, P = coco_keypoints.shape[0], coco_keypoints.shape[1]
        h36m = np.zeros_like(coco_keypoints)
        for i in range(N):
            for p in range(P):
                if coco_keypoints[i, p].max() > 0:
                    h36m[i, p] = coco_to_h36m(coco_keypoints[i, p])
        return h36m
    else:
        N = coco_keypoints.shape[0]
        h36m = np.zeros_like(coco_keypoints)
        for i in range(N):
            h36m[i] = coco_to_h36m(coco_keypoints[i])
        return h36m


def run_videopose3d(h36m_keypoints, video_width, video_height, model_path='VideoPose3D/pretrained_h36m_detectron_coco.bin'):
    """Run VideoPose3D inference on H36M keypoints."""
    print("\n[Phase 2/4] Loading VideoPose3D model...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
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
    print("  Model loaded!")
    
    receptive_field = 243
    pad = receptive_field // 2
    
    is_multi = h36m_keypoints.ndim == 4
    
    if is_multi:
        N, P = h36m_keypoints.shape[0], h36m_keypoints.shape[1]
        predictions_3d = np.zeros((N, P, 17, 3), dtype=np.float32)
        
        print(f"\n[Phase 3/4] Running 3D inference ({P} tracks)...")
        
        for p in range(P):
            person_2d = h36m_keypoints[:, p, :, :]
            
            if person_2d.max() == 0:
                continue
            
            person_norm = normalize_screen_coordinates(person_2d.copy(), w=video_width, h=video_height)
            left_pad = np.tile(person_norm[0:1], (pad, 1, 1))
            right_pad = np.tile(person_norm[-1:], (pad, 1, 1))
            person_padded = np.concatenate([left_pad, person_norm, right_pad], axis=0)
            
            for i in range(N):
                window = person_padded[i : i + receptive_field]
                input_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                
                with torch.no_grad():
                    pred = model(input_tensor)
                
                predictions_3d[i, p] = pred.squeeze().numpy()
            
            print(f"  Person {p+1}/{P} done")
    
    else:
        N = h36m_keypoints.shape[0]
        print(f"\n[Phase 3/4] Running 3D inference...")
        
        keypoints_norm = normalize_screen_coordinates(h36m_keypoints, w=video_width, h=video_height)
        
        left_pad = np.tile(keypoints_norm[0:1], (pad, 1, 1))
        right_pad = np.tile(keypoints_norm[-1:], (pad, 1, 1))
        keypoints_padded = np.concatenate([left_pad, keypoints_norm, right_pad], axis=0)
        
        predictions_3d = []
        for i in range(N):
            window = keypoints_padded[i : i + receptive_field]
            input_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            
            with torch.no_grad():
                pred = model(input_tensor)
            
            predictions_3d.append(pred.squeeze().numpy())
            
            if (i + 1) % 30 == 0:
                print(f"  Frame {i+1}/{N}...")
        
        predictions_3d = np.stack(predictions_3d, axis=0)
    
    print(f"  Output shape: {predictions_3d.shape}")
    return predictions_3d


def get_bbox(kps_17x2):
    """Get bounding box from keypoints."""
    valid = kps_17x2[(kps_17x2[:, 0] > 0) & (kps_17x2[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return np.array([valid[:,0].min(), valid[:,1].min(),
                     valid[:,0].max(), valid[:,1].max()])


def iou(a, b):
    """Compute IoU between two bboxes."""
    xi1, yi1 = max(a[0],b[0]), max(a[1],b[1])
    xi2, yi2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def track_people(raw_kps):
    """IoU-based tracking for consistent person IDs."""
    N, max_people = raw_kps.shape[0], raw_kps.shape[1]
    tracked = np.zeros_like(raw_kps)
    tracked[0] = raw_kps[0]
    prev_boxes = [get_bbox(raw_kps[0, p]) for p in range(max_people)]
    
    for i in range(1, N):
        frame_kps = raw_kps[i]
        det_boxes, det_indices = [], []
        for d in range(max_people):
            b = get_bbox(frame_kps[d])
            if b is not None:
                det_boxes.append(b)
                det_indices.append(d)
        
        if not det_boxes:
            continue
        
        cost = np.ones((max_people, len(det_boxes)))
        for t in range(max_people):
            if prev_boxes[t] is None:
                continue
            for di, db in enumerate(det_boxes):
                cost[t, di] = 1.0 - iou(prev_boxes[t], db)
        
        t_ids, d_ids = linear_sum_assignment(cost)
        assigned_t, assigned_d = set(), set()
        
        for t_id, d_id in zip(t_ids, d_ids):
            orig = det_indices[d_id]
            tracked[i, t_id] = frame_kps[orig]
            prev_boxes[t_id] = det_boxes[d_id]
            assigned_t.add(t_id)
            assigned_d.add(d_id)
        
        leftover = [det_indices[d] for d in range(len(det_boxes)) if d not in assigned_d]
        empty = [t for t in range(max_people) if t not in assigned_t]
        for slot, orig in zip(empty, leftover):
            tracked[i, slot] = frame_kps[orig]
            prev_boxes[slot] = get_bbox(frame_kps[orig])
    
    return tracked


def smooth_track(kps_track, sigma=2):
    """Apply Gaussian smoothing within detection runs."""
    N = kps_track.shape[0]
    smoothed = kps_track.copy()
    
    has_detection = np.array([kps_track[i].max() > 0 for i in range(N)])
    
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
    
    for (start, end) in runs:
        if end - start < 3:
            continue
        for joint_idx in range(17):
            for coord in range(2):
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


def filter_short_tracks(tracked, min_frames=10):
    """Remove short detection runs."""
    N, max_people = tracked.shape[0], tracked.shape[1]
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


CONNECTIONS = [
    ("left_wrist",  "left_elbow"),
    ("right_wrist", "right_elbow"),
    ("right_elbow", "neck"),
    ("left_elbow",  "neck"),
    ("neck",        "torso"),
    ("torso",       "left_knee"),
    ("torso",       "right_knee"),
    ("left_knee",   "left_ankle"),
    ("right_knee",  "right_ankle"),
    ("nose",        "neck"),
    ("nose",        "left_eye"),
    ("nose",        "right_eye"),
]

COLORS = [
    (0,   255,   0),
    (255,   0,   0),
    (0,     0, 255),
    (0,   165, 255),
    (128,   0, 128),
    (0,   255, 255),
]


def build_pts(kps, processed_w, processed_h, orig_w, orig_h):
    """Build keypoints dict with scaling to original resolution."""
    sx = orig_w / processed_w
    sy = orig_h / processed_h
    
    def sc(x, y):
        return (int(x * sx), int(y * sy))
    
    ls = sc(kps[5][0],  kps[5][1])
    rs = sc(kps[6][0],  kps[6][1])
    lh = sc(kps[11][0], kps[11][1])
    rh = sc(kps[12][0], kps[12][1])
    
    named = {
        "nose":        (kps[0][0],  kps[0][1]),
        "left_eye":    (kps[1][0],  kps[1][1]),
        "right_eye":   (kps[2][0],  kps[2][1]),
        "left_elbow":  (kps[7][0],  kps[7][1]),
        "right_elbow": (kps[8][0],  kps[8][1]),
        "left_wrist":  (kps[9][0],  kps[9][1]),
        "right_wrist": (kps[10][0], kps[10][1]),
        "left_knee":   (kps[13][0], kps[13][1]),
        "right_knee":  (kps[14][0], kps[14][1]),
        "left_ankle":  (kps[15][0], kps[15][1]),
        "right_ankle": (kps[16][0], kps[16][1]),
    }
    
    pts = {k: sc(v[0], v[1]) for k, v in named.items() if v[0] > 0 and v[1] > 0}
    
    if kps[5][0] > 0 and kps[6][0] > 0:
        pts["neck"] = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    if kps[5][0] > 0 and kps[6][0] > 0 and kps[11][0] > 0 and kps[12][0] > 0:
        pts["torso"] = ((ls[0]+rs[0]+lh[0]+rh[0])//4,
                        (ls[1]+rs[1]+lh[1]+rh[1])//4)
    return pts


def render_output(smoothed_kps, video_path, output_path, video_info, mode='smooth'):
    """Render 2D skeleton visualization."""
    N, max_people = smoothed_kps.shape[0], smoothed_kps.shape[1]
    
    print(f"\n[Phase 4/4] Rendering {mode} output...")
    
    cap = cv2.VideoCapture(video_path)
    fps = video_info['fps']
    if fps < 5 or fps > 120:
        fps = 30
    
    width_orig = video_info['width_orig']
    height_orig = video_info['height_orig']
    width_proc = video_info['width']
    height_proc = video_info['height']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width_orig, height_orig))
    
    for i in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        
        canvas = frame.copy()
        
        for p in range(max_people):
            kps = smoothed_kps[i, p]
            
            if kps.max() == 0:
                continue
            
            color = COLORS[p % len(COLORS)]
            pts = build_pts(kps, width_proc, height_proc, width_orig, height_orig)
            
            for point in pts.values():
                cv2.circle(canvas, point, 4, color, -1)
            for p1, p2 in CONNECTIONS:
                if p1 in pts and p2 in pts:
                    cv2.line(canvas, pts[p1], pts[p2], color, 2)
        
        out.write(canvas)
        if (i+1) % 30 == 0:
            print(f"  Frame {i+1}/{N}...")
    
    cap.release()
    out.release()
    print(f"\n✓ Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified 3D Pose Estimation Pipeline')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--mode', choices=['smooth', 'raw'], default='smooth', help='smooth = temporal smoothing')
    parser.add_argument('--smoothing', type=float, default=2.0, help='Smoothing sigma')
    parser.add_argument('--max-people', type=int, default=6, help='Maximum people to track')
    parser.add_argument('--export-npy', action='store_true', help='Export intermediate .npy files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Unified 3D Pose Estimation Pipeline")
    print("=" * 60)
    
    print("\nLoading YOLO11n-pose model...")
    model = YOLO("yolo11n-pose.pt")
    print("  Model loaded!")
    
    coco_kps, video_info = extract_2d_poses(args.input, model, max_people=args.max_people)
    
    if args.export_npy:
        np.save("coco_keypoints.npy", coco_kps)
        print(f"  Saved: coco_keypoints.npy")
    
    h36m_kps = convert_to_h36m(coco_kps)
    
    if args.export_npy:
        np.save("h36m_keypoints.npy", h36m_kps)
        print(f"  Saved: h36m_keypoints.npy")
    
    predictions_3d = run_videopose3d(h36m_kps, video_info['width'], video_info['height'])
    
    if args.export_npy:
        np.save("predictions_3d.npy", predictions_3d)
        print(f"  Saved: predictions_3d.npy")
    
    if args.mode == 'smooth':
        print("\nApplying tracking and smoothing...")
        tracked = track_people(coco_kps)
        tracked = filter_short_tracks(tracked, min_frames=10)
        
        smoothed = np.zeros_like(tracked)
        for p in range(args.max_people):
            smoothed[:, p] = smooth_track(tracked[:, p], sigma=args.smoothing)
        
        render_kps = smoothed
    else:
        render_kps = coco_kps
    
    render_output(render_kps, args.input, args.output, video_info, mode=args.mode)
    print("\nDone!")


if __name__ == "__main__":
    main()
