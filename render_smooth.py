import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

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

YOLO_W = 1440
YOLO_H = 810


def get_bbox(kps_17x2):
    valid = kps_17x2[(kps_17x2[:, 0] > 0) & (kps_17x2[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return np.array([valid[:,0].min(), valid[:,1].min(),
                     valid[:,0].max(), valid[:,1].max()])


def iou(a, b):
    xi1, yi1 = max(a[0],b[0]), max(a[1],b[1])
    xi2, yi2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def track_people(raw_kps):
    N, max_people = raw_kps.shape[0], raw_kps.shape[1]
    tracked   = np.zeros_like(raw_kps)
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
        empty    = [t for t in range(max_people) if t not in assigned_t]
        for slot, orig in zip(empty, leftover):
            tracked[i, slot] = frame_kps[orig]
            prev_boxes[slot] = get_bbox(frame_kps[orig])

    return tracked


def smooth_track(kps_track, sigma=2):
    """
    Smooth each person's track but ONLY within frames where
    they are actually detected. Zero frames stay zero.
    No interpolation across gaps. No bleeding.
    """
    N = kps_track.shape[0]
    smoothed = kps_track.copy()

    # Find which frames this person actually has ANY detection
    has_detection = np.array([kps_track[i].max() > 0 for i in range(N)])

    if has_detection.sum() < 3:
        return smoothed

    # Find contiguous detection runs
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

    # Smooth each run independently
    for (start, end) in runs:
        if end - start < 3:
            continue
        for joint_idx in range(17):
            for coord in range(2):
                segment = kps_track[start:end, joint_idx, coord].astype(np.float64)
                valid   = segment > 0
                if valid.sum() < 2:
                    continue
                # Only smooth the valid portion within this run
                indices = np.arange(len(segment))
                # Small gap fill within the run
                filled = np.interp(indices, indices[valid], segment[valid])
                smooth = gaussian_filter1d(filled, sigma=sigma)
                # Only write back to frames that had actual detections
                result = smoothed[start:end, joint_idx, coord]
                result[valid] = smooth[valid]
                smoothed[start:end, joint_idx, coord] = result

    return smoothed.astype(np.float32)


def build_pts(kps, sx, sy):
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

def filter_short_tracks(tracked, min_frames=10):
    """
    For each person slot, find all contiguous detection runs.
    Any run shorter than min_frames gets wiped to zero.
    This kills flash glitches where YOLO detects someone for
    just a few frames then loses them.
    """
    N, max_people = tracked.shape[0], tracked.shape[1]
    filtered = tracked.copy()

    for p in range(max_people):
        # Find which frames have a detection for this person
        has_det = np.array([tracked[i, p].max() > 0 for i in range(N)])

        # Find contiguous runs
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
                    # Wipe this short run to zero
                    filtered[run_start:i, p] = 0.0
                in_run = False

    return filtered

def render_smooth(raw_kps_path="yolo_keypoints_trial1.npy",
                  video_path="trial1.mp4",
                  output_path="output_smooth_trial1.mp4",
                  sigma=2):

    raw_kps = np.load(raw_kps_path)
    if raw_kps.ndim == 3:
        raw_kps = raw_kps[:, np.newaxis, :, :]

    N, max_people = raw_kps.shape[0], raw_kps.shape[1]
    print(f"Loaded {N} frames, {max_people} people slots")

    print("Tracking...")
    tracked = track_people(raw_kps)

    print("Filtering short tracks...")
    tracked = filter_short_tracks(tracked, min_frames=10)

    print("Smoothing (detection-aware, no bleeding)...")
    smoothed_all = np.zeros_like(tracked)
    for p in range(max_people):
        smoothed_all[:, p] = smooth_track(tracked[:, p], sigma=sigma)
        print(f"  Person {p+1}/{max_people} done")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 5 or fps > 120:
        fps = 30
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = VW / YOLO_W, VH / YOLO_H

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (VW, VH))

    print(f"Rendering {N} frames...")
    for i in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        canvas = frame.copy()

        for p in range(max_people):
            kps = smoothed_all[i, p]

            # CRITICAL: only draw if this frame actually had a detection
            # Check raw tracked (pre-smooth) to decide whether to draw
            if tracked[i, p].max() == 0:
                continue

            color = COLORS[p % len(COLORS)]
            pts   = build_pts(kps, sx, sy)

            for point in pts.values():
                cv2.circle(canvas, point, 4, color, -1)
            for p1, p2 in CONNECTIONS:
                if p1 in pts and p2 in pts:
                    cv2.line(canvas, pts[p1], pts[p2], color, 2)

        out.write(canvas)
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{N} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    render_smooth()