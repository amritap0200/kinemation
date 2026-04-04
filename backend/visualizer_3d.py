"""3D pose visualization module"""

import os
import sys
import cv2
import numpy as np

# Ensure this module's directory is on sys.path for sibling imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from mediapipe_to_h36m import H36M_CONNECTIONS

PERSON_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 255, 128), (128, 255, 0),
]


# Get color based on depth
def depth_to_color(z_normalized, base_color):
    brightness = 0.7 - 0.3 * z_normalized
    brightness = np.clip(brightness, 0.4, 1.0)
    return tuple(int(c * brightness) for c in base_color)


# Draw 3D skeleton with depth coloring
def draw_3d_skeleton(canvas, keypoints_3d, color=(0, 255, 0), video_width=None, video_height=None, draw_depth_colors=True):
    h, w = canvas.shape[:2]
    video_width = video_width or w
    video_height = video_height or h
    
    if keypoints_3d is None or keypoints_3d.max() == 0:
        return canvas
    
    points_2d = []
    z_values = []
    
    for i in range(17):
        x, y, z = keypoints_3d[i]
        if x == 0 and y == 0:
            points_2d.append(None)
            z_values.append(0)
            continue
        px = int((x + 1) * video_width / 2)
        py = int((y + 1) * video_height / 2)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        points_2d.append((px, py))
        z_values.append(z)
    
    valid_z = [z for z in z_values if z != 0]
    if valid_z:
        z_min, z_max = min(valid_z), max(valid_z)
        z_range = z_max - z_min if z_max != z_min else 1.0
    else:
        z_min, z_range = 0, 1.0
    
    for idx_a, idx_b in H36M_CONNECTIONS:
        if points_2d[idx_a] is not None and points_2d[idx_b] is not None:
            if draw_depth_colors:
                avg_z = (z_values[idx_a] + z_values[idx_b]) / 2
                z_norm = (avg_z - z_min) / z_range * 2 - 1
                line_color = depth_to_color(z_norm, color)
            else:
                line_color = color
            cv2.line(canvas, points_2d[idx_a], points_2d[idx_b], line_color, 2)
    
    for i, point in enumerate(points_2d):
        if point is not None:
            if draw_depth_colors:
                z_norm = (z_values[i] - z_min) / z_range * 2 - 1
                joint_color = depth_to_color(z_norm, color)
            else:
                joint_color = color
            radius = int(4 - z_norm) if draw_depth_colors else 4
            radius = np.clip(radius, 3, 6)
            cv2.circle(canvas, point, radius, joint_color, -1)
    
    return canvas


# Draw 2D skeleton in H36M format
def draw_2d_skeleton_h36m(canvas, keypoints_2d, color=(0, 255, 0)):
    if keypoints_2d is None or keypoints_2d.max() == 0:
        return canvas
    
    for idx_a, idx_b in H36M_CONNECTIONS:
        pt_a = keypoints_2d[idx_a]
        pt_b = keypoints_2d[idx_b]
        if pt_a[0] > 0 and pt_a[1] > 0 and pt_b[0] > 0 and pt_b[1] > 0:
            pt_a = (int(pt_a[0]), int(pt_a[1]))
            pt_b = (int(pt_b[0]), int(pt_b[1]))
            cv2.line(canvas, pt_a, pt_b, color, 2)
    
    for point in keypoints_2d:
        if point[0] > 0 and point[1] > 0:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, color, -1)
    
    return canvas


# Visualizer class
class Visualizer3D:
    def __init__(self, video_width, video_height, max_people=10):
        self.video_width = video_width
        self.video_height = video_height
        self.max_people = max_people
    
    def render_frame(self, keypoints_3d_all, mode='skeleton', background=None):
        if mode == 'overlay' and background is not None:
            canvas = background.copy()
        else:
            canvas = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        for person_id in range(self.max_people):
            if keypoints_3d_all[person_id].max() == 0:
                continue
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            canvas = draw_3d_skeleton(canvas, keypoints_3d_all[person_id], color=color,
                                      video_width=self.video_width, video_height=self.video_height)
        return canvas
    
    def render_side_by_side(self, keypoints_2d_all, keypoints_3d_all, original_frame=None, labels=True):
        if original_frame is not None:
            left = original_frame.copy()
        else:
            left = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        for person_id in range(self.max_people):
            if keypoints_2d_all[person_id].max() == 0:
                continue
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            left = draw_2d_skeleton_h36m(left, keypoints_2d_all[person_id], color)
        
        right = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        for person_id in range(self.max_people):
            if keypoints_3d_all[person_id].max() == 0:
                continue
            color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
            right = draw_3d_skeleton(right, keypoints_3d_all[person_id], color=color,
                                     video_width=self.video_width, video_height=self.video_height)
        
        if labels:
            cv2.putText(left, "2D Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(right, "3D Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return np.hstack([left, right])
