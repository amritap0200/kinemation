# Kinemation

A computer vision pipeline for converting human movement in video into temporally coherent, smooth animated stick figures.

Kinemation detects body joints from monocular video, enforces temporal consistency across frames, and maps the resulting motion data to temporally coherent and smooth stick figure animations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Approaches Explored](#approaches-explored)
  - [Pose Estimation](#pose-estimation)
  - [Temporal Smoothing and Coherence](#temporal-smoothing-and-coherence)
- [Work Completed](#work-completed)
- [Work In Progress](#work-in-progress)
- [Roadmap](#roadmap)
- [Setup and Installation](#setup-and-installation)
- [Team](#team)
- [References](#references)

---

## Project Overview

Standard pose estimation models operate frame-by-frame, producing skeletal estimates that are accurate in isolation but temporally incoherent in sequence - joints flicker, limbs snap between positions, and the resulting animation is visually noisy. Kinemation addresses this by treating video as a temporal signal rather than a collection of independent images.

The longer-term goal extends beyond geometric accuracy. By integrating body-language-based emotion recognition, Kinemation aims to produce stick figures that not only move like the subject but also express like them - mapping inferred emotional state to visual properties of the animation such as posture, joint expressiveness, motion dynamics, and rendering style.

---

## Pipeline Architecture

```
INPUT VIDEO
     |
     v
2D Pose Estimator (MediaPipe BlazePose)
     |  N x 33 landmarks per frame
     v
Keypoint Adapter (MediaPipe -> H36M format)
     |  N x 17 x 2 keypoint sequence
     v
Temporal Smoothing Layer (VideoPose3D TCN)
     |  smooth 3D poses: N x 17 x 3
     v
Stick Figure Renderer
     |  visual parameters
     v
OUTPUT VIDEO
```

---

## Approaches Explored

### Pose Estimation

The following methods were surveyed and evaluated for suitability in the Kinemation pipeline. Evaluation criteria included real-time performance, keypoint accuracy, ease of integration with downstream modules, and community support.

#### Tools and Libraries Evaluated

| Tool | Type | Keypoints | Status |
|------|------|-----------|--------|
| MediaPipe BlazePose | 2D/3D, real-time | 33 | In use |
| OpenPose | 2D, bottom-up | 25 | Surveyed |
| OpenCV DNN | 2D, lightweight | 18 | Surveyed |
| AlphaPose | 2D, top-down | 17/26 | Surveyed |
| RTMPose | 2D, real-time | 17 | Planned |
| HRNet | 2D, top-down | 17 | Surveyed |
| ViTPose | 2D, transformer | 17 | Surveyed |

#### Key Papers Reviewed

- [*Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields*](https://arxiv.org/abs/1611.08050) - Cao et al. (CVPR 2017) - Introduces Part Affinity Fields for bottom-up multi-person pose estimation; the foundational OpenPose paper.
- [*Deep High-Resolution Representation Learning*](https://arxiv.org/abs/1908.07919) - Sun et al. (CVPR 2019) - HRNet maintains high-resolution feature maps throughout the network rather than recovering from downsampled representations.
- [*Stacked Hourglass Networks for Human Pose Estimation*](https://arxiv.org/abs/1603.06937) - Newell et al. (ECCV 2016) - Introduced the hourglass architecture and heatmap-based keypoint prediction, still widely used as a backbone.
- [*TransPose: Keypoint Localization via Transformer*](https://arxiv.org/abs/2012.14214) - Yang et al. (2021) - Uses transformer attention maps to explicitly show which image regions each keypoint attends to.
- [*ViTPose: Simple Vision Transformer Baselines*](https://arxiv.org/abs/2204.12484) - Xu et al. (NeurIPS 2022) - Applies a plain ViT backbone to pose estimation, achieving state-of-the-art results with a simple and scalable architecture.
- [*RTMPose: Real-Time Multi-Person Pose Estimation*](https://arxiv.org/abs/2303.07399) - Jiang et al. (2023) - Optimised for real-time multi-person inference; currently one of the strongest practical options for deployment.
- [*BlazePose: On-device Real-time Body Pose Tracking*](https://arxiv.org/abs/2006.10204) - Bazarevsky et al. (2020) - Google's on-device pose model used in MediaPipe; prioritises speed and cross-platform compatibility.

---

### Temporal Smoothing and Coherence

Temporal coherence is the primary active research focus. Three papers were studied in depth, representing distinct approaches to the problem.

#### Paper 1 - Temporal Bundle Adjustment

[*Exploiting Temporal Context for 3D Human Pose Estimation in the Wild*](https://arxiv.org/abs/1905.04668) - Arnab, Doersch & Zisserman (CVPR 2019)

Treats the entire video as a single global optimization problem. Per-frame 2D keypoints are fed into an HMR model to produce SMPL mesh estimates (beta - body shape, theta - joint angles). Bundle Adjustment then jointly minimizes a compound loss E = E_R + E_T + E_P across all frames simultaneously using L-BFGS. Body shape beta is held constant across frames to enforce anatomical consistency. Robust to noisy detections via Huber loss and camera shake via hinge loss.

- Strengths: globally consistent, principled, handles real-world noise well
- Limitations: requires full video upfront, computationally heavy, not suitable for real-time use

#### Paper 2 - Temporal Convolutional Network (Primary Implementation Target)

[*3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training*](https://arxiv.org/abs/1811.11742) - Pavllo et al. (CVPR 2019)

Takes a sequence of 2D keypoints (T x J x 2) and uses a dilated 1D TCN with exponentially increasing dilation rates (1, 2, 4, 8...) to infer smooth 3D poses for the center frame of each window. Temporal smoothness is learned from data rather than explicitly optimized. Includes a semi-supervised extension using a back-projection loss, enabling training on unlabeled video without 3D ground truth. Inference is a single forward pass, making it significantly more practical than bundle adjustment for a real pipeline.

- Strengths: near-real-time, detector-agnostic, plug-and-play with MediaPipe outputs, semi-supervised capability
- Limitations: less globally consistent than bundle adjustment; receptive field of 243 frames requires padding on short clips

#### Paper 3 - Bidirectional 2D Temporal Refinement

[*Deep Dual Consecutive Network for Human Pose Estimation*](https://arxiv.org/abs/2103.07254) - Liu et al. (CVPR 2021)

Operates purely in 2D. Takes a triplet of frames (t-k, t, t+k) and uses a Pose Temporal Merger (PTM) built on deformable convolutions to warp and align neighboring heatmaps to the target frame before merging. A Pose Refine Machine (PRM) then fuses the merged temporal features with the original single-frame heatmap to produce a corrected output. Occlusion recovery is an emergent property - joints hidden at frame t but visible at t+/-k are automatically recovered through the PTM-PRM pipeline without any explicit occlusion modeling.

- Strengths: stays in 2D, natural occlusion handling, architecturally elegant
- Limitations: operates on intermediate feature maps rather than final keypoint coordinates, making it non-trivial to integrate with arbitrary 2D detectors

#### Planned Upgrade - MotionBERT

[*MotionBERT: A Unified Perspective on Learning Human Motion Representations*](https://arxiv.org/abs/2210.06551) - Zhu et al. (ICCV 2023)

Transformer-based temporal model accepting the same N x 17 x 2 input format as VideoPose3D, making it a straightforward upgrade once the base temporal pipeline is established. Demonstrates consistent accuracy improvements over TCN-based approaches, particularly on fast motion and occluded joints.

---

## Work Completed

- Base 2D pose estimation pipeline using MediaPipe BlazePose, with per-frame keypoint extraction on arbitrary video input
- Stick figure renderer connecting skeletal joint detections to a 2D drawing layer with full bone connectivity
- Literature survey covering 35+ papers across pose estimation paradigms including top-down, bottom-up, heatmap-based, regression-based, transformer-based, 3D, and video-based approaches
- In-depth study of three temporal smoothing papers (Arnab et al., Pavllo et al., Liu et al.) covering global optimization, learned TCN-based smoothing, and bidirectional 2D refinement
- Keypoint adapter mapping MediaPipe's 33-landmark format to the Human3.6M 17-joint format required by VideoPose3D (in progress)
- VideoPose3D repository set up with dependencies resolved and pretrained checkpoint downloaded (in progress)

---

## Work In Progress

- Full video-to-3D-pose inference pipeline: MediaPipe extraction - H36M adapter - dilated TCN - smooth 3D output
- Jitter metric for quantitative evaluation of temporal smoothness (mean acceleration magnitude across joints) for before/after comparison
- 3D-to-2D back-projection to feed smooth poses back into the existing stick figure renderer
- Edge case handling: interpolation for missing detections, sequence padding for short clips, causal mode for faster inference

---

## Roadmap

**Phase 1 - Temporal Smoothing (current)**
Integrate VideoPose3D as the temporal smoothing backbone. Benchmark against raw MediaPipe output using the jitter metric. Produce side-by-side demo videos. Evaluate MotionBERT as an upgrade path once the baseline is stable.

**Phase 2 - 3D Stick Figure Rendering**
Extend the stick figure renderer to work in 3D space, enabling viewpoint manipulation and more expressive animation using Open3D or matplotlib 3D axes.

**Phase 3 - Applying Temporal Smoothing Techniques to Achieve Temporal Consistency of Video Outputs**

**Phase 4 - Evaluation and Demo**
End-to-end evaluation on a curated test set. Quantitative benchmarks on temporal smoothness, 3D accuracy, and emotion classification. Final demo production.

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/your-org/kinemation
cd kinemation

# Install dependencies
pip install -r requirements.txt

# Clone and set up VideoPose3D
git clone https://github.com/facebookresearch/VideoPose3D
cd VideoPose3D
pip install -r requirements.txt

# Download pretrained checkpoint
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
```

### Running the Base Pipeline

```bash
python run_pipeline.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Running with Temporal Smoothing

```bash
python run_pipeline.py --input path/to/video.mp4 --output path/to/output.mp4 --smooth --checkpoint pretrained_h36m_detectron_coco.bin
```

---

## Team

**Mentor:** [Maaya Mohan](https://github.com/maayamohan)

**Mentees:**
- [Amrita Pradeep](https://github.com/amritap0200)
- [Hemaksh Breja](https://github.com/Hemaksh-b)
- [Kurias Joji](https://github.com/kurj17)
- [Navya S Gurupadmath](https://github.com/Navya2022)
- [Saisree Vaishnavi](https://github.com/svr-arch)
- [Yatin R](https://github.com/T1n777)

---

## References

| Paper | Venue | Summary |
|-------|-------|---------|
| [OpenPose](https://arxiv.org/abs/1611.08050) - Cao et al. | CVPR 2017 | Bottom-up multi-person pose estimation using Part Affinity Fields to associate detected keypoints into skeletons |
| [Stacked Hourglass Networks](https://arxiv.org/abs/1603.06937) - Newell et al. | ECCV 2016 | Introduced the hourglass architecture and heatmap-based keypoint prediction that most modern pose estimators build on |
| [HRNet](https://arxiv.org/abs/1908.07919) - Sun et al. | CVPR 2019 | Maintains high-resolution feature representations throughout the network for more precise keypoint localisation |
| [ViTPose](https://arxiv.org/abs/2204.12484) - Xu et al. | NeurIPS 2022 | Plain vision transformer backbone for pose estimation, scalable and state-of-the-art on COCO benchmarks |
| [RTMPose](https://arxiv.org/abs/2303.07399) - Jiang et al. | 2023 | Real-time multi-person pose estimation optimised for practical deployment with strong accuracy-speed tradeoff |
| [BlazePose](https://arxiv.org/abs/2006.10204) - Bazarevsky et al. | 2020 | On-device real-time pose model powering MediaPipe; designed for mobile and cross-platform use |
| [Temporal Bundle Adjustment](https://arxiv.org/abs/1905.04668) - Arnab et al. | CVPR 2019 | Treats full video as a global optimization problem using SMPL meshes and L-BFGS to enforce temporal consistency |
| [VideoPose3D](https://arxiv.org/abs/1811.11742) - Pavllo et al. | CVPR 2019 | Dilated temporal convolutional network over 2D keypoint sequences for smooth 3D pose estimation in video |
| [DCPose](https://arxiv.org/abs/2103.07254) - Liu et al. | CVPR 2021 | Bidirectional 2D temporal refinement using deformable convolutions to recover occluded joints across neighboring frames |
| [MotionBERT](https://arxiv.org/abs/2210.06551) - Zhu et al. | ICCV 2023 | Transformer-based unified motion representation model, stronger than TCN-based approaches on fast motion and occlusion |