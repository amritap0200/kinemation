\# OpenPose Evaluation and Documentation



\## Overview



This section documents the research, setup process, evaluation, and performance analysis of OpenPose for multi-person 2D pose estimation within the Kinemation project.



OpenPose is a bottom-up real-time pose estimation framework that uses Part Affinity Fields (PAFs) to associate detected keypoints with individuals in multi-person scenes.



Repository:

https://github.com/CMU-Perceptual-Computing-Lab/openpose



Primary Paper:

\[OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1812.08008)



---



\## Environment Used



Local setup attempts were time consuming and relied on CUDA GPU, hence final evaluation was performed using:



->Google Colab

&nbsp;  - Tesla T4 GPU

&nbsp;  - CUDA-enabled configuration

&nbsp;  - Pretrained BODY\_25 model



---



\## Evaluation Summary



\- Image inference successful

\- Video inference successful

\- Multi-person detection functional

\- GPU acceleration required for practical performance

\- CPU-only execution significantly slower and not viable for real-time use



---



\## Deployment Considerations



OpenPose performs efficiently with CUDA-enabled GPU hardware.



For pipeline integration and deployment, GPU availability is strongly recommended, as CPU-based systems introduce severe performance degradation.



Further performance analysis is documented in the Results section.

