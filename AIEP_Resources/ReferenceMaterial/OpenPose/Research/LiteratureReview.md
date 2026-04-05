\# Literature Review 



Research on OpenPose, it's underlying mechanisms of pose estimation, center-keypoint grouping, continuous heatmap regression vs discrete heatmap prediction, and alternate approaches, as well as the use of TCNs for further temporal stabilization.



\## 1. \[OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1812.08008)



OpenPose introduces a bottom-up approach for multi-person 2D pose estimation using Part Affinity Fields (PAFs). Unlike top-down approaches that detect individuals first and then estimate pose, OpenPose detects keypoints globally and then associates them using learned vector fields.



Key Contributions:

\- Real-time multi-person detection

\- Bottom-up architecture

\- Efficient grouping using PAFs

\- GPU-accelerated performance



Strengths:

\- Robust multi-person tracking

\- Real-time capability with GPU

\- Open-source and widely adopted



Limitations:

\- High GPU dependency

\- Slower performance on CPU

\- Large memory usage for high resolutions



---



\## 2. \[The Center of Attention: Center-Keypoint Grouping via Attention for Multi-Person Pose Estimation](https://arxiv.org/pdf/2110.05132)



This approach improves grouping mechanisms by using attention-based strategies to associate keypoints more robustly in crowded scenes.



Relevance:

\- Addresses grouping weaknesses in bottom-up pipelines

\- Improves robustness in complex scenes



---



\## 3. \[Continuous Heatmap Regression for Pose Estimation via Implicit Neural Representation (https://proceedings.neurips.cc/paper\_files/paper/2024/file/b90cb10d4dae058dd167388e76168c1b-Paper-Conference.pdf)



This paper proposes continuous heatmap regression instead of discrete heatmap prediction, improving spatial precision and reducing quantization errors.



Relevance:

\- Higher precision pose estimation

\- More stable regression formulation



---



\## 4. \[Temporal Convolutional Networks and Forecasting](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)

&nbsp;     \[Temporal Convolutional Network](https://www.sciencedirect.com/topics/computer-science/temporal-convolutional-network)



TCNs are used for sequence modeling and forecasting. In pose estimation, they are useful for smoothing pose predictions over time in videos.



Relevance:

\- Useful for temporal stabilization

\- Can reduce jitter in frame-by-frame inference



---



\## 5. \[Bottom-up approaches - Hierarchical Graphical Clustering (HGG)](https://www.mdpi.com/1424-8220/23/7/3725#:~:text=Bottom%2Dup%20approaches%20face%20the,Hierarchical%20Graphical%20Clustering%20(HGG)).



Hierarchical Graphical Clustering improves grouping efficiency in multi-person scenarios.



Relevance:

\- Alternative grouping strategy

\- May offer improvements in crowded scenes

