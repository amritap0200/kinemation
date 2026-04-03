\# OpenPose Setup and Configuration



\## Local Attempt



Initial attempts to configure OpenPose locally were slow performance wise and wasn't viable for training and testing, as:



\- CUDA GPU dependency

\- Lack of dedicated GPU

\- Compatibility issues during configuration



OpenPose requires CUDA-enabled hardware for practical real-time performance.



---



\## Google Colab Setup



Environment:

\- Google Colab

\- Tesla T4 GPU

\- CUDA 12.x

\- cuDNN enabled



Steps:

1\. Clone OpenPose repository

2\. Install dependencies

3\. Configure CMake

4\. Build Caffe backend

5\. Compile OpenPose

6\. Download pretrained models

7\. Test image inference

8\. Test video inference



All build steps completed successfully on GPU environment.

