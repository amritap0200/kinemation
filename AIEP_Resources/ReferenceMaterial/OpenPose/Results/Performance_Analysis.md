\# Performance Analysis



Hardware:

\- Tesla T4 GPU (Colab)



Model:

\- BODY\_25 pretrained model



Observations:



\- Real-time inference achievable with GPU

\- Multi-person detection stable

\- Video processing successful

\- CPU-based performance significantly slower

\- GPU strongly recommended for deployment



Resolution Testing:

Lower net resolution improves speed

Higher net resolution improves accuracy but increases latency



Conclusion:

OpenPose is viable for real-time applications only when GPU acceleration is available.

