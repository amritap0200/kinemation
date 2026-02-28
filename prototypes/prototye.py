import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
protoFile = os.path.join(script_dir, 'models', 'pose_deploy_linevec.prototxt')
weightsFile = os.path.join(script_dir, 'models', 'pose_iter_440000.caffemodel')

print(f"Prototxt size: {os.path.getsize(protoFile)} bytes")
print(f"Caffemodel size: {os.path.getsize(weightsFile)} bytes")

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
print("Model loaded successfully!")

body_parts = { 0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle",
    14: "REye", 15: "LEye", 16: "REar", 17: "LEar"}

pose_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]]

img_path = os.path.join(script_dir, 'person.jpg')
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load image at {img_path}")
    exit()
h, w = img.shape[:2]
print(f"Image loaded: {w}x{h}")

inwidth, inheight = 368, 368
blob = cv2.dnn.blobFromImage(img, 1.0/255, (inwidth, inheight), (0,0,0), swapRB=False, crop=False)
print(f"Blob shape: {blob.shape}")

net.setInput(blob)
output = net.forward()
print(f"Output shape: {output.shape}")

num_points = 18
points = []
threshold = 0.1

for i in range(num_points):
    heatmap = output[0,i,:,:]
    _,confidence,_,point = cv2.minMaxLoc(heatmap)

    x = int((w*point[0])/output.shape[3])
    y = int((h*point[1])/output.shape[2])

    if confidence > threshold:
        points.append((x,y))
        cv2.circle(img,(x,y),8,(0,255,255),thickness = -1)
        cv2.putText(img,f'{i}',(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    else:
        points.append(None)

for pair in pose_pairs:
    partA, partB = pair
    if points[partA] and points[partB]:
        cv2.line(img, points[partA],points[partB],(0,255,0),3)

cv2.imshow("OpenPose Result",img)
cv2.imwrite("output2.png",img)
cv2.save
cv2.waitKey(0)
cv2.destroyAllWindows()