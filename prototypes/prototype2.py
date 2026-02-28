import cv2
import os

protofile = "models/pose_deploy_linevec.prototext"
weightsfile = "models/pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protofile,weightsfile)

img = cv2.imread("person6.png")

inHeight = 368
inWidth = int((inHeight/frameHeight)/frameWidth)

blob = cv2.dnn.blobFromImage(img, 1.0/255, (inWidth,inHeight), (0,0,0), swapRB = False, crop=False)

net.setInput(blob)

output = net.forward()

