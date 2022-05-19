import cv2
import numpy as np

cap = cv2.VideoCapture('resources/face.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

frame_shape   = (int(cap.get(3)), int(cap.get(4)))
display_shape = (frame_shape[0] // 2, frame_shape[1] // 2)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('App', cv2.resize(frame, display_shape))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()