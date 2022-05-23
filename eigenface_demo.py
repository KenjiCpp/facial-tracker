import numpy as np
import cv2
from utils import FaceNormalizer, Eigenface
from os.path import exists

fm = FaceNormalizer()
ef = Eigenface()

if exists('resources/eigenface_texture_generator.dat'):
    print('>>> Loading existing Eigenface Texture Generator...')
    ef.load('resources/eigenface_texture_generator.dat')
else:
    print('>>> Creating and fitting new Eigenface Texture Generator...')
    cap = cv2.VideoCapture('resources/face.mp4')
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    normalize_faces = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        normalize_faces.append(fm.normalize(cv2.resize(frame, np.array(frame.shape[1::-1]) // 8)).flatten())
    cap.release()
    ef.fit(normalize_faces)
    ef.save('resources/eigenface_texture_generator.dat')

while True:
    cv2.imshow("Demo Eigenface - Mean", cv2.resize(ef.mean_img, (360, 360)))
    if cv2.waitKey(25) & 0xFF == ord('q'):        
        cv2.destroyAllWindows()
        break
for i in range(65):
    img = ef.eigenfaces[i]
    while True:
        cv2.imshow("Demo Eigenface - Eigenvector", cv2.resize(img / np.max(img), (360, 360)))
        if cv2.waitKey(25) & 0xFF == ord('q'):        
            break
cv2.destroyAllWindows()
