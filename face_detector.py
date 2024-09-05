# face_detector.py
import cv2
import numpy as np
from scipy.ndimage import zoom

class FaceDetector:
    def __init__(self, haarcascade_path):
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return gray, detected_faces
    
    def extract_face_features(self, gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))
        extracted_face = gray[y + vertical_offset:y + h, x + horizontal_offset:x - horizontal_offset + w]
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face
