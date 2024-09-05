import cv2
import numpy as np
from scipy.ndimage import zoom

class FaceDetector:
    """
    Class for detecting and extracting facial features from an image using a Haar cascade classifier.

    Attributes:
    face_cascade: The Haar cascade classifier for face detection.

    Methods:
    __init__(self, haarcascade_path):
        Initializes the class with the path to the Haar cascade classifier.

    detect_faces(self, frame):
        Detects faces in the provided image and converts the image to grayscale.

    extract_face_features(self, gray, detected_face, offset_coefficients):
        Extracts and normalizes facial features from the detected face region.

    Parameters:
    haarcascade_path (str): Path to the Haar cascade XML file.
    frame (numpy.ndarray): Image where faces will be detected.
    gray (numpy.ndarray): Grayscale image used for feature extraction.
    detected_face (tuple): Coordinates of the detected face in the image.
    offset_coefficients (tuple): Offset coefficients to adjust the extracted face region.
    """

    def __init__(self, haarcascade_path):
        """
        Initializes the class with the path to the Haar cascade classifier.

        Parameters:
        haarcascade_path (str): Path to the Haar cascade XML file.
        """
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
    
    def detect_faces(self, frame):
        """
        Detects faces in the provided image and converts the image to grayscale.

        Parameters:
        frame (numpy.ndarray): Image where faces will be detected.

        Returns:
        tuple: Grayscale image and coordinates of the detected faces.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return gray, detected_faces
    
    def extract_face_features(self, gray, detected_face, offset_coefficients):
        """
        Extracts and normalizes facial features from the detected face region.

        Parameters:
        gray (numpy.ndarray): Grayscale image used for feature extraction.
        detected_face (tuple): Coordinates of the detected face in the image.
        offset_coefficients (tuple): Offset coefficients to adjust the extracted face region.

        Returns:
        numpy.ndarray: Normalized facial features extracted from the face region.
        """
        (x, y, w, h) = detected_face
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))
        extracted_face = gray[y + vertical_offset:y + h, x + horizontal_offset:x - horizontal_offset + w]
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face
