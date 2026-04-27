import os
import sys
import io
import logging
import warnings

# Suppress ALL MediaPipe/TFLite C++ noise before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel']      = '3'
os.environ['GLOG_logtostderr']      = '0'
warnings.filterwarnings('ignore')

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

class FaceAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Face Landmarker (modern API)"""
        # Get model path - download if not exists
        model_path = self._get_face_landmarker_model()

        # Create Face Landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )

        # Silence C++ glog/TFLite warnings at OS file-descriptor level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_fd2    = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        finally:
            os.dup2(old_fd2, 2)
            os.close(devnull_fd)
            os.close(old_fd2)

        # Landmark indices for different face regions
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 382, 374]
        self.LEFT_CHEEK = [234, 227, 216, 206, 205, 50]
        self.RIGHT_CHEEK = [454, 447, 446, 436, 425, 280]
        self.FOREHEAD = [10, 109, 67, 297, 338]
        self.NOSE = [6, 131, 48, 219, 279]
        self.CHIN = [175, 200, 152, 377, 400]

    def _get_face_landmarker_model(self):
        """Download face landmarker model if not present"""
        model_dir = os.path.expanduser('~/.mediapipe_models')
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'face_landmarker.task')

        # If model exists, return path
        if os.path.exists(model_path):
            return model_path

        # Download model
        print("[MediaPipe] Downloading face_landmarker.task model...")
        try:
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print(f"[MediaPipe] Model downloaded to {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download face_landmarker.task: {e}\n"
                f"Download manually from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task\n"
                f"Place it in: {model_path}"
            )

        return model_path

    def detect_landmarks(self, image):
        """Detect face landmarks using modern MediaPipe API"""
        h, w = image.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        results = self.face_landmarker.detect(mp_image)

        # Check if landmarks were detected
        if not results.face_landmarks:
            return None

        # Extract landmarks from first detected face
        landmarks = results.face_landmarks[0]
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        return points

    def compute_shadow_map(self, image):
        """
        Detect shadows by analyzing brightness in face regions.
        Dark regions = high shadow intensity.
        """
        h, w = image.shape[:2]
        landmarks = self.detect_landmarks(image)

        if landmarks is None:
            return np.ones((h, w), dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shadow_map = np.ones((h, w), dtype=np.float32)

        regions = {
            'left_eye': self.LEFT_EYE,
            'right_eye': self.RIGHT_EYE,
            'left_cheek': self.LEFT_CHEEK,
            'right_cheek': self.RIGHT_CHEEK,
            'forehead': self.FOREHEAD,
            'nose': self.NOSE,
            'chin': self.CHIN
        }

        for region_name, indices in regions.items():
            region_points = landmarks[indices].astype(np.int32)
            region_points = np.clip(region_points, 0, [w-1, h-1])

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [region_points], 1)

            region_gray = gray * mask
            region_sum = region_gray.sum()
            region_pixels = mask.sum()

            if region_pixels > 0:
                avg_brightness = region_sum / region_pixels
                shadow_intensity = 1.0 - avg_brightness
                shadow_map[mask > 0] = shadow_intensity

        shadow_map = cv2.GaussianBlur(shadow_map, (15, 15), 0)
        shadow_map = np.clip(shadow_map, 0, 1)

        return shadow_map

    def compute_light_direction_map(self, image):
        """
        Detect light direction by comparing left/right sides.
        Returns map indicating where light is coming from.
        """
        h, w = image.shape[:2]
        landmarks = self.detect_landmarks(image)

        if landmarks is None:
            return np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        light_map = np.zeros((h, w), dtype=np.float32)

        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]

        left_brightness = left_half.mean()
        right_brightness = right_half.mean()

        if left_brightness > right_brightness:
            light_direction = 1.0
            bright_area = left_half > left_half.mean()
            light_map[:, :w//2][bright_area] = 1.0
        else:
            light_direction = -1.0
            bright_area = right_half > right_half.mean()
            light_map[:, w//2:][bright_area] = 1.0

        light_map = cv2.GaussianBlur(light_map, (21, 21), 0)

        return light_map

    def compute_face_confidence_map(self, image):
        """
        Create a confidence map for face regions.
        Face = 1.0, edges fade to 0, background = 0
        """
        h, w = image.shape[:2]
        landmarks = self.detect_landmarks(image)

        if landmarks is None:
            return np.zeros((h, w), dtype=np.float32)

        face_contour = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]

        face_points = np.array([landmarks[i] for i in face_contour if i < len(landmarks)], dtype=np.int32)

        # Extend forehead upward
        face_top = face_points[:, 1].min()
        face_bottom = face_points[:, 1].max()
        face_height = face_bottom - face_top
        forehead_ext = int(face_height * 0.18)
        threshold_y = face_top + int(face_height * 0.40)

        for i in range(len(face_points)):
            if face_points[i, 1] <= threshold_y:
                face_points[i, 1] = max(0, face_points[i, 1] - forehead_ext)

        confidence_map = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(confidence_map, [face_points], 1.0)
        confidence_map = cv2.GaussianBlur(confidence_map, (15, 15), 0)

        return confidence_map

    def create_analysis_map(self, image):
        """
        Combine shadow, light direction, and face confidence maps.
        Returns a 3-channel map for the model.
        """
        shadow_map = self.compute_shadow_map(image)
        light_map = self.compute_light_direction_map(image)
        confidence_map = self.compute_face_confidence_map(image)

        analysis_map = np.stack([shadow_map, light_map, confidence_map], axis=2)

        return analysis_map.astype(np.float32)
