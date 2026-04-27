import cv2
import numpy as np

class FaceAligner:
    def __init__(self, output_size=256):
        self.output_size = output_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.template = np.array([
            [112.0, 115.0], [152.0, 115.0], [80.0, 160.0], [184.0, 160.0], [132.0, 190.0]
        ], dtype=np.float64)
    
    def detect_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None, None
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        if len(eyes) < 2:
            cx, cy = x + w/2, y + h/2
            left_eye = np.array([cx - w*0.3, cy - h*0.1], dtype=np.float64)
            right_eye = np.array([cx + w*0.3, cy - h*0.1], dtype=np.float64)
        else:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            left_eye = np.array([x + ex1 + ew1/2, y + ey1 + eh1/2], dtype=np.float64)
            right_eye = np.array([x + ex2 + ew2/2, y + ey2 + eh2/2], dtype=np.float64)
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye
        nose = np.array([x + w*0.5, y + h*0.55], dtype=np.float64)
        mouth_left = np.array([x + w*0.35, y + h*0.75], dtype=np.float64)
        mouth_right = np.array([x + w*0.65, y + h*0.75], dtype=np.float64)
        key_points = np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float64)
        all_landmarks = [
            [left_eye[0]/image.shape[1], left_eye[1]/image.shape[0]],
            [right_eye[0]/image.shape[1], right_eye[1]/image.shape[0]],
            [nose[0]/image.shape[1], nose[1]/image.shape[0]],
            [mouth_left[0]/image.shape[1], mouth_left[1]/image.shape[0]],
            [mouth_right[0]/image.shape[1], mouth_right[1]/image.shape[0]],
            [x/image.shape[1], y/image.shape[0]],
            [(x+w)/image.shape[1], y/image.shape[0]],
            [(x+w)/image.shape[1], (y+h)/image.shape[0]],
            [x/image.shape[1], (y+h)/image.shape[0]]
        ]
        return key_points, all_landmarks
    
    def compute_transform(self, src_points):
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(self.template, axis=0)
        src_dist = np.mean(np.linalg.norm(src_points - src_center, axis=1))
        dst_dist = np.mean(np.linalg.norm(self.template - dst_center, axis=1))
        scale = dst_dist / src_dist if src_dist > 0 else 1.0
        tx = dst_center[0] - scale * src_center[0]
        ty = dst_center[1] - scale * src_center[1]
        return np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float64)
    
    def align_image(self, image, M):
        return cv2.warpAffine(image, M, (self.output_size, self.output_size), 
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def generate_mask(self, landmarks, M):
        size = self.output_size
        h, w = size, size
        src_points = np.array([
            [landmarks[5][0]*w, landmarks[5][1]*h],
            [landmarks[6][0]*w, landmarks[6][1]*h],
            [landmarks[7][0]*w, landmarks[7][1]*h],
            [landmarks[8][0]*w, landmarks[8][1]*h]
        ], dtype=np.float64)
        dst = np.float64([[0,0], [w,0], [w,h], [0,h]])
        M_mask = cv2.getPerspectiveTransform(src_points, dst)
        mask = np.zeros((size, size), dtype=np.uint8)
        pts = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.int32)
        cv2.fillConvexHull(mask, pts, 255)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask / 255.0
    
    def align_pair(self, input_img, target_img):
        key_points, all_landmarks = self.detect_landmarks(input_img)
        if key_points is None:
            return None, None, None
        M = self.compute_transform(key_points)
        aligned_input = self.align_image(input_img, M)
        aligned_target = self.align_image(target_img, M)
        mask = np.ones((self.output_size, self.output_size), dtype=np.float64)
        return aligned_input, aligned_target, mask
    
    def align_single(self, image):
        key_points, _ = self.detect_landmarks(image)
        if key_points is None:
            return None, None
        M = self.compute_transform(key_points)
        return self.align_image(image, M), M
    
    def augment(self, image):
        if np.random.rand() < 0.3:
            image = np.clip(image * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.7, 1.3)
            image = np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            h, w = image.shape[:2]
            grad = np.linspace(1.0, 0.4, w if np.random.randint(2) else h)
            if grad.ndim == 1:
                grad = grad.reshape(1, -1) if h > w else grad.reshape(-1, 1)
            image = (image * grad[:, :, np.newaxis]).astype(np.uint8)
        return image
    
    def to_tensor(self, image):
        import torch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(image).permute(2, 0, 1).float()
    
    def tensor_to_image(self, tensor):
        import torch
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)