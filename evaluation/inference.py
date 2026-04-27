import sys
sys.path.insert(0, '.')

import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from models_Relighty import UNet
from models_Relighty.unet_v2 import UNetV2
from utils.face_analyzer import FaceAnalyzer

class ShadowRemovalModel:
    def __init__(self, checkpoint_path="checkpoints/best_model.pth", use_v2=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_analyzer = None
        self.use_v2 = False

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            model_state = checkpoint['model_state_dict']

            has_attention = any('attention' in k for k in model_state.keys())
            has_bottleneck = any('bottleneck' in k for k in model_state.keys())

            if has_attention or has_bottleneck:
                print("✓ Detected: UNetV2 checkpoint (Face-Aware, 6-channel input)")
                self.model = UNetV2(input_channels=6, output_channels=3).to(self.device)
                self.face_analyzer = FaceAnalyzer()
                self.use_v2 = True
            else:
                print("✓ Detected: UNet checkpoint (Legacy, 3-channel input)")
                self.model = UNet().to(self.device)
                self.use_v2 = False

            try:
                self.model.load_state_dict(model_state)
                print(f"✓ Loaded: {checkpoint_path}")
            except RuntimeError as e:
                print(f"✗ Failed to load checkpoint: {e}")
                print(f"Creating untrained model...")
                if self.use_v2:
                    self.model = UNetV2(input_channels=6, output_channels=3).to(self.device)
                    self.face_analyzer = FaceAnalyzer()
                else:
                    self.model = UNet().to(self.device)
        else:
            print(f"✗ No checkpoint found: {checkpoint_path}")
            print(f"Using untrained model. Please train first:")
            print(f"  python train/scripts/main.py")
            self.model = UNetV2(input_channels=6, output_channels=3).to(self.device)
            self.face_analyzer = FaceAnalyzer()
            self.use_v2 = True

        self.model.eval()
        self.output_size = 256
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def create_face_neck_mask(self, image):
        if self.face_analyzer is None:
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        return self.face_analyzer.compute_face_confidence_map(image)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None, None, None
        x, y, w, h = faces[0]
        face_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 4)
        if len(eyes) >= 2:
            ex1, ey1, _, _ = eyes[0]
            ex2, ey2, _, _ = eyes[1]
            left_eye = np.array([x + ex1, y + ey1], dtype=np.float64)
            right_eye = np.array([x + ex2, y + ey2], dtype=np.float64)
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye
        else:
            cx, cy = x + w/2, y + h/2
            left_eye = np.array([cx - w*0.3, cy - h*0.15], dtype=np.float64)
            right_eye = np.array([cx + w*0.3, cy - h*0.15], dtype=np.float64)
        src_center = (left_eye + right_eye) / 2
        dst_center = np.array([112.0, 115.0], dtype=np.float64)
        eye_dist = np.linalg.norm(right_eye - left_eye)
        dst_dist = np.linalg.norm(np.array([152.0, 115.0]) - np.array([112.0, 115.0]))
        scale = dst_dist / eye_dist if eye_dist > 0 else 1.0
        tx = dst_center[0] - scale * src_center[0]
        ty = dst_center[1] - scale * src_center[1]
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float64)
        aligned = cv2.warpAffine(image, M, (self.output_size, self.output_size),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        raw_mask = self.create_face_neck_mask(aligned)
        mask = cv2.warpAffine(raw_mask, M, (self.output_size, self.output_size),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = np.clip(mask, 0, 1).astype(np.float32)
        tensor = self.to_tensor(aligned)
        return tensor, aligned, mask
    
    def to_tensor(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    
    def tensor_to_image(self, tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    def process(self, image):
        tensor, aligned, mask = self.preprocess(image)
        if tensor is None:
            print("No face detected, returning original")
            return image
        tensor = tensor.to(self.device)

        with torch.no_grad():
            if self.use_v2 and self.face_analyzer:
                analysis_map = self.face_analyzer.create_analysis_map(aligned)
                analysis_tensor = torch.from_numpy(analysis_map).permute(2, 0, 1).unsqueeze(0).to(self.device)
                output = self.model(tensor, analysis_tensor)
            else:
                output = self.model(tensor)

        result = self.tensor_to_image(output)
        if mask is not None:
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            result = (result * mask_3ch + aligned * (1 - mask_3ch)).astype(np.uint8)
        return result

def process_directory(input_dir, output_dir, checkpoint="checkpoints/best_model.pth"):
    os.makedirs(output_dir, exist_ok=True)
    model = ShadowRemovalModel(checkpoint, use_v2=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nProcessing {len(files)} images from {input_dir}")
    print(f"Model: {'UNetV2 (Face-Aware)' if model.use_v2 else 'UNet (Legacy)'}\n")
    for f in files:
        image = cv2.imread(os.path.join(input_dir, f))
        if image is None:
            print(f"✗ Failed to load: {f}")
            continue
        try:
            result = model.process(image)
            name, ext = os.path.splitext(f)
            output_name = f"{name}_test{ext}"
            cv2.imwrite(os.path.join(output_dir, output_name), result)
            print(f"✓ Saved: {output_name}")
        except Exception as e:
            print(f"✗ Error {f}: {e}")
    print("\n✓ Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="Results/input", help="Input folder")
    parser.add_argument("-o", "--output", default="Results/output", help="Output folder")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/best_model.pth")
    args = parser.parse_args()
    
    os.makedirs(args.input, exist_ok=True)
    process_directory(args.input, args.output, args.checkpoint)