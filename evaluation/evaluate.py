import sys
sys.path.insert(0, '.')

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from models_Relighty import UNet
from utils import FaceAligner
from data import FaceDataset
from torch.utils.data import DataLoader
from utils.losses import SSIMLoss

def calculate_metrics(pred, target):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    pred_norm = pred / 255.0
    target_norm = target / 255.0
    std_pred = np.std(pred_norm)
    std_target = np.std(target_norm)
    ssim = (1 - np.mean((pred_norm - target_norm) ** 2) / (std_pred * std_target + 1e-8)) * 0.5 + 0.5
    return {'MAE': mae, 'MSE': mse, 'SSIM': ssim}

def evaluate(checkpoint_path, input_dir, target_dir, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    aligner = FaceAligner()
    dataset = FaceDataset(input_dir, target_dir, aligner, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    ssim_fn = SSIMLoss().to(device)
    metrics_sum = {'MAE': 0, 'MSE': 0, 'SSIM': 0}
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    print(f"Evaluating {len(dataset)} images...")
    with torch.no_grad():
        for i, (inputs, targets, masks) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            for j in range(outputs.shape[0]):
                pred_img = aligner.tensor_to_image(outputs[j])
                target_img = aligner.tensor_to_image(targets[j])
                m = calculate_metrics(pred_img, target_img)
                for k in metrics_sum:
                    metrics_sum[k] += m[k]
                if output_dir:
                    idx = i * 8 + j
                    if idx < len(dataset.files):
                        cv2.imwrite(os.path.join(output_dir, dataset.files[idx]), pred_img)
    n = len(dataset)
    print(f"\nResults ({checkpoint_path}):")
    print(f"MAE: {metrics_sum['MAE']/n:.4f}")
    print(f"MSE: {metrics_sum['MSE']/n:.4f}")
    print(f"SSIM: {metrics_sum['SSIM']/n:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("-i", "--input", default="dataset_no_bg/val/input")
    parser.add_argument("-t", "--target", default="dataset_no_bg/val/target")
    parser.add_argument("-o", "--output", default="evaluation/outputs")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.input, args.target, args.output)