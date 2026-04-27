import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import numpy as np
from rembg import remove, new_session

MODEL_PATH = "BiRefNet-portrait-epoch_150.onnx"

def remove_bg(input_path, output_path=None):
    with open(input_path, 'rb') as f:
        input_image = f.read()
    
    try:
        if os.path.exists(MODEL_PATH):
            session = new_session(MODEL_PATH)
        else:
            session = new_session("birefnet-portrait")
        output = remove(input_image, session=session)
        
        nparr = np.frombuffer(output, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        if len(img.shape) > 2 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            mask = alpha.astype(np.float32) / 255.0
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = mask.astype(np.float32) / 255.0
            mask = np.clip(mask, 0.05, 1.0)
            rgb = img[:, :, :3]
            result = np.zeros_like(rgb)
            for c in range(3):
                result[:, :, c] = (rgb[:, :, c].astype(np.float32) * mask).astype(np.uint8)
            result = np.dstack([result, (mask * 255).astype(np.uint8)])
        else:
            result = img
        
        if output_path:
            png_path = os.path.splitext(output_path)[0] + '.png'
            cv2.imwrite(png_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return png_path
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing {len(files)} images...")
    
    for i, f in enumerate(files):
        try:
            remove_bg(os.path.join(input_dir, f), os.path.join(output_dir, f))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} done")
        except Exception as e:
            print(f"Error {f}: {e}")
    
    print(f"Done: {len(files)} images")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found!")
        return
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {MODEL_PATH}")
    
    os.makedirs(args.output, exist_ok=True)
    
    for sub in ['input', 'target']:
        inp = os.path.join(args.input, sub)
        out = os.path.join(args.output, sub)
        if os.path.exists(inp):
            print(f"\n{'='*50}")
            print(f"Processing: {sub}/")
            os.makedirs(out, exist_ok=True)
            process_folder(inp, out)
        else:
            print(f"Skip: {sub}/ not found")

if __name__ == "__main__":
    main()