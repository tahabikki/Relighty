import sys
import os
import warnings

# ── Path setup ───────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Suppress ALL warnings before any imports ─────────────────────────────────
# TensorFlow / MediaPipe C++ logs (glog)
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['GLOG_minloglevel']         = '3'   # suppress W/I/E from C++ MediaPipe
os.environ['GLOG_logtostderr']         = '0'

# PyTorch CUDA memory allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # safe alternative

# Suppress Python-level warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Redirect stderr to swallow remaining C++ noise during imports
import io
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()

import gc
import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import tempfile
import shutil
import logging

# Restore stderr after imports
sys.stderr = _stderr_backup

# Suppress Python logging for noisy libs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from data import FaceDataset
from models_Relighty.shadow_remover import create_model
from utils.shadow_loss import ShadowRemovalLoss


def train(model_name='unet'):
    print("=" * 60)
    print("SHADOW REMOVAL - TRAINING")
    print("=" * 60)

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config.get('training', {})
    loss_cfg  = config.get('loss',     {})
    data_cfg  = config.get('data',     {})
    ckpt_cfg  = config.get('checkpoint', {})

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        if train_cfg.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            print(f"cuDNN benchmark : ON")

    # ── Hyper-params (all from config) ──────────────────────────────────
    batch_size         = train_cfg.get('batch_size',         8)
    learning_rate      = train_cfg.get('learning_rate',      5e-5)
    encoder_lr_factor  = train_cfg.get('encoder_lr_factor',  0.1)
    epochs             = train_cfg.get('epochs',             150)
    gradient_clip      = train_cfg.get('gradient_clip',      1.0)
    use_amp            = train_cfg.get('mixed_precision',     True) and torch.cuda.is_available()
    scheduler_patience = train_cfg.get('scheduler_patience', 10)
    scheduler_factor   = train_cfg.get('scheduler_factor',   0.5)
    scheduler_min_lr   = train_cfg.get('scheduler_min_lr',   1e-6)

    num_workers        = data_cfg.get('num_workers',         2)
    pin_memory         = data_cfg.get('pin_memory',          True)
    persistent_workers = data_cfg.get('persistent_workers',  True) and num_workers > 0

    print(f"Batch size     : {batch_size}")
    print(f"Learning rate  : {learning_rate}  (encoder × {encoder_lr_factor})")
    print(f"Epochs         : {epochs}")
    print(f"Workers        : {num_workers}")
    print(f"Mixed precision: {'ON  (~2x faster)' if use_amp else 'OFF'}")

    # ── Data ────────────────────────────────────────────────────────────
    split_dir  = config.get('data', {}).get('split_dir', 'split/splited_dataset_data')
    train_list = os.path.join(split_dir, "train_input.txt")
    val_list   = os.path.join(split_dir, "val_input.txt")

    if not os.path.exists(train_list):
        print(f"ERROR: {train_list} not found — run split/split_files.py first")
        return

    print("\nLoading datasets...")
    output_size = config.get('data', {}).get('output_size', 256)
    train_ds = FaceDataset(train_list, output_size=output_size, augment=True)
    val_ds   = FaceDataset(val_list,   output_size=output_size, augment=False)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = create_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    model_display = model_name.upper() if model_name.lower() == 'resunet++' else 'ShadowRemovalNet'
    print(f"\nModel : {model_display}  ({total_params/1e6:.1f}M params)")

    # ── Loss ────────────────────────────────────────────────────────────
    loss_fn = ShadowRemovalLoss(
        l1_weight         = loss_cfg.get('l1_weight',         1.00),
        perceptual_weight = loss_cfg.get('perceptual_weight', 0.10),
        ssim_weight       = loss_cfg.get('ssim_weight',       0.00),
        identity_weight   = loss_cfg.get('identity_weight',   0.00),
        color_weight      = loss_cfg.get('color_weight',      0.00),
    )

    # ── Optimizer (encoder slower, decoder full LR) ──────────────────────
    # Get encoder parameters (works for both UNet and ResUNet++)
    enc_param_list = []
    for attr in ['enc0', 'pool', 'enc1', 'enc2', 'enc3', 'enc4']:
        if hasattr(model, attr):
            enc_param_list.extend(getattr(model, attr).parameters())
    enc_ids    = {id(p) for p in enc_param_list}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    enc_params = list(enc_param_list)

    optimizer = optim.Adam([
        {'params': enc_params, 'lr': learning_rate * encoder_lr_factor},
        {'params': dec_params, 'lr': learning_rate},
    ], betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor   = scheduler_factor,
        patience = scheduler_patience,
        min_lr   = scheduler_min_lr,   # never go below this — keeps model trainable
    )

    # ── Mixed Precision Scaler (controlled by config) ────────────────────
    scaler = GradScaler(device='cuda', enabled=use_amp)

    # ── Resume ──────────────────────────────────────────────────────────
    ckpt_dir   = ckpt_cfg.get('dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss  = float('inf')
    start_epoch = 0

    ckpt_prefix = f"shadow_removal_{model_name.lower()}"
    latest = os.path.join(ckpt_dir, f"{ckpt_prefix}_latest.pth")
    if os.path.exists(latest):
        # Load on CPU first to avoid CUDA memory fragmentation
        gc.collect()
        torch.cuda.empty_cache()
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"\nResumed from epoch {start_epoch}  (best loss {best_loss:.6f})")

    print("\n" + "=" * 60)
    print("Training started...")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Training Loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, masks, _ in pbar:
            inputs  = inputs.to(device,  non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks   = masks.to(device,   non_blocking=True)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            # ── Forward pass under AMP ──────────────────────────────────
            with autocast(device_type='cuda', enabled=use_amp):
                output = model(inputs)
                loss   = loss_fn(output, targets, masks)

            # ── Backward pass with scaler ───────────────────────────────
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})
        gc.collect()

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks, _ in val_loader:
                inputs  = inputs.to(device,  non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                masks   = masks.to(device,   non_blocking=True)
                with autocast(device_type='cuda', enabled=use_amp):
                    output = model(inputs)
                    loss   = loss_fn(output, targets, masks)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        is_best     = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        dec_lr = optimizer.param_groups[1]['lr']
        print(f"\n{'='*55}")
        print(f"  Epoch   : {epoch+1} / {epochs}")
        print(f"  Train   : {train_loss:.5f}")
        print(f"  Val     : {val_loss:.5f}  {'<-- NEW BEST' if is_best else ''}")
        print(f"  Best    : {best_loss:.5f}")
        print(f"  LR      : encoder={optimizer.param_groups[0]['lr']:.1e}  decoder={dec_lr:.1e}")
        print(f"{'='*55}")

        # ── Save Checkpoints ─────────────────────────────────────────────
        ckpt = {
            'epoch'              : epoch,
            'model_name'         : model_name,
            'model_state_dict'   : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict'  : scaler.state_dict(),
            'best_loss'          : best_loss,
        }
        
        def save_checkpoint_safe(filepath, checkpoint):
            """Save checkpoint using atomic write with temporary file."""
            try:
                # Free memory before saving
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create parent directory if needed
                os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
                
                # Save to temporary file first
                temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath) or '.', 
                                                       prefix='.tmp_', suffix='.pth')
                os.close(temp_fd)
                
                try:
                    torch.save(checkpoint, temp_path)
                    # Atomic move
                    shutil.move(temp_path, filepath)
                    return True
                except Exception as temp_err:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    raise temp_err
            except Exception as e:
                return False
        
        try:
            if save_checkpoint_safe(os.path.join(ckpt_dir, f"{ckpt_prefix}_latest.pth"), ckpt):
                pass  # Latest saved successfully
            else:
                print(f"  [WARN] failed to save latest checkpoint")

            if is_best:
                if save_checkpoint_safe(os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth"), ckpt):
                    print("  [BEST] saved best model")
                else:
                    print(f"  [WARN] failed to save best checkpoint")

            if (epoch + 1) % ckpt_cfg.get('save_every', 10) == 0:
                if save_checkpoint_safe(os.path.join(ckpt_dir,
                           f"{ckpt_prefix}_epoch_{epoch+1:03d}.pth"), ckpt):
                    print(f"  [CKPT] saved epoch {epoch+1}")
                else:
                    print(f"  [WARN] failed to save epoch {epoch+1} checkpoint")
        except Exception as e:
            print(f"  [WARN] checkpoint operation failed: {e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss : {best_loss:.6f}")
    print(f"Best model    : checkpoints/{ckpt_prefix}_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train shadow removal model')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'resunet++'],
                        help='Model architecture: unet (29M params) or resunet++ (50M params, 2x better quality)')
    args = parser.parse_args()

    train(model_name=args.model)
