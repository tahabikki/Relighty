import torch
import torch.nn as nn
import numpy as np

class ColorConsistencyLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        pred_mean = pred.mean(dim=[2, 3], keepdim=True)
        target_mean = target.mean(dim=[2, 3], keepdim=True)
        pred_std = pred.std(dim=[2, 3], keepdim=True)
        target_std = target.std(dim=[2, 3], keepdim=True) + 1e-8

        pred_normalized = (pred - pred_mean) / (pred_std + 1e-8)
        target_normalized = (target - target_mean) / target_std

        loss = nn.functional.mse_loss(pred_normalized, target_normalized)

        if mask is not None:
            mask_expanded = mask.expand_as(pred)
            loss = (nn.functional.mse_loss(pred_normalized * mask_expanded, target_normalized * mask_expanded, reduction='none').mean() * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        self.register_buffer('kernel', kernel)

    def forward(self, pred, target, mask=None):
        kernel = self.kernel.to(pred.device)
        loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            pred_edge = nn.functional.conv2d(pred_c, kernel.unsqueeze(0).unsqueeze(0), padding=1)
            target_edge = nn.functional.conv2d(target_c, kernel.unsqueeze(0).unsqueeze(0), padding=1)
            if mask is not None:
                edge_loss = (torch.abs(pred_edge - target_edge) * mask).sum() / (mask.sum() + 1e-8)
            else:
                edge_loss = nn.functional.l1_loss(pred_edge, target_edge)
            loss += edge_loss
        return loss / pred.shape[1]

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        sigma = 1.5
        kernel = torch.tensor([np.exp(-(i - window_size//2)**2 / (2 * sigma**2)) for i in range(window_size)], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.outer(kernel).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('window', kernel)

    def forward(self, img1, img2, mask=None):
        C1, C2 = 0.01**2, 0.03**2
        window = self.window.to(img1.device)
        mu1 = nn.functional.conv2d(img1, window, padding=self.window_size//2, groups=img1.shape[1])
        mu2 = nn.functional.conv2d(img2, window, padding=self.window_size//2, groups=img2.shape[1])
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu12 = mu1 * mu2
        sigma1_sq = nn.functional.conv2d(img1**2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2**2, window, padding=self.window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = nn.functional.conv2d(img1*img2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu12
        ssim = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        loss = 1 - ssim
        if mask is not None:
            mask_expanded = mask.expand_as(loss)
            return (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        return loss.mean()

def l1_loss(pred, target, mask=None):
    if mask is not None:
        mask = mask.expand_as(pred)
        return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-8)
    return nn.functional.l1_loss(pred, target)

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.5, ssim_weight=0.8, perceptual_weight=0.3, edge_weight=0.2, color_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.color_loss = ColorConsistencyLoss()
    
    def forward(self, pred, target, mask=None):
        loss = l1_loss(pred, target, mask) * self.l1_weight
        loss += self.ssim_loss(pred, target, mask) * self.ssim_weight
        loss += self.edge_loss(pred, target, mask) * self.edge_weight
        loss += self.color_loss(pred, target, mask) * self.color_weight
        return loss

    def get_metrics(self, pred, target, mask=None):
        return {
            'l1': l1_loss(pred, target, mask).item(),
            'ssim': (1 - self.ssim_loss(pred, target, mask).item())
        }