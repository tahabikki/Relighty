import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss.
    Compares high-level features (texture, structure) not just pixels.
    Prevents blurry outputs and preserves facial details.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features)[:9]).eval()    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg.features)[9:16]).eval()  # relu3_3
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        self.slice1.to(pred.device)
        self.slice2.to(pred.device)

        # Normalize to ImageNet stats
        mean = self.mean.to(pred.device)
        std  = self.std.to(pred.device)
        pred_n   = (pred   - mean) / std
        target_n = (target - mean) / std

        # Compare VGG features
        f1_p = self.slice1(pred_n);    f1_t = self.slice1(target_n)
        f2_p = self.slice2(f1_p);     f2_t = self.slice2(f1_t)

        return nn.functional.l1_loss(f1_p, f1_t) + \
               nn.functional.l1_loss(f2_p, f2_t)


class ShadowRemovalLoss(nn.Module):
    """
    Simple 2-loss combination:
      L1         → pixel accuracy on face region
      Perceptual → texture/detail realism (VGG features)

    Keeping it simple = stable training = better convergence.
    """
    def __init__(self,
                 l1_weight        = 1.0,
                 perceptual_weight= 0.1,
                 # kept for config compatibility, ignored
                 ssim_weight=0, identity_weight=0,
                 color_weight=0):
        super().__init__()
        self.w_l1   = l1_weight
        self.w_perc = perceptual_weight
        self.perc   = PerceptualLoss()
        print(f"Loss: L1 (weight={l1_weight}) + Perceptual (weight={perceptual_weight})")

    def forward(self, pred, target, mask=None):
        # L1 on face region only
        if mask is not None:
            m  = mask.expand_as(pred)
            l1 = (torch.abs(pred - target) * m).sum() / (m.sum() + 1e-8)
        else:
            l1 = nn.functional.l1_loss(pred, target)

        # Perceptual on full output (VGG needs natural image, not masked)
        perc = self.perc(pred, target)

        return self.w_l1 * l1 + self.w_perc * perc

    def get_metrics(self, pred, target, mask=None):
        if mask is not None:
            m  = mask.expand_as(pred)
            l1 = (torch.abs(pred - target) * m).sum() / (m.sum() + 1e-8)
        else:
            l1 = nn.functional.l1_loss(pred, target)
        return {'l1': l1.item()}
