import torch
import torch.nn as nn

class UNetV2(nn.Module):
    """
    Improved U-Net with face analysis input.
    Takes image (3ch) + face analysis map (3ch: shadow, light, confidence) = 6ch input
    Outputs corrected image (3ch)
    """
    def __init__(self, input_channels=6, output_channels=3):
        super(UNetV2, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def attention_block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 8, ch, 1),
                nn.Sigmoid()
            )

        self.enc1 = conv_block(input_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 1024)

        self.bottleneck = conv_block(1024, 1024)
        self.attention_bottleneck = attention_block(1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, output_channels, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)

    def forward(self, x, analysis_map=None):
        if analysis_map is not None:
            x = torch.cat([x, analysis_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        b = self.bottleneck(e5)
        b = b * self.attention_bottleneck(b)

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.final(d1)
