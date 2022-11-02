import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.unet import UNet


class WNetGenerator(nn.Module):
    def __init__(self, seg_in_channels, seg_out_channels, color_in_channels, color_out_channels, use_biliear=True):
        super().__init__()
        self.segmentator = UNet(seg_in_channels, seg_out_channels, use_biliear)
        self.colorizer = UNet(color_in_channels, color_out_channels, use_biliear)

    def forward(self, x):
        x = self.color(x)
        return x
    
    def segment(self, x, tau=1.0, deterministic=False):
        logits = self.segmentator(x)

        if deterministic:
            max_idx = torch.argmax(logits, 1, keepdim=True)
            one_hot = torch.zeros(logits.shape, device=logits.device)
            one_hot.scatter_(1, max_idx, 1)
            t = one_hot
        else:
            t = F.gumbel_softmax(logits, tau, hard=True, dim=1)
        
        seg = t.view(*logits.shape)
        return logits, seg
    
    def color(self, x):
        x = self.colorizer(x)
        x = torch.tanh(x)
        return x
