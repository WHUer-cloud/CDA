import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Mamba ----------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d, device=device))

    def forward(self, x):
        x = x.to(device)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class GroupedLinear(nn.Module):
    def __init__(self, in_f, out_f, groups):
        super().__init__()
        assert in_f % groups == 0
        self.g = groups
        self.ls = nn.ModuleList(
            [nn.Linear(in_f // groups, out_f // groups) for _ in range(groups)]
        )

    def forward(self, x):
        x = torch.split(x, x.shape[-1] // self.g, dim=-1)
        return torch.cat([l(xx) for l, xx in zip(self.ls, x)], dim=-1)

class DepthwiseMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        g = int(d ** 0.5)
        while d % g: g -= 1
        self.dw = GroupedLinear(d, d, g)

    def forward(self, x):
        return F.silu(self.dw(x))

class GSU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.in_proj = nn.Linear(d, 2 * d, device=device)
        self.conv1d = nn.Conv1d(d, d, 4, groups=d, padding=3, device=device)
        self.dw_mlp = DepthwiseMLP(d)
        self.norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, d, device=device)

    def forward(self, x):
        B, L, _ = x.shape
        x, z = self.in_proj(x).chunk(2, -1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = self.dw_mlp(x)
        return self.out_proj(self.norm(x) * F.silu(z))

class GSULayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gsu = GSU(d)
        self.norm = RMSNorm(d)

    def forward(self, x):
        return x + self.gsu(self.norm(x))

# ---------- CNN ----------
class ConvBlock(nn.Module):
    def __init__(self, c_in, k, filters, stride):
        super().__init__()
        f1, f2, f3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(c_in, f1, 1, stride),
            nn.BatchNorm2d(f1), nn.GELU(),
            nn.Conv2d(f1, f2, k, 1, 1),
            nn.BatchNorm2d(f2), nn.GELU(),
            nn.Conv2d(f2, f3, 1, 1),
            nn.BatchNorm2d(f3), nn.GELU()
        )
        self.short = nn.Sequential(
            nn.Conv2d(c_in, f3, 1, stride), nn.BatchNorm2d(f3), nn.GELU()
        )

    def forward(self, x):
        return self.stage(x) + self.short(x)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.expand = nn.Linear(dim, dim // scale)
        self.norm = nn.LayerNorm(dim // scale, device=device)

    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)          # B,HW,C
        x = self.norm(self.expand(x))              # B,HW,C//2
        return x

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        w = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.conv1 = w.conv1
        self.bn1 = w.bn1
        self.relu = w.relu
        self.maxpool = w.maxpool
        self.layer1 = w.layer1
        self.layer2 = w.layer2
        self.layer3 = w.layer3

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        e1 = self.layer1(x)   # 256,64,64
        e2 = self.layer2(e1)  # 512,32,32
        e3 = self.layer3(e2)  # 1024,16,16
        return e1, e2, e3

# ---------- OCBE + Decoder ----------
class GCM(nn.Module):
    def __init__(self):
        super().__init__()
        from Transformer import Fusion   # 本地文件
        self.fusion = Fusion()
        self.local1 = ConvBlock(256, 3, [256, 512, 1024], 4)
        self.local2 = ConvBlock(512, 3, [512, 512, 1024], 2)
        self.merge = nn.Conv2d(3072, 1024, 1)
        self.res = ConvBlock(1024, 3, [512, 1024, 2048], 2)

    def forward(self, x1, x2, x3):
        g = self.fusion(x1, x2, x3)
        l1 = self.local1(x1)
        l2 = self.local2(x2)
        l = self.merge(torch.cat([l1, l2, x3], dim=1))
        return self.res(g + l)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l3 = nn.Sequential(PatchExpand2D(2048, 2), GSULayer(1024))
        self.l2 = nn.Sequential(PatchExpand2D(1024, 2), GSULayer(512))
        self.l1 = nn.Sequential(PatchExpand2D(512, 2), GSULayer(256))

    def to_2d(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        return x.permute(0, 2, 1).view(B, C, H, H)

    def forward(self, x):
        s3 = self.to_2d(self.l3(x))   # 1024,16,16
        s2 = self.to_2d(self.l2(s3))  # 512,32,32
        s1 = self.to_2d(self.l1(s2))  # 256,64,64
        return s1, s2, s3, \
               s2, s3, \
               s1, s3, \
               s1, s2

class GCMAndDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcm = GCM()
        self.dec = Decoder()
        # 通道对齐
        self.c11 = nn.Conv2d(512, 256, 3, 1, 1)
        self.c12 = nn.Conv2d(1024, 256, 3, 1, 1)
        self.c21 = nn.Conv2d(256, 512, 3, 1, 1)
        self.c22 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.c31 = nn.Conv2d(256, 1024, 3, 1, 1)
        self.c32 = nn.Conv2d(512, 1024, 3, 1, 1)

    def forward(self, e1, e2, e3):
        f = self.gcm(e1, e2, e3)
        s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n = self.dec(f)
        # 统一插值到目标分辨率
        l1n = self.c12(F.interpolate(l1n, size=64, mode='bilinear', align_corners=True))
        l1m = self.c11(F.interpolate(l1m, size=64, mode='bilinear', align_corners=True))
        l2m = self.c21(F.interpolate(l2m, size=32, mode='bilinear', align_corners=True))
        l2n = self.c22(F.interpolate(l2n, size=32, mode='bilinear', align_corners=True))
        l3m = self.c31(F.interpolate(l3m, size=16, mode='bilinear', align_corners=True))
        l3n = self.c32(F.interpolate(l3n, size=16, mode='bilinear', align_corners=True))
        return s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n