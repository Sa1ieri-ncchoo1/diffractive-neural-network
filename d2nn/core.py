"""Numerical definitions extracted unchanged from the supplied train.py.

Importing this module does not construct datasets, train, or download files.
See docs/PROVENANCE.md for the source inventory and extraction details.
"""
from __future__ import annotations
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Diffraction(nn.Module):
    def __init__(self, λ=532e-9, N_pixels=128, pixel_size=2e-6, distance=torch.tensor([0.005]), scale_factor=2,
                 device='cpu'):
        super(Diffraction, self).__init__()
        self.device = device
        self.scale_factor = scale_factor
        self.N_pixels = N_pixels * scale_factor
        self.pixel_size = pixel_size

        if isinstance(distance, torch.Tensor):
            distance_value = distance.item() if distance.numel() == 1 else distance[0].item()
        else:
            distance_value = float(distance)
        distance_tensor = torch.tensor(distance_value, dtype=torch.float32, device=device).view(1, 1, 1, 1)
        self.register_buffer("distance", distance_tensor)

        fx = torch.fft.fftshift(torch.fft.fftfreq(self.N_pixels, d=self.pixel_size)).to(device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(self.N_pixels, d=self.pixel_size)).to(device)
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

        argument = (2 * torch.pi) ** 2 * ((1 / λ) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        tmp_complex = tmp.to(torch.complex64)
        kz = torch.where(argument >= 0, tmp_complex, 1j * tmp_complex).to(self.device)
        self.register_buffer("kz", kz)

    def forward(self, E):
        if not torch.is_complex(E):
            E = E.to(torch.complex64)
        E = E.to(self.device)

        original_h, original_w = E.size(2), E.size(3)
        pad_h = (original_h * (self.scale_factor - 1)) // 2
        pad_w = (original_w * (self.scale_factor - 1)) // 2
        pad_size = (pad_w, pad_w, pad_h, pad_h)
        E_padded = torch.nn.functional.pad(E, pad_size, mode='constant', value=0)

        c = torch.fft.fftshift(torch.fft.fft2(E_padded, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        phase = torch.exp(1j * self.kz * self.distance)
        angular_spectrum = torch.fft.ifft2(
            torch.fft.ifftshift(c * phase, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho"
        )

        padded_h, padded_w = E_padded.size(2), E_padded.size(3)
        return angular_spectrum[:, :, pad_h:padded_h - pad_h, pad_w:padded_w - pad_w]


class RingDetector(nn.Module):
    def __init__(self, num_regions: int, height: int, width: int, ring_distance_px: float,
                 region_radius_px: int) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.height = int(height)
        self.width = int(width)
        self.ring_distance_px = float(ring_distance_px)
        self.region_radius_px = int(region_radius_px)

        masks = self._precompute_masks()
        self.register_buffer("masks", masks, persistent=False)

    def _precompute_masks(self) -> torch.Tensor:
        H, W = self.height, self.width
        cx0, cy0 = W // 2, H // 2
        masks = torch.zeros(self.num_regions, H, W, dtype=torch.float32)
        angles = torch.linspace(0.0, 2.0 * math.pi, steps=self.num_regions + 1)[:-1]
        for r in range(self.num_regions):
            cx = int(round(cx0 + self.ring_distance_px * math.cos(float(angles[r]))))
            cy = int(round(cy0 + self.ring_distance_px * math.sin(float(angles[r]))))
            if 0 <= cx < W and 0 <= cy < H:
                masks[r] = self._circular_mask(H, W, center=(cy, cx), radius=self.region_radius_px)
        return masks

    @staticmethod
    def _circular_mask(H: int, W: int, center: Tuple[int, int], radius: int) -> torch.Tensor:
        cy, cx = center
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        return (((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2).float()

    def normalize_intensity(self, E: torch.Tensor, out_size: Optional[int] = None) -> torch.Tensor:
        Int = torch.abs(E) ** 2
        if out_size is not None:
            B, C, H, W = Int.shape
            start_h = H // 2 - out_size // 2
            end_h = H // 2 + out_size // 2
            start_w = W // 2 - out_size // 2
            end_w = W // 2 + out_size // 2
            Int = Int[:, :, start_h:end_h, start_w:end_w]
        max_val = Int.amax(dim=(2, 3), keepdim=True)
        return Int / (max_val + 1e-8)

    def forward(self, E: torch.Tensor, normalize: bool = False, flatten: bool = True) -> torch.Tensor:
        B, C, H, W = E.shape
        I_norm = self.normalize_intensity(E, out_size=None)
        I_norm = I_norm.unsqueeze(2)
        masks = self.masks.to(I_norm.device).unsqueeze(0).unsqueeze(0)
        region_intensity = I_norm * masks
        region_sums = region_intensity.sum(dim=(-1, -2))

        if normalize:
            region_sums = region_sums / (region_sums.sum(dim=-1, keepdim=True) + 1e-12)
        if flatten:
            region_sums = region_sums.reshape(B, C * self.num_regions)
        return region_sums


class D2NNPhaseOnly(nn.Module):
    def __init__(self, num_layers: int, wavelength: float, n_pixels: int, pixel_size: float, distances: List[float],
                 device: torch.device) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.n_pixels = int(n_pixels)
        self.device = device

        self.phase = nn.ParameterList([
            nn.Parameter(torch.zeros((self.n_pixels, self.n_pixels), dtype=torch.float32))
            for _ in range(self.num_layers)
        ])

        self.layers = nn.ModuleList([
            Diffraction(λ=wavelength, N_pixels=n_pixels, pixel_size=pixel_size,
                        distance=torch.tensor([distances[i]], dtype=torch.float32, device=device), device=device)
            for i in range(self.num_layers)
        ])
        self.last = Diffraction(λ=wavelength, N_pixels=n_pixels, pixel_size=pixel_size,
                                distance=torch.tensor([distances[-1]], dtype=torch.float32, device=device),
                                device=device)
        self.to(device)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        E = E.to(self.device).to(torch.complex64)
        for i, prop in enumerate(self.layers):
            E = prop(E) * torch.exp(1j * math.pi * torch.tanh(self.phase[i]))
        E = self.last(E)
        return E


class D2NNOpticalClassifier(nn.Module):
    def __init__(self, d2nn: D2NNPhaseOnly, detector: RingDetector) -> None:
        super().__init__()
        self.d2nn = d2nn
        self.detector = detector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E_in = x.to(self.d2nn.device).to(torch.complex64)
        E_out = self.d2nn(E_in)
        region = self.detector(E_out, normalize=False, flatten=False).squeeze(1)
        return region / (region.sum(-1, keepdim=True) + 1e-12)

    def forward_intensity(self, x: torch.Tensor) -> torch.Tensor:
        """返回 max-归一化后的 100×100 输出强度图，shape (B, 1, H, W)。"""
        E_in = x.to(self.d2nn.device).to(torch.complex64)
        E_out = self.d2nn(E_in)  # (B, 1, H, W) complex
        I = torch.abs(E_out) ** 2  # (B, 1, H, W) real
        max_val = I.amax(dim=(2, 3), keepdim=True)
        I_norm = I / (max_val + 1e-8)
        return I_norm


def mse_onehot_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    num_classes = logits.shape[1]
    target = F.one_hot(y.long(), num_classes=num_classes).float().to(logits.device)
    return F.mse_loss(logits, target)


def contrast_loss(logits, y, margin=0.1):
    B = logits.shape[0]
    correct = logits[torch.arange(B), y]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B), y] = False
    wrong_max = logits.masked_fill(~mask, -float('inf')).max(dim=1).values
    return F.relu(wrong_max + margin - correct).mean()


def pixel_mse_loss(I_norm: torch.Tensor, y: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """
    I_norm: (B, 1, H, W)  —— max-归一化后的输出强度图
    y:      (B,)          —— 类别标签
    target_masks: (num_classes, H, W)  —— detector.masks，每类一个圆盘
    """
    target = target_masks[y].unsqueeze(1)  # (B, 1, H, W)
    return F.mse_loss(I_norm, target)
