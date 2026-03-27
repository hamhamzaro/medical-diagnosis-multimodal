"""
models/cnn3d.py
---------------
3D Convolutional Neural Network for volumetric CT scan analysis.

Architecture:
    - 4x Conv3D blocks (BatchNorm + ReLU + MaxPool3D)
    - U-Net-style skip connections for segmentation head
    - Global Average Pooling → embedding for fusion
    - Classification head: pathology detection (multi-label)

Usage:
    from src.models.cnn3d import CNN3D, build_cnn3d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── Conv Block ───────────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    """
    Basic 3D convolution block: Conv3D → BatchNorm → ReLU (×2).

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        dropout:      Dropout rate after second conv.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Encoder ──────────────────────────────────────────────────────────────────

class Encoder3D(nn.Module):
    """
    3D CNN encoder: hierarchical feature extraction with downsampling.

    Input:  (batch, 1, D, H, W) — single-channel CT volume
    Output: embedding vector (batch, embed_dim) + skip connections

    Args:
        in_channels: Number of input channels (1 for grayscale CT).
        base_filters: Number of filters in first conv block (doubles each level).
        embed_dim:   Final embedding dimension.
        dropout:     Dropout rate per block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock3D(in_channels, f, dropout)
        self.enc2 = ConvBlock3D(f, f * 2, dropout)
        self.enc3 = ConvBlock3D(f * 2, f * 4, dropout)
        self.enc4 = ConvBlock3D(f * 4, f * 8, dropout)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool3d(1)   # Global Average Pooling

        self.projection = nn.Sequential(
            nn.Linear(f * 8, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, D, H, W)
        Returns:
            embedding: (batch, embed_dim)
            skips:     List of skip connection tensors [e1, e2, e3, e4]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Global embedding
        gap = self.gap(e4).flatten(1)           # (batch, f*8)
        embedding = self.projection(gap)         # (batch, embed_dim)

        return embedding, [e1, e2, e3, e4]


# ─── Segmentation Head (U-Net decoder) ───────────────────────────────────────

class SegmentationHead3D(nn.Module):
    """
    U-Net-style decoder for 3D volumetric segmentation.

    Uses skip connections from the encoder to recover spatial resolution.

    Args:
        base_filters: Must match encoder base_filters.
        n_classes:    Number of segmentation classes (1 = binary).
    """

    def __init__(self, base_filters: int = 32, n_classes: int = 1):
        super().__init__()
        f = base_filters

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)  # f*4 + f*4 skip

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)  # f*2 + f*2 skip

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)      # f + f skip

        self.out_conv = nn.Conv3d(f, n_classes, kernel_size=1)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            skips: [e1, e2, e3, e4] from encoder
        Returns:
            segmentation map: (batch, n_classes, D, H, W)
        """
        e1, e2, e3, e4 = skips

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ─── Classification Head ──────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    MLP classification head on top of the encoder embedding.

    Args:
        embed_dim:  Input embedding dimension.
        n_classes:  Number of output classes (multi-label).
        dropout:    Dropout rate.
    """

    def __init__(self, embed_dim: int = 256, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


# ─── Full CNN3D Model ─────────────────────────────────────────────────────────

class CNN3D(nn.Module):
    """
    Full 3D CNN model for CT scan analysis.

    Combines:
    - Encoder3D: hierarchical feature extraction
    - SegmentationHead3D: U-Net decoder (optional)
    - ClassificationHead: pathology multi-label classification

    Args:
        in_channels:    Input channels (1 for CT).
        base_filters:   Filter count for first encoder block.
        embed_dim:      Embedding dimension.
        n_classes:      Number of pathology classes.
        with_segmentation: Whether to include segmentation head.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        embed_dim: int = 256,
        n_classes: int = 5,
        with_segmentation: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.with_segmentation = with_segmentation

        self.encoder = Encoder3D(in_channels, base_filters, embed_dim, dropout)
        self.classifier = ClassificationHead(embed_dim, n_classes, dropout)

        if with_segmentation:
            self.segmentor = SegmentationHead3D(base_filters, n_classes=1)

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, 1, D, H, W) — normalized CT volume

        Returns:
            Dict with keys:
                'embedding':     (batch, embed_dim)
                'logits':        (batch, n_classes) — classification logits
                'segmentation':  (batch, 1, D, H, W) — optional segmentation mask
        """
        embedding, skips = self.encoder(x)
        logits = self.classifier(embedding)

        out = {"embedding": embedding, "logits": logits}

        if self.with_segmentation:
            out["segmentation"] = self.segmentor(skips)

        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for each class."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)["logits"])


# ─── Grad-CAM 3D ──────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D CNNs.

    Generates volumetric saliency maps showing which voxels
    contributed most to a given class prediction.

    Usage:
        cam = GradCAM3D(model, target_layer=model.encoder.enc4)
        heatmap = cam(volume_tensor, class_idx=0)
    """

    def __init__(self, model: CNN3D, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a given class.

        Args:
            x:         Input volume (1, 1, D, H, W).
            class_idx: Target class index.

        Returns:
            heatmap: (D, H, W) normalized saliency map.
        """
        self.model.eval()
        x.requires_grad_(True)

        out = self.model(x)
        score = out["logits"][0, class_idx]
        self.model.zero_grad()
        score.backward()

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu()


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_cnn3d(
    n_classes: int = 5,
    with_segmentation: bool = True,
    pretrained_path: Optional[str] = None
) -> CNN3D:
    """Instantiate CNN3D, optionally loading pretrained weights."""
    model = CNN3D(
        in_channels=1,
        base_filters=32,
        embed_dim=256,
        n_classes=n_classes,
        with_segmentation=with_segmentation,
        dropout=0.2
    )
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from {pretrained_path}")

    total = sum(p.numel() for p in model.parameters())
    print(f"CNN3D | params: {total:,} | classes: {n_classes} | seg: {with_segmentation}")
    return model


if __name__ == "__main__":
    model = build_cnn3d(n_classes=5, with_segmentation=True)
    x = torch.randn(2, 1, 64, 128, 128)
    out = model(x)
    print(f"Input:         {x.shape}")
    print(f"Embedding:     {out['embedding'].shape}")
    print(f"Logits:        {out['logits'].shape}")
    print(f"Segmentation:  {out['segmentation'].shape}")
    print("Smoke test passed ✅")
