"""
models/bert_encoder.py + fusion.py
------------------------------------
BERT-based clinical text encoder and multimodal fusion model.

BertClinicalEncoder:
    Fine-tunes bert-base-uncased on clinical reports.
    Outputs [CLS] embedding for downstream fusion.

MultimodalFusion:
    Late fusion of CNN3D image embedding + BERT text embedding.
    Learnable attention weights per modality.
    Final MLP classifier.

Usage:
    from src.models.bert_encoder import BertClinicalEncoder, MultimodalFusion
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Optional


# ─── BERT Clinical Encoder ────────────────────────────────────────────────────

class BertClinicalEncoder(nn.Module):
    """
    Fine-tuned BERT encoder for clinical report understanding.

    Uses bert-base-uncased as backbone. Extracts [CLS] token embedding
    from the last hidden state as the document-level representation.

    Args:
        model_name:    HuggingFace model identifier.
        embed_dim:     Output projection dimension.
        freeze_layers: Number of BERT layers to freeze (0 = fine-tune all).
        dropout:       Dropout on projection head.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embed_dim: int = 256,
        freeze_layers: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size  # 768

        # Freeze first N transformer layers
        if freeze_layers > 0:
            modules_to_freeze = [
                self.bert.embeddings,
                *self.bert.encoder.layer[:freeze_layers]
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        # Projection head: 768 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"BertClinicalEncoder | trainable params: {n_trainable:,} | "
              f"frozen layers: {freeze_layers}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len) — tokenized report
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
            token_type_ids: (batch, seq_len) — optional segment IDs

        Returns:
            embedding: (batch, embed_dim) — [CLS] token projection
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_embedding)

    def encode_texts(
        self,
        texts: list[str],
        max_length: int = 512,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Convenience method: tokenize + encode a list of clinical reports.

        Args:
            texts:      List of raw clinical report strings.
            max_length: Maximum token length (truncates longer reports).
            device:     Target device.

        Returns:
            embeddings: (n_texts, embed_dim)
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            return self.forward(**encoded)


# ─── Multimodal Fusion ────────────────────────────────────────────────────────

class ModalityAttention(nn.Module):
    """
    Learnable soft attention over modalities.

    Computes a scalar weight per modality, normalized via softmax.
    Allows the model to up-weight the more informative modality
    depending on the input (e.g., weight text more when image is noisy).

    Args:
        embed_dim:    Shared embedding dimension.
        n_modalities: Number of input modalities (default 2: image + text).
    """

    def __init__(self, embed_dim: int = 256, n_modalities: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, n_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of (batch, embed_dim) tensors.
        Returns:
            fused: (batch, embed_dim) — attention-weighted sum.
        """
        stacked = torch.stack(embeddings, dim=1)       # (batch, n_mod, embed_dim)
        concat = torch.cat(embeddings, dim=-1)          # (batch, n_mod * embed_dim)
        weights = self.attention(concat).unsqueeze(-1)  # (batch, n_mod, 1)
        fused = (stacked * weights).sum(dim=1)          # (batch, embed_dim)
        return fused


class MultimodalFusion(nn.Module):
    """
    Late fusion model: CNN3D image embedding + BERT text embedding → diagnosis.

    Architecture:
        image_embedding (batch, embed_dim)  ─┐
                                              ├→ ModalityAttention → MLP → logits
        text_embedding  (batch, embed_dim)  ─┘

    Args:
        embed_dim:   Shared embedding dimension (must match both encoders).
        n_classes:   Number of diagnostic classes (multi-label).
        dropout:     Dropout rate in classifier.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_classes: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()

        self.modality_attention = ModalityAttention(embed_dim, n_modalities=2)

        # Additional cross-modal interaction (bilinear)
        self.cross_modal = nn.Bilinear(embed_dim, embed_dim, embed_dim)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),   # fused + cross-modal
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes)
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"MultimodalFusion | params: {total:,} | classes: {n_classes}")

    def forward(
        self,
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image_embedding: (batch, embed_dim) from CNN3D encoder
            text_embedding:  (batch, embed_dim) from BERT encoder

        Returns:
            Dict with:
                'logits':         (batch, n_classes)
                'fused':          (batch, embed_dim) — attention-fused embedding
                'cross_modal':    (batch, embed_dim) — bilinear cross-modal features
                'modality_weights': (batch, 2) — attention weights [image, text]
        """
        # Attention-weighted fusion
        fused = self.modality_attention([image_embedding, text_embedding])

        # Cross-modal interaction (bilinear)
        cross = self.cross_norm(self.cross_modal(image_embedding, text_embedding))

        # Concatenate and classify
        combined = torch.cat([fused, cross], dim=-1)
        logits = self.classifier(combined)

        # Retrieve modality weights for interpretability
        concat = torch.cat([image_embedding, text_embedding], dim=-1)
        weights = self.modality_attention.attention(concat)

        return {
            "logits": logits,
            "fused": fused,
            "cross_modal": cross,
            "modality_weights": weights
        }

    def predict_proba(
        self,
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return sigmoid probabilities for each class."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(image_embedding, text_embedding)["logits"])


# ─── End-to-End Pipeline ──────────────────────────────────────────────────────

class MultimodalDiagnosticPipeline(nn.Module):
    """
    Full end-to-end pipeline combining CNN3D + BERT + Fusion.

    Wraps all three components for inference and joint training.

    Args:
        cnn3d:        CNN3D model instance.
        bert_encoder: BertClinicalEncoder instance.
        fusion:       MultimodalFusion instance.
    """

    def __init__(self, cnn3d, bert_encoder: BertClinicalEncoder, fusion: MultimodalFusion):
        super().__init__()
        self.cnn3d = cnn3d
        self.bert_encoder = bert_encoder
        self.fusion = fusion

    def forward(
        self,
        volume: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            volume:         (batch, 1, D, H, W) — CT volume
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            Dict with logits, fused embedding, segmentation, modality_weights.
        """
        cnn_out = self.cnn3d(volume)
        image_emb = cnn_out["embedding"]
        text_emb = self.bert_encoder(input_ids, attention_mask)

        fusion_out = self.fusion(image_emb, text_emb)
        fusion_out["segmentation"] = cnn_out.get("segmentation")
        return fusion_out


if __name__ == "__main__":
    from cnn3d import build_cnn3d

    print("Testing BertClinicalEncoder...")
    bert_enc = BertClinicalEncoder(embed_dim=256, freeze_layers=8)
    dummy_ids = torch.randint(0, 30522, (2, 128))
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    text_emb = bert_enc(dummy_ids, dummy_mask)
    print(f"  Text embedding: {text_emb.shape}")

    print("\nTesting MultimodalFusion...")
    fusion = MultimodalFusion(embed_dim=256, n_classes=5)
    img_emb = torch.randn(2, 256)
    out = fusion(img_emb, text_emb)
    print(f"  Logits:           {out['logits'].shape}")
    print(f"  Modality weights: {out['modality_weights']}")

    print("\nAll tests passed ✅")
