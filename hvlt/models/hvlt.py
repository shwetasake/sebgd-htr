"""
HVLT: Hierarchical Vision-Language Transformer
Full 5-stage pipeline as described in the paper.

Stage I  : Word-level cropping (handled in dataset)
Stage II : TPS-STN geometric rectification
Stage III: Gated CNN + Swin Transformer encoder
Stage IV : Artifact Classification Gate (ACG)
Stage V  : RoBERTa cross-attention decoder

Total params: ~142M (matching paper Table IV)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tps_stn  import TPSSTN
from models.encoder  import HierarchicalVisionEncoder
from models.decoder  import PositionalBridge, RoBERTaCrossAttentionDecoder
from data.dataset    import VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX, MAX_SEQ_LEN


class HVLT(nn.Module):
    """
    Hierarchical Vision-Language Transformer for Handwritten Text Recognition.

    Hyperparameters (Table IV):
        TPS fiducial points: K=16
        Swin layer distribution: [2, 2, 18, 2]
        Swin attention heads: [4, 8, 16, 32]
        Language decoder: RoBERTa-base (L=12, H=768)
        Cross-attention heads: 12
        Sequence dimension d_model: 768
        Total parameters: ~142M
        Output classes: 99
        Max sequence length: 25 tokens
        Optimizer: Adam, lr=5e-5
        ACG dropout: 0.3
        ACG auxiliary loss weight λ: 0.1
    """

    def __init__(
        self,
        img_height:          int   = 32,
        img_width:           int   = 128,
        num_fiducial:        int   = 16,
        d_model:             int   = 768,
        n_heads:             int   = 12,
        n_layers:            int   = 12,
        max_seq_len:         int   = MAX_SEQ_LEN,
        vis_seq_len:         int   = 256,
        vocab_size:          int   = VOCAB_SIZE,
        acg_dropout:         float = 0.3,
        acg_lambda:          float = 0.1,
        pretrained_swin:     bool  = True,
        pretrained_roberta:  bool  = True,
    ):
        super().__init__()
        self.acg_lambda = acg_lambda

        # ── Stage II: TPS-STN ──────────────────────────────────────────────────
        self.tps_stn = TPSSTN(
            num_fiducial=num_fiducial,
            img_h=img_height,
            img_w=img_width,
        )

        # ── Stage III + IV: Hierarchical Vision Encoder + ACG ─────────────────
        self.encoder = HierarchicalVisionEncoder(
            pretrained=pretrained_swin,
            acg_dropout=acg_dropout,
        )

        # ── Stage V: Positional Bridge + RoBERTa Decoder ──────────────────────
        self.pos_bridge = PositionalBridge(
            in_channels=768,
            d_model=d_model,
            vis_seq_len=vis_seq_len,
        )
        self.decoder = RoBERTaCrossAttentionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            vis_seq_len=vis_seq_len,
            pretrained_roberta=pretrained_roberta,
        )

    def forward(
        self,
        images:       torch.Tensor,  # (B, 3, H, W)
        target_tokens: torch.Tensor, # (B, T) for teacher forcing
        acg_labels:   torch.Tensor = None,  # (B,) binary: 1=artifact, 0=normal
    ):
        """
        Training forward pass.

        Returns:
            logits:   (B, T, vocab_size) — for cross-entropy loss
            acg_gate: (B, 1)             — for BCE auxiliary loss
        """
        # Stage II: Geometric rectification
        rectified = self.tps_stn(images)           # (B, 3, H, W)

        # Stage III + IV: Visual encoding + ACG
        vis_feats, acg_gate = self.encoder(rectified)  # (B, H', W', 768), (B, 1)

        # Stage V: Bridge + decode
        vis_memory = self.pos_bridge(vis_feats)    # (B, T_vis, 768)
        logits = self.decoder(vis_memory, target_tokens)  # (B, T, vocab_size)

        return logits, acg_gate

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference: greedy autoregressive decoding.

        Returns:
            token_ids: (B, max_seq_len)
        """
        self.eval()

        rectified  = self.tps_stn(images)
        vis_feats, _ = self.encoder(rectified)
        vis_memory = self.pos_bridge(vis_feats)
        preds      = self.decoder.greedy_decode(vis_memory)

        return preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Loss Function ─────────────────────────────────────────────────────────────

class HVLTLoss(nn.Module):

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        acg_lambda: float = 0.1
    ):
        super().__init__()

        self.acg_lambda = acg_lambda

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX,
            reduction="mean"
        )

        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,         # (B, T, V)
        targets: torch.Tensor,        # (B, T)
        acg_gate: torch.Tensor,       # (B, 1)
        acg_labels: torch.Tensor = None,
    ) -> dict:

        B, T, V = logits.shape

        # Flatten for token CE
        pred = logits.reshape(-1, V)      # (B*T, V)
        tgt  = targets.reshape(-1)        # (B*T)

        # Sequence loss
        l_seq = self.ce_loss(pred, tgt)

        # ACG loss
        l_acg = torch.tensor(
            0.0,
            device=logits.device
        )

        if acg_labels is not None:

            gate = acg_gate.squeeze(1)    # (B,)

            l_acg = self.bce_loss(
                gate,
                acg_labels.float()
            )

        # Total loss
        l_total = l_seq + self.acg_lambda * l_acg

        return {
            "loss":  l_total,
            "l_seq": l_seq,
            "l_acg": l_acg,
        }