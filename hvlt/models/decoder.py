"""
Stage V: Multimodal Fusion and Decoding
Corrected implementation with:

✓ Proper causal masking
✓ Proper autoregressive decoding
✓ Correct padding masks
✓ Stable TransformerDecoder usage
✓ RoBERTa weight initialization
✓ Correct inference behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from data.dataset import (
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    MAX_SEQ_LEN,
)


# ─── Positional Bridge ─────────────────────────────────────────────────────────

class PositionalBridge(nn.Module):

    def __init__(
        self,
        in_channels: int = 768,
        d_model: int = 768,
        vis_seq_len: int = 256,
    ):
        super().__init__()

        self.vis_seq_len = vis_seq_len
        self.d_model = d_model

        # Channel projection
        if in_channels != d_model:
            self.channel_proj = nn.Linear(in_channels, d_model)
        else:
            self.channel_proj = nn.Identity()

        # Height-wise pooling
        self.pool = nn.AdaptiveAvgPool2d((1, vis_seq_len))

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, vis_seq_len, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor):

        # x: (B, H, W, C)

        B, H, W, C = x.shape

        # (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Height-wise pooling
        x = self.pool(x)          # (B, C, 1, T_vis)

        x = x.squeeze(2)          # (B, C, T_vis)

        x = x.permute(0, 2, 1)    # (B, T_vis, C)

        x = self.channel_proj(x)

        x = x + self.pos_embed

        return x


# ─── Decoder ───────────────────────────────────────────────────────────────────

class RoBERTaCrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = MAX_SEQ_LEN,
        vis_seq_len: int = 256,
        dropout: float = 0.1,
        pretrained_roberta: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # ── Token embeddings ────────────────────────────────────────────────
        self.token_embed = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_IDX,
        )

        self.pos_embed = nn.Embedding(
            max_seq_len + 5,
            d_model,
        )

        self.embed_drop = nn.Dropout(dropout)

        # ── Transformer decoder ─────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=3072,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # ── Output projection ───────────────────────────────────────────────
        self.output_proj = nn.Linear(
            d_model,
            vocab_size,
        )

        # ── RoBERTa init ────────────────────────────────────────────────────
        if pretrained_roberta:
            self._init_from_roberta()

    # ────────────────────────────────────────────────────────────────────────

    def _init_from_roberta(self):

        try:

            print("  Loading RoBERTa-base weights...")

            roberta = RobertaModel.from_pretrained(
                "roberta-base"
            )

            roberta_state = roberta.state_dict()

            # Token embeddings
            roberta_embed = roberta_state.get(
                "embeddings.word_embeddings.weight"
            )

            if roberta_embed is not None:

                min_vocab = min(
                    self.vocab_size,
                    roberta_embed.shape[0],
                )

                self.token_embed.weight.data[:min_vocab] = (
                    roberta_embed[:min_vocab]
                )

            # Position embeddings
            roberta_pos = roberta_state.get(
                "embeddings.position_embeddings.weight"
            )

            if roberta_pos is not None:

                min_pos = min(
                    self.max_seq_len + 5,
                    roberta_pos.shape[0],
                )

                self.pos_embed.weight.data[:min_pos] = (
                    roberta_pos[:min_pos]
                )

            del roberta

            print("  RoBERTa initialization complete.")

        except Exception as e:

            print(f"  [WARN] RoBERTa init failed: {e}")

    # ────────────────────────────────────────────────────────────────────────

    def _generate_causal_mask(
        self,
        T: int,
        device,
    ):

        # True above diagonal = masked
        mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1,
        ).bool()

        return mask

    # ────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        visual_memory: torch.Tensor,   # (B, T_vis, D)
        target_tokens: torch.Tensor,   # (B, T)
    ):

        B, T = target_tokens.shape

        device = target_tokens.device

        # ── Position IDs ───────────────────────────────────────────────────
        positions = torch.arange(
            T,
            device=device,
        ).unsqueeze(0).expand(B, T)

        # ── Embeddings ─────────────────────────────────────────────────────
        tgt_embed = (
            self.token_embed(target_tokens)
            + self.pos_embed(positions)
        )

        tgt_embed = self.embed_drop(tgt_embed)

        # ── Causal mask (VERY IMPORTANT) ───────────────────────────────────
        tgt_mask = self._generate_causal_mask(
            T,
            device,
        )

        # ── Padding mask ───────────────────────────────────────────────────
        tgt_key_padding_mask = (
            target_tokens == PAD_IDX
        )

        # ── Decode ─────────────────────────────────────────────────────────
        decoded = self.decoder(
            tgt=tgt_embed,
            memory=visual_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # ── Output logits ──────────────────────────────────────────────────
        logits = self.output_proj(decoded)

        return logits

    # ────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        visual_memory: torch.Tensor,
        max_len: int = MAX_SEQ_LEN,
    ):

        B = visual_memory.shape[0]

        device = visual_memory.device

        # Start token
        generated = torch.full(
            (B, 1),
            SOS_IDX,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            B,
            dtype=torch.bool,
            device=device,
        )

        for step in range(max_len):

            logits = self.forward(
                visual_memory,
                generated,
            )

            next_token = logits[:, -1, :].argmax(dim=-1)

            # Keep finished sequences padded
            next_token = next_token.masked_fill(
                finished,
                PAD_IDX,
            )

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)],
                dim=1,
            )

            finished = finished | (
                next_token == EOS_IDX
            )

            # Stop early if all finished
            if finished.all():
                break

        # Remove SOS
        generated = generated[:, 1:]

        return generated