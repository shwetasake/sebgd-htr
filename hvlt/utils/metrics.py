"""
Evaluation Metrics as defined in the HVLT paper.

CAR (Character Accuracy Rate) — Equation 1:
    CAR = (1/M) * sum_i [ 1 - edit(g_i, y_hat_i) / |g_i| ]

WAR (Word Accuracy Rate) — Equation 2:
    WAR = (1/M) * sum_i [ edit(g_i, y_hat_i) == 0 ]

Where edit(·,·) is the Levenshtein edit distance, M is total samples.
"""

import editdistance
import torch
from data.dataset import decode_tokens, EOS_IDX, PAD_IDX, SOS_IDX


def compute_car(predictions: list[str], ground_truths: list[str]) -> float:
    """
    Character Accuracy Rate (Eq. 1).
    CAR = mean_i [1 - edit(g_i, y_hat_i) / |g_i|]
    """
    total = 0.0
    for pred, gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            # empty GT → perfect if pred is also empty
            total += 1.0 if len(pred) == 0 else 0.0
            continue
        ed = editdistance.eval(pred, gt)
        total += max(0.0, 1.0 - ed / len(gt))
    return total / max(len(predictions), 1)


def compute_war(predictions: list[str], ground_truths: list[str]) -> float:
    """
    Word Accuracy Rate (Eq. 2).
    WAR = mean_i [edit(g_i, y_hat_i) == 0]
    """
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / max(len(predictions), 1)


def decode_batch_predictions(token_ids: torch.Tensor) -> list[str]:
    """
    Convert (B, T) token ID tensor to list of decoded strings.
    Stops at EOS, ignores PAD and SOS.
    """
    results = []
    for row in token_ids:
        indices = row.cpu().tolist()
        text = decode_tokens(indices)
        results.append(text)
    return results


class MetricTracker:
    """Running average tracker for training/validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss  = 0.0
        self.total_l_seq = 0.0
        self.total_l_acg = 0.0
        self.n_batches   = 0
        self.all_preds   = []
        self.all_gts     = []

    def update(self, loss_dict: dict, preds: list[str], gts: list[str]):
        self.total_loss  += loss_dict["loss"].item()
        self.total_l_seq += loss_dict["l_seq"].item()
        self.total_l_acg += loss_dict.get("l_acg", torch.tensor(0.0)).item()
        self.n_batches   += 1
        self.all_preds.extend(preds)
        self.all_gts.extend(gts)

    def compute(self) -> dict:
        n = max(self.n_batches, 1)
        car = compute_car(self.all_preds, self.all_gts)
        war = compute_war(self.all_preds, self.all_gts)
        return {
            "loss":     self.total_loss / n,
            "l_seq":    self.total_l_seq / n,
            "l_acg":    self.total_l_acg / n,
            "CAR":      car * 100,
            "WAR":      war * 100,
        }
