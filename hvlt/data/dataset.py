"""
ICDAR Word-Level English Dataset Loader for HVLT
Supports train.txt format: "image_path label" per line
e.g. "image/74_1.jpg hello"
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── Character Vocabulary ──────────────────────────────────────────────────────
# 99 output classes as per paper: upper+lower letters, digits, punctuation + special tokens
CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
# Special tokens
PAD_TOKEN  = "<PAD>"   # idx 0
SOS_TOKEN  = "<SOS>"   # idx 1
EOS_TOKEN  = "<EOS>"   # idx 2
UNK_TOKEN  = "<UNK>"   # idx 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
VOCAB = SPECIAL_TOKENS + list(CHARSET)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)  # ~99 classes as in paper

PAD_IDX = CHAR2IDX[PAD_TOKEN]
SOS_IDX = CHAR2IDX[SOS_TOKEN]
EOS_IDX = CHAR2IDX[EOS_TOKEN]
UNK_IDX = CHAR2IDX[UNK_TOKEN]

MAX_SEQ_LEN = 25  # as per paper


def encode_label(text: str) -> list[int]:
    """Encode text to list of token indices."""
    tokens = [SOS_IDX]
    for ch in text:
        tokens.append(CHAR2IDX.get(ch, UNK_IDX))
    tokens.append(EOS_IDX)
    return tokens


def decode_tokens(indices: list[int]) -> str:
    """Decode token indices to string, stopping at EOS."""
    chars = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        chars.append(IDX2CHAR.get(idx, ""))
    return "".join(chars)


# ─── Dataset Class ─────────────────────────────────────────────────────────────

class ICDARWordDataset(Dataset):
    """
    Reads train.txt where each line is:
        relative/path/to/image.jpg LABEL
    e.g.:
        image/74_1.jpg hello
    """

    def __init__(
        self,
        txt_file: str,
        root_dir: str,
        img_height: int = 32,
        img_width: int = 128,
        augment: bool = False,
    ):
        self.root_dir   = root_dir
        self.img_height = img_height
        self.img_width  = img_width
        self.augment    = augment
        self.samples    = []

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                img_rel, label = parts
                img_path = os.path.join(root_dir, img_rel)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    print(f"[WARN] Missing image: {img_path}")

        print(f"  Loaded {len(self.samples)} samples from {txt_file}")

        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Augmentation transforms (applied before resize)
        self.aug_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.augment:
            image = self.aug_transform(image)

        image = self.base_transform(image)

        # Encode label
        encoded = encode_label(label)
        # Pad / truncate to MAX_SEQ_LEN+2 (SOS + chars + EOS)
        target_len = MAX_SEQ_LEN + 2
        if len(encoded) > target_len:
            encoded = encoded[:target_len - 1] + [EOS_IDX]
        else:
            encoded = encoded + [PAD_IDX] * (target_len - len(encoded))

        target = torch.tensor(encoded, dtype=torch.long)
        return image, target, label


# ─── Collate & Loaders ─────────────────────────────────────────────────────────

def collate_fn(batch):
    images, targets, labels = zip(*batch)
    images  = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets, list(labels)


def get_dataloaders(
    train_txt, root_dir, val_split=0.15,
    batch_size=32, img_height=32, img_width=128, num_workers=4,
):
    # Load full list of samples first
    temp = ICDARWordDataset(train_txt, root_dir, img_height, img_width, augment=False)
    all_samples = temp.samples

    # Split indices
    n_total = len(all_samples)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples   = [all_samples[i] for i in val_indices]

    # Create SEPARATE dataset instances — train with aug, val without
    train_ds = ICDARWordDataset.__new__(ICDARWordDataset)
    train_ds.root_dir   = root_dir
    train_ds.img_height = img_height
    train_ds.img_width  = img_width
    train_ds.augment    = True
    train_ds.samples    = train_samples
    train_ds.base_transform = temp.base_transform
    train_ds.aug_transform  = temp.aug_transform

    val_ds = ICDARWordDataset.__new__(ICDARWordDataset)
    val_ds.root_dir   = root_dir
    val_ds.img_height = img_height
    val_ds.img_width  = img_width
    val_ds.augment    = False
    val_ds.samples    = val_samples
    val_ds.base_transform = temp.base_transform
    val_ds.aug_transform  = temp.aug_transform

    print(f"  Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader