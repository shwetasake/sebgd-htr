"""
HVLT Inference Script
=====================
Run predictions on:
  (a) A single image
  (b) A folder of images (the unlabeled test set)
  (c) An image list file

For the ICDAR test set without ground truth:
  → Use this script to generate predictions only.
  → No CAR/WAR can be computed (no GT labels).
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import csv

sys.path.insert(0, str(Path(__file__).parent))

from models.hvlt   import HVLT
from data.dataset  import decode_tokens, VOCAB_SIZE, MAX_SEQ_LEN


def get_transform(img_height: int = 32, img_width: int = 128):
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str, device: torch.device) -> HVLT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})

    model = HVLT(
        img_height=cfg.get("img_height", 32),
        img_width=cfg.get("img_width", 128),
        num_fiducial=cfg.get("num_fiducial", 16),
        d_model=cfg.get("d_model", 768),
        n_heads=cfg.get("n_heads", 12),
        n_layers=cfg.get("n_layers", 12),
        vis_seq_len=cfg.get("vis_seq_len", 256),
        acg_dropout=cfg.get("acg_dropout", 0.3),
        pretrained_swin=False,     # Don't reload during inference
        pretrained_roberta=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"val_CAR={ckpt.get('val_car', '?'):.2f}%, "
          f"val_WAR={ckpt.get('val_war', '?'):.2f}%")
    return model


@torch.no_grad()
def predict_images(
    model: HVLT,
    image_paths: list[str],
    device: torch.device,
    img_height: int = 32,
    img_width:  int = 128,
    batch_size: int = 32,
) -> list[tuple[str, str]]:
    """Returns list of (image_path, predicted_text)."""
    transform = get_transform(img_height, img_width)
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs  = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(transform(img))
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")
                batch_imgs.append(torch.zeros(3, img_height, img_width))

        imgs_tensor = torch.stack(batch_imgs).to(device)
        token_ids   = model.predict(imgs_tensor)      # (B, T)

        for path, tokens in zip(batch_paths, token_ids):
            text = decode_tokens(tokens.cpu().tolist())
            results.append((path, text))

        print(f"  Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

    return results


def main():
    parser = argparse.ArgumentParser(description="HVLT Inference")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--image",       type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image_dir",   type=str, default=None,
                        help="Directory of images (test set)")
    parser.add_argument("--image_list",  type=str, default=None,
                        help="Text file with one image path per line")
    parser.add_argument("--output_csv",  type=str, default="predictions.csv")
    parser.add_argument("--img_height",  type=int, default=32)
    parser.add_argument("--img_width",   type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = sorted([
            str(p) for p in Path(args.image_dir).rglob("*")
            if p.suffix.lower() in exts
        ])
        print(f"Found {len(image_paths)} images in {args.image_dir}")
    elif args.image_list:
        with open(args.image_list) as f:
            image_paths = [l.strip() for l in f if l.strip()]
    else:
        parser.error("Provide --image, --image_dir, or --image_list")

    # Run predictions
    print(f"\nRunning predictions on {len(image_paths)} images...")
    results = predict_images(
        model, image_paths, device,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
    )

    # Save to CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "prediction"])
        writer.writerows(results)

    print(f"\nSaved {len(results)} predictions → {args.output_csv}")

    # Print first 10 samples
    print("\nSample predictions:")
    for path, pred in results[:10]:
        print(f"  {os.path.basename(path):30s} → '{pred}'")


if __name__ == "__main__":
    main()
