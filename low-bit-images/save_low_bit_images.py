"""Save a random training image at different bit depths: 8, 4, 2, and 1 bit."""

import os
import random
import numpy as np
from PIL import Image

TRAIN_DIR = "/home/pf4636/imagenet/train"
OUTPUT_DIR = "low-bit-images"


def quantize(img_array: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits - 1
    quantized = np.round(img_array / 255.0 * levels) / levels * 255
    return quantized.astype(np.uint8)


def pick_random_image(train_dir: str) -> str:
    classes = os.listdir(train_dir)
    cls = random.choice(classes)
    cls_dir = os.path.join(train_dir, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    return os.path.join(cls_dir, random.choice(images))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_path = pick_random_image(TRAIN_DIR)
    print(f"Selected image: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    img.save(os.path.join(OUTPUT_DIR, "original.png"))

    for bits, label in [(8, "uint8"), (4, "uint4"), (2, "uint2"), (1, "uint1_binary")]:
        quantized = quantize(img_array, bits)
        out = Image.fromarray(quantized)
        out.save(os.path.join(OUTPUT_DIR, f"{label}.png"))
        print(f"Saved {label}: {2**bits} levels per channel")


if __name__ == "__main__":
    main()
