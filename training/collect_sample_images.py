import os
from torchvision import datasets


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")               # MNIST download (gitignored)
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "test_images")  # committed PNGs

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = datasets.MNIST(DATA_DIR, train=False, download=True)
for i in range(100):
    img, label = ds[i]
    img.save(os.path.join(OUTPUT_DIR, f"mnist_{i}_label_{label}.png"))

print(f"Saved 100 samples to {os.path.abspath(OUTPUT_DIR)}")
