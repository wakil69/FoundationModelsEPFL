import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_train import ExploratoryDataset
from models import resnet50

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------------------
# Load pretrained model
# -----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("→ Loading pretrained ResNet-50...")
model = resnet50(pretrained=True, num_classes=10, device=device).to(device)
model.eval()
print("✓ Model loaded.\n")

# -----------------------------------------
# Transform (same as your code)
# -----------------------------------------
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    ),
])

# -----------------------------------------
# Load corrupted dataset
# -----------------------------------------
print("→ Loading exploratory corrupted dataset...")
corrupt_dataset = ExploratoryDataset(
    "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/exploratory_data.pkl",
    transform=test_tf,
    full=True
)

loader = DataLoader(corrupt_dataset, batch_size=512, shuffle=False, num_workers=4)

print(f"✓ Loaded {len(corrupt_dataset)} samples.\n")

# -----------------------------------------
# Sweep thresholds
# -----------------------------------------
thresholds = np.arange(0.0, 1.01, 0.05)
counts = []

print("→ Sweeping thresholds...\n")

with torch.no_grad():
    # Precompute logits once (no need to redo for each threshold)
    print("Computing logits once for all thresholds...")
    all_logits = []
    for imgs, _ in tqdm(loader, desc="Forward pass"):
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1)
    max_conf, preds = probs.max(dim=1)

    print("\n→ Computing thresholds...\n")

    for t in thresholds:
        count = (max_conf >= t).sum().item()
        counts.append(count)
        print(f"Threshold {t:.2f} → {count} images")

# -----------------------------------------
# Plot results
# -----------------------------------------
plt.figure(figsize=(10,5))
plt.plot(thresholds, counts, marker='o')
plt.xlabel("Confidence Threshold")
plt.ylabel("Number of Selected Images")
plt.title("Effect of Confidence Threshold on Pseudo-Label Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_vs_count.png")

print("\n=====================================")
print("Plot saved as threshold_vs_count.png")
print("=====================================\n")
