import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_train import ExploratoryDataset
from models import resnet50
import pickle
import numpy as np
from tqdm import tqdm

# ----------------------------
# Load pretrained model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("→ Loading pretrained ResNet-50...")
model = resnet50(
    pretrained=True,
    num_classes=10,
    device=device
).to(device)

model.eval()
print("✓ Model loaded and set to eval mode.\n")

# ----------------------------
# Transform for inference
# ----------------------------
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    ),
])

# ----------------------------a
# Load corrupted dataset
# ----------------------------
print("→ Loading exploratory (corrupted) dataset...")
corrupt_dataset = ExploratoryDataset(
    "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/exploratory_data.pkl",
    transform=test_tf,
    full=True
)

total_corrupted = len(corrupt_dataset)
print(f"✓ Loaded {total_corrupted} corrupted samples.\n")

loader = DataLoader(corrupt_dataset, batch_size=512, shuffle=False, num_workers=4)

# ----------------------------
# Pseudo-labeling
# ----------------------------
threshold = 0.9 # 0.75
pseudo_labeled = []

print("→ Starting pseudo-label generation...")
print(f"  Threshold = {threshold}\n")

running_total = 0

with torch.no_grad():
    for batch_idx, (imgs, _) in enumerate(tqdm(loader, desc="Pseudo-labeling")):

        print(f"\n--- Batch {batch_idx+1} ---")
        imgs = imgs.to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)

        confidences, preds = probs.max(dim=1)
        mask = confidences >= threshold

        num_selected = mask.sum().item()
        print(f"Batch size: {len(imgs)}")
        print(f"High-confidence samples (>= {threshold}): {num_selected}")

        if num_selected == 0:
            print("→ No confident samples in this batch.")
            continue

        selected_imgs = imgs[mask].cpu()
        selected_preds = preds[mask].cpu()
        selected_confs = confidences[mask].cpu()

        # Print confidence stats
        print(f"  Max confidence in batch: {confidences.max().item():.4f}")
        print(f"  Mean confidence in batch: {confidences.mean().item():.4f}")

        for i, (img_tensor, label) in enumerate(zip(selected_imgs, selected_preds)):
            # undo normalization → uint8 numpy array
            img = img_tensor.clone()
            img[0] = img[0] * 0.2471 + 0.4914
            img[1] = img[1] * 0.2435 + 0.4822
            img[2] = img[2] * 0.2616 + 0.4465
            img = img.clamp(0, 1)

            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            pseudo_labeled.append((img_np, int(label)))
            running_total += 1

        print(f"Running total pseudo-labeled so far: {running_total}")

# ----------------------------
# Save PKL file
# ----------------------------
save_path = "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/pseudo_labeled_corrupted.pkl"

with open(save_path, "wb") as f:
    pickle.dump(pseudo_labeled, f)

print("\n====================================")
print(f"✓ Finished pseudo-labeling.")
print(f"✓ Total high-confidence samples saved: {len(pseudo_labeled)}")
print(f"✓ Saved to: {save_path}")
print("====================================\n")

