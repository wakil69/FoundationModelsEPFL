import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataset_train import CIFAR10Dataset, PseudoLabeledDataset
from models import resnet50
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
import os


# ==========================================================
# TRANSFORMS
# ==========================================================
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    ),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    ),
])


# ==========================================================
# LOAD PSEUDO-LABELED DATA AND SPLIT
# ==========================================================
PSEUDO_PATH = "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/pseudo_labeled_corrupted.pkl"

print("→ Loading pseudo-labeled corrupted data...")
with open(PSEUDO_PATH, "rb") as f:
    pseudo_data = pickle.load(f)  # list of (img, label)

imgs = np.array([img for (img, lbl) in pseudo_data])
lbls = np.array([lbl for (img, lbl) in pseudo_data])

total_pseudo = len(imgs)
print(f"Total pseudo-labeled samples: {total_pseudo}")

TRAIN_PSEUDO = int(total_pseudo*0.8) #400000
assert TRAIN_PSEUDO <= total_pseudo, "Not enough pseudo-labeled samples."

rng = np.random.default_rng(42)
indices = rng.permutation(total_pseudo)

train_idx = indices[:TRAIN_PSEUDO]
test_idx = indices[TRAIN_PSEUDO:]

train_pseudo_imgs = imgs[train_idx]
train_pseudo_lbls = lbls[train_idx]

test_pseudo_imgs = imgs[test_idx]
test_pseudo_lbls = lbls[test_idx]

print(f"Pseudo-labeled training samples: {len(train_pseudo_imgs)}")
print(f"Pseudo-labeled test samples:     {len(test_pseudo_imgs)}")


# Build datasets
pseudo_train = PseudoLabeledDataset(train_pseudo_imgs, train_pseudo_lbls, transform=train_tf)
pseudo_test  = PseudoLabeledDataset(test_pseudo_imgs,  test_pseudo_lbls,  transform=test_tf)


# ==========================================================
# CIFAR-10 TRAIN AND TEST
# ==========================================================
print("→ Loading CIFAR-10 train...")
cifar_train = CIFAR10Dataset(
    "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/cifar-10-batches-py",
    train=True,
    transform=train_tf
)

print("→ Loading CIFAR-10 test...")
cifar_test = CIFAR10Dataset(
    "/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part1/data/cifar-10-batches-py",
    train=False,
    transform=test_tf
)

train_dataset = ConcatDataset([cifar_train, pseudo_train])
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)

test_loader_clean = DataLoader(cifar_test, batch_size=1024, shuffle=False, num_workers=4)

test_loader_corrupted = DataLoader(pseudo_test, batch_size=1024, shuffle=False, num_workers=4)


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total


# ==========================================================
# TRAINING
# ==========================================================
def train_model(device="cuda"):

    print("\n→ Loading pretrained ResNet-50")
    model = resnet50(pretrained=True, num_classes=10, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    best_corrupt_acc = 0.0
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    save_path = f"resnet50_best_corrupted_{job_id}.pt"

    for epoch in range(10):

        # ---------------------------
        # TRAINING
        # ---------------------------
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # ---------------------------
        # EVALUATION (EVERY EPOCH)
        # ---------------------------
        clean_acc   = evaluate(model, test_loader_clean, device)
        corrupt_acc = evaluate(model, test_loader_corrupted, device)

        print(
            f"Epoch {epoch}: "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"CIFAR10_acc={clean_acc:.4f} | "
            f"Corrupted_acc={corrupt_acc:.4f}"
        )

        # ---------------------------
        # SAVE BEST MODEL BASED ON CORRUPTED ACCURACY
        # ---------------------------
        if corrupt_acc > best_corrupt_acc:
            best_corrupt_acc = corrupt_acc
            torch.save(model.state_dict(), save_path)
            print(f"🔥 New BEST (corrupted) model saved → {save_path} "
                  f"(corrupted_acc={best_corrupt_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best CORRUPTED accuracy = {best_corrupt_acc:.4f}")
    print(f"Model saved to: {save_path}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(device)
