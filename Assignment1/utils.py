
from typing import Union
from pathlib import Path
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset


try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    print("sklearnex not installed, using standard sklearn")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE


default_transform = T.Compose([
    T.ToImage(),  # convert to a torch image tensor
    T.ToDtype(torch.float32, scale=True),  # scale [0,255] → [0,1] and cast to float32
])

class SimCLRTransform:

    def __init__(self, size=64, s=0.5, blur_p=0.5):
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        k = 3 if size <= 32 else 5
        base = [
            T.ToImage(),
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p),
            T.ToDtype(torch.float32, scale=True)
        ]
        self.train_transform = T.Compose(base)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def simclr_collate_fn(batch):
    xs1, xs2, ys = [], [], []
    for (x1, x2), y in batch:
        xs1.append(x1)
        xs2.append(x2)
        ys.append(y)
    return torch.stack(xs1), torch.stack(xs2), torch.tensor(ys)

class ImageDatasetNPZ(Dataset):
    def __init__(self, data_path: Union[str, Path], transform=default_transform):
        self.load_from_npz(data_path)
        self.transform = transform

    def load_from_npz(self, data_path: Union[str, Path]):
        data = np.load(data_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
def extract_features_and_labels(model, dataloader, normalize=False):
    """
    Extract features and labels from a dataloader using the given model.
    model: an encoder model taking as input a batch of images (batch_size, channels, height, width) and outputing either a batch of feature vectors (batch_size, feature_dim) or a list/tuple in which the first element is the batch of feature vectors (batch_size, feature_dim)
    dataloader: a PyTorch dataloader providing batches of (images, labels)
    returns: features (num_samples, feature_dim), labels (num_samples,)
    """
    features = []
    labels = []

    device = next(model.parameters()).device

    for batch in tqdm(dataloader, disable=True):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            feats = model.get_features(x)
        features.append(feats.cpu())
        labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    if normalize:
        features = F.normalize(features, dim=1)

    return features, labels

def run_knn_probe(train_features, train_labels, test_features, test_labels, return_preds=False):
    """
    Runs a k-NN probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_features, train_labels)
    test_preds = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    if return_preds:
        return accuracy * 100, test_preds
    return accuracy * 100

def run_linear_probe(train_features, train_labels, test_features, test_labels):
    """
    Runs a linear probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logreg.fit(train_features, train_labels)
    test_preds = logreg.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy * 100

def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

############################ For Visualisation 

map_clscloc = pd.read_csv("/home/benabdi/CS461_Assignment/data/map_clsloc.txt", sep=" ", header = None)

def get_true_label(label_id):
    row = map_clscloc[map_clscloc[1] == (label_id+1)]
    if row.empty:
        return None  # handle case where label_id not found
    label_value = row.iloc[0, 2]
    return label_value

def visualize_predictions(model, train_loader, val_loader, save_path, title, device="cuda", num_samples=10):
    """
    Shows sample validation images with their predicted (k-NN) labels.
    """
    model.eval().to(device)

    train_features, train_labels = extract_features_and_labels(model, train_loader)
    val_features, val_labels = extract_features_and_labels(model, val_loader)

    train_features,train_labels,val_features,val_labels = train_features.numpy(),train_labels.numpy(),val_features.numpy(),val_labels.numpy()

    _, preds = run_knn_probe(
        train_features,
        train_labels,
        val_features,
        val_labels,
        return_preds=True
    )
    indices = np.random.choice(len(val_labels), num_samples, replace=False) # randomly choose label classes
    images = [val_loader.dataset[i][0] for i in indices] # load first image of these indices
    preds = preds[indices] # extract prediction labels 
    true_labels = val_labels[indices] # extract true labels

    # Map the number labels to the actual text labe
    plt.figure(figsize=(14, 8))
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        true_label_text = get_true_label(true_labels[i])
        predicted_label_text = get_true_label(preds[i])
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pred: {preds[i]} ({predicted_label_text}), \n True: {true_labels[i]} ({true_label_text})")
    plt.suptitle(title)
    plt.tight_layout()

    # --- Save figure ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight") 
        print(f"Saved visualization to {save_path}")

    plt.show()

def visualize_tsne(model, data_loader, save_path, device="cuda"):
    """
    Visualize learned image embeddings using t-SNE (no subsampling).

    Args:
        model (torch.nn.Module): Trained encoder or full SimCLR model.
        data_loader (DataLoader): DataLoader with plain (non-augmented) images.
        device (str): "cuda" or "cpu".
    """
    model.eval().to(device)

    # Extract features and labels
    features, labels = extract_features_and_labels(model, data_loader, normalize=True)
    features, labels = features.cpu().numpy(), labels.cpu().numpy()

    print("Running t-SNE embedding ")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, verbose=1)

    embeddings = tsne.fit_transform(features)

    # Plot the t-SNE projection
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    plt.title("t-SNE Visualization of Learned Representations")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved t-SNE plot to {save_path}")

def visualize_tsne_highlight_labels(model, data_loader, save_path=None, device="cuda", number = 20):
    """
    Visualize learned image embeddings using t-SNE, coloring 10 random labels.

    Args:
        model (torch.nn.Module): Trained encoder or full SimCLR model.
        data_loader (DataLoader): DataLoader with plain (non-augmented) images.
        save_path (str, optional): Where to save the figure.
        device (str): "cuda" or "cpu".
    """
    model.eval().to(device)

    # Extract features and labels
    features, labels = extract_features_and_labels(model, data_loader, normalize=True)
    features, labels = features.cpu().numpy(), labels.cpu().numpy()

    print("Running t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, verbose=1)
    embeddings = tsne.fit_transform(features)

    # Choose up to 10 random labels
    unique_labels = np.unique(labels)
    chosen_labels = random.sample(list(unique_labels), number)

    # Create a color map for the chosen labels
    cmap = plt.cm.get_cmap("tab20", number)
    label_to_color = {label: cmap(i) for i, label in enumerate(chosen_labels)}

    # Assign colors: gray for others
    colors = np.array([
        label_to_color[label] if label in chosen_labels else (0.8, 0.8, 0.8, 0.3)
        for label in labels
    ])

    # Plot the t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        s=6,
        alpha=0.8
    )

    # Add legend for the colored labels
    for label in chosen_labels:
        label_text = get_true_label(label)
        plt.scatter([], [], color=label_to_color[label], label=f"Label {label} ({label_text})")
    plt.legend(title="Highlighted Labels", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("t-SNE Visualization (20 Random Labels Highlighted)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    print(f"Saved t-SNE plot to {save_path}" if save_path else "Displayed t-SNE plot")

    """
    Visualize learned image embeddings using UMAP (no subsampling).

    Args:
        model (torch.nn.Module): Trained encoder or full SimCLR model.
        data_loader (DataLoader): DataLoader with plain (non-augmented) images.
        device (str): "cuda" or "cpu".
    """
    model.eval().to(device)

    # Extract features and labels
    features, labels = extract_features_and_labels(model, data_loader, normalize=True)
    features, labels = features.cpu().numpy(), labels.cpu().numpy()

    print("Running UMAP embedding...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embeddings = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    plt.title("UMAP Visualization of Learned Representations")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    if save == True:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved UMAP plot to {save_path}")