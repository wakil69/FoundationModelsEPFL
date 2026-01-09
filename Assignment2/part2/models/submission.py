import torch
import torch.nn as nn

class Submission(nn.Module):
    """
    Downstream classifier
    Receives precomputed patch embeddings (3072-dim) and predicts a patient label.
    Uses:
        • A small MLP to transform each patch embedding
        • Mean pooling to aggregate patch features per patient (MIL)
        • A linear classifier on top of the pooled representation
    """
    def __init__(self, embed_dim=3072, hidden_dim=512, num_classes=7):
        """
        Input:
            embed_dim (int): Dimension of each patch embedding (fixed at 3072).
            hidden_dim (int): Hidden size for the MLP.
            num_classes (int): Number of diagnosis classes.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Each patch (3072-dim) is processed independently.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 3072 → 512
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 512 → 512
            nn.ReLU()
        )

        # After pooling patch embeddings, classify patient into 7 classes.
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patches, patch_counts):
        """
        Forward pass for a batch of multiple patients.

        Args:
            patches (Tensor):
                Shape (total_patches_in_batch, 3072)
                All patches from every patient concatenated together.

            patch_counts (List[int]):
                A list where entry i = number of patches belonging to patient i.
                Example:
                    patches = [p1_1, p1_2, p1_3, p2_1, p2_2]
                    patch_counts = [3, 2]

        Workflow:
            1. Run the MLP on every patch.
            2. Split the resulting embeddings into patients using patch_counts.
            3. Mean-pool patches of each patient → patient representation.
            4. Apply final classifier.
        """

        feats = self.mlp(patches) # Patch-level transformation

        patient_embeds = []
        idx = 0
        for count in patch_counts:
            # Take patch features of one patient and average them
            pooled = feats[idx:idx+count].mean(dim=0)
            patient_embeds.append(pooled)
            idx += count

        patient_embeds = torch.stack(patient_embeds)
        logits = self.classifier(patient_embeds) # Patient-level classification

        if not self.training:
            return torch.softmax(logits, dim=1)

        return logits


    @classmethod
    def load_weights(cls, path, device="cpu"):
        """
        Loads a saved checkpoint and reconstructs a model with the right dimensions.
        """
        ckpt = torch.load(path, map_location=device)

        # Recreate model with stored architecture parameters
        model = cls(
            embed_dim=ckpt.get("embed_dim", 3072),
            hidden_dim=ckpt.get("hidden_dim", 512),
            num_classes=ckpt.get("num_classes", 7))

        # Load learned weights
        model.load_state_dict(ckpt["state_dict"])
        print(f"[Submission] Loaded weights from {path}")
        return model

def submission_collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Each batch consists of several patients.
    Each patient has:
        - A variable number of patches
        - A single label

    This function:
        • concatenates all patches into a single tensor
        • records how many patches come from each patient
        • stacks all patient labels

    Returns:
        all_patches: Tensor of shape (total_patches, 3072)
        counts: list of patch counts per patient
        labels: Tensor of shape (num_patients,) with class indices
    """
    all_patches = []
    counts = []
    labels = []

    for patches, label in batch:
        counts.append(patches.shape[0]) # number of patches for this patient
        all_patches.append(patches)
        labels.append(label)

    # Concatenate patches from all patients
    all_patches = torch.cat(all_patches, dim=0)

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return all_patches, counts, labels
