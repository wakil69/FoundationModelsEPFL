import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import os

from dataset import ImageDataset
from models.submission import Submission, submission_collate_fn

# --------------------------------------------------------
# METRICS
# --------------------------------------------------------
def calculate_metrics(outputs, predictions, ground_truth):
    """
    Compute evaluation metrics for multiclass classification.
    
    Args:
        outputs (np.ndarray): Probability distributions (N × num_classes)
        predictions (np.ndarray): Predicted class indices
        ground_truth (np.ndarray): True class indices

    Returns:
        dict: accuracy, balanced_accuracy, macro-F1, ROC-AUC (OVO)
    """
    accuracy = accuracy_score(ground_truth, predictions)
    balanced_acc = balanced_accuracy_score(ground_truth, predictions)
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    roc_auc_macro = roc_auc_score(ground_truth,outputs,average='macro',multi_class='ovo')
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1_macro,
        'roc_auc': roc_auc_macro
    }

# --------------------------------------------------------
# TRAINING LOOP FOR ONE FOLD
# --------------------------------------------------------
def train_one_fold(model, train_loader, val_loader, device, num_classes, fold_id):
    """
    Train a model on a single fold of k-fold cross validation.

    Args:
        model: Submission model instance
        train_loader: DataLoader for training patients
        val_loader: DataLoader for validation patients
        device: 'cuda' or 'cpu'
        num_classes: number of output classes
        fold_id: ID of fold (1..5)

    Returns:
        best_f1 (float): Best validation F1 score in this fold
        best_state (dict): State dict of the best model for this fold
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = -1.0
    best_state = None

    epochs = 50  

    for epoch in range(1, epochs + 1):
        model.train()

        for patches, counts, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            logits = model(patches, counts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()

        all_probs = []   # predicted probability distributions
        all_preds = []   # predicted classes
        all_labels = []  # ground truth

        with torch.no_grad():
            for patches, counts, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)

                logits = model(patches, counts)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                gt = labels.cpu().numpy()

                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(gt)

        # Concatenate all validation results
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = calculate_metrics(all_probs, all_preds, all_labels, num_classes)

        print(f"[Fold {fold_id}] Epoch {epoch} | "
              f"Val F1: {metrics['f1_score']:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f}")

        # Track best model for this fold
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_state = model.state_dict()

    return best_f1, best_state

# --------------------------------------------------------
# MAIN 5-FOLD TRAINING
# --------------------------------------------------------

def train_kfold():
    """
    Perform 5-fold cross-validation, where each fold:
        - Creates a new Submission MLP model
        - Trains it on 80% of the patients
        - Validates on 20%
        - Tracks the best F1 score per fold
    After all folds:
        - Saves ONLY the best model across all 5 folds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageDataset(
        dataset_path="/home/elhoujja/CS461_Assignment2/cs461_assignment2_data/part2/data/"
    )

    num_classes = len(np.unique(dataset.labels))

    # 5-fold CV ensures patient-level stratification by ID
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    best_global_f1 = -1.0
    best_global_state = None

    fold_id = 1
    patient_ids = dataset.unique_patient_ids

    # convert patient-level indexing to numpy
    patient_ids = np.array(patient_ids)

    for train_idx, val_idx in kfold.split(patient_ids):

        print(f"\n========== Training Fold {fold_id} ==========\n")

        # Subset of patients
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]

        # Map patient IDs to dataset indices
        train_indices = [i for i, pid in enumerate(dataset.unique_patient_ids) if pid in train_patients]
        val_indices = [i for i, pid in enumerate(dataset.unique_patient_ids) if pid in val_patients]

        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=4,
            shuffle=True,
            collate_fn=submission_collate_fn
        )

        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=4,
            shuffle=False,
            collate_fn=submission_collate_fn
        )

        # New model for each fold
        model = Submission().to(device)

        # Train this fold
        fold_best_f1, fold_best_state = train_one_fold(
            model, train_loader, val_loader, device, num_classes, fold_id
        )

        print(f"[Fold {fold_id}] Best F1: {fold_best_f1:.4f}")

        # Track best across folds
        if fold_best_f1 > best_global_f1:
            best_global_f1 = fold_best_f1
            best_global_state = fold_best_state

        fold_id += 1

    # --------------------------------------------------------
    # Save only the BEST model from all 5 folds
    # --------------------------------------------------------
    job_id = os.environ.get("SLURM_JOB_ID", "kfold")
    save_path = f"/home/elhoujja/CS461_Assignment2/cs461_assignment2_submission/part2/best_kfold_model_{job_id}.pt"
    torch.save({
        'state_dict': best_global_state,
        'embed_dim': model.embed_dim,
        'hidden_dim': model.hidden_dim, 
        'num_classes': model.num_classes
    }, save_path)

    print("\n============================================")
    print(f"Best model across all folds saved to:\n{save_path}")
    print(f"Best F1 across all folds: {best_global_f1:.4f}")
    print("============================================\n")


if __name__ == "__main__":
    train_kfold()
