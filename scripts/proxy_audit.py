"""Proxy Audit: Detecting demographic leakage in features.

This script audits the input features to determine if they contain enough
information to accurately predict protected demographic groups (e.g., race, income).
If a simple classifier can predict the demographic group from the features with
high accuracy (AUC >> 0.5), it means the features are acting as a proxy for the
protected attribute.

In a policing context, this is dangerous because the model can reconstruct the
protected attribute and bias its predictions, even if the attribute itself is
excluded from the feature set.
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProxyClassifier(nn.Module):
    """Simple MLP to predict demographic groups from features."""
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def run_proxy_audit(
    features: torch.Tensor,
    groups: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 256,
) -> float:
    """Run the proxy audit.

    Args:
        features: (N, F) tensor of features.
        groups: (N,) tensor of integer group labels.
        num_epochs: Training epochs for the auditor.
        batch_size: Batch size.

    Returns:
        The ROC-AUC score of the auditor (macro-averaged).
    """
    N, F = features.shape
    num_classes = len(torch.unique(groups))
    
    logger.info(f"Starting Proxy Audit on {N} samples, {F} features, {num_classes} groups.")
    
    # Train/test split (80/20)
    indices = torch.randperm(N)
    split = int(0.8 * N)
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    train_dataset = TensorDataset(features[train_idx], groups[train_idx])
    test_dataset = TensorDataset(features[test_idx], groups[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ProxyClassifier(F, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.debug(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
            
    # Evaluate
    model.eval()
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_true.append(y_batch)
            
    all_probs = torch.cat(all_probs).numpy()
    all_true = torch.cat(all_true).numpy()
    
    try:
        # Macro-averaged ROC AUC for multiclass
        from sklearn.metrics import roc_auc_score  # lazy import: optional dep
        auc = roc_auc_score(all_true, all_probs, multi_class="ovo", average="macro")
        logger.info(f"Proxy Audit ROC-AUC: {auc:.4f}")
        
        if auc > 0.65:
            logger.warning("HIGH PROXY RISK: Features leak significant demographic info!")
        elif auc < 0.55:
            logger.info("LOW PROXY RISK: Features do not leak demographic info.")
        else:
            logger.info("MODERATE PROXY RISK.")
    except ValueError as e:
        logger.error(f"Could not compute AUC (maybe only one class in test?): {e}")
        auc = 0.5
        
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy Audit")
    parser.add_argument("--samples", type=int, default=10000, help="Dummy samples")
    parser.add_argument("--features", type=int, default=20, help="Dummy features")
    parser.add_argument("--classes", type=int, default=4, help="Dummy classes")
    args = parser.parse_args()
    
    print("--- Running Dummy Proxy Audit ---")
    # Generate dummy data
    features = torch.randn(args.samples, args.features)
    # Random groups (no leakage) -> AUC should be ~0.5
    groups_random = torch.randint(0, args.classes, (args.samples,))
    
    print("Test 1: Random features (No leakage)")
    run_proxy_audit(features, groups_random)
    
    # Leaky groups (strong proxy) -> AUC should be high
    # Let group equal the sign of the first feature, quantized
    groups_leaky = (features[:, 0] > 0).long()
    
    print("\nTest 2: Leaky features (Strong proxy)")
    run_proxy_audit(features, groups_leaky)
