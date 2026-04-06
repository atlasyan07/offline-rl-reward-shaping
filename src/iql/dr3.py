"""DR3: Deep Representation Rank Regularization.

Prevents representation collapse by maintaining feature diversity.

Reference: https://arxiv.org/abs/2209.08016
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DR3Loss(nn.Module):
    """DR3 regularization loss.

    Encourages high-rank representations by penalizing low singular values
    of the feature covariance matrix.

    Loss = -log(det(Cov(Z) + εI))
         ≈ -Σ log(σ_i² + ε)

    where σ_i are singular values of the feature matrix Z.

    This prevents collapse by:
    1. Encouraging features to span a high-dimensional subspace
    2. Avoiding degenerate solutions where all states map to similar features
    3. Maintaining diversity in learned representations
    """

    def __init__(self, eps: float = 1e-4, alpha: float = 1.0):
        """Initialize DR3 loss.

        Args:
            eps: Regularization term for numerical stability
            alpha: Weight for DR3 loss term
        """
        super().__init__()
        self.eps = float(eps)
        self.alpha = float(alpha)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute DR3 loss on batch of features.

        Args:
            features: (B, D) batch of feature vectors

        Returns:
            loss: scalar DR3 loss
        """
        batch_size = int(features.size(0))
        feature_dim = int(features.size(-1))

        # Normalize features (zero mean, unit variance per dimension)
        features = features - features.mean(dim=0, keepdim=True)
        features = features / (features.std(dim=0, keepdim=True) + 1e-8)

        # Compute covariance matrix: Cov = (1/B) Z^T Z
        # Shape: (D, D)
        cov = (features.T @ features) / batch_size

        # Add regularization: Cov + εI
        identity = torch.eye(feature_dim, device=features.device, dtype=features.dtype)
        cov_reg = cov + self.eps * identity

        # Compute eigenvalues (more stable than det for high dimensions)
        eigenvalues = torch.linalg.eigvalsh(cov_reg)

        # DR3 loss: -log(det(Cov + εI)) = -Σ log(λ_i)
        # Negative log encourages large eigenvalues (high rank)
        loss = -torch.log(eigenvalues + 1e-8).sum()

        return self.alpha * loss

    def get_metrics(self, features: torch.Tensor) -> dict:
        """Compute diagnostic metrics for representation collapse.

        Args:
            features: (B, D) batch of feature vectors

        Returns:
            Dictionary with:
                - rank: Effective rank (normalized)
                - top_eigenvalue: Largest eigenvalue
                - eigenvalue_ratio: Ratio of largest to smallest eigenvalue
                - feature_std: Average std across dimensions
        """
        batch_size = int(features.size(0))
        feature_dim = int(features.size(-1))

        # Normalize features
        features = features - features.mean(dim=0, keepdim=True)
        features_std = features.std(dim=0).mean().item()

        # Covariance
        cov = (features.T @ features) / batch_size
        identity = torch.eye(feature_dim, device=features.device, dtype=features.dtype)
        cov_reg = cov + self.eps * identity

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov_reg)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive only

        # Effective rank: exp(entropy of normalized eigenvalues)
        probs = eigenvalues / eigenvalues.sum()
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        effective_rank = torch.exp(entropy).item()
        normalized_rank = effective_rank / feature_dim

        # Eigenvalue ratio (condition number)
        eigenvalue_ratio = (eigenvalues.max() / (eigenvalues.min() + 1e-8)).item()

        return {
            "rank": normalized_rank,
            "top_eigenvalue": eigenvalues.max().item(),
            "eigenvalue_ratio": eigenvalue_ratio,
            "feature_std": features_std,
        }
