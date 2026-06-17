"""CIVIC-SAFE Model package — complete spatiotemporal ZINB forecasting system.

Architecture variants:
  1. Sequential: GATv2 (spatial) → Causal Transformer (temporal) → MFFM → ZINB Head
  2. Unified:    Spatiotemporal Graph Transformer → MFFM → ZINB Head (joint space-time)

Modules:
  civicsafe_model:  Master model assembly (sequential architecture)
  spatial:          GATv2 spatial encoder with dual adjacency
  temporal:         Causal Transformer temporal encoder
  st_transformer:   Spatiotemporal Graph Transformer (unified architecture)
  feature_mixer:    Multi-Factor Feature Mixer (MFFM)
  zinb_head:        3-parameter ZINB projection head
  zinb_loss:        Numerically stable ZINB NLL loss
  dataset:          Sliding-window dataset with chronological splits
  graph:            Dual adjacency graph builder
"""

from __future__ import annotations

from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
from civicsafe.models.feature_mixer import FeatureMixer
from civicsafe.models.graph import (
    build_adjacency_from_panel,
    build_adjacency_from_synthetic,
)
from civicsafe.models.spatial import SpatialEncoder
from civicsafe.models.st_transformer import SpatiotemporalTransformer
from civicsafe.models.temporal import SinusoidalPositionalEncoding, TemporalEncoder
from civicsafe.models.zinb_head import ZINBHead
from civicsafe.models.zinb_loss import ZINBLoss

__all__ = [
    "CivicSafeModel",
    "CrimeWindowDataset",
    "FeatureMixer",
    "SinusoidalPositionalEncoding",
    "SpatialEncoder",
    "SpatiotemporalTransformer",
    "TemporalEncoder",
    "ZINBHead",
    "ZINBLoss",
    "build_adjacency_from_panel",
    "build_adjacency_from_synthetic",
    "create_chronological_splits",
]
