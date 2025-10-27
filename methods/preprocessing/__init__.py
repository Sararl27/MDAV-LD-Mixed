"""
Preprocessing utilities for data anonymization methods.

This module provides various preprocessing tools including:
- Autoencoder-based dimensionality reduction
- Data normalization and scaling utilities
- Feature extraction and transformation functions
"""

# Import from training module
from .training import TrainConfig, train_encoder

# Import model architectures from subdirectory
from .models.autoencoder import Encoder, Decoder, Autoencoder

# Import utility functions
from .utils import *
