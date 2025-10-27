"""
Autoencoder training utilities for dimensionality reduction in data anonymization.

This module provides training functionality for autoencoders.

Classes:
    TrainConfig: Configuration dataclass for training parameters
    
Functions:
    train_encoder: Train an autoencoder and return the encoder model
"""

import numpy as np
import pandas as pd
import os
import logging
from dataclasses import dataclass
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from .models.autoencoder import Autoencoder

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    Configuration class for autoencoder training parameters.
    This class encapsulates all the hyperparameters and settings needed
    for training an autoencoder, including data splitting, architecture
    parameters, training parameters, and early stopping criteria.
    Attributes:
        val_size (float): Validation set size as fraction of training data
        test_size (float): Test set size as fraction of total data      
        latent_dim (int): Dimensionality of the latent space
        auto_reduction (str): Strategy for automatic latent_dim ("sqrt" or "cube")
        max_epochs (int): Maximum number of training epochs
        batch_size (int): Training batch size
        inference_batch_size (int): Batch size for inference/prediction
        underfit_loss_thresh (float): Threshold for detecting underfitting
        overfit_loss_thresh (float): Threshold for detecting overfitting
        test_loss_thresh (float): Maximum acceptable test loss
        stored_models_path (str, optional): Directory to save the trained model
        verbose (int): Verbosity level for training output
    """

    val_size: float = 0.1
    test_size: float = 0.2
    latent_dim: int = None  # Automatically set if None
    auto_reduction: str = "cube"  # Strategy for automatic latent_dim: "sqrt" or "cube"
    max_epochs: int = 100
    batch_size: int = 1024
    inference_batch_size: int = 1024  # Batch size for inference
    underfit_loss_thresh: float = 0.05
    overfit_loss_thresh: float = 0.05
    test_loss_thresh: float = 0.1
    stored_models_path: Optional[str] = None #Directory path
    verbose: int = 0

def train_encoder(dataset, raw_input_dim, cfg: TrainConfig = TrainConfig(), target_dtype=np.float32, attempts=3, get_history=False, accept_model_over_threshold=False) -> Any:
    """
    Train an autoencoder on the given dataset and return the encoder model.

    Parameters:
        dataset: DataFrame or numpy array containing the training data (should be preprocessed)
        raw_input_dim: Original input dimension before preprocessing (used to cap latent_dim)
        cfg (TrainConfig): Configuration object with all hyperparameters and thresholds
        target_dtype: Target data type for numpy arrays (default: np.float32)
        attempts (int): Maximum number of training attempts (default: 5)
        get_history (bool): Whether to return training history along with the model

    Returns:
        encoder: The trained encoder sub-model
        history (optional): Training history if get_history=True
        
    Raises:
        ValueError: If training fails after all attempts due to underfitting,
                   overfitting, or poor test performance
    """
    # Convert input data to numpy array with specified dtype
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy(dtype=target_dtype)
    else:
        dataset = np.asarray(dataset, dtype=target_dtype)

    if cfg.latent_dim is None:
        if dataset.shape[1] == 0:
            raise ValueError(f"Dataset has no features (shape: {dataset.shape}). Cannot determine latent_dim.")
        # Compute automatic latent dimension based on selected strategy
        if cfg.auto_reduction.lower() == "sqrt":
            auto_latent = int(np.ceil(np.sqrt(raw_input_dim)))
        elif cfg.auto_reduction.lower() == "cube":
            auto_latent = int(np.ceil(np.cbrt(raw_input_dim)))
        else:
            raise ValueError(f"Unknown auto_reduction strategy '{cfg.auto_reduction}'. Use 'sqrt' or 'cube'.")
        
        if cfg.verbose > 0:
            logger.info(
                f"Setting latent_dim to {cfg.latent_dim} (strategy={cfg.auto_reduction}, auto={auto_latent}, cap={raw_input_dim}, preprocessed_dim={dataset.shape[1]})"
            )
    elif cfg.latent_dim > raw_input_dim:
        raise ValueError(f"latent_dim ({cfg.latent_dim}) cannot exceed raw_input_dim ({raw_input_dim}). Dataset shape: {dataset.shape}")
    # Additional validation
    if cfg.latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {cfg.latent_dim}. Dataset shape: {dataset.shape}")

    if cfg.verbose > 0:
        logger.info(f"Training autoencoder with config: {cfg}")

    # 1) Split into train/test - work directly with numpy arrays
    X_train, X_test = train_test_split(dataset, test_size=cfg.test_size)
    X_train, X_val = train_test_split(X_train, test_size=cfg.val_size)
    input_dim = X_train.shape[1]

    def make_ds_from_generator(arr, batch_size):
        # Build Dataset from the batch‐generator
        return tf.data.Dataset.from_tensor_slices((arr, arr)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds_from_generator(X_train, cfg.batch_size); del X_train
    val_ds = make_ds_from_generator(X_val, cfg.batch_size); del X_val
    test_ds = make_ds_from_generator(X_test, cfg.batch_size); del X_test 

    # Build callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min"
        )
    ]
    if cfg.stored_models_path:
        callbacks.append(
            ModelCheckpoint(
                f'{cfg.stored_models_path}/autoencoder.keras', monitor='val_loss',
                save_best_only=True, mode='min')
        )
    if cfg.verbose > 0:
        logger.info(f"Starting training with {attempts} attempts")
    for _ in range(attempts):

        strategy = (tf.distribute.MirroredStrategy() if len(tf.config.experimental.list_physical_devices('GPU')) > 1 
                    else None)
        if strategy:
            with strategy.scope():
                model = Autoencoder(original_dim=input_dim, latent_dim=cfg.latent_dim)
                model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        else:
            model = Autoencoder(original_dim=input_dim, latent_dim=cfg.latent_dim)
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

        # 3) Fit
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.max_epochs,
            verbose=cfg.verbose,
            callbacks=callbacks
        ).history

        # 4) Assess losses
        best_epoch = int(np.argmin(history['val_loss']))
        train_loss = history['loss'][best_epoch]
        val_loss = history['val_loss'][best_epoch]

        # 2. Determine fitting status
        is_underfitting = (train_loss > cfg.underfit_loss_thresh and
                        val_loss > cfg.underfit_loss_thresh)
        is_overfitting  = (val_loss - train_loss) > cfg.overfit_loss_thresh
        test_loss = model.evaluate(test_ds, verbose=cfg.verbose)

        # 3. If neither under- nor overfitting, check test loss
        if not (is_underfitting or is_overfitting):
            if test_loss <= cfg.test_loss_thresh:
                if cfg.stored_models_path:
                    model.encoder.save(f"{cfg.stored_models_path}/encoder.keras")

                if cfg.verbose:
                    logger.info(f"Training successful - Losses: train={train_loss:.4f}, val={val_loss:.4f}, test={test_loss:.4f}")
                if get_history:
                    return model.encoder, history
                return model.encoder
            
        if cfg.verbose > 0:  
            if is_underfitting:
                logger.warning(f"Underfitting detected (train={train_loss:.4f}, val={val_loss:.4f}, "
                          f"threshold={cfg.underfit_loss_thresh})")
            if is_overfitting:
                logger.warning(f"Overfitting detected (val_loss - train_loss = {val_loss - train_loss:.4f}, "
                          f"threshold={cfg.overfit_loss_thresh})")
            logger.info(f"Increasing max_epochs from {cfg.max_epochs} to {min(cfg.max_epochs * 2, 500)}")
        
        cfg.max_epochs = min(cfg.max_epochs * 2, 500)  # Limit max epochs to prevent infinite loop
    
    error_msgs = []
    if is_underfitting:
        error_msgs.append(f"Underfitting (train={train_loss:.4f}, val={val_loss:.4f}, threshold={cfg.underfit_loss_thresh})")
    if is_overfitting:
        error_msgs.append(f"Overfitting (Δ={val_loss - train_loss:.4f}, threshold={cfg.overfit_loss_thresh})")
    if test_loss > cfg.test_loss_thresh:
        error_msgs.append(f"Test loss too high ({test_loss:.4f})")
    if accept_model_over_threshold:
        logger.warning(f"Accepting model despite thresholds due to accept_model_over_threshold=True. Error: " + "; ".join(error_msgs))
        return model.encoder
    raise ValueError(f"Training failed after {attempts} attempts: " + "; ".join(error_msgs))
