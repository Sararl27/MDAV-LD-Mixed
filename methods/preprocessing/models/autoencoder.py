"""
Autoencoder architecture models for dimensionality reduction.

Classes:
    Encoder: Neural network encoder model
    Decoder: Neural network decoder model  
    Autoencoder: Complete autoencoder combining encoder and decoder
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.models import Sequential
import keras
from typing import Optional, List

from ..utils import compute_hidden_dims

mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)


@keras.saving.register_keras_serializable(package='autoencoder')
class Encoder(Model):
    """
    Neural network encoder that maps input data to a lower-dimensional latent space.
    
    The encoder uses a series of fully connected layers with optional dropout
    and batch normalization to progressively reduce the dimensionality of the
    input data to the specified latent dimension.
    
    Args:
        original_dim (int): Size of input vectors
        latent_dim (int): Size of latent vectors to encode to
        hidden_dims (List[int], optional): List of hidden layer sizes
        hidden_layers (int, optional): Number of hidden layers to create
        first_hidden (int, optional): Size of the first hidden layer
        dropout (float): Dropout rate to apply after each hidden layer
        use_batchnorm (bool): Whether to apply batch normalization
        activation (str): Activation function for hidden layers
        verbose (int): Verbosity level for debugging
    """
    
    def __init__(
        self,
        original_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        hidden_layers: Optional[int] = None,
        first_hidden: Optional[int] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        activation: str = "relu",
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize the encoder network architecture.
        
        Creates a sequential model with the specified architecture including
        hidden layers, dropout, and batch normalization as configured.
        """
        kwargs.pop('name', None)
        super().__init__(name="encoder", **kwargs)
        self.input_dim = original_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or compute_hidden_dims(latent_dim, original_dim, hidden_layers, first_hidden)
        
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        # Build the encoder architecture
        layers_list = [layers.InputLayer(shape=(self.input_dim,))]
        for h in self.hidden_dims:
            layers_list.append(layers.Dense(h, activation=self.activation))
            if self.dropout > 0:
                layers_list.append(layers.Dropout(self.dropout))
        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Dense(self.latent_dim, activation="tanh"))

        self.encoder = Sequential(layers_list, name="encoder")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, original_dim)
            
        Returns:
            tf.Tensor: Encoded tensor of shape (batch_size, latent_dim)
        """
        return self.encoder(inputs)
    
    def get_config(self):
        """Get configuration for saving/loading the model."""
        return {
            'original_dim': self.input_dim,
            'latent_dim':   self.latent_dim,
            'hidden_dims':  self.hidden_dims,
            'dropout':      self.dropout,
            'use_batchnorm':self.use_batchnorm,
            'activation':   self.activation,}
    
    @classmethod
    def from_config(cls, config):
        """Create an Encoder instance from its configuration."""
        return cls(**config)


@keras.saving.register_keras_serializable(package='autoencoder')
class Decoder(Model):
    """
    Neural network decoder that reconstructs original data from latent representations.
    
    The decoder mirrors the encoder architecture in reverse, progressively increasing
    the dimensionality from the latent space back to the original input dimension.
    
    Args:
        latent_dim (int): Size of latent input vectors
        original_dim (int): Size of output vectors to reconstruct
        hidden_dims (List[int], optional): List of hidden layer sizes
        hidden_layers (int, optional): Number of hidden layers to create
        first_hidden (int, optional): Size of the first hidden layer
        dropout (float): Dropout rate to apply after each hidden layer
        use_batchnorm (bool): Whether to apply batch normalization
        activation (str): Activation function for hidden layers
    """
    
    def __init__(
        self,
        latent_dim: int,
        original_dim: int,
        hidden_dims: Optional[List[int]] = None,
        hidden_layers: Optional[int] = None,
        first_hidden: Optional[int] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        activation: str = "relu",
        **kwargs
    ):
        """
        Initialize the decoder network architecture.
        
        Creates a sequential model that mirrors the encoder in reverse,
        reconstructing the original data from the latent representation.
        """
        kwargs.pop('name', None)
        super().__init__(name="decoder", **kwargs)
        self.latent_dim = latent_dim
        self.output_dim = original_dim
        self.hidden_dims = hidden_dims or compute_hidden_dims(latent_dim, original_dim, hidden_layers, first_hidden, inverted=True)
        
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        # Build the decoder architecture
        layers_list = [layers.InputLayer(shape=(self.latent_dim,))]
        for h in self.hidden_dims:
            layers_list.append(layers.Dense(h, activation=self.activation))
            if self.dropout > 0:
                layers_list.append(layers.Dropout(self.dropout))
        if self.use_batchnorm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Dense(self.output_dim, activation="tanh"))

        self.decoder = Sequential(layers_list, name="decoder")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            inputs (tf.Tensor): Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            tf.Tensor: Reconstructed tensor of shape (batch_size, original_dim)
        """
        return self.decoder(inputs)
    
    def get_config(self):
        return {
            'latent_dim':    self.latent_dim,
            'original_dim':  self.output_dim,
            'hidden_dims':   self.hidden_dims,
            'dropout':       self.dropout,
            'use_batchnorm': self.use_batchnorm,
            'activation':    self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='autoencoder')
class Autoencoder(Model):
    """
    Complete autoencoder combining encoder and decoder models.
    
    This class ties the encoder and decoder together into one end-to-end
    autoencoder that can be trained to learn a compressed representation
    of the input data and reconstruct it.
    
    Args:
        encoder (Encoder, optional): Pre-built encoder model
        decoder (Decoder, optional): Pre-built decoder model
        **kwargs: Arguments to pass to Encoder/Decoder constructors if not provided
    """
    
    def __init__(
            self, 
            encoder: Encoder = None,
            decoder: Decoder = None,
            **kwargs):
        super().__init__(name="autoencoder")
        self.encoder = encoder if encoder is not None else Encoder(**kwargs)
        self.decoder = decoder if decoder is not None else Decoder(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the complete autoencoder.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, original_dim)
            
        Returns:
            tf.Tensor: Reconstructed tensor of shape (batch_size, original_dim)
        """
        z = self.encoder(inputs)
        return self.decoder(z)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder)
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        enc = keras.saving.deserialize_keras_object(config['encoder'])
        dec = keras.saving.deserialize_keras_object(config['decoder'])
        return cls(encoder=enc, decoder=dec)