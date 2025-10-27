"""
Preprocessing utilities for data preparation and transformation.

This module provides utility functions for data preprocessing including:
- Tabular data preprocessing with automatic categorical/numerical detection
- Feature scaling and normalization
- Neural network architecture utilities
- Data type handling and conversion functions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
from typing import Optional, List, Type
import category_encoders as ce


def safe_to_numeric(series):
    """
    Safely convert a pandas Series to numeric, returning the original series if conversion fails.
    
    This function replaces the deprecated pd.to_numeric(errors='ignore') pattern.
    
    Args:
        series (pd.Series): Input pandas Series to convert
        
    Returns:
        pd.Series: Converted numeric series or original series if conversion fails
    """
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def detect_column_types(data, QI=None):
    """
    Detect or extract categorical and numerical column information.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        QI (QuasiIdentifiers, optional): QI object with column type information
        
    Returns:
        tuple: (categorical_columns, numerical_columns) as lists of column names
    """
    if QI is None:
        # Ensure DataFrame is numeric where possible
        data = data.apply(safe_to_numeric)
        
        # Detect column types
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = [
            col for col in data.columns
            if col not in cat_cols and np.issubdtype(data[col].dtype, np.number)
        ]
    else:
        cat_cols = [data.columns[i] for i in QI.get_categorical_columns() if i < len(data.columns)]
        num_cols = [data.columns[i] for i in QI.get_numerical_columns() if i < len(data.columns)]
    
    return cat_cols, num_cols


def encode_categorical_columns(data, cat_cols, encoder_type="onehot", target_dtype=np.float32):
    """
    Handle categorical column encoding.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        cat_cols (list): List of categorical column names
        encoder_type (str): "onehot" or "binary"
        target_dtype (np.dtype): Target data type
        
    Returns:
        pd.DataFrame: Encoded categorical data
    """
    if not cat_cols:
        return pd.DataFrame(index=data.index, dtype=target_dtype)
    
    if encoder_type.lower() == "binary":
        encoder = ce.BinaryEncoder(cols=cat_cols, return_df=True)
        df_encoded = encoder.fit_transform(data[cat_cols])
        df_encoded = df_encoded.astype(target_dtype)
        df_encoded.index = data.index
    else:
        encoder = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
        encoded = encoder.fit_transform(data[cat_cols].astype(str))
        df_encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(cat_cols),
            index=data.index,
            dtype=target_dtype
        )
    
    return df_encoded


def scale_numerical_columns(data, num_cols, scaler_type="StandardScaler", target_dtype=np.float32):
    """
    Handle numerical column scaling.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        num_cols (list): List of numerical column names
        scaler_type (str): "StandardScaler", "MinMaxScaler", or "RobustScaler"
        target_dtype (np.dtype): Target data type
        
    Returns:
        pd.DataFrame: Scaled numerical data
        
    Raises:
        ValueError: If columns cannot be converted to numeric
    """
    if not num_cols:
        return pd.DataFrame(index=data.index, dtype=target_dtype)
    
    # Ensure numeric conversion
    for col in num_cols:
        data.loc[:, col] = safe_to_numeric(data[col]).astype(np.float32)
    
    # Select scaler based on type
    scaler_type_lower = scaler_type.lower()
    if scaler_type_lower == "standardscaler":
        scaler = StandardScaler()
    elif scaler_type_lower == "minmaxscaler":
        scaler = MinMaxScaler()
    elif scaler_type_lower == "robustscaler":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'. Must be 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'.")
    
    try:
        scaled = scaler.fit_transform(data[num_cols])
        return pd.DataFrame(scaled, columns=num_cols, index=data.index, dtype=target_dtype)
    except ValueError as e:
        raise ValueError(f"Failed to convert numerical columns {num_cols} to numeric. "
                        f"Check if QI classification is correct. Error: {e}")


def preprocess_tabular_df(data, num_scaler="StandardScaler", cat_encoder="onehot", convert_to_numpy=True, target_dtype=np.float32, QI=None):
    """
    Main preprocessing function for tabular data.
    
    This function orchestrates the preprocessing pipeline by:
    1. Converting input to DataFrame format
    2. Detecting column types (categorical vs numerical)
    3. Encoding categorical columns
    4. Scaling numerical columns
    5. Returning data in the requested format
    
    Args:
        data (pd.DataFrame or np.ndarray): Input tabular data
        num_scaler (str): "StandardScaler", "MinMaxScaler", or "RobustScaler"
        cat_encoder (str): "onehot" for OneHotEncoder or "binary" for BinaryEncoder
        convert_to_numpy (bool): Whether to return numpy array or DataFrame
        target_dtype (np.dtype): Target numpy dtype for numeric data (default: np.float32)
        QI (QuasiIdentifiers, optional): QI object with column type information
        only_fix_data_types (bool): If True, only fix data types without encoding/scaling

    Returns:
        Processed data (pd.DataFrame or np.ndarray depending on convert_to_numpy)
        Always returns data in target_dtype when convert_to_numpy=True
    """
    # Input validation and conversion
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
    elif not isinstance(data, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame or a NumPy ndarray. Received: {type(data)}")
    
    # Detect column types
    cat_cols, num_cols = detect_column_types(data, QI)
    
    # Process columns
    df_cat = encode_categorical_columns(data, cat_cols, cat_encoder, target_dtype)
    df_num = scale_numerical_columns(data, num_cols, num_scaler, target_dtype)
    
    # Combine results
    processed_data = pd.concat([df_cat, df_num], axis=1)
    
    # Return in requested format
    return processed_data.to_numpy(dtype=target_dtype) if convert_to_numpy else processed_data



def compute_hidden_dims(
    lower_dim: int,
    higher_dim: int,
    hidden_layers: Optional[int] = None,
    first_hidden: Optional[int] = None,
    inverted: bool = False
) -> List[int]:
    """
    Automatically compute hidden layer dimensions for neural networks.
    
    This function creates a list of hidden layer sizes by linearly interpolating
    between the first hidden layer size and the output dimension. It's designed
    to create smooth transitions in neural network architectures.
    
    Args:
        lower_dim (int): Smaller dimension (typically latent or input)
        higher_dim (int): Larger dimension (typically input or output)
        hidden_layers (int, optional): Number of hidden layers to create
        first_hidden (int, optional): Size of the first hidden layer
        inverted (bool): If True, return dimensions in descending order (for decoders)
        
    Returns:
        List[int]: List of hidden layer dimensions
        
    Raises:
        ValueError: If lower_dim > higher_dim or first_hidden is out of range
    """
    if lower_dim > higher_dim:
        raise ValueError(f"lower_dim ({lower_dim}) must be <= higher_dim ({higher_dim})")

    # Auto-select number of layers based on expansion factor
    if hidden_layers is None:
        expansion_factor = higher_dim / lower_dim
        if expansion_factor <= 25:
            hidden_layers = 1
        elif expansion_factor <= 75:
            hidden_layers = 2
        else:
            hidden_layers = 3

    # Determine first_hidden if not specified (use midpoint)
    if first_hidden is None:
        first_hidden = (lower_dim + higher_dim) // 2
    if not (lower_dim <= first_hidden <= higher_dim):
        raise ValueError(f"first_hidden ({first_hidden}) must be between lower_dim ({lower_dim}) and higher_dim ({higher_dim})")

    # Compute step size and layer dimensions using linear interpolation
    step = (higher_dim - first_hidden) / (hidden_layers + 1)
    dims = [
        int(first_hidden + step * (i + 1))
        for i in range(hidden_layers)
    ]

    return list(reversed(dims)) if inverted else dims


