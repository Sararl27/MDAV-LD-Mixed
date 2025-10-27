"""
Base classes and utilities for data anonymization methods.

This module provides the foundational abstract classes and data structures
that all anonymization algorithms inherit from. It includes runtime monitoring
and data partitioning capabilities.

Classes:
    BasePartition: Base class for data partitions
    Base: Abstract base class for all anonymization algorithms
"""

from abc import ABC, abstractmethod
import time
import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

from .preprocessing.utils import safe_to_numeric, detect_column_types

if TYPE_CHECKING:
    from .schema import QuasiIdentifiers


class BasePartition(ABC):
    """
    Base class for data partitions used in anonymization algorithms.
    
    This class provides a common interface for handling subsets of data
    during the anonymization process, including indexing and size management.
    
    Attributes:
        data (np.ndarray): The data points in this partition
        indices (np.ndarray): Original indices of the data points
    """
    
    def __init__(self, data, indices=None):
        """
        Initialize a data partition.
        
        Args:
            data (array-like): Data points for this partition
            indices (array-like, optional): Original indices of the data points.
        """
        self.data = np.asarray(data)
        self.indices = np.asarray(indices) if indices is not None else np.arange(len(data))

    @property
    def size(self):
        """Get the number of data points in this partition."""
        return self.data.shape[0]

    @property
    def shape(self):
        """Get the shape of the data in this partition."""
        return self.data.shape


class Base(ABC):
    """
    Abstract base class for all anonymization algorithms.
    
    This class provides common functionality including runtime monitoring,
    configuration management, and the basic interface that all anonymization
    methods must implement.
    
    Attributes:
        max_runtime_seconds (float): Maximum allowed runtime in seconds
        start_check_time (float): When to start checking runtime limits
    """
    QI: Optional['QuasiIdentifiers'] = None
    generalization_technique: dict[str, list[str]] = {'numerical': ['centroid', 'median', 'mode', 'range', 'mask'],
                                       'categorical': ['mode', 'mask']}

    def __init__(self, max_runtime_hours, start_check_time_seconds):
        """
        Initialize the base anonymization algorithm.
        
        Args:
            max_runtime_hours (float): Maximum runtime allowed in hours
            start_check_time_seconds (float): When to start checking runtime limits
        """
        self.max_runtime_seconds = max_runtime_hours * 3600 if max_runtime_hours is not None else None
        self.start_check_time = start_check_time_seconds if max_runtime_hours is not None else None
    @abstractmethod
    def anonymize(self):
        """Run the anonymization algorithm. Must be implemented by subclasses."""
        pass

    def _configure_check_runtime(self):
        """Initialize runtime monitoring."""
        self.estimated_time = None
        self.start_time = time.perf_counter()

    def check_runtime_exceed(self, estimated_time_func, extra_args=None):
        """
        Check if the algorithm is likely to exceed the maximum runtime.
        
        Args:
            estimated_time_func (callable): Function to estimate remaining time
            extra_args (dict, optional): Additional arguments for the estimation function
            
        Returns:
            bool: True if runtime limit is likely to be exceeded
        """
        if not self.max_runtime_seconds:
            return False

        elapsed_time = time.perf_counter() - self.start_time

        if elapsed_time < (self.start_check_time or 0):
            return False

        if extra_args:
            estimated_time, completed_fraction = estimated_time_func(elapsed_time, **extra_args)
        else:
            estimated_time, completed_fraction = estimated_time_func(elapsed_time)
        
        exceed_runtime = estimated_time > self.max_runtime_seconds

        if exceed_runtime:
            self.estimated_time = f"Execution terminated due to exceeding the maximum allowed runtime. Only {completed_fraction * 100:.2f}% of the data has " + \
                                  f"been processed in {int(elapsed_time/3600)} hours. Estimated time to would be done it: {estimated_time/3600:.4f} hours."
        else:
            self.estimated_time = estimated_time

        return exceed_runtime
    
    
    def generalize_clusters(self, X_original, clusters, QI):
        """
        Apply generalization to clusters for k-anonymity and L-diversity compliance.
        
        This method processes each cluster by applying the specified generalization technique
        to quasi-identifier columns while preserving sensitive attributes unchanged.
        
        Args:
            X_original (pd.DataFrame): Original dataset before clustering
            clusters (list): List of clusters containing data point indices
            QI (QuasiIdentifiers): Object containing QI column indices, generalization_technique and types
                - generalization_technique (list): Generalization method for each QI column:
                    - 'centroid': Replace with cluster mean (numerical data)
                    - 'median': Replace with cluster median (numerical data)
                    - 'mode': Replace with most frequent value (any data type)
                    - 'range': Replace with value range, e.g., "20-30" (numerical) or mode (categorical)
                    - 'mask': Replace with masked values using asterisks, e.g., "123**"
                    - 'permutation': Randomly permute values within the cluster (preserves distribution)
            
        Returns:
            pd.DataFrame: Generalized dataset with only QI (generalized) + sensitive (preserved) columns
        """
        # Detect column types for proper processing
        if isinstance(X_original, np.ndarray):
            X_original = pd.DataFrame(X_original, columns=[f"col_{i}" for i in range(X_original.shape[1])])
        elif not isinstance(X_original, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame or a NumPy ndarray. Received: {type(X_original)}")

        categorical_cols, numerical_cols = detect_column_types(X_original, QI)
    
        # Apply safe numeric conversion to numerical columns
        if numerical_cols:
            for col in numerical_cols:
                X_original[col] = safe_to_numeric(X_original[col])

        # Store original DataFrame structure for result reconstruction
        original_columns = X_original.columns
        

        # Process each cluster independently
        for cluster in clusters:
            # Convert cluster indices to numpy array for efficient indexing
            indices = np.array(cluster, dtype=np.int64)
            
            # Skip empty clusters (edge case handling)
            if len(indices) == 0:
                continue
            
            # Extract data points belonging to this cluster
            cluster_data = X_original.iloc[indices]
            
            # Apply generalization to each quasi-identifier column
            for column_idx, column_name in enumerate(original_columns):
                # Skip columns not are QI
                if column_idx not in QI.allowed:
                    continue
                try:
                    # Apply generalization based on column type
                    if column_name in numerical_cols:
                        original_dtype = X_original[column_name].dtype
                        generalized_value = self._generalize_numerical_column(cluster_data[column_name], QI.get_generalization_technique(column_idx)).astype(original_dtype)
                    elif column_name in categorical_cols:
                        generalized_value = self._generalize_categorical_column(cluster_data[column_name], QI.get_generalization_technique(column_idx))
                    else:
                        raise ValueError(f"Column {column_name} is neither numerical nor categorical. Check QI classification.\n - Numerical columns: {numerical_cols}\n - Categorical columns: {categorical_cols}")
                except Exception as e:
                    raise RuntimeError(f"Error generalizing column '{column_name}' in cluster with indices {indices}: {e}") 
                # Apply the generalized value to all points in the cluster
                X_original.loc[indices, column_name] = generalized_value
        
        # Return the generalized dataset as a DataFrame
        return X_original


    def _generalize_numerical_column(self, column_values, generalization_method):
        """
        Apply generalization to numerical column values.
        
        Args:
            column_values (array-like): Values from a numerical column within a cluster
            generalization_method (str): Method to use ('centroid', 'median', 'mode', 'range', 'mask')
            
        Returns:
            Generalized value to replace all values in the cluster
        """
        if generalization_method == 'centroid':
            return np.mean(column_values.astype(np.float32))
        elif generalization_method == 'median':
            return np.median(column_values.astype(np.float32))
        elif generalization_method == 'mode':
            # Find most common value efficiently
            try:
                unique_vals, counts = np.unique(column_values, return_counts=True)
            except TypeError:
                unique_vals, counts = np.unique(column_values.astype(np.float32), return_counts=True)
            
            if len(unique_vals) > 0:
                mode_idx = np.argmax(counts)
                return unique_vals[mode_idx]
            else:
                return column_values[0] if len(column_values) > 0 else None
        elif generalization_method == 'range':
            min_val = np.min(column_values)
            max_val = np.max(column_values)
            if min_val == max_val:
                return str(int(min_val))
            else:
                return f"{int(min_val)}–{int(max_val)}"
        elif generalization_method == 'mask':
            # For numerical values, mask with asterisks (keep first digits)
            str_values = [str(int(val)) for val in column_values.astype(np.float32)]
            if len(str_values) > 0:
                # Find common prefix
                common_prefix = str_values[0]
                for val_str in str_values[1:]:
                    temp_prefix = ""
                    for j, char in enumerate(common_prefix):
                        if j < len(val_str) and val_str[j] == char:
                            temp_prefix += char
                        else:
                            break
                    common_prefix = temp_prefix
                
                # Create masked value
                max_len = max(len(s) for s in str_values)
                mask_len = max_len - len(common_prefix)
                return common_prefix + "*" * mask_len
            else:
                return "*"
        elif generalization_method == 'permutation':
            try:
                return np.random.permutation(column_values)
            except Exception:
                # Fallback: return original values if permutation fails
                raise RuntimeError("Permutation failed for numerical column.")
        else:
            raise ValueError(f"Unknown generalization method for numerical data: {generalization_method}")

    def _generalize_categorical_column(self, column_values, generalization_method):
        """
        Apply generalization to categorical column values.
        
        Args:
            column_values (array-like): Values from a categorical column within a cluster
            generalization_method (str): Method to use ('mode', 'mask')
            
        Returns:
            Generalized value to replace all values in the cluster
        """
        if generalization_method == 'mode':
            # Find most common value efficiently
            try:
                unique_vals, counts = np.unique(column_values, return_counts=True)
            except TypeError:
                # Handle mixed types by converting to strings first
                column_values_str = [str(val) for val in column_values]
                unique_vals, counts = np.unique(column_values_str, return_counts=True)
                
            if len(unique_vals) > 0:
                mode_idx = np.argmax(counts)
                return unique_vals[mode_idx]
            else:
                return column_values[0] if len(column_values) > 0 else None
        elif generalization_method == 'mask':
            # For categorical values, mask with asterisks
            str_values = [str(val) for val in column_values]
            if len(str_values) > 0:
                # Find common prefix
                common_prefix = str_values[0]
                for val_str in str_values[1:]:
                    temp_prefix = ""
                    for j, char in enumerate(common_prefix):
                        if j < len(val_str) and val_str[j] == char:
                            temp_prefix += char
                        else:
                            break
                    common_prefix = temp_prefix
                
                # Create masked value
                max_len = max(len(s) for s in str_values)
                mask_len = max_len - len(common_prefix)
                return common_prefix + "*" * mask_len
            else:
                return "*"
        elif generalization_method == 'permutation':
            try:
                if len(column_values) <= 1:
                    return column_values
                # It can leave some elements in their original positions, which is fine in itself. However, if we want to ensure that no element ever remains fixed, 
                # an attacker could exploit repeated permutations: by observing many independent permutations, they may identify positions where certain elements have 
                # never been placed and use that information to infer the original values
                return np.random.permutation(column_values) 
            except Exception:
                raise RuntimeError("Permutation failed for categorical column.")
        else:
            raise ValueError(f"Unknown generalization method for categorical data: {generalization_method}")

