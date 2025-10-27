"""
Utility Functions for Data Generation and Processing

This module provides utility functions for generating synthetic datasets,
managing experimental data, calculating time estimates, and processing
anonymized data for the methods comparison framework.

Functions:
    check_variable_existence: Check if an instance has a specific attribute
    calculate_end_time_str: Calculate and format estimated completion time
    delete_file: Safely delete a file if it exists
    get_data: Load or generate dataset based on parameters
    get_anonymized_data: Process partitioned data into clusters and centroids
    estimate_dataset_size: Estimate memory requirements for datasets
    _generate_data: Generate synthetic classification datasets
    _compute_centroid: Compute centroids for data partitions
    _extract_cluster: Extract cluster data from partitions
"""

import logging
from sklearn.datasets import make_classification
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from joblib import Parallel, delayed
from ucimlrepo import fetch_ucirepo
import time
from .write import store_generated_data
from .read import read_generated_data

logger = logging.getLogger(__name__)

def check_variable_existence(instance, variable_name):
    """
    Check if an instance has a specific attribute.
    
    Parameters:
        instance: Object instance to check
        variable_name (str): Name of the attribute to check for
        
    Returns:
        bool: True if attribute exists and is not None, False otherwise
    """
    return getattr(instance, variable_name, None) is not None


def calculate_end_time_str(start_time_obj, t_list, repetitions):
    """
    Calculate and format estimated completion time for experiments.
    
    Uses the average of previous execution times to estimate when
    the current batch of repetitions will complete.
    
    Parameters:
        start_time_obj (datetime): When the current batch started
        t_list (list): List of previous execution times
        repetitions (int): Total number of repetitions planned
        
    Returns:
        str: Formatted estimated completion time (YYYY-MM-DD HH:MM) or "-"
    """
    if not t_list:
        return "-"
    duration_seconds = np.mean(t_list) * repetitions

    # Add the duration as a timedelta
    end_time = start_time_obj + timedelta(seconds=duration_seconds)

    # Return the result in "YYYY-MM-DD HH:MM" format
    return end_time.strftime("%Y-%m-%d %H:%M")


def delete_file(file_path):
    """
    Safely delete a file if it exists.
    
    Parameters:
        file_path (str): Path to the file to delete
    """
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def _generate_data(n_samples, n_features=2, n_informative=2, verbose=False):
    """
    Generate synthetic classification dataset.
    
    Creates a synthetic dataset using scikit-learn's make_classification
    with specified parameters for reproducible experiments.
    
    Parameters:
        n_samples (int): Number of samples to generate
        n_features (int): Total number of features (default: 2)
        n_informative (int): Number of informative features (default: 2)
        verbose (bool): Whether to print timing information
        
    Returns:
        numpy.ndarray: Generated feature matrix X
    """
    if verbose:
        init_time = datetime.now()
        print("Generating data...", end=" ")

    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_repeated=0,
        n_informative=n_informative,
        n_clusters_per_class=2,
        random_state=42  # For reproducibility
    )

    if verbose:
        end_time = datetime.now()
        print(f"Done. Time: {end_time - init_time} [ST: {init_time.strftime('%Y-%m-%d %H:%M:%S')}, ET: {end_time.strftime('%Y-%m-%d %H:%M:%S')}]")    

    return X


def estimate_dataset_size(n_samples, n_features, unit="B"):
    """
    Estimate the memory size of a dataset.
    
    Calculates the approximate memory requirements for storing a dataset
    in memory, considering float64 precision for features.
    
    Parameters:
        n_samples (int): Number of samples in the dataset
        n_features (int): Number of features per sample
        unit (str): Unit for size calculation ("B", "KB", "MB", "GB")
        
    Returns:
        float: Estimated size in the specified unit
        
    Raises:
        ValueError: If an invalid unit is specified
    """
    # Size of X (float64 = 8 bytes per point)
    size_bytes = n_samples * n_features * 8  
    
    # Unit conversion
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    
    if unit not in units:
        raise ValueError(f"Invalid unit. Use one of {list(units.keys())}")
    
    return size_bytes / units[unit]


def get_data(file_path, n=None, n_features=None, ucimlrepo_id=None, verbose=False, min_size_store=1_000_000, MAX_TRIES=10):
    """
    Load or generate dataset based on parameters.
    
    Attempts to load data from UCI ML Repository or from a cached file.
    If neither is available, generates synthetic data and optionally caches it.
    
    Parameters:
        file_path (str): Path to look for/store cached data
        n (int): Number of samples needed
        n_features (int): Number of features needed
        ucimlrepo_id (int, optional): UCI ML Repository dataset ID
        verbose (bool): Whether to print verbose output
        min_size_store (int): Minimum size to cache generated data
        
    Returns:
        numpy.ndarray: Dataset with shape (n, n_features)
    """
    if ucimlrepo_id is not None:
        tries = 0
        while tries < MAX_TRIES:
            try:
                X = fetch_ucirepo(id=ucimlrepo_id).data.features
                break  # Exit loop if successful
            except Exception as e:
                logger.warning(f"Failed to fetch UCI dataset ID {ucimlrepo_id}: {e}. Retrying...")
                time.sleep(3)
                tries += 1
        if tries == MAX_TRIES:
            logger.error(f"Exceeded maximum retries ({MAX_TRIES}) for UCI dataset ID {ucimlrepo_id}. Proceeding to other methods.")
            raise Exception(f"Failed to fetch UCI dataset ID {ucimlrepo_id} after {MAX_TRIES} attempts.")
    else:
        X = read_generated_data(file_path)
        
    
    if X is None:        
        X = _generate_data(n, n_features=n_features, verbose=verbose)
        # Store if size is greater than min_size_store
        if len(X) >= min_size_store:
            store_generated_data(file_path, X, verbose=verbose)
    
    return X


def _compute_centroid(X, partition, numeric_cols):
    """
    Compute the centroid for a data partition.
    
    Parameters:
        X (numpy.ndarray): Full dataset
        partition (list): Indices of samples in this partition
        numeric_cols (numpy.ndarray): Indices of numeric columns
        
    Returns:
        numpy.ndarray: Centroid coordinates for numeric columns
    """
    # try: 
    return np.nanmean(X[partition][:, numeric_cols].astype(float), axis=0)
    # except Exception:
    #     return np.nanmean(X[partition].astype(float), axis=0)


def _extract_cluster(X, partition):
    """
    Extract cluster data from a partition.
    
    Parameters:
        X (numpy.ndarray): Full dataset
        partition (list): Indices of samples in this partition
        
    Returns:
        numpy.ndarray: Data samples in this cluster
    """
    return X[partition]


def get_anonymized_data(X, partitions, QI=None, centroids=True, clusters=True):
    """
    Process partitioned data into clusters and centroids.
    
    Converts the output of anonymization algorithms into clusters and
    optionally computes centroids for numeric columns in parallel.
    
    Parameters:
        X (numpy.ndarray): Original dataset
        partitions (list): List of partitions (lists of sample indices)
        QI: Quasi-identifier specification object
        centroids (bool): Whether to compute centroids
        clusters (bool): Whether to extract cluster data
        
    Returns:
        tuple: (clusters_result, centroids_result) where:
            - clusters_result: List of cluster data arrays (if clusters=True)
            - centroids_result: Array of centroids (if centroids=True)
    """
    # Detect numerical columns from QI specification
    numeric_cols = QI.get_numerical_columns()

    centroids_result = None
    if centroids and numeric_cols.size > 0:            
        centroids_result = np.array(Parallel(n_jobs=-2, prefer="threads")(
            delayed(_compute_centroid)(X, partition, numeric_cols)
            for partition in partitions
        ))

    clusters_result = None
    if clusters:
        clusters_result = Parallel(n_jobs=-2, prefer="threads")(
            delayed(_extract_cluster)(X, partition)
            for partition in partitions
        )

    return clusters_result, centroids_result

