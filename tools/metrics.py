"""
Anonymization evaluation metrics for data privacy and performance assessment.

This module provides metrics to evaluate the quality and performance of
anonymization algorithms, focusing on information loss (NCP) and execution
performance metrics including speedup calculations.

Functions:
    calculate_ncp: Normalized Certainty Penalty for information loss
    calculate_performance_metrics: Runtime and efficiency metrics
    calculate_speedup_metrics: Speedup analysis for parallel algorithms
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Union

# Set up logger
logger = logging.getLogger(__name__)



def calculate_ncp(
    instance, 
    X: Union[np.ndarray, 'pd.DataFrame'], 
    clusters: List[np.ndarray],
    X_gen: Optional[Union[np.ndarray, 'pd.DataFrame']] = None,
    weights: Optional[Dict[int, float]] = None
) -> Optional[float]:    
    """
    Calculate NCP (Normalized Certainty Penalty) for anonymized data.
    
    NCP measures information loss in anonymized data by comparing the precision
    of generalized values to the original domain ranges.
    
    Args:
        instance: Algorithm instance with QI (quasi-identifier) information
        X: Original dataset (numpy array or DataFrame)
        clusters: List of clusters/groups from anonymization (list of index arrays)
        X_gen: Optional generalized dataset. If provided, uses true generalized ranges.
               For numerical QIs: each cell should be [min, max]
               For categorical QIs: each cell should be iterable of categories
        weights: Optional dict mapping attribute index -> weight (should sum to 1)
        use_sa: Whether to include sensitive attributes (default False for standard NCP)
    
    Returns:
        float: NCP value between 0 (no information loss) and 1 (maximum loss)
        None: If calculation fails or invalid inputs
    """
    try:
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, 'values') else np.asarray(X)
        
        # Prepare generalized array if provided
        X_gen_array = None
        if X_gen is not None:
            X_gen_array = X_gen.values if hasattr(X_gen, 'values') else np.asarray(X_gen)
        
        if not hasattr(instance, 'QI') or not hasattr(instance.QI, 'allowed'):
            logger.warning("Instance missing QI.allowed attribute")
            return None

        all_indices = np.array(instance.QI.allowed, dtype=int)
        
        
        if all_indices.max() >= X_array.shape[1]:
            raise ValueError(f"Attribute index out of bounds. Max index {all_indices.max()} >= number of columns {X.shape[1]}.")
        

        m = len(all_indices)
        if m == 0 or not clusters:
            return 0.0
        
        # Set up weights (uniform if not provided)
        if weights is None:
            weights = {idx: 1.0/m for idx in all_indices}
        else:
            # Normalize weights to sum to 1
            total_w = sum(weights.get(idx, 0) for idx in all_indices)
            if abs(total_w - 1.0) > 1e-6:
                weights = {idx: weights.get(idx, 0)/total_w for idx in all_indices}
        
        n_records = len(X_array)
        total_ncp = 0.0
        
        # Process each attribute
        for attr_idx in all_indices:
            try:
                # Get weight for this attribute
                weight = weights.get(attr_idx, 1.0/m)
                

                original_col = X_array[:, attr_idx]
                attr_ncp = 0.0
                
                if instance.QI.is_numerical(attr_idx):
                    # Numerical processing
                    original_float = original_col.astype(float)
                    finite_original = original_float[~np.isnan(original_float)]
                    
                    if len(finite_original) == 0:
                        continue
                        
                    domain_range = np.ptp(finite_original)
                    if domain_range == 0:
                        continue
                    
                    # Calculate NCP for each cluster
                    for cluster in clusters:
                        cluster_size = len(cluster)
                        if cluster_size == 0:
                            continue
                        
                        # Use X_gen if available and valid, otherwise fallback to original method
                        if X_gen_array is not None:
                            try:
                                gen_val = X_gen_array[cluster[0], attr_idx]
                                if hasattr(gen_val, '__len__') and len(gen_val) == 2:
                                    low, high = float(gen_val[0]), float(gen_val[1])
                                    if not (np.isnan(low) or np.isnan(high)):
                                        cluster_range = high - low
                                        cluster_ncp = cluster_range / domain_range
                                        attr_ncp += cluster_ncp * (cluster_size / n_records)
                                        continue
                            except (IndexError, TypeError, ValueError):
                                pass  # Fallback to original method
                        
                        # Fallback: calculate range from original data in cluster
                        cluster_data = original_float[cluster]
                        finite_cluster = cluster_data[~np.isnan(cluster_data)]
                        if len(finite_cluster) <= 1:
                            cluster_ncp = 0.0
                        else:
                            cluster_range = np.ptp(finite_cluster)
                            cluster_ncp = cluster_range / domain_range
                        
                        attr_ncp += cluster_ncp * (cluster_size / n_records)
                
                else:
                    # Categorical processing
                    try:
                        total_categories = len(np.unique(original_col))
                    except TypeError:
                        # Handle mixed types
                        original_col_str = np.array([str(x) for x in original_col])
                        total_categories = len(np.unique(original_col_str))
                    
                    if total_categories <= 1:
                        continue
                    
                    # Calculate NCP for each cluster
                    for cluster in clusters:
                        cluster_size = len(cluster)
                        if cluster_size == 0:
                            continue
                        
                        # Use X_gen if available and valid, otherwise fallback to original method
                        if X_gen_array is not None:
                            try:
                                gen_val = X_gen_array[cluster[0], attr_idx]
                                if hasattr(gen_val, '__len__'):
                                    k_cats = len(gen_val)
                                    cluster_ncp = k_cats / total_categories
                                    attr_ncp += cluster_ncp * (cluster_size / n_records)
                                    continue
                            except (IndexError, TypeError, ValueError):
                                pass  # Fallback to original method
                        
                        # Fallback: calculate categories from original data in cluster
                        cluster_data = original_col[cluster]
                        try:
                            cluster_categories = len(np.unique(cluster_data))
                        except TypeError:
                            cluster_data_str = np.array([str(x) for x in cluster_data])
                            cluster_categories = len(np.unique(cluster_data_str))
                        
                        cluster_ncp = cluster_categories / total_categories
                        attr_ncp += cluster_ncp * (cluster_size / n_records)
                
                total_ncp += weight * attr_ncp
                
            except Exception as e:
                raise Exception(f"Error processing attribute {attr_idx}: {e}")
        
        return float(total_ncp)
        
    except Exception as e:
        raise Exception(f"Error calculating NCP: {e}")
