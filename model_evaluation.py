"""
Model Evaluation Module

This module provides functionality for evaluating anonymization and clustering
models with comprehensive performance metrics. It handles model execution,
timing measurements, and calculation of quality metrics like NCP (Normalized
Certainty Penalty).

The module supports:
- Multiple repetitions for statistical reliability
- Progress tracking with estimated completion times
- Memory management for large datasets
- Timeout handling for long-running algorithms
- Comprehensive metric calculation including clustering quality

Functions:
    run_model_evaluation: Main evaluation function for anonymization models
"""

import time
from datetime import datetime
import numpy as np
import logging
from tqdm.auto import tqdm
# Set up logger
logger = logging.getLogger(__name__)

from tools.read import read_from_pkl_tmp
from tools.metrics import (
    calculate_ncp
)
from tools.utils import (
    check_variable_existence, 
    calculate_end_time_str, 
    get_data
)
from tools.write import write_to_pkl_tmp


def run_model_evaluation(
    instance, 
    model, 
    X_path_file, 
    n_features, 
    n, 
    k, 
    file_path_tmp, 
    repetitions=3, 
    attributeSchema=None, 
    max_runtime_seconds=1800, 
    ucimlrepo_id=None,
    show_progress=True,
    return_indices_data=True,
):
    """
    Execute and evaluate an anonymization model with comprehensive metrics.
    
    This function runs the specified model multiple times to gather performance
    statistics and calculates comprehensive quality metrics including privacy,
    utility, cluster quality, and performance measures.
    
    Parameters:
        instance: The anonymization algorithm instance to evaluate
        model (str): Name/identifier of the model being evaluated
        X_path_file (str): Path to the dataset file
        n_features (int): Number of features in the dataset
        n (int): Number of samples to use from the dataset
        k (int): Anonymization parameter (e.g., k-anonymity level)
        file_path_tmp (str): Path for temporary file storage
        repetitions (int): Number of times to run the algorithm (default: 3)
        QI: Quasi-identifier specification object
        max_runtime_seconds (int): Maximum runtime before timeout (default: 1000)
        ucimlrepo_id: UCI ML Repository dataset ID (optional)
        show_progress (bool): Whether to show internal progress bar (default: True)
    
    Returns:
        tuple: (result_dict, clusters, centroids, instance) where:
            - result_dict: Dictionary containing comprehensive evaluation metrics:
                * Model: Model name
                * Number_of_Clusters: Count of generated clusters
                * NCP: Normalized Certainty Penalty
                * Elapsed_Times: List of execution times
                * Privacy_Metrics: Privacy-related metrics (information loss, k-anonymity compliance)
                * Utility_Metrics: Data utility preservation metrics
                * Cluster_Quality_Metrics: Clustering quality assessment
                # * Performance_Metrics: Runtime and efficiency metrics
            - clusters: List of data clusters/partitions
            - centroids: Cluster centroids (if applicable)
            - instance: The algorithm instance (with any updated state)
    
    Raises:
        ValueError: If repetitions < 1
    """
    if repetitions < 1:
        raise ValueError("Number of repetitions must be greater than 0.")

    key = f"n_features_{n_features}_n_{n}_k_{k}_model_{model}"

    # Load previous timing results if available
    time_list, i = read_from_pkl_tmp(file_path_tmp, key)
    init_time = datetime.now()
    
    # Create internal progress bar that will disappear when finished
    pbar = tqdm(total=3, desc=f"Evaluating '{model}'", disable=not show_progress, leave=False)
    
    # Execute algorithm multiple times
    for i in range(i, repetitions):
        pbar.set_postfix_str(
            f"([ST: {init_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"ET: {calculate_end_time_str(init_time, time_list, repetitions)}]), rep {i + 1}/{repetitions}"
        )

        # Load data and run algorithm
        X = get_data(X_path_file, n, n_features, ucimlrepo_id, verbose=False)
        start_time = time.perf_counter()
        
        # Use anonymize method for all MDAV implementations
        generalized, clusters = instance.anonymize(X, k, attributeSchema=attributeSchema, return_indices_data=return_indices_data)

        elapsed_time = time.perf_counter() - start_time
        time_list.append(elapsed_time)
        del X
            
        # Handle cases where algorithm failed to produce clusters
        if not clusters and check_variable_existence(instance, "estimated_time"):
            pbar.set_postfix_str(
                f"Algorithm failed to produce clusters ([ST: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], "
                f"estimated_time: {np.ceil(instance.estimated_time)} seconds)"
            )
            pbar.close()  # Clean up progress bar before returning
            return {
                "Model": model,
                "Number_of_Clusters": 0, 
                "Mean_Distance": None, 
                "NCP": None, 
                "Elapsed_Times": [], 
                "Information": instance.estimated_time
            }, None, None, instance

        # Handle timeout scenarios
        if elapsed_time > max_runtime_seconds: 
            break  # Only one run for long execution
        elif i < repetitions - 1:
            del clusters  # Clean up memory
            # Save intermediate results for long runs
            if elapsed_time > 60:
                write_to_pkl_tmp(file_path_tmp, time_list, key)

    pbar.update(1)
    del attributeSchema
    
    pbar.set_postfix_str(
        f"Calculating metrics ([ST: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], Loading data)"
    )
    X = get_data(X_path_file, n, n_features, ucimlrepo_id, verbose=False)

    pbar.update(1)  # Update after data processing
        
    # Calculate comprehensive metrics
    pbar.set_postfix_str(f"Calculating metrics ([ST: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], Computing metrics)")
    
    if clusters is not None:
        # Calculate NCP using clusters for efficiency
        try:
            ncp = calculate_ncp(instance, X, clusters)
        except Exception as e:
            logger.warning(f"Could not calculate NCP: {e}")
            ncp = None
        # Disclosure and record-linkage metrics removed from evaluation
        disclosure_metrics = {}
        rl_metrics = {}
    else:
        logger.warning("No clusters available for NCP calculation")
        ncp = None
        disclosure_metrics = {}
        rl_metrics = {}
    
    # Performance Metrics - Enhanced with runtime analysis
    # performance_metrics = calculate_performance_metrics(
    #     time_list, 
    #     getattr(instance, 'aditional_info', None)
    # )
    
    # Throughput Metrics - Data processing efficiency
    # if X is not None and time_list:
    #     throughput_metrics = calculate_throughput_metrics(
    #         data_size=len(X),
    #         execution_times=time_list,
    #         size_unit="records"
    #     )
    # else:
    #     throughput_metrics = {}
    
    pbar.update(1)  # Final update
    del X

    # Compile comprehensive results
    
    sizes_cluster = [len(c) for c in clusters] if clusters else []
    result = {
        "Model": model,
        "Number_of_Clusters": len(clusters) if clusters else 0,
        "Max_&_Min_Cluster_Size": {
            "Max": np.max(sizes_cluster) if clusters else 0,
            "Min": np.min(sizes_cluster) if clusters else 0
        },
        "Elapsed_Times": time_list,
        "NCP": ncp,

    }

    # Add additional information if available
    if check_variable_existence(instance, "estimated_time"):
        result["Information"] = f"Estimated time to finish: {np.ceil(instance.estimated_time)} seconds."

    if check_variable_existence(instance, "aditional_info"):
        result["Aditional_Information"] = instance.aditional_info

    # Clean up progress bar
    pbar.close()

    return result, clusters, instance

