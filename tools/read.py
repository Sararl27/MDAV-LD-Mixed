"""
Data Reading and Directory Management Utilities

This module provides utilities for reading configuration files, managing
output directories, loading experimental results, and handling temporary
data storage for the anonymization methods comparison framework.

Functions:
    load_dirs: Set up and manage output directory structure
    get_parameters: Extract experimental parameters from JSON config
    load_existing_results_for_n: Load previously computed results
    read_from_pkl_tmp: Read temporary data from pickle files
    read_generated_data: Load generated datasets from files
    _find_file: Internal utility for finding directories by pattern
"""

import json
from operator import ge
import os
from datetime import datetime
import pickle
import pandas as pd


def _find_file(output_dir, file):
    """
    Find a directory containing a specific file pattern.
    
    Parameters:
        output_dir (str): Directory to search in
        file (str): File pattern to look for in directory names
        
    Returns:
        str or None: Path to the first matching directory, or None if not found
    """
    return next(
        (os.path.join(output_dir, d) for d in os.listdir(output_dir)
         if os.path.isdir(os.path.join(output_dir, d)) and os.path.join(output_dir, d).endswith(file)),
        None
    )


def load_dirs(method, base_dir, local_dir=None, restore_output=None, add_postfix=None, restore_or_create=False):
    """
    Set up and manage the directory structure for experimental outputs.
    
    Creates necessary directories for storing results, images, and temporary
    files. Handles both new runs and restoration of previous experiments.
    
    Parameters:
        method (str): Name of the anonymization method
        base_dir (str): Base directory for the project
        local_dir (str, optional): Local directory for data files
        restore_output (str, optional): Pattern to restore previous output
        add_postfix (str, optional): Postfix to add to run directory name
        
    Returns:
        tuple: (images_dir, clusters_images_dir, results_path, tmp_path, X_path)
            - images_dir: Directory for general images
            - clusters_images_dir: Directory for cluster visualization images
            - results_path: Path to results JSON file
            - tmp_path: Path to temporary data file
            - X_path: Path to dataset files
    """
    X_path = local_dir or base_dir
    # os.makedirs(X_path, exist_ok=True)

    output_dir = os.path.join(base_dir, 'outputs', method)
    os.makedirs(output_dir, exist_ok=True)

    if restore_output:
        run_dir = _find_file(output_dir, restore_output)
        if not run_dir: 
            if not restore_or_create:
                raise ValueError(f"Directory containing '{restore_output}' not found in {output_dir}")
            
            print(f"Directory containing '{restore_output}' not found. Creating new directory with this name.")
            run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d%H%M')}_{restore_output}")
    else:
        run_dir = _find_file(output_dir, add_postfix) if add_postfix else None
        if not run_dir:
            add_postfix = f"_{add_postfix}" if add_postfix else ""
            run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d%H%M')}{add_postfix}")

    images_dir = os.path.join(run_dir, 'images')
    results_path = os.path.join(run_dir, f'{method}_results.json')
    tmp_path = os.path.join(run_dir, 'tmp_elapsed_times.pkl')

    os.makedirs(images_dir, exist_ok=True)

    return images_dir, results_path, tmp_path, X_path

def get_parameters(file_path):
    """
    Extract experimental parameters from a JSON configuration file.
    
    Reads the first entry from a JSON file containing experimental configuration
    and extracts key parameters for running anonymization experiments.
    
    Parameters:
        file_path (str): Path to the JSON configuration file
        
    Returns:
        tuple: (n_features, repetitions, n_iterations, k_iterations, 
                n_iterations_range, k_iterations_range, ucimlrepo_id, 
                QI_numerical, QI_categorical)
                
    Raises:
        ValueError: If the configuration file is not found
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)[0]
        n_features = data.get("N_Features", None)
        repetitions = data.get("Repetitions", None)
        n_iterations, n_list = data.get("N_Metrics", (None, [None, None]))
        k_iterations, k_list = data.get("K_Metrics", (None, [None, None]))
        ucimlrepo_id = data.get("ucimlrepo_id", None)
        columns_names = data.get("columns", None)
        
        try:
            QI_types = [None] * n_features
            generalization_technique = [None] * n_features
            for idx, tpe, gen in data.get("QI", []):
                if idx < n_features:
                    QI_types[idx] = tpe
                    generalization_technique[idx] = gen
                else:
                    raise ValueError(f"Index {idx} in QI exceeds n_features {n_features}")
        except Exception as e: # For backward compatibility
            QI = data.get("QI", None)
            if QI is not None:
                QI_types = QI.get("types", None)
                generalization_technique = QI.get("generalization_technique", None)

        try: 
            sensitive_attributes_names = [None] * n_features
            for idx, sa in data.get("sensitive_attributes_names", []):
                if idx < n_features:
                    sensitive_attributes_names[idx] = sa
                else: 
                    raise ValueError(f"Index {idx} in sensitive_attributes_names exceeds n_features {n_features}")
                    
        except Exception as e: # For backward compatibility
            sensitive_attributes_names = data.get("sensitive_attributes_names", None)
        return n_features, repetitions, n_iterations, k_iterations, n_list, k_list, ucimlrepo_id, columns_names, QI_types, sensitive_attributes_names, generalization_technique
   
    raise ValueError(f"Configuration file {file_path} not found")


def load_existing_results_for_n(file_path, n, k=None, k_list=None):
    """
    Load existing experimental results for a specific sample size.
    
    Retrieves previously computed results from a JSON file to avoid
    recomputing experiments that have already been completed.
    
    Parameters:
        file_path (str): Path to the results JSON file
        n (int): Number of samples to check results for
        k (int, optional): Specific k value to check
        k_list (list, optional): List of k values to count results for
        
    Returns:
        set or int: Set of completed model names (if k specified) or 
                   count of completed experiments (if k_list specified)
                   
    Raises:
        ValueError: If neither k nor k_list is provided
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)[1].get(str(n), {})  

        if k is not None:
            return set(entry["Model"] for entry in data.get(str(k), []))
        elif k_list is not None:
            return sum(map(lambda k: len(data.get(str(k), [])), k_list)), set(entry["Model"] for k in k_list for entry in data.get(str(k), []))
        else:
            raise ValueError("Either k or k_list must be provided.")
    return []


def read_from_pkl_tmp(file_path, key):
    """
    Read temporary data from a pickle file.
    
    Loads previously saved timing data from temporary pickle files.
    If the key doesn't match or file is corrupted, removes the file.
    
    Parameters:
        file_path (str): Path to the pickle file
        key (str): Expected key for data validation
        
    Returns:
        tuple: (data_list, data_length) where data_list contains the
               stored data and data_length is the number of entries
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if data[0] == key:
            return data[1], len(data[1])
        os.remove(file_path)
        
    return [], 0


def read_generated_data(file_path, to_dataframe=True):
    """
    Load generated datasets from CSV or pickle files.
    
    Reads previously generated synthetic datasets, supporting both
    CSV and pickle formats. Returns None if file doesn't exist or is empty.
    
    Parameters:
        file_path (str): Path to the data file
        
    Returns:
        numpy.ndarray or None: Loaded dataset as numpy array, or None if
                               file not found or empty
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        if file_path.endswith('.csv'):
            if to_dataframe:
                return pd.read_csv(file_path)
            else:
                return pd.read_csv(file_path).to_numpy()
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data
    return None


def check_test_completion_status(file_path, n, methods, k_list, verbose=True):
    """
    Check the completion status of tests for a specific sample size.
    
    Provides detailed information about which method-k combinations have been completed
    and which are still pending. This is more comprehensive than just counting results.
    
    Parameters:
        file_path (str): Path to the results JSON file
        n (int): Number of samples to check results for
        methods (dict): Dictionary of method names and instances to test
        k_list (array-like): List/array of k values to test
        verbose (bool): Whether to print detailed status information
        
    Returns:
        dict: Dictionary containing:
            - 'all_completed': bool indicating if all tests are done
            - 'total_tests': int total number of method-k combinations
            - 'completed_tests': int number of completed combinations
            - 'completion_rate': float percentage of completion (0-100)
            - 'missing_tests': list of (method, k) tuples that are missing
            - 'completed_methods': set of methods that are fully completed
            - 'partial_methods': dict of methods with missing k values
    """
    # Initialize result structure
    result = {
        'all_completed': False,
        'total_tests': len(methods) * len(k_list),
        'completed_tests': 0,
        'completion_rate': 0.0,
        'missing_tests': [],
        'completed_methods': set(),
        'partial_methods': {}
    }
    
    # Load existing results if file exists
    existing_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if len(data) > 1:
                    existing_data = data[1].get(str(n), {})
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            if verbose:
                print(f"Warning: Could not load existing results: {e}")
            existing_data = {}
    
    # Check each method-k combination
    for method_name in methods.keys():
        completed_k_values = set()
        missing_k_values = []
        
        for k in k_list:
            # Get existing results for this k value
            k_results = existing_data.get(str(k), [])
            completed_models = {entry.get("Model", "") for entry in k_results if isinstance(entry, dict)}
            
            if method_name in completed_models:
                completed_k_values.add(k)
                result['completed_tests'] += 1
            else:
                missing_k_values.append(k)
                result['missing_tests'].append((method_name, k))
        
        # Categorize method completion status
        if len(missing_k_values) == 0:
            result['completed_methods'].add(method_name)
        elif len(completed_k_values) > 0:
            result['partial_methods'][method_name] = missing_k_values
    
    # Calculate completion statistics
    result['completion_rate'] = (result['completed_tests'] / result['total_tests']) * 100 if result['total_tests'] > 0 else 0
    result['all_completed'] = len(result['missing_tests']) == 0
    
    # Print detailed status if verbose
    if verbose:
        print(f"\nTest Completion Status for n={n}")
        print("=" * 60)
        print(f"Total tests required: {result['total_tests']} ({len(methods)} methods × {len(k_list)} k values)")
        print(f"Completed tests: {result['completed_tests']}")
        print(f"Completion rate: {result['completion_rate']:.1f}%")
        
        if result['all_completed']:
            print("All tests completed!")
        else:
            print(f"Missing {len(result['missing_tests'])} tests")
            
            if result['completed_methods']:
                print(f"\n - Fully completed methods ({len(result['completed_methods'])}):")
                for method in sorted(result['completed_methods']):
                    print(f"   - {method}")

            if result['partial_methods']:
                print(f"\n - Partially completed methods ({len(result['partial_methods'])}):")
                for method, missing_k in result['partial_methods'].items():
                    completed_k = len(k_list) - len(missing_k)
                    print(f"   - {method}: {completed_k}/{len(k_list)} completed, missing k={missing_k}")
            
            # Show methods with no results
            no_results_methods = [m for m in methods.keys() 
                                if m not in result['completed_methods'] and m not in result['partial_methods']]
            if no_results_methods:
                print(f"\n - Methods with no results ({len(no_results_methods)}):")
                for method in sorted(no_results_methods):
                    print(f"   - {method}")
    
    return result


def should_skip_test_iteration(file_path, n, methods, k_list, verbose=False):
    """
    Determine if a test iteration should be skipped based on completion status.
    
    This is a convenience function that wraps check_test_completion_status
    and returns a simple boolean for use in loops.
    
    Parameters:
        file_path (str): Path to the results JSON file
        n (int): Number of samples to check
        methods (dict): Dictionary of methods to test
        k_list (array-like): List of k values to test
        verbose (bool): Whether to print status information
        
    Returns:
        bool: True if all tests are completed and iteration should be skipped
    """
    return check_test_completion_status(file_path, n, methods, k_list, verbose)['all_completed']