"""
MDAV with L-diversity support for mixed data types (Iterative version).

This implementation combines the iterative approach of MDAV_Mixed with
L-diversity support from MDAV_LD_Mixed_recursive, ensuring each cluster
contains at least L distinct sensitive attribute values.
"""

from enum import unique
import heapq
import numpy as np
import pandas as pd
import os
from keras.models import load_model
import time
import logging
import tensorflow as tf
from typing import Optional

from .base import IterativeMDAV, Partition, Centroid
from ..preprocessing import preprocess_tabular_df, train_encoder, TrainConfig

logger = logging.getLogger(__name__)


class LDiversityCentroid(Centroid):
    """
    Extended Centroid class that includes sensitive attribute value for L-diversity support.
    """
    
    def __init__(self, center, index_center, sensitive_attr_value=None, precomputed_center=None):
        """
        Initialize a centroid with its coordinates, metadata, and sensitive attribute value.
        
        Args:
            center (array-like): Coordinates of the centroid point
            index_center (int): Original index of this point in the dataset
            sensitive_attr_value (str, optional): Sensitive attribute value for L-diversity
            precomputed_center (array-like, optional): Precomputed optimization values
        """
        super().__init__(center, index_center, precomputed_center)
        self.sensitive_attr_value = sensitive_attr_value


class LDiversityPartition(Partition):
    @staticmethod
    def preprocess_sensitive_attrs(sensitive_attrs):
        """
        Efficiently convert sensitive_attrs (DataFrame, Series, array, or list) to a 1D array of strings for L-diversity.
        NaN, '?', or None are replaced by '?'.
        """
        sensitive_attrs = np.asarray(sensitive_attrs, dtype=object)

        # Convert to string for uniform whitespace handling, but still detect NaN via pandas
        as_str = sensitive_attrs.astype(str)

        # Treat NaN, '?', empty strings and whitespace-only strings as missing -> map to '?'
        mask = pd.isna(sensitive_attrs) | (as_str == '?') | (np.char.strip(as_str) == "")

        sensitive_attrs = np.where(mask, '?', as_str).astype(str)

        if sensitive_attrs.ndim == 1:
            return sensitive_attrs.squeeze()

        return np.asarray([",".join(row) for row in sensitive_attrs])


    def __init__(self, data, precomputed, indices=None, sensitive_attrs=None):
        """
        Initialize partition with L-diversity support.
        
        Args:
            data: Cluster data points (numpy array)
            precomputed: Precomputed distance norms for efficiency  
            indices: Original indices of data points (optional)
            sensitive_attrs: Sensitive attribute values for each data point
        """
        super().__init__(data, precomputed, indices)
        # Convert sensitive_attrs to ndarray or set to None if absent/empty to ensure safe boolean masking
        if sensitive_attrs is not None:
            arr = np.asarray(sensitive_attrs)
            self.sensitive_attrs = arr if arr.size > 0 else None
        else:
            self.sensitive_attrs = None

        # self._l_diversity_flag = None

    # @property
    # def l_diversity_flag(self):
    #     if self._l_diversity_flag is None:
    #         raise Exception("L-diversity flag not set for this partition. First check L-diversity with check_l_diversity().")
    #     return self._l_diversity_flag

    def get_diversity_count(self, return_counts=False, return_inverse=False):
        """Get the number of distinct values in sensitive attributes (already preprocessed as strings)."""
        
        if self.sensitive_attrs is None:
            raise Exception("Sensitive attributes not set for this partition.")
        
        if not return_counts and not return_inverse:
            return len(np.unique(self.sensitive_attrs)) 

        return np.unique(self.sensitive_attrs, return_inverse=return_inverse, return_counts=return_counts)

    def check_l_diversity(self, l):
        """Check if partition satisfies L-diversity constraint."""
        return self.get_diversity_count() >= l
        # self._l_diversity_flag = self.get_diversity_count() >= l
        # return self._l_diversity_flag


class MDAV_LD_Mixed(IterativeMDAV):
    """
    Iterative MDAV implementation with L-diversity support for mixed data types.
    
    This implementation ensures that each cluster contains at least L distinct values
    for sensitive attributes, providing both k-anonymity and L-diversity privacy guarantees.
    Uses the iterative approach (non-recursive) from MDAV_Mixed.
    """

    def __init__(self, l_diversity: Optional[int] = None,
                 reduce_dimensionality: bool = True, 
                 encoder_config=TrainConfig(), 
                 target_dtype=np.float16, 
                 numerical_scaler="standardscaler",
                 categorical_encoder="onehot",
                 enable_time_monitoring: bool = True,
                 max_suppression_ratio: float = 0.5,
                 **kwargs):
        """
        Initialize MDAV with L-diversity support (iterative version).
        
        Args:
            l_diversity: Required L-diversity level (minimum distinct sensitive values per cluster)
            reduce_dimensionality: Whether to use autoencoder for dimensionality reduction
            encoder_config: Configuration for autoencoder training
            target_dtype: Data type for internal computations (float16 for memory efficiency)
            numerical_scaler: Scaler for numerical features ("standardscaler", "minmax", etc.)
            categorical_encoder: Encoder for categorical features ("binary", "onehot", etc.)
            enable_time_monitoring: Whether to track execution time for different phases
            max_suppression_ratio: Maximum fraction of records that can be suppressed
            **kwargs: Additional arguments passed to parent IterativeMDAV class
        """
        super().__init__(**kwargs)
        self.reduce_dimensionality = reduce_dimensionality
        self.encoder_config = encoder_config
        self.target_dtype = target_dtype
        self.numerical_scaler = numerical_scaler
        self.categorical_encoder = categorical_encoder
        self.enable_time_monitoring = enable_time_monitoring

        # Epsilon for floating point comparisons (concise): use target dtype if floating, else fall back
        self.eps = np.finfo(target_dtype).eps if np.issubdtype(target_dtype, np.floating) else np.finfo(np.float32).eps

        # L-diversity configuration
        self.l_diversity = l_diversity
        self.max_suppression_ratio = max_suppression_ratio
        self.SA = None
        self.suppressed_records = []
        self.suppressed_count = 0  # Always track count for statistics

        # Configure GPU memory growth for TensorFlow
        import tensorflow as tf
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    # =====================================
    # Core MDAV Algorithm Methods
    # =====================================

    def distance(self, x, y, precomputed):
        """
        Calculate distances between x and each row in y.
        """
        den = max(np.sqrt(np.einsum('i,i->', x, x, dtype=self.target_dtype)), self.eps) * precomputed
        return np.einsum('ij,j->i', y, -x, dtype=self.target_dtype) / den

    def pop(self, partition, i):
        """
        Remove and return a point from the partition.
        
        Args:
            partition (LDiversityPartition): The partition to modify
            i (int): Index of the point to remove
            
        Returns:
            tuple: (modified_partition, removed_centroid)
        """
        # Get sensitive attribute value if available
            
        centroid = LDiversityCentroid(partition.data[i].copy(), partition.indices[i], 
                                    sensitive_attr_value=partition.sensitive_attrs[i] if partition.sensitive_attrs is not None else None)

        # Create boolean mask to remove element
        mask = np.ones(len(partition.data), dtype=bool)
        mask[i] = False
        
        partition.data = partition.data[mask]
        partition.precomputed = partition.precomputed[mask]
        partition.indices = partition.indices[mask]
        # Handle sensitive attributes
        if partition.sensitive_attrs is not None:
            partition.sensitive_attrs = partition.sensitive_attrs[mask]
        return partition, centroid

    @staticmethod
    def _can_form_future_ld(counts, clusters_remaining, l):
        """Verify if we can build `clusters_remaining` future clusters with ≥ l distinct values.
        
        Optimized version with adaptive strategy:
        - Small/challenging cases: Use precise greedy simulation
        - Large balanced cases: Use mathematical bounds analysis for speed
        
        Performance characteristics:
        - Small clusters (≤20): O(n × m × log(m)) with full sort for precision
        - Large clusters (>20): O(m log m) + O(n × m/10 × log(m)) with periodic resort
        
        Args:
            counts: Array of counts for each distinct sensitive attribute value
            clusters_remaining: Number of future clusters to form
            l: Minimum distinct values required per cluster (l-diversity parameter)
            
        Returns:
            bool: True if basic necessary conditions are met, False otherwise
        """
                
        # Convert to numpy array and filter positive counts
        counts_array = np.asarray(counts)
        positive_counts = counts_array[counts_array > 0]
        
        # Quick early exit checks
        num_distinct = len(positive_counts)
        if num_distinct < l:
            return False
        
        if clusters_remaining <= 0:
            return True
        
           
        # For challenging distributions with few distinct SA values, or small cluster counts,
        # use the precise greedy simulation. Small-num-distinct distributions (e.g. 2-4
        # distinct SA values) need exact simulation irrespective of clusters_remaining
        # to avoid false negatives in feasibility checks.
        if num_distinct <= l + 2 or clusters_remaining <= 20:
            return MDAV_LD_Mixed._can_form_future_ld_small(positive_counts, clusters_remaining, l)
        
        # For larger, more balanced cases, use mathematical bounds analysis (much faster)
        return MDAV_LD_Mixed._can_form_future_ld_large(positive_counts, clusters_remaining, l)
    
    @staticmethod
    def _can_form_future_ld_small(positive_counts, clusters_remaining, l):
        """Optimized NumPy-based simulation for small/challenging cases.
        
        Uses full sorting in each iteration for maximum precision.
        Best for: ≤20 clusters or challenging distributions (< few distinct values).
        """
        # Convert to numpy array and sort in descending order (largest first)
        available = np.sort(np.asarray(positive_counts, dtype=np.int32))[::-1]
        
        for _ in range(clusters_remaining):
            # Check if we have enough distinct values
            if np.sum(available > 0) < l:
                return False
            
            # Decrement the l largest values by 1
            available[:l] -= 1
            
            # Resort to maintain descending order (ensures correctness)
            # For small arrays, sorting is very fast and ensures precision
            available = np.sort(available)[::-1]
            
            # Remove zeros to keep array compact
            available = available[available > 0]
        
        return True

    @staticmethod 
    def _can_form_future_ld_large(positive_counts, clusters_remaining, l):
        """Mathematical bounds analysis for large cluster counts (>20).
        
        Uses periodic resorting (every 10 iterations) and early-exit optimization
        for balanced distributions. Much faster than full simulation.
        """
        # Sort in descending order for greedy analysis
        sorted_counts = np.sort(positive_counts)[::-1].astype(np.int64)
        
        # Check if we can distribute l-diversity requirement across all clusters
        min_records_needed = clusters_remaining * l
        total_available = int(sorted_counts.sum())
        
        if total_available < min_records_needed:
            return False
        
        # For very large scenarios with balanced distributions, use mathematical upper bound check
        if clusters_remaining > 200 and len(sorted_counts) >= l:
            # Conservative estimate: if the sum of the l largest values is much larger than needed
            top_l_sum = int(sorted_counts[:l].sum())
            if top_l_sum >= clusters_remaining * 2:  # Conservative threshold
                return True
        
        # For all other cases, use precise greedy simulation with periodic resorting
        working_counts = sorted_counts.copy()
        
        for cluster_idx in range(clusters_remaining):
            # Ensure we have enough distinct values
            nonzero_counts = working_counts[working_counts > 0]
            if len(nonzero_counts) < l:
                return False
            
            # Check if we can take l values from the current distribution
            if len(nonzero_counts) >= l and nonzero_counts[l-1] > 0:
                # Take 1 from each of the top l values
                working_counts[:l] -= 1
                
                # Clean up and resort periodically for accuracy (every 10 iterations)
                # This balances performance vs precision
                if cluster_idx % 10 == 0 or cluster_idx == clusters_remaining - 1:
                    working_counts = working_counts[working_counts > 0]
                    if len(working_counts) > 0:
                        working_counts = np.sort(working_counts)[::-1]
            else:
                return False
        
        return True


    def _select_nearest_neighbors_with_ldiversity(self, partition, k_minus_1, dist_to_center, center_sa_value, l):
        """
        Select the k-1 companions for `center` ensuring L-diversity and future feasibility.
        1. First, select the closest candidates that contribute new sensitive values (L-distinct values).
        2. Once diversity is satisfied, fill the cluster with the remaining closest candidates.
        3. Ensure that future clusters can also satisfy L-diversity.
        
        Uses adaptive distance adjustment to reserve rare SA values for future clusters:
        - Common SA values: distance reduced (higher priority, use them first)
        - Rare SA values: distance increased (lower priority, reserve for future)
        
        Args:
            partition: Data partition with sensitive attributes
            k: Cluster size 
            dist_to_center: Precomputed distances to center
            center_sa_value: Sensitive attribute value of the center (REQUIRED for proper L-diversity calculation)
            l: L-diversity parameter (assumed to be > 0)
        """

        batch_size = 3 * (k_minus_1 + 1)  # Number of candidates to consider in each batch
        total_candidates = len(partition.indices)
        if total_candidates == 0 or k_minus_1 <= 1:
            return []

        # Prepare masks and counters
        sa_values = partition.sensitive_attrs
        unique_sas, counts = np.unique(sa_values, return_counts=True)
        value_to_idx = {value: pos for pos, value in enumerate(unique_sas)}
    
        if len(unique_sas) < l:
            raise RuntimeError(
                f"Insufficient distinct sensitive values ({len(unique_sas)}) to satisfy L-diversity (L={l})."
            )    

        selected_indices = []  # Indices selected for the cluster
        selected_mask = np.zeros(total_candidates, dtype=bool)  # True if already selected
        selected_unique_mask = np.zeros(len(unique_sas), dtype=bool)  # True if sensitive value is already in the cluster
        unique_selected_count = 0  # Number of distinct sensitive values in the cluster
        if center_sa_value is None:
            raise ValueError("Center's sensitive attribute value must be provided for L-diversity selection.")
        # If center's SA value is present among candidates, count it towards L-diversity
        elif center_sa_value in value_to_idx:
            selected_unique_mask[value_to_idx[center_sa_value]] = True
            unique_selected_count = 1
        # If center's SA value is not present among candidates, it still counts as a unique value
        else:
            # Center has a unique value not present in the candidates
            # This still counts towards L-diversity, so we need L-1 more values from candidates
            unique_selected_count = 1

        def candidate_batches():
            remaining_mask = np.ones(total_candidates, dtype=bool)  # True if not yet considered in any batch
            """Yield batches of candidates not yet selected or considered, sorted by adjusted distance."""
            while True:
                candidates = np.where((~selected_mask) & (remaining_mask))[0]
                if candidates.size == 0:
                    break
                if len(selected_indices) >= k_minus_1:
                    break
                take = min(candidates.size, batch_size)
                # Use adjusted distances that prioritize rare SA values
                batch = candidates[np.argpartition(dist_to_center[candidates], take - 1)][:take]
                remaining_mask[batch] = False
                yield batch[np.argsort(dist_to_center[batch])]

        def attempt_select(candidate_idx, require_new_unique=False):
            """
            Try to select the candidate:
            - If require_new_unique=True, only accept if it brings a new sensitive value.
            - Modifies counts in-place and reverts if selection is not valid.
            """
            nonlocal unique_selected_count
            # Early exit if already enough selected or already picked
            if len(selected_indices) >= k_minus_1 or selected_mask[candidate_idx]:
                return False
            sa_value = sa_values[candidate_idx]
            sa_pos = value_to_idx[sa_value]
            is_new_unique = not selected_unique_mask[sa_pos]
            # If we require a new unique value and this is not, skip
            if require_new_unique and not is_new_unique:
                return False
            # If no more available for this value, skip
            if counts[sa_pos] <= 0:
                return False

            # Simulate selection by decrementing counts 
            counts[sa_pos] -= 1
            next_unique_selected_count = unique_selected_count + (1 if is_new_unique else 0)
            remaining_slots = max(k_minus_1 - (len(selected_indices) + 1), 0)
            unique_needed = max(l - next_unique_selected_count, 0)
            revert = False

            # Check if we can still satisfy L-diversity with remaining slots for this cluster
            if unique_needed > 0:
                available_unique = np.count_nonzero((counts > 0) & (~selected_unique_mask))
                # If this candidate is_new_unique and still has more left, discount one
                if is_new_unique and counts[sa_pos] > 0:
                    available_unique -= 1
                # If not enough unique values left to satisfy L-diversity, revert
                if remaining_slots < unique_needed or available_unique < unique_needed:
                    revert = True
            # Check if future clusters can still satisfy L-diversity
            if not revert:
                # Calculate how many complete clusters can be formed with remaining records
                # After completing current cluster, we'll have (counts.sum() - remaining_slots) records left
                # Note: k here is k-1 (companions only), so k+1 = self.k (full cluster size)
                clusters_remaining = (int(counts.sum()) - remaining_slots) // (k_minus_1 + 1)
                if not self._can_form_future_ld(counts, clusters_remaining, l):
                    revert = True

            if revert:  # Revert counts if selection is not valid
                counts[sa_pos] += 1
                return False

            # All checks passed, update selection state
            selected_indices.append(candidate_idx)
            selected_mask[candidate_idx] = True
            if is_new_unique:
                selected_unique_mask[sa_pos] = True
                unique_selected_count += 1
            return True

        num_batches = 0
        filling_records = [] 
        # 1. Prioritize diversity: select new sensitive values until L-diversity is satisfied
        for batch in candidate_batches():
            if len(selected_indices) >= k_minus_1 or unique_selected_count >= l:
                break
            for idx in batch:
                if unique_selected_count >= l or len(selected_indices) >= k_minus_1:
                    break
                attempt_select(idx, require_new_unique=True)
            num_batches += 1
            # Save up to batch_size unselected records in total
            if len(filling_records) < batch_size:
                to_add = batch[~selected_mask[batch]]
                remaining = batch_size - len(filling_records)
                if len(to_add) > remaining:
                    filling_records.extend(to_add[:remaining])
                else:
                    filling_records.extend(to_add)
        logger.debug(f"_select_nearest_neighbors_with_ldiversity: Batches needed: {num_batches}. Candidates left: {total_candidates - len(selected_indices)}. Unique selected: {unique_selected_count}/{l}.")
        
        # 2. If L-diversity is satisfied but cluster is not full, fill with closest remaining candidates
        if unique_selected_count >= l and len(selected_indices) < k_minus_1:
            num_batches = 0
            filling_records = [filling_records] if filling_records else []

            def fill_from_batches(batches):
                nonlocal num_batches

                for batch in batches:
                    if len(selected_indices) >= k_minus_1:
                        break
                    for idx in batch:
                        if len(selected_indices) >= k_minus_1:
                            break
                        if selected_mask[idx]:
                            continue
                        attempt_select(idx)
                    num_batches += 1

            # First try with saved filling_records, avoiding re-calculating batches
            fill_from_batches(filling_records)
            #  Worst case, if with filling_records we still need more, rerun candidate_batches()
            if len(selected_indices) < k_minus_1: 
                logger.warning(f"Filling cluster with additional candidates to reach k, re-evaluating candidate batches. Candidates left: {total_candidates - len(selected_indices)}. Unique selected: {unique_selected_count}/{l}.")
                fill_from_batches(candidate_batches())

            logger.debug(f"_select_nearest_neighbors_with_ldiversity: L-diversity satisfied, filling cluster, batches needed: {num_batches}")
        # print({unique_sas[i]: int(counts[i]) for i in range(len(unique_sas)) if counts[i] > 0})
        if len(selected_indices) < k_minus_1:
            # Diagnostic information for debugging
            records_left_after_cluster = int(counts.sum()) - (k_minus_1 - len(selected_indices))
            future_clusters_possible = records_left_after_cluster // (k_minus_1 + 1)
            current_diversity = unique_selected_count
            mask = (counts > 0) & (~selected_unique_mask)
            unique_values_with_counts = {unique_sas[i]: int(counts[i]) for i in range(len(unique_sas)) if counts[i] > 0}
            raise RuntimeError(
                f"Unable to guarantee future L-diversity: not enough feasible neighbors.\n"
                f"  Required companions: {k_minus_1}, obtained: {len(selected_indices)}\n"
                f"  Current cluster diversity: {current_diversity}/{l}\n"
                f"  Remaining candidates: {total_candidates - len(selected_indices)}\n"
                f"  Diverse values available: {np.count_nonzero(mask)}\n"
                f"  Future clusters possible: {future_clusters_possible}\n"
                f"  Available SA values with counts: {unique_values_with_counts}\n"
                f"  This error occurs when selecting any remaining candidate would prevent "
                f"future clusters from satisfying L-diversity={l}."
            )
           
        return np.array(selected_indices[:k_minus_1], dtype=int)

    def cluster(self, partition: LDiversityPartition, center: LDiversityCentroid, dist_to_center=None):
        """
        Build one cluster of size k around `center` with L-diversity constraints.
        
        Args:
            partition (LDiversityPartition): Remaining data partition with sensitive attributes
            center (LDiversityCentroid): Center point for the cluster with sensitive attribute value
            dist_to_center (np.ndarray, optional): Precomputed distances to center
            
        Returns:
            tuple: (remaining_partition, cluster_indices)
        """
        if dist_to_center is None:
            dist_to_center = self.distance(center.data, partition.data, partition.precomputed)
        

        # If l-diversity is not enabled or no sensitive attributes
        if self.l_diversity is None:
            nearest =  np.argpartition(dist_to_center, self.k-1)[:self.k-1]
        else:
            # Select points with L-diversity constraints, using center's SA value from the centroid
            nearest = self._select_nearest_neighbors_with_ldiversity(partition, self.k-1, dist_to_center, center.sensitive_attr_value, self.l_diversity)
        
        # Create cluster indices array
        cluster_indices = np.empty(self.k, dtype=partition.indices.dtype)
        cluster_indices[0] = center.index; cluster_indices[1:] = partition.indices[nearest]

        # Remove selected points from partition
        mask = np.ones(len(partition.data), dtype=bool)
        mask[nearest] = False

        partition.data = partition.data[mask]
        if partition.precomputed is not None:
            partition.precomputed = partition.precomputed[mask]
        partition.indices = partition.indices[mask]

        # Update sensitive attributes - remove selected points
        if partition.sensitive_attrs is not None:
            partition.sensitive_attrs = partition.sensitive_attrs[mask]
        return partition, cluster_indices

    def farthest_centroid(self, partition):
        """Find point farthest from centroid of partition."""
        xm = np.mean(partition.data, axis=0, dtype=self.target_dtype)
        distances = self.distance(xm, partition.data, partition.precomputed)
        xri = np.argmax(distances)

        return self.pop(partition, xri)

    # =====================================
    # MDAV Clustering with L-Diversity (Iterative)
    # =====================================

    def mdav(self, partition):
        """
        Single MDAV pass on current subset X/precomputed with L-diversity constraints.
        
        Args:
            partition (LDiversityPartition): Data partition to process with sensitive attributes
            
        Returns:
            list: List of cluster indices
        """
        # Apply L-diversity preprocessing

        clusters = []

        while partition.size >= 3 * self.k:
            if self.check_runtime_exceed(partition.size):
                return None
            
            # Calculate the farthest point from the centroid
            partition, centers_xr = self.farthest_centroid(partition)
            
            # Furthest point from xr
            dist_to_xr = self.distance(centers_xr.data, partition.data, partition.precomputed)
            xsi = np.argmax(dist_to_xr)
            dist_to_xr = np.concatenate((dist_to_xr[:xsi], dist_to_xr[xsi+1:]))
            partition, centers_xs = self.pop(partition, xsi)

            # Cluster of xr
            partition, cluster = self.cluster(partition, centers_xr, dist_to_center=dist_to_xr)
            clusters.append(cluster)
            del dist_to_xr, centers_xr, cluster

            # Cluster of xs
            partition, cluster = self.cluster(partition, centers_xs)
            clusters.append(cluster)
            del centers_xs, cluster


        if 2 * self.k <= partition.size < 3 * self.k:

            # Calculate the farthest point from the centroid
            partition, centers_xr = self.farthest_centroid(partition)
            
            # Cluster of xr
            partition, cluster = self.cluster(partition, centers_xr)
            clusters.append(cluster)

        if 2 * self.k > partition.size > 0:
            # Rest of the points
            # If l-diversity is required, check if the leftover partition satisfies it
            if self.l_diversity is not None and not partition.check_l_diversity(self.l_diversity):
                self.suppressed_records.extend(partition.indices)
                self.suppressed_count += len(partition.indices)
            else:
                clusters.append(partition.indices)

        return clusters


    def anonymize(self, X, k, attributeSchema, l=None, model_path=None, return_indices_data=False):
        """
        Entry point for L-diversity MDAV anonymization (iterative version).
        
        Args:
            X: Input dataset
            k (int): Anonymity parameter
            attributeSchema: Schema containing QI and SA specifications
            l (int, optional): L-diversity parameter. If None, uses instance default.
            model_path (str, optional): Path to pre-trained encoder model
            return_indices_data (bool): If True, return (generalized_data, clusters).
                                          If False, return only generalized_data
        
        Returns:
            pandas.DataFrame or tuple: Generalized data or (clusters, generalized_data)
        """
        self.SA = attributeSchema.SA if attributeSchema.SA is not None else None
        self.l_diversity = l if l is not None else self.l_diversity
        self.configure(X.shape, k, attributeSchema)

        if self.QI is None or len(self.QI.allowed) == 0:
            raise ValueError("At least one quasi-identifier (QI) must be specified for MDAV.")

        tmp_path = self._save_original_data(X)

        sensitive_attrs = None
        if self.SA is not None and self.l_diversity is not None:
            if len(self.SA.allowed) == 0 and self.l_diversity is not None:
                raise Exception(f"L-diversity requires at least one sensitive attribute. SA allowed: {self.SA.allowed}")
            
            sensitive_attrs = self.SA.keep_selected_columns(X)
            sensitive_attrs = LDiversityPartition.preprocess_sensitive_attrs(sensitive_attrs)

        X, QI = self.QI.keep_selected_columns(X, return_allowed_copy=True)

        if self.enable_time_monitoring:
            self.aditional_info = {}
            start = time.perf_counter()

        # Preprocess with consistent target_dtype output
        self._raw_input_dim = X.shape[1]
        X = preprocess_tabular_df(X, target_dtype=self.target_dtype, convert_to_numpy=True, 
                                num_scaler=self.numerical_scaler, cat_encoder=self.categorical_encoder, QI=QI)
        if self.enable_time_monitoring:
            self.aditional_info['Preprocess'] = time.perf_counter() - start
            self.aditional_info['Preprocessed shape'] = X.shape

        if self.reduce_dimensionality:
            X = self._get_reduced_data(X, model_path)
        if self.enable_time_monitoring:
            if self.reduce_dimensionality:
                self.aditional_info['Reduced shape'] = X.shape
            start = time.perf_counter()
            
        # Create L-diversity partition
        precomputed = np.sqrt(np.einsum('ij,ij->i', X, X, dtype=self.target_dtype))
        np.maximum(precomputed, self.eps, out=precomputed, dtype=self.target_dtype)
        clusters = self.mdav(LDiversityPartition(X.astype(self.target_dtype), precomputed=precomputed.astype(self.target_dtype), sensitive_attrs=sensitive_attrs))

        if self.enable_time_monitoring:
            self.aditional_info["MDAV"] = time.perf_counter() - start

        # Generalize clusters while preserving L-diversity
        if self.enable_time_monitoring:
            start = time.perf_counter()
            
        X = self._load_original_data(tmp_path)
        X, QI = self.QI.keep_selected_columns(X, self.SA.allowed if self.SA else None, return_allowed_copy=True)

        # Use the base class method with sensitive columns awareness
        X = self.generalize_clusters(X, clusters, QI)

        if self.suppressed_count > 0 and self.suppressed_records:
            X = X.drop(index=self.suppressed_records, errors='ignore')

        if self.enable_time_monitoring:
            self.aditional_info["Generalization"] = time.perf_counter() - start
            if self.l_diversity is not None:
                self.aditional_info["Suppressed"] = self.suppressed_count

        return (X, clusters) if return_indices_data else X


    # =====================================
    # Dimensionality Reduction Methods
    # =====================================

    def _make_encoder_fn(self, encoder: tf.keras.Model):
        """Compile a tf.function for fast inference."""
        @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=self.target_dtype)])
        def encode_fn(x):
            return encoder(x, training=False)
        return encode_fn

    def _get_model(self, X, model_path, encoder_config, accept_model_over_threshold=False):
        """
        Load or train an encoder model with improved error handling.
        
        Args:
            X: Input data for training
            model_path: Path to pre-trained model
            encoder_config: Configuration for encoder training
            accept_model_over_threshold: Whether to accept models over threshold
            
        Returns:
            Compiled tf.function for encoder inference
        """
        if not model_path:
            if self.enable_time_monitoring:
                start = time.perf_counter()

            model = train_encoder(X, self._raw_input_dim, encoder_config, target_dtype=self.target_dtype, accept_model_over_threshold=accept_model_over_threshold)
            if self.enable_time_monitoring:
                self.aditional_info["Training"] = time.perf_counter() - start
        elif os.path.exists(model_path):
            model = load_model(model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

        return self._make_encoder_fn(model)
        
    def _get_reduced_data(self, X, model_path, accept_model_over_threshold=False):
        """
        Run encoder(X) with multi-GPU support and improved memory management.
        
        Args:
            X: Input data to encode
            model_path: Path to pre-trained model
            accept_model_over_threshold: Whether to accept models over threshold
            
        Returns:
            np.ndarray: Encoded data with target_dtype
        """
        # Check for multi-GPU availability
        from  copy import deepcopy as _deepcopy
        encoder_config = _deepcopy(self.encoder_config)

        strategy = (tf.distribute.MirroredStrategy() if len(tf.config.experimental.list_physical_devices('GPU')) > 1 
                    else None)
                    

        if strategy:
            with strategy.scope():
                encode_fn = self._get_model(X, model_path, encoder_config, accept_model_over_threshold=accept_model_over_threshold)
        else:
            encode_fn = self._get_model(X, model_path, encoder_config, accept_model_over_threshold=accept_model_over_threshold)

        if self.enable_time_monitoring:
            start = time.perf_counter()

        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.batch(encoder_config.inference_batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        if strategy:
            ds = strategy.experimental_distribute_dataset(ds)

        encoded_chunks = []
        try:
            for batch in ds:
                encoded_batch = encode_fn(batch)
                # Convert to numpy and ensure target dtype
                encoded_chunks.append(encoded_batch.numpy().astype(self.target_dtype))
        except Exception as e:
            raise RuntimeError(f"Error during data encoding: {str(e)}")
        finally:
            # Clean up dataset reference
            del ds

        # Concatenate results efficiently
        result = np.vstack(encoded_chunks).astype(self.target_dtype)
        
        if self.enable_time_monitoring:
            self.aditional_info["Reduction"] = time.perf_counter() - start

        return result
