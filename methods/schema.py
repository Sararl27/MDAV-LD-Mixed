"""
Data schema and attribute management for anonymization methods.

This module provides classes for managing dataset schemas, quasi-identifiers,
and sensitive attributes used in privacy-preserving data processing.

Classes:
    QuasiIdentifiers: Handler for quasi-identifier attributes
    SensitiveAttributes: Handler for sensitive attributes
    AttributeSchema: Schema definition for datasets with QI and SA attributes
    AttributeSchema: Deprecated alias for AttributeSchema (for backward compatibility)
"""

import numpy as np
import pandas as pd

# Define generalization techniques to avoid circular import
GENERALIZATION_TECHNIQUES = {
    'numerical': ['centroid', 'median', 'mode', 'range', 'mask', 'permutation'],
    'categorical': ['mode', 'mask', 'permutation']
}


class QuasiIdentifiers:
    """
    Handler for quasi-identifier attributes in privacy-preserving data processing.
    
    Quasi-identifiers are attributes that, when combined, can potentially identify
    individuals in a dataset. This class manages these attributes, their types
    (numerical vs categorical), and associated generalization trees.
    
    Attributes:
        allowed (np.ndarray): Indices of columns that are quasi-identifiers
        types (np.ndarray): Data types of QI columns (0=numerical, 1=categorical)
        trees (np.ndarray): Generalization trees for categorical QI columns (optional)
    """
    __slots__ = ('allowed', 'types', 'trees', 'generalization_technique')

    _types_QI = {"numerical": 0, "categorical": 1, None: None}

    def __init__(self, QI=None, generalization_technique=None, trees=None, allowed_indices=None, data_types=None):
        """
        Initialize quasi-identifier configuration.
        
        Args:
            QI (list, optional): List specifying QI types for each column.
                Example: ["numerical", "categorical", "numerical", None, None, "categorical"]
                for a dataset with 6 columns where columns 0, 1, 2, and 5 are QI.
            
            trees (list, optional): Generalization trees for categorical quasi-identifiers.
                Example: [None, GenTree(), None, None, None, GenTree()]
                for columns 1 and 5 having categorical generalization trees.
            
            allowed_indices (np.ndarray, optional): Direct specification of QI column indices
            data_types (np.ndarray, optional): Direct specification of QI data types
        """

        if QI is not None:
            
            # Process each element individually to preserve None values
            processed_QI = []
            for x in QI:
                if x is None:
                    processed_QI.append(None)
                elif isinstance(x, str):
                    processed_QI.append(x.lower())
                else:
                    processed_QI.append(x)
            QI = np.array(processed_QI, dtype=object)

            trees = np.array(trees, dtype=object) if trees is not None else None

            if np.any(~np.isin(QI, list(self._types_QI.keys()))):
                raise ValueError(f"Invalid categories for the quasi-identifiers. Valid categories are 'numerical', 'categorical' and None. Got {QI}.")
 
            # Extract indices of non-None values (these are the QI columns)
            self.allowed = np.nonzero(QI != None)[0].astype(np.int32)
            # Convert types: 1 for categorical, 0 for numerical
            self.types = np.where(QI[self.allowed] == "categorical", 1, 0).astype(np.int8)

            self.trees = trees[self.allowed] if trees is not None else None
            
            self.generalization_technique = self._validate_generalization(generalization_technique, QI) if generalization_technique is not None else np.array([], dtype=object)
        
        
        else:
            self.allowed = allowed_indices if allowed_indices is not None else np.array([], dtype=np.int32)
            self.types = data_types if data_types is not None else np.array([], dtype=np.int8)
            self.trees = trees if trees is not None else None
            self.generalization_technique = generalization_technique if generalization_technique is not None else np.array([], dtype=object)


    @property
    def size(self):
        return self.allowed.size

    def delete(self, idx):
        """
        Delete the quasi-identifier at the given index.
        
        Args:
            idx (int): Index of the QI to remove
        """
        self.allowed = np.delete(self.allowed, idx)
        self.types = np.delete(self.types, idx)
        if self.trees is not None:
            self.trees = np.delete(self.trees, idx)

    def keep_selected_columns(self, D, extra_columns=None, return_allowed_copy=False):
        """
        Select and return only quasi-identifier columns (and optional extra columns) from dataset.
        
        This method filters the dataset to keep only the specified columns,
        optionally returning an updated copy of the QuasiIdentifiers object.
        
        Args:
            D (pd.DataFrame or np.ndarray): Input dataset
            extra_columns (array-like, optional): Additional column indices to keep
            return_allowed_copy (bool): Whether to return updated QI object
            
        Returns:
            pd.DataFrame or np.ndarray: Filtered dataset
            QuasiIdentifiers (optional): Updated QI object if return_allowed_copy=True
        """
        # Determine columns to keep
        if extra_columns is None:
            selected_columns = self.allowed.astype(int)
        else:
            selected_columns = np.sort(np.unique(np.concatenate((self.allowed, extra_columns)))).astype(int)

        # Validate selected columns exist in D
        if np.any(selected_columns > D.shape[1]):
            raise IndexError("One or more column indices are out of bounds.")

        # Create mapping from original indices to new indices
        index_mapping = {col: idx for idx, col in enumerate(selected_columns)}
        QI = np.array([index_mapping[col] for col in self.allowed])

        # Filter dataset based on selected columns
        if isinstance(D, pd.DataFrame):
            D = D.iloc[:, selected_columns]
        elif isinstance(D, np.ndarray):
            D = D[:, selected_columns]
        else:
            raise TypeError("D must be a pandas DataFrame or a NumPy ndarray.")

        if return_allowed_copy:
            return D, self.copy(allowed_indices=QI)
        return D

    def copy(self, allowed_indices=None, data_types=None, generalization_technique=None, replace_tree_at=None):
        """
        Create a copy of this QuasiIdentifiers object with optional modifications.
        
        Args:
            allowed_indices (np.ndarray, optional): New allowed indices
            data_types (np.ndarray, optional): New data types
            replace_tree_at (tuple, optional): (new_root, dimension) to replace tree
            
        Returns:
            QuasiIdentifiers: Copy of this object with modifications applied
        """

        trees = np.copy(self.trees) if self.trees is not None else None
        if replace_tree_at:
            new_root, dimension = replace_tree_at
            if trees is None or dimension >= len(trees):
                raise ValueError("Invalid tree replacement target.")
            trees[dimension] = trees[dimension].find(new_root)

        # Preserve generalization_technique when copying if not supplied

        return QuasiIdentifiers(
            allowed_indices=np.copy(self.allowed if allowed_indices is None else allowed_indices),
            data_types=np.copy(self.types if data_types is None else data_types),
            generalization_technique=np.copy(self.generalization_technique if generalization_technique is None else generalization_technique),
            trees=trees
        )
    
    def get_generalization_technique(self, idx):
        """
        Get the generalization technique for the quasi-identifier at the given index.
        
        Args:
            idx (int): Index to check - can be either:
                      - Absolute column index (will be converted to relative QI index)
                      - Relative QI index (direct index into self.types)

        Returns:
            str: Generalization technique for the specified QI
        """
        # Check if idx is an absolute column index that exists in allowed indices
        if idx in self.allowed:
            # Convert absolute index to relative QI index
            return self.generalization_technique[np.where(self.allowed == idx)[0][0]]
        # Otherwise, assume it's already a relative QI index
        elif 0 <= idx < len(self.types):
            return self.generalization_technique[idx]
        else:
            raise IndexError(f"Index {idx} is neither a valid absolute column index nor a valid relative QI index")

    def get_numerical_columns(self):
        """
        Get the column indices of all numerical quasi-identifiers.
        
        Returns:
            np.ndarray: Indices of numerical QI columns
        """
        
        return self.allowed[self.types == 0]

    def get_categorical_columns(self):
        """
        Get the column indices of all categorical quasi-identifiers.
        
        Returns:
            np.ndarray: Indices of categorical QI columns
        """
        return self.allowed[self.types == 1]
        
    def is_numerical(self, idx):
        """
        Check if the quasi-identifier at the given index is numerical.
        
        Args:
            idx (int): Index to check - can be either:
                      - Absolute column index (will be converted to relative QI index)
                      - Relative QI index (direct index into self.types)
            
        Returns:
            bool: True if numerical, False if categorical
        """
        # Check if idx is an absolute column index that exists in allowed indices
        if idx in self.allowed:
            # Convert absolute index to relative QI index
            return self.types[np.where(self.allowed == idx)[0][0]] == 0
        # Otherwise, assume it's already a relative QI index
        elif 0 <= idx < len(self.types):
            return self.types[idx] == 0
        else:
            raise IndexError(f"Index {idx} is neither a valid absolute column index nor a valid relative QI index")
    
    def is_categorical(self, idx):
        """
        Check if the quasi-identifier at the given index is categorical.
        
        Args:
            idx (int): Index to check - can be either:
                      - Absolute column index (will be converted to relative QI index)
                      - Relative QI index (direct index into self.types)
            
        Returns:
            bool: True if categorical, False if numerical
        """
        # Check if idx is an absolute column index that exists in allowed indices
        if idx in self.allowed:
            # Convert absolute index to relative QI index
            return self.types[np.where(self.allowed == idx)[0][0]] == 1
        # Otherwise, assume it's already a relative QI index
        elif 0 <= idx < len(self.types):
            return self.types[idx] == 1
        else:
            raise IndexError(f"Index {idx} is neither a valid absolute column index nor a valid relative QI index")

    def change_categorical_to_numerical_indexes(self):
        """
        Convert all categorical quasi-identifiers to numerical type.
        
        This method modifies the types of all quasi-identifiers to numerical,
        which can be useful for certain algorithms that only handle numerical data.
        """
        self.types = np.zeros_like(self.types, dtype=np.int8)

    def _validate_generalization(self, generalization_technique, QI):
        """
        Private helper: validate the provided generalization_technique array and
        return the techniques for the QI columns (as an ndarray).

        Parameters
        ----------
        generalization_technique : array-like
            Full-array of techniques for all dataset columns.
        QI : np.ndarray or None
            Original QI array (if provided) used to check expected length.

        Returns
        -------
        np.ndarray
            Array of generalization techniques corresponding to `self.allowed`.
        """
        generalization_technique = np.asarray(generalization_technique, dtype=object)

        # If QI was passed as an array, require matching length.
        if generalization_technique.size != QI.size:
            raise ValueError(f"Length of generalization_technique must match number of columns in QI. Expected {QI.size}, got {generalization_technique.size}.")
       
        # Validate entries only for QI columns
        for abs_idx, rel_idx in zip(self.allowed, range(self.allowed.size)):
            if abs_idx < len(generalization_technique):
                tech = generalization_technique[abs_idx]
                if tech is None:
                    continue
                if self.types[rel_idx] == 1:
                    # categorical
                    if tech not in GENERALIZATION_TECHNIQUES['categorical']:
                        raise ValueError(
                            f"Invalid generalization type '{tech}' for categorical QI at index {abs_idx}. "
                            f"Valid: {GENERALIZATION_TECHNIQUES['categorical']}."
                        )
                else:
                    # numerical
                    if tech not in GENERALIZATION_TECHNIQUES['numerical']:
                        raise ValueError(
                            f"Invalid generalization type '{tech}' for numerical QI at index {abs_idx}. "
                            f"Valid: {GENERALIZATION_TECHNIQUES['numerical']}."
                        )

        # Return the techniques for the allowed QI columns (keep original ordering)
        return generalization_technique[self.allowed]


class SensitiveAttributes(QuasiIdentifiers):
    """
    Handler for sensitive attributes with identical interface to QuasiIdentifiers.
    
    This class inherits from QuasiIdentifiers to provide consistent column management
    for sensitive attributes used in L-diversity calculations. It supports the same
    flexible input formats as QuasiIdentifiers.
    """
    
    def __init__(self, sensitive_attrs=None, trees=None, allowed_indices=None, data_types=None):
        super().__init__(QI=sensitive_attrs, trees=trees, allowed_indices=allowed_indices, data_types=data_types, generalization_technique=None)


class AttributeSchema:
    """
    Schema definition for datasets with quasi-identifiers and sensitive attributes.
    
    This class provides a unified interface for managing dataset schemas used in
    privacy-preserving data processing, including quasi-identifiers, sensitive
    attributes, and generalization trees.
    
    Attributes:
        QI (QuasiIdentifiers): Quasi-identifier attributes
        SA (SensitiveAttributes): Sensitive attributes for L-diversity
        trees (Optional[List[Type]]): Generalization trees for categorical attributes
    """
    __slots__ = ('QI', 'SA')
    
    def __init__(self, QI, generalization_technique, SA=None, trees=None):
        """
        Initialize the data schema.
        
        Args:
            QI: Quasi-identifier configuration
            SA: Sensitive attributes configuration (optional)
            trees: Generalization trees for categorical attributes (optional)
        """
        if not (isinstance(QI, (list, np.ndarray, QuasiIdentifiers)) and len(QI) > 0):
            raise TypeError("QI must be a non-empty list, ndarray, or QuasiIdentifiers.")
        if not (isinstance(generalization_technique, (list, np.ndarray)) and len(generalization_technique) > 0):
            raise TypeError("generalization_technique must be a non-empty list or ndarray.")
        if SA is not None and not (isinstance(SA, (list, np.ndarray, SensitiveAttributes)) and len(SA) > 0):
            raise TypeError("SA must be a non-empty list, ndarray, or SensitiveAttributes.")
        if trees is not None and not (isinstance(trees, (list, np.ndarray)) and len(trees) > 0):
            raise TypeError("trees must be a non-empty list or ndarray of generalization trees.")

        # Initialize QI and SA objects, pass generalization_technique as numpy array through to QI
        self.QI = QuasiIdentifiers(QI, trees=trees, generalization_technique=generalization_technique)
        self.SA = SensitiveAttributes(SA) if SA is not None else None

        # Check that QI and SA don't point to the same column indices
        if self.SA is not None:
            qi_indices = set(self.QI.allowed)
            sa_indices = set(self.SA.allowed)
            overlapping_indices = qi_indices.intersection(sa_indices)
            if overlapping_indices:
                raise ValueError(f"QI and SA cannot point to the same column indices. Overlapping indices: {sorted(overlapping_indices)}")


    @property
    def size(self):
        """
        Get the number of quasi-identifier attributes.
        
        Returns:
            int: Number of QI attributes
        """
        return self.QI.size
    
    @property
    def generalization_technique(self):
        """
        Get the generalization techniques for quasi-identifiers.
        
        Returns:
            np.ndarray: Array of generalization techniques
        """
        return self.QI.generalization_technique

    @property
    def trees(self):
        """
        Get the generalization trees for categorical attributes.
        
        Returns:
            np.ndarray: Array of generalization trees
        """
        return self.QI.trees

    def __repr__(self) -> str:
        return f"AttributeSchema(QI={type(self.QI)}, SA={type(self.SA) if self.SA is not None else None}, generalization_technique={type(self.generalization_technique) if self.generalization_technique is not None else None}, trees={type(self.trees) if self.trees is not None else None})"

    # Note: generalization technique validation moved to QuasiIdentifiers.check_generalization_technique


def create_attribute_schema(columns_names, QI, generalization_technique, sensitive_attributes_types=None, trees=None, verbose=True):
    """
    Create AttributeSchema with detailed logging and preprocessing.
    
    This helper function provides debugging output and handles the preprocessing
    needed to separate quasi-identifiers from sensitive attributes before
    creating an AttributeSchema object.
    
    Args:
        columns_names (list): Names of all QI columns
        QI (list): QI types for each column
        generalization_technique (list): Generalization techniques for each column
        sensitive_attributes_types (list, optional): Types of sensitive attribute columns
        verbose (bool, optional): Whether to print detailed debugging information. Defaults to True.
        
    Returns:
        AttributeSchema: Configured schema object
    """
    
    # Process sensitive attributes if provided
    if sensitive_attributes_types:
        if len(sensitive_attributes_types) != len(QI):
            raise ValueError("Length of sensitive_attributes must match length of QI.")
        # Check for overlapping indices
        QI_modified = []
        for idx, (sa, qi) in enumerate(zip(sensitive_attributes_types, QI)):
            if sa is not None and qi is not None:
                raise ValueError(f"Column index {idx} cannot be both QI and SA. Found QI='{qi}' and SA='{sa}'.")
            elif sa is None:
                QI_modified.append(qi)
            else:
                QI_modified.append(None)
    else:
        # If no sensitive attributes are configured, keep original QI
        QI_modified = QI
        sensitive_attributes_types = None

    # Create AttributeSchema with updated QI, generalization techniques, and sensitive attribute types
    attributeSchema = AttributeSchema(QI_modified, generalization_technique, sensitive_attributes_types)

    # Display configuration summary
    if verbose:
        print(f"\n{'='*80}")
        print("AttributeSchema Configuration")
        print(f"{'='*80}")
        
        # Count active QI and SA
        qi_count = len([qi for qi in QI_modified if qi is not None])
        sa_count = len([sa for sa in (sensitive_attributes_types or []) if sa is not None])
        total_cols = len(columns_names)
        
        print(f"Total columns    : {total_cols}")
        print(f"QI attributes    : {qi_count}/{total_cols}")
        print(f"SA attributes    : {sa_count}/{total_cols}")
        print(f"Other columns    : {total_cols - qi_count - sa_count}/{total_cols}")
        print(f"{'-'*80}", end=(f"\n" if trees is None else f"{'-'*20}\n"))
        
        # Show detailed column information
        print(f"\n{'Column Name':<25} {'Type':<15} {'Generalization':<20} {'Role':<10} {'Index':<10}", end=("\n" if trees is None else f" {'Tree':<10}\n"))
    
        print(f"{'-'*80}")
        for i, name in enumerate(columns_names):
            gen_tech = generalization_technique[i] if i < len(generalization_technique) and generalization_technique[i] is not None else "N/A"

            # Determine role
            if sensitive_attributes_types and sensitive_attributes_types[i] is not None:
                role = "SA"
                col_type = sensitive_attributes_types[i] if sensitive_attributes_types[i] is not None else "N/A"
            elif QI_modified[i] is not None:
                role = "QI"
                # Use QI_modified (processed) instead of original QI to get the correct type
                col_type = QI_modified[i]
            else:
                role = "Other"
                col_type = "N/A"

            print(f"{name:<25} {col_type:<15} {gen_tech:<20} {role:<10} {i:<10}", end=("\n" if trees is None else f" {trees[i] if i < len(trees) else None:<10}"))

        print(f"{'-'*80}", end=(f"\n" if trees is None else f"{'-'*20}\n"))
        
        # Show internal schema details
        if attributeSchema.QI.allowed.size > 0:
            print(f"QI indices       : {attributeSchema.QI.allowed.tolist()}")
            print(f"QI types         : {['numerical' if t == 0 else 'categorical' for t in attributeSchema.QI.types]}")
            print(f"QI techniques    : {attributeSchema.QI.generalization_technique.tolist()}")
            if attributeSchema.QI.trees is not None:
                print(f"QI trees         : {[type(tree).__name__ if tree is not None else None for tree in attributeSchema.QI.trees]}")
        
        if attributeSchema.SA is not None and attributeSchema.SA.allowed.size > 0:
            print(f"\nSA indices       : {attributeSchema.SA.allowed.tolist()}")
            print(f"SA types         : {['numerical' if t == 0 else 'categorical' for t in attributeSchema.SA.types]}")
            if attributeSchema.SA.trees is not None:
                print(f"SA trees         : {[type(tree).__name__ if tree is not None else None for tree in attributeSchema.SA.trees]}")
        
        print(f"{'='*80}\n")
    
    return attributeSchema
    
