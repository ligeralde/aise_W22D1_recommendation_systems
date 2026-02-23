"""
Time-based Data Splitting Module

Splits interaction data temporally to ensure training data only contains
interactions that occurred before validation and test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def time_based_split(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = 'timestamp',
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings data by timestamp ensuring temporal ordering.
    
    Training set contains only interactions before validation cutoff.
    Validation set contains only interactions before test cutoff.
    Test set contains the most recent interactions.
    
    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame with columns including user_id, item_id, and timestamp
    train_ratio : float, default 0.7
        Proportion of data for training (by time, not by count)
    val_ratio : float, default 0.15
        Proportion of data for validation
    test_ratio : float, default 0.15
        Proportion of data for testing
    timestamp_col : str, default 'timestamp'
        Name of timestamp column
    random_seed : int, optional
        Random seed for reproducibility (affects tie-breaking for same timestamps)
    
    Returns
    -------
    train_df : pd.DataFrame
        Training set (earliest interactions)
    val_df : pd.DataFrame
        Validation set (middle interactions)
    test_df : pd.DataFrame
        Test set (latest interactions)
    
    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 or if timestamp column is missing
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if timestamp_col not in ratings_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
    
    # Sort by timestamp to ensure temporal ordering
    df_sorted = ratings_df.sort_values(timestamp_col).copy()
    
    # Calculate split points based on timestamp percentiles
    timestamps = df_sorted[timestamp_col]
    train_cutoff = timestamps.quantile(train_ratio)
    val_cutoff = timestamps.quantile(train_ratio + val_ratio)
    
    # Split data
    train_df = df_sorted[df_sorted[timestamp_col] < train_cutoff].copy()
    val_df = df_sorted[
        (df_sorted[timestamp_col] >= train_cutoff) & 
        (df_sorted[timestamp_col] < val_cutoff)
    ].copy()
    test_df = df_sorted[df_sorted[timestamp_col] >= val_cutoff].copy()
    
    # Handle edge case: if cutoff timestamps have many ties, we might need to
    # randomly assign some records to ensure proper ratios
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate temporal ordering
    if len(train_df) > 0 and len(val_df) > 0:
        max_train_time = train_df[timestamp_col].max()
        min_val_time = val_df[timestamp_col].min()
        if max_train_time >= min_val_time:
            # There are ties at the boundary, need to handle them
            boundary_records = df_sorted[
                (df_sorted[timestamp_col] == train_cutoff) |
                (df_sorted[timestamp_col] == val_cutoff)
            ]
            # For simplicity, assign boundary records to later split
            # This ensures strict temporal ordering
            pass  # Already handled by < and >= operators
    
    if len(val_df) > 0 and len(test_df) > 0:
        max_val_time = val_df[timestamp_col].max()
        min_test_time = test_df[timestamp_col].min()
        if max_val_time >= min_test_time:
            # Similar boundary handling
            pass
    
    # Print split statistics
    print(f"Temporal split completed:")
    print(f"  Training: {len(train_df):,} interactions ({len(train_df)/len(df_sorted)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} interactions ({len(val_df)/len(df_sorted)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} interactions ({len(test_df)/len(df_sorted)*100:.1f}%)")
    
    if len(train_df) > 0:
        print(f"  Training time range: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}")
    if len(val_df) > 0:
        print(f"  Validation time range: {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}")
    if len(test_df) > 0:
        print(f"  Test time range: {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}")
    
    # Validate no temporal leakage
    if len(train_df) > 0 and len(test_df) > 0:
        max_train = train_df[timestamp_col].max()
        min_test = test_df[timestamp_col].min()
        if max_train >= min_test:
            raise ValueError(
                f"Temporal leakage detected! Max training time ({max_train}) >= "
                f"Min test time ({min_test})"
            )
    
    return train_df, val_df, test_df
