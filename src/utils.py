"""
Utility Functions

Helper functions for the recommendation systems lesson.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def get_top_items(df: pd.DataFrame, item_col: str = 'item_id', 
                  score_col: str = 'rating', n: int = 10) -> pd.DataFrame:
    """
    Get top N items by score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with item and score columns
    item_col : str
        Item ID column name
    score_col : str
        Score column name
    n : int
        Number of top items to return
    
    Returns
    -------
    top_items : pd.DataFrame
        Top N items sorted by score
    """
    return df.nlargest(n, score_col)[[item_col, score_col]]


def filter_by_users(df: pd.DataFrame, user_ids: List[int], 
                    user_col: str = 'user_id') -> pd.DataFrame:
    """
    Filter DataFrame to include only specified users.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with user column
    user_ids : list of int
        User IDs to include
    user_col : str
        User ID column name
    
    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered DataFrame
    """
    return df[df[user_col].isin(user_ids)].copy()
