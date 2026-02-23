"""
MovieLens Data Loading Module

Loads and parses MovieLens 100K dataset files:
- u.data: User ratings (user_id, item_id, rating, timestamp)
- u.item: Movie metadata (item_id, title, release_date, etc.)
"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional


def load_movielens_100k(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 100K dataset.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing ml-100k folder. If None, looks for 'ml-100k' in current directory.
    
    Returns
    -------
    ratings_df : pd.DataFrame
        DataFrame with columns: user_id, item_id, rating, timestamp
    items_df : pd.DataFrame
        DataFrame with columns: item_id, title, release_date, and other metadata
    
    Raises
    ------
    FileNotFoundError
        If u.data or u.item files cannot be found
    """
    if data_dir is None:
        # Try common locations
        possible_dirs = ['ml-100k', 'data/ml-100k', '../ml-100k']
        data_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(os.path.join(dir_path, 'u.data')):
                data_dir = dir_path
                break
        
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find ml-100k directory. "
                "Please download from https://grouplens.org/datasets/movielens/100k/ "
                "and extract to 'ml-100k' folder, or specify data_dir parameter."
            )
    
    u_data_path = os.path.join(data_dir, 'u.data')
    u_item_path = os.path.join(data_dir, 'u.item')
    
    if not os.path.exists(u_data_path):
        raise FileNotFoundError(f"Rating file not found: {u_data_path}")
    if not os.path.exists(u_item_path):
        raise FileNotFoundError(f"Item file not found: {u_item_path}")
    
    # Load ratings: user_id, item_id, rating, timestamp
    # File is tab-separated, no header
    ratings_df = pd.read_csv(
        u_data_path,
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    # Load items: item_id | title | release_date | video_release_date | 
    #              IMDb URL | unknown | Action | Adventure | Animation | 
    #              Children's | Comedy | Crime | Documentary | Drama | 
    #              Fantasy | Film-Noir | Horror | Musical | Mystery | 
    #              Romance | Sci-Fi | Thriller | War | Western
    # File is pipe-separated, no header
    item_columns = [
        'item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    items_df = pd.read_csv(
        u_item_path,
        sep='|',
        header=None,
        names=item_columns,
        encoding='latin-1'
    )
    
    # Convert timestamp to datetime for easier handling
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # Ensure item_id types match
    ratings_df['item_id'] = ratings_df['item_id'].astype(int)
    items_df['item_id'] = items_df['item_id'].astype(int)
    
    # Validate data
    if ratings_df.empty:
        raise ValueError("Ratings DataFrame is empty")
    if items_df.empty:
        raise ValueError("Items DataFrame is empty")
    
    # Check for missing values in critical columns
    if ratings_df[['user_id', 'item_id', 'rating']].isnull().any().any():
        raise ValueError("Ratings data contains missing values in critical columns")
    
    print(f"Loaded {len(ratings_df):,} ratings from {ratings_df['user_id'].nunique():,} users")
    print(f"Loaded {len(items_df):,} items")
    print(f"Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
    
    return ratings_df, items_df
