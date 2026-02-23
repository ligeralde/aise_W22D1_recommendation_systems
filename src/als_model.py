"""
ALS (Alternating Least Squares) Recommendation Model

Matrix factorization using implicit library for collaborative filtering.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import implicit
from scipy.sparse import csr_matrix
import os


class ALSModel:
    """
    ALS model using implicit library.
    
    Implements matrix factorization with alternating least squares optimization.
    Default configuration: factors=32, iterations=10, optimized for CPU.
    """
    
    def __init__(
        self,
        factors: int = 32,
        iterations: int = 10,
        regularization: float = 0.1,
        alpha: float = 40.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize ALS model.
        
        Parameters
        ----------
        factors : int, default 32
            Number of latent factors
        iterations : int, default 10
            Number of ALS iterations
        regularization : float, default 0.1
            Regularization parameter
        alpha : float, default 40.0
            Confidence weight (higher = more weight to observed interactions)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize implicit ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
            use_gpu=False,  # CPU-only for lesson requirements
            num_threads=0  # Use all available CPU threads
        )
        
        self.user_factors_ = None
        self.item_factors_ = None
        self.user_id_map_ = None
        self.item_id_map_ = None
        self.reverse_user_map_ = None
        self.reverse_item_map_ = None
    
    def _build_interaction_matrix(
        self,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        rating_col: str = 'rating'
    ) -> Tuple[csr_matrix, dict, dict]:
        """
        Build sparse interaction matrix from DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Interaction data
        user_col : str
            User ID column name
        item_col : str
            Item ID column name
        rating_col : str
            Rating column name
        
        Returns
        -------
        matrix : csr_matrix
            Sparse matrix of shape (n_users, n_items)
        user_id_map : dict
            Mapping from original user_id to matrix index
        item_id_map : dict
            Mapping from original item_id to matrix index
        """
        # Create mappings
        unique_users = sorted(df[user_col].unique())
        unique_items = sorted(df[item_col].unique())
        
        user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Build sparse matrix
        # implicit library expects confidence scores, not raw ratings
        # We'll use alpha * rating as confidence
        row_indices = [user_id_map[user_id] for user_id in df[user_col]]
        col_indices = [item_id_map[item_id] for item_id in df[item_col]]
        # Convert ratings to confidence scores
        confidence_scores = self.alpha * df[rating_col].values
        
        matrix = csr_matrix(
            (confidence_scores, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        return matrix, user_id_map, item_id_map
    
    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        rating_col: str = 'rating'
    ):
        """
        Fit ALS model on training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions
        user_col : str
            User ID column name
        item_col : str
            Item ID column name
        rating_col : str
            Rating column name
        """
        print(f"Building interaction matrix from {len(train_df):,} interactions...")
        matrix, user_id_map, item_id_map = self._build_interaction_matrix(
            train_df, user_col, item_col, rating_col
        )
        
        # Store mappings
        self.user_id_map_ = user_id_map
        self.item_id_map_ = item_id_map
        self.reverse_user_map_ = {v: k for k, v in user_id_map.items()}
        self.reverse_item_map_ = {v: k for k, v in item_id_map.items()}
        
        print(f"Matrix shape: {matrix.shape[0]:,} users × {matrix.shape[1]:,} items")
        print(f"Training ALS model (factors={self.factors}, iterations={self.iterations})...")
        
        # Fit model
        self.model.fit(matrix)
        
        # Extract factors
        self.user_factors_ = self.model.user_factors
        self.item_factors_ = self.model.item_factors
        
        print(f"Training complete. User factors shape: {self.user_factors_.shape}")
        print(f"Item factors shape: {self.item_factors_.shape}")
    
    def predict(
        self,
        user_ids: List[int],
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate top-K recommendations for users.
        
        Parameters
        ----------
        user_ids : list of int
            User IDs to generate recommendations for
        k : int, default 10
            Number of recommendations per user
        exclude_items : list of int, optional
            Item IDs to exclude from recommendations
        
        Returns
        -------
        recommendations : pd.DataFrame
            DataFrame with columns: user_id, item_id, score, rank
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        recommendations = []
        
        for user_id in user_ids:
            if user_id not in self.user_id_map_:
                # Cold start: user not in training data
                # Skip or use popularity fallback
                continue
            
            user_idx = self.user_id_map_[user_id]
            
            # Compute scores using learned factors: user_factor @ item_factors^T
            user_factor = self.user_factors_[user_idx]
            scores = np.dot(self.item_factors_, user_factor)
            
            # Get top-K items (excluding specified items)
            item_indices = np.arange(len(scores))
            
            if exclude_items:
                # Map exclude_items to indices
                exclude_indices = [
                    self.item_id_map_[item_id] 
                    for item_id in exclude_items 
                    if item_id in self.item_id_map_
                ]
                # Set scores for excluded items to -inf
                scores[exclude_indices] = -np.inf
            
            # Get top-K items
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_scores = scores[top_k_indices]
            
            # Convert item indices back to item IDs
            user_recs = []
            for rank, (item_idx, score) in enumerate(zip(top_k_indices, top_k_scores), start=1):
                if score == -np.inf:
                    continue  # Skip excluded items
                item_id = self.reverse_item_map_[item_idx]
                user_recs.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'score': float(score),
                    'rank': rank
                })
            
            recommendations.extend(user_recs)
        
        return pd.DataFrame(recommendations)
    
    def save_factors(self, output_dir: str = 'artifacts'):
        """
        Save user and item factors to numpy files.
        
        Parameters
        ----------
        output_dir : str, default 'artifacts'
            Directory to save factor files
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise ValueError("Model must be fitted before saving factors")
        
        os.makedirs(output_dir, exist_ok=True)
        
        user_factors_path = os.path.join(output_dir, 'user_factors.npy')
        item_factors_path = os.path.join(output_dir, 'item_factors.npy')
        
        np.save(user_factors_path, self.user_factors_)
        np.save(item_factors_path, self.item_factors_)
        
        print(f"Saved user factors to {user_factors_path}")
        print(f"Saved item factors to {item_factors_path}")
    
    def load_factors(self, factors_dir: str = 'artifacts'):
        """
        Load user and item factors from numpy files.
        
        Parameters
        ----------
        factors_dir : str, default 'artifacts'
            Directory containing factor files
        """
        user_factors_path = os.path.join(factors_dir, 'user_factors.npy')
        item_factors_path = os.path.join(factors_dir, 'item_factors.npy')
        
        if not os.path.exists(user_factors_path):
            raise FileNotFoundError(f"User factors not found: {user_factors_path}")
        if not os.path.exists(item_factors_path):
            raise FileNotFoundError(f"Item factors not found: {item_factors_path}")
        
        self.user_factors_ = np.load(user_factors_path)
        self.item_factors_ = np.load(item_factors_path)
        
        # Note: This doesn't restore the full model state, just factors
        # For full prediction, you'd need to also restore mappings
        print(f"Loaded user factors: {self.user_factors_.shape}")
        print(f"Loaded item factors: {self.item_factors_.shape}")
