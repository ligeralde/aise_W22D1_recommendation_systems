"""
Baseline Recommendation Models

Simple baseline models for comparison with more sophisticated approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class PopularityModel:
    """
    Popularity-based recommendation model.
    
    Recommends items based on their overall popularity (interaction count or average rating).
    Fast baseline that should run in < 5 seconds.
    """
    
    def __init__(self, ranking_method: str = 'count'):
        """
        Initialize popularity model.
        
        Parameters
        ----------
        ranking_method : str, default 'count'
            How to rank items: 'count' (by number of interactions) or 
            'rating' (by average rating)
        """
        if ranking_method not in ['count', 'rating']:
            raise ValueError(f"ranking_method must be 'count' or 'rating', got {ranking_method}")
        
        self.ranking_method = ranking_method
        self.item_scores_ = None
        self.item_rankings_ = None
    
    def fit(self, train_df: pd.DataFrame, rating_col: str = 'rating'):
        """
        Fit the popularity model on training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions with columns: user_id, item_id, rating
        rating_col : str, default 'rating'
            Name of rating column
        """
        if self.ranking_method == 'count':
            # Rank by number of interactions
            item_counts = train_df.groupby('item_id').size()
            self.item_scores_ = item_counts.to_dict()
        else:  # ranking_method == 'rating'
            # Rank by average rating
            item_ratings = train_df.groupby('item_id')[rating_col].mean()
            self.item_scores_ = item_ratings.to_dict()
        
        # Create ranking (higher score = better)
        sorted_items = sorted(
            self.item_scores_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.item_rankings_ = {item_id: rank + 1 for rank, (item_id, _) in enumerate(sorted_items)}
        
        print(f"Fitted popularity model with {len(self.item_scores_)} items")
        print(f"Top 5 items: {[item_id for item_id, _ in sorted_items[:5]]}")
    
    def predict(self, user_ids: Optional[List[int]] = None, k: int = 10) -> pd.DataFrame:
        """
        Generate top-K recommendations.
        
        Parameters
        ----------
        user_ids : list of int, optional
            Users to generate recommendations for. If None, uses all items
            (same recommendations for all users in popularity model)
        k : int, default 10
            Number of recommendations per user
        
        Returns
        -------
        recommendations : pd.DataFrame
            DataFrame with columns: user_id, item_id, score, rank
        """
        if self.item_rankings_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get top-K items
        sorted_items = sorted(
            self.item_scores_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # If no user_ids provided, we still need to return a DataFrame
        # For popularity model, recommendations are the same for all users
        if user_ids is None:
            # Return top-K items without user_id (or with a dummy user_id)
            # This is a design choice - for consistency, we'll require user_ids
            raise ValueError("user_ids must be provided")
        
        # Create recommendations for each user
        recommendations = []
        for user_id in user_ids:
            for rank, (item_id, score) in enumerate(sorted_items, start=1):
                recommendations.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'score': score,
                    'rank': rank
                })
        
        return pd.DataFrame(recommendations)
    
    def recommend_for_user(self, user_id: int, k: int = 10, 
                          exclude_items: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate top-K recommendations for a single user.
        
        Parameters
        ----------
        user_id : int
            User ID
        k : int, default 10
            Number of recommendations
        exclude_items : list of int, optional
            Item IDs to exclude from recommendations
        
        Returns
        -------
        recommendations : pd.DataFrame
            DataFrame with columns: user_id, item_id, score, rank
        """
        if self.item_rankings_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get top-K items, excluding specified items
        sorted_items = sorted(
            self.item_scores_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if exclude_items:
            sorted_items = [(item_id, score) for item_id, score in sorted_items 
                          if item_id not in exclude_items]
        
        top_k = sorted_items[:k]
        
        recommendations = []
        for rank, (item_id, score) in enumerate(top_k, start=1):
            recommendations.append({
                'user_id': user_id,
                'item_id': item_id,
                'score': score,
                'rank': rank
            })
        
        return pd.DataFrame(recommendations)
