"""
Evaluation Metrics for Recommendation Systems

Implements Recall@K, NDCG@K, Precision@K and leakage detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import ndcg_score
import json
import os


def recall_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    Compute Recall@K metric.
    
    Recall@K = (number of relevant items in top-K) / (total relevant items)
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendations with columns: user_id, item_id, rank
    ground_truth : pd.DataFrame
        Ground truth interactions with columns: user_id, item_id
    k : int, default 10
        Number of top recommendations to consider
    
    Returns
    -------
    recall : float
        Average Recall@K across all users
    """
    # Filter to top-K recommendations
    top_k_recs = recommendations[recommendations['rank'] <= k].copy()
    
    # Group ground truth by user
    gt_by_user = ground_truth.groupby('user_id')['item_id'].apply(set).to_dict()
    
    recalls = []
    for user_id in top_k_recs['user_id'].unique():
        user_recs = set(top_k_recs[top_k_recs['user_id'] == user_id]['item_id'].values)
        
        if user_id in gt_by_user:
            user_gt = gt_by_user[user_id]
            if len(user_gt) > 0:
                recall = len(user_recs & user_gt) / len(user_gt)
                recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def precision_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    """
    Compute Precision@K metric.
    
    Precision@K = (number of relevant items in top-K) / K
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendations with columns: user_id, item_id, rank
    ground_truth : pd.DataFrame
        Ground truth interactions with columns: user_id, item_id
    k : int, default 10
        Number of top recommendations to consider
    
    Returns
    -------
    precision : float
        Average Precision@K across all users
    """
    # Filter to top-K recommendations
    top_k_recs = recommendations[recommendations['rank'] <= k].copy()
    
    # Group ground truth by user
    gt_by_user = ground_truth.groupby('user_id')['item_id'].apply(set).to_dict()
    
    precisions = []
    for user_id in top_k_recs['user_id'].unique():
        user_recs = top_k_recs[top_k_recs['user_id'] == user_id]['item_id'].values[:k]
        
        if user_id in gt_by_user:
            user_gt = gt_by_user[user_id]
            if len(user_recs) > 0:
                precision = len(set(user_recs) & user_gt) / len(user_recs)
                precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def ndcg_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain) metric.
    
    Uses scikit-learn's ndcg_score implementation.
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendations with columns: user_id, item_id, score, rank
    ground_truth : pd.DataFrame
        Ground truth interactions with columns: user_id, item_id
    k : int, default 10
        Number of top recommendations to consider
    
    Returns
    -------
    ndcg : float
        Average NDCG@K across all users
    """
    # Filter to top-K recommendations
    top_k_recs = recommendations[recommendations['rank'] <= k].copy()
    
    # Group ground truth by user
    gt_by_user = ground_truth.groupby('user_id')['item_id'].apply(set).to_dict()
    
    # Get all unique items for building relevance vectors
    all_items = set(recommendations['item_id'].unique()) | set(ground_truth['item_id'].unique())
    all_items = sorted(all_items)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(all_items)}
    
    ndcgs = []
    for user_id in top_k_recs['user_id'].unique():
        user_recs = top_k_recs[top_k_recs['user_id'] == user_id].sort_values('rank')
        
        if user_id not in gt_by_user:
            continue
        
        user_gt = gt_by_user[user_id]
        
        # Build relevance vector (1 if item in ground truth, 0 otherwise)
        y_true = np.zeros(len(all_items))
        for item_id in user_gt:
            if item_id in item_to_idx:
                y_true[item_to_idx[item_id]] = 1
        
        # Build score vector for recommendations
        y_score = np.zeros(len(all_items))
        for _, row in user_recs.iterrows():
            item_id = row['item_id']
            if item_id in item_to_idx:
                # Use score if available, otherwise use inverse rank
                score = row.get('score', 1.0 / row['rank'])
                y_score[item_to_idx[item_id]] = score
        
        # Compute NDCG
        if np.sum(y_true) > 0:  # Only compute if there are relevant items
            ndcg = ndcg_score([y_true], [y_score], k=k)
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def check_temporal_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    recommendations: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> Dict[str, bool]:
    """
    Check for temporal leakage in recommendations.
    
    Validates that recommendations don't include items that users interacted with
    in training data for the same time window (should be excluded).
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions with timestamp
    test_df : pd.DataFrame
        Test interactions with timestamp
    recommendations : pd.DataFrame
        Recommendations with columns: user_id, item_id
    timestamp_col : str, default 'timestamp'
        Timestamp column name
    
    Returns
    -------
    leakage_report : dict
        Dictionary with leakage check results
    """
    # Group training interactions by user
    train_by_user = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    # Check if recommendations include training items for same users
    leakage_count = 0
    total_recs = 0
    
    for user_id in recommendations['user_id'].unique():
        user_recs = set(recommendations[recommendations['user_id'] == user_id]['item_id'].values)
        
        if user_id in train_by_user:
            train_items = train_by_user[user_id]
            # Items that appear in both training and recommendations
            overlap = user_recs & train_items
            if len(overlap) > 0:
                leakage_count += len(overlap)
        
        total_recs += len(user_recs)
    
    leakage_rate = leakage_count / total_recs if total_recs > 0 else 0.0
    
    # Check temporal ordering
    if len(train_df) > 0 and len(test_df) > 0:
        max_train_time = train_df[timestamp_col].max()
        min_test_time = test_df[timestamp_col].min()
        temporal_order_valid = max_train_time < min_test_time
    else:
        temporal_order_valid = True
    
    return {
        'has_leakage': leakage_count > 0,
        'leakage_count': leakage_count,
        'leakage_rate': leakage_rate,
        'temporal_order_valid': temporal_order_valid,
        'total_recommendations': total_recs
    }


def evaluate_model(
    recommendations: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    k_values: List[int] = [5, 10, 20],
    timestamp_col: str = 'timestamp'
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Computes multiple metrics and performs sanity checks.
    
    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendations with columns: user_id, item_id, score, rank
    test_df : pd.DataFrame
        Test set interactions
    train_df : pd.DataFrame, optional
        Training set interactions (for leakage detection)
    k_values : list of int, default [5, 10, 20]
        K values for metrics
    timestamp_col : str, default 'timestamp'
        Timestamp column name
    
    Returns
    -------
    results : dict
        Dictionary containing all evaluation metrics
    """
    results = {
        'model': 'unknown',
        'metrics': {},
        'leakage_check': {},
        'sanity_checks': {}
    }
    
    # Compute metrics for each K
    for k in k_values:
        recall = recall_at_k(recommendations, test_df, k=k)
        precision = precision_at_k(recommendations, test_df, k=k)
        ndcg = ndcg_at_k(recommendations, test_df, k=k)
        
        results['metrics'][f'recall@{k}'] = float(recall)
        results['metrics'][f'precision@{k}'] = float(precision)
        results['metrics'][f'ndcg@{k}'] = float(ndcg)
    
    # Leakage detection
    if train_df is not None:
        leakage_report = check_temporal_leakage(train_df, test_df, recommendations, timestamp_col)
        results['leakage_check'] = leakage_report
    
    # Sanity checks
    recall_10 = results['metrics'].get('recall@10', 0.0)
    results['sanity_checks'] = {
        'recall@10_not_zero': recall_10 > 0.0,
        'recall@10_value': float(recall_10),
        'num_recommendations': len(recommendations),
        'num_users': recommendations['user_id'].nunique() if len(recommendations) > 0 else 0,
        'num_items': recommendations['item_id'].nunique() if len(recommendations) > 0 else 0
    }
    
    return results


def save_evaluation_results(results: Dict, output_path: str = 'artifacts/offline_eval_rec.json'):
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    results : dict
        Evaluation results dictionary
    output_path : str, default 'artifacts/offline_eval_rec.json'
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {output_path}")
