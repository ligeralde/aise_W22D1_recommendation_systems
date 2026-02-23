# Recommendation Systems Lesson

Interactive repository for learning recommendation systems using the MovieLens 100K dataset.

## Overview

This repository implements a complete recommendation system workflow including:
- Data loading and exploration
- Temporal data splitting
- Baseline recommendation models (popularity-based)
- ALS (Alternating Least Squares) matrix factorization
- Model evaluation with Recall@K, NDCG@K, and Precision@K
- Artifact generation and validation

## Requirements

- Python 3.10+
- Local venv or conda environment
- Jupyter or VS Code notebooks

## Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n recsys python=3.10
conda activate recsys
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

**Option A: Automated Download**

```bash
python setup_data.py
```

**Option B: Manual Download**

1. Visit https://grouplens.org/datasets/movielens/100k/
2. Download `ml-100k.zip`
3. Extract to `ml-100k/` folder in project root

**Backup Dataset**

If MovieLens 100K is unavailable, you can use MovieLens "small" dataset:
- https://grouplens.org/datasets/movielens/latest/
- Download the "small" version (stable link, lightweight)

### 4. Verify Setup

```python
# Quick import test
import numpy as np
import pandas as pd
import scipy
import sklearn
import implicit
import torch
import matplotlib

print("✓ All dependencies installed successfully")
```

## Usage

### Run the Main Notebook

```bash
jupyter notebook notebooks/main_lesson.ipynb
```

Or open in VS Code and run the notebook interactively.

### Expected Workflow

1. **Data Loading**: Parse `u.data` and `u.item` into pandas DataFrames
2. **Data Splitting**: Temporal split ensuring training < validation < test
3. **Baseline Model**: Train popularity model (< 5 seconds)
4. **ALS Model**: Train with factors=32, iterations=10 (~5-10 minutes on CPU)
5. **Evaluation**: Compute Recall@K, NDCG@K, Precision@K
6. **Artifacts**: Generate all required output files

### Expected Artifacts

After running the notebook, you should have:

```
artifacts/
├── offline_eval_rec.json      # Evaluation metrics
├── user_factors.npy           # ALS user embeddings
├── item_factors.npy           # ALS item embeddings
└── rec_candidates.parquet     # Recommendations (user_id, item_id, score, rank)
```

## Project Structure

```
aise_W22D1_recommendation_systems/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup_data.py             # Data download script
├── notebooks/
│   └── main_lesson.ipynb     # Main interactive notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # MovieLens data parsing
│   ├── data_split.py         # Temporal splitting
│   ├── baselines.py          # Popularity model
│   ├── als_model.py          # ALS implementation
│   ├── evaluation.py         # Metrics and evaluation
│   └── utils.py              # Helper functions
└── artifacts/                # Generated outputs (gitignored)
```

## Key Requirements

### Temporal Splitting
- Training data must only contain interactions **earlier** than validation/test
- Validated automatically in the notebook

### Baseline Model
- Popularity model runs in **< 5 seconds**
- Ranks items by interaction count or average rating

### ALS Model
- Configuration: `factors=32`, `iterations=10`
- CPU-optimized (no GPU required)
- Runs within **~5-10 minutes** on typical CPU

### Evaluation Metrics
- **Recall@K**: Fraction of relevant items found in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain (using scikit-learn)
- **Precision@K**: Fraction of top-K items that are relevant

### Sanity Checks
- ✅ Recall@10 is not zero
- ✅ No training-heldout leakage for same user/time window
- ✅ Temporal ordering validated (train < val < test)

## Instructor Prep Checklist

### Dataset Access
- [ ] Download `ml-100k.zip` from GroupLens
- [ ] Verify you can parse `u.data` and `u.item` into pandas
- [ ] Test data loading: `load_movielens_100k('ml-100k')`

### Environment Setup
- [ ] Create virtual environment: `python -m venv .venv`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Import test: `numpy`, `pandas`, `scikit-learn`, `scipy`, `implicit`, `torch`
- [ ] Run notebook end-to-end on CPU

### Validation
- [ ] **Split**: Confirm training only uses interactions earlier than validation/test cutoff
- [ ] **Baselines**: Popularity model runs in < 5 seconds
- [ ] **ALS**: factors=32, iterations=10, runs within ~5-10 minutes on typical CPU

### Output Validation
- [ ] `artifacts/offline_eval_rec.json` created
- [ ] `artifacts/user_factors.npy`, `artifacts/item_factors.npy` created
- [ ] `artifacts/rec_candidates.parquet` includes `user_id`, `item_id`, `score`, `rank`

### Sanity Checks
- [ ] Recall@10 is not zero
- [ ] Candidate list has no training-heldout leakage for same user/time window

## Core References

- **implicit library**: [GitHub](https://github.com/benfred/implicit) | [Documentation](https://implicit.readthedocs.io/)
- **PyTorch**: [Get Started (Locally)](https://pytorch.org/get-started/locally/)
- **scikit-learn NDCG**: [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- **MovieLens Dataset**: [GroupLens](https://grouplens.org/datasets/movielens/)

## Troubleshooting

### Import Errors
If you get import errors in the notebook, ensure:
1. Virtual environment is activated
2. `src/` directory is in Python path (handled automatically in notebook)
3. All dependencies installed: `pip install -r requirements.txt`

### Data Not Found
If data loading fails:
1. Verify `ml-100k/` folder exists in project root
2. Check that `u.data` and `u.item` files are present
3. Run `python setup_data.py` to download automatically

### Performance Issues
- ALS training may take longer on slower CPUs (10-15 minutes is acceptable)
- If popularity model takes > 5 seconds, check data size and system resources

## License

This is an educational repository. The MovieLens dataset is provided by GroupLens Research under their terms of use.
# aise_W22D1_recommendation_systems
