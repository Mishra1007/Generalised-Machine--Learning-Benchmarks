#!/usr/bin/env python
"""Debug CBS frame and aggregation for Edu-Primary."""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Load the normalized results
results_dir = Path('results/Edu-Primary')
normalized_path = results_dir / 'normalized_results.csv'
cbs_path = results_dir / 'cbs_scores.csv'

normalized = pd.read_csv(normalized_path)
cbs_scores = pd.read_csv(cbs_path)

print("Normalized results shape:", normalized.shape)
print("Normalized results columns:", normalized.columns.tolist())
print("Normalized results:\n", normalized)

print("\n\nCBS scores shape:", cbs_scores.shape)
print("CBS scores columns:", cbs_scores.columns.tolist())
print("CBS scores:\n", cbs_scores)

# Build CBS frame like in the validation code
CBS_METRICS = [
    'f1',
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'time_score',
    'stability_score',
]

from metrics.cbs import CBS_WEIGHTS

# Check if columns match
print("\n\nChecking column names:")
print("CBS_WEIGHTS keys:", list(CBS_WEIGHTS.keys()))
print("Normalized columns matching CBS_METRICS:", [m for m in CBS_METRICS if m in normalized.columns])
print("Normalized columns NOT in CBS_METRICS:", [c for c in normalized.columns if c not in CBS_METRICS and c != 'model'])

# Test compute_weighted_scores
def normalize_weights(weights):
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError('CBS weights must sum to a positive value')
    return {k: float(v) / total for k, v in weights.items()}

def compute_weighted_scores(frame, weights):
    normalized = normalize_weights(weights)
    missing = [metric for metric in normalized if metric not in frame.columns]
    if missing:
        print(f"ERROR: Missing CBS metric columns: {missing}")
        print(f"Available columns: {frame.columns.tolist()}")
        raise ValueError(f'Missing CBS metric columns: {missing}')
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for metric, weight in normalized.items():
        score = score + frame[metric].astype(float) * weight
    return score.astype(float)

print("\n\nTesting compute_weighted_scores:")
print("\nMetric values in normalized:")
for metric in CBS_METRICS:
    col_values = normalized[metric].values
    print(f"  {metric}: {col_values} (dtype: {normalized[metric].dtype})")

print("\nChecking for NaN values:")
for metric in CBS_METRICS:
    nan_count = normalized[metric].isna().sum()
    print(f"  {metric}: {nan_count} NaN values")

print("\nNormalized weights:")
norm_weights = normalize_weights(CBS_WEIGHTS)
for metric, weight in norm_weights.items():
    print(f"  {metric}: {weight}")

try:
    scores = compute_weighted_scores(normalized, CBS_WEIGHTS)
    print("\nComputed scores:")
    print(scores)
    
    # Debug: manually compute scores step by step
    print("\nManual computation:")
    norm_weights = normalize_weights(CBS_WEIGHTS)
    manual_score = pd.Series(0.0, index=normalized.index, dtype=float)
    for metric, weight in norm_weights.items():
        values = normalized[metric].astype(float) * weight
        print(f"  {metric} ({weight:.4f}): {values.values}")
        manual_score = manual_score + values
    print(f"  Manual score: {manual_score.values}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
