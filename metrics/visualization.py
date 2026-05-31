"""Visualization utilities for benchmarking results.

Produces publication-quality plots:
- Model vs Accuracy
- Model vs F1
- Model vs CBS
- ROC-AUC comparison
- Training Time comparison
- Heatmap of normalized metrics

All plots are saved automatically (PNG and PDF) under an output directory per-dataset.
"""

from typing import Dict, Any, List, Tuple
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None
    logging.getLogger(__name__).warning('seaborn not available; falling back to matplotlib styles')

from metrics.normalization import normalize_model_summaries
from metrics.cbs import compute_cbs

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_fig(fig, out_path_base: str):
    png = out_path_base + '.png'
    pdf = out_path_base + '.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _bar_plot(models: List[str], values: List[float], ylabel: str, title: str, out_path_base: str):
    if sns is not None:
        sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(values)[::-1]
    models_sorted = [models[i] for i in order]
    vals_sorted = [values[i] for i in order]

    if sns is not None:
        palette = sns.color_palette('deep', len(models_sorted))
        sns.barplot(x=models_sorted, y=vals_sorted, palette=palette, ax=ax)
    else:
        colors = plt.get_cmap('tab10').colors
        bar_colors = [colors[i % len(colors)] for i in range(len(models_sorted))]
        ax.bar(models_sorted, vals_sorted, color=bar_colors)

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
    # Annotate
    span = (max(vals_sorted) - min(vals_sorted)) if vals_sorted else 1
    for i, v in enumerate(vals_sorted):
        ax.text(i, v + (0.01 * span), f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return _save_fig(fig, out_path_base)


def _heatmap_normalized(normalized_map: Dict[str, Dict[str, float]], out_path_base: str, title: str = 'Normalized Metrics'):
    if sns is not None:
        sns.set_theme(style='white')
    df = pd.DataFrame.from_dict(normalized_map, orient='index')
    # Order columns for readability
    cols_order = ['f1', 'roc_auc', 'accuracy', 'precision', 'recall', 'time_score', 'stability_score']
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df)), max(4, 0.6 * len(df.columns))))
    if sns is not None:
        cmap = sns.color_palette('vlag', as_cmap=True)
        sns.heatmap(df, annot=True, fmt='.3f', cmap=cmap, cbar_kws={'label': 'Normalized Score'}, ax=ax)
    else:
        # Matplotlib fallback
        im = ax.imshow(df.values, aspect='auto', cmap='RdYlBu')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Score')
        # annotation
        for (i, j), val in np.ndenumerate(df.values):
            ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=9)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=10)

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, out_path_base)


def generate_plots_for_dataset(dataset_name: str, models_summaries: Dict[str, Dict[str, Any]], out_dir: str = 'plots') -> Dict[str, Tuple[str, str]]:
    """Generate all required plots for a single dataset.

    Returns mapping plot_name -> (png_path, pdf_path)
    """
    out_root = os.path.join(out_dir, dataset_name)
    _ensure_dir(out_root)

    # Normalize and compute CBS
    normalized = normalize_model_summaries(models_summaries)
    cbs_map = compute_cbs(models_summaries)

    # Prepare lists
    model_names = list(models_summaries.keys())

    # Accuracy
    acc_vals = []
    f1_vals = []
    roc_vals = []
    time_vals = []
    cbs_vals = []

    for m in model_names:
        s = models_summaries[m].get('overall', {})
        acc = s.get('accuracy_mean') or s.get('accuracy') or 0.0
        f1 = s.get('f1_mean') or s.get('f1') or 0.0
        roc = s.get('roc_auc_mean') or s.get('roc_auc') or 0.0
        # time: prefer total_time_mean else sum
        tt = s.get('total_time_mean')
        if tt is None:
            tt = (s.get('train_time_mean') or 0.0) + (s.get('eval_time_mean') or 0.0)
        acc_vals.append(float(acc))
        f1_vals.append(float(f1))
        roc_vals.append(float(roc))
        time_vals.append(float(tt))
        cbs_vals.append(float(cbs_map[m]['cbs']))

    plots = {}

    # Model vs Accuracy
    plots['model_vs_accuracy'] = _bar_plot(model_names, acc_vals, 'Accuracy', f'{dataset_name}: Model vs Accuracy', os.path.join(out_root, f'{dataset_name}_model_vs_accuracy'))

    # Model vs F1
    plots['model_vs_f1'] = _bar_plot(model_names, f1_vals, 'F1 Score', f'{dataset_name}: Model vs F1', os.path.join(out_root, f'{dataset_name}_model_vs_f1'))

    # Model vs CBS
    plots['model_vs_cbs'] = _bar_plot(model_names, cbs_vals, 'CBS (Composite Benchmark Score)', f'{dataset_name}: Model vs CBS', os.path.join(out_root, f'{dataset_name}_model_vs_cbs'))

    # ROC-AUC comparison
    plots['roc_auc'] = _bar_plot(model_names, roc_vals, 'ROC-AUC', f'{dataset_name}: ROC-AUC Comparison', os.path.join(out_root, f'{dataset_name}_roc_auc'))

    # Training Time comparison (log-scale if wide spread)
    # Decide scale: if max/min ratio > 100, use log scale
    max_t = max(time_vals) if time_vals else 0.0
    min_t = min([t for t in time_vals if t > 0]) if any(t > 0 for t in time_vals) else 0.0
    fig_time, ax_time = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(time_vals)[::-1]
    models_sorted = [model_names[i] for i in order]
    vals_sorted = [time_vals[i] for i in order]
    if sns is not None:
        sns.barplot(x=models_sorted, y=vals_sorted, palette='muted', ax=ax_time)
    else:
        colors = plt.get_cmap('tab10').colors
        bar_colors = [colors[i % len(colors)] for i in range(len(models_sorted))]
        ax_time.bar(models_sorted, vals_sorted, color=bar_colors)
    ax_time.set_title(f'{dataset_name}: Training Time Comparison', fontsize=14, weight='bold')
    ax_time.set_ylabel('Training Time (s)', fontsize=12)
    ax_time.set_xlabel('Model', fontsize=12)
    ax_time.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
    if min_t > 0 and max_t / min_t > 100:
        ax_time.set_yscale('log')
    for i, v in enumerate(vals_sorted):
        ax_time.text(i, v * 1.01 if v != 0 else 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    fig_time.tight_layout()
    plots['training_time'] = _save_fig(fig_time, os.path.join(out_root, f'{dataset_name}_training_time'))

    # Heatmap of normalized metrics
    plots['heatmap_normalized'] = _heatmap_normalized(normalized, os.path.join(out_root, f'{dataset_name}_heatmap_normalized'), title=f'{dataset_name}: Normalized Metrics Heatmap')

    logger.info(f"Generated {len(plots)} plots for dataset '{dataset_name}' in {out_root}")
    return plots


def generate_plots_for_multiple_datasets(all_results: Dict[str, Dict[str, Dict[str, Any]]], out_dir: str = 'plots'):
    """Generate plots for multiple datasets.

    all_results: mapping dataset_name -> (mapping model_name -> summary)
    """
    all_paths = {}
    for dataset, models in all_results.items():
        paths = generate_plots_for_dataset(dataset, models, out_dir=out_dir)
        all_paths[dataset] = paths
    return all_paths
