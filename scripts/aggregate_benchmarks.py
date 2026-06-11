import argparse
import sys
from pathlib import Path

import pandas as pd

from analysis.reports import write_significance_artifacts
from analysis.significance import global_significance_analysis

def build_multidataset_matrix(results_dir: Path, primary_metric: str = 'test_score') -> pd.DataFrame:
    dfs = []
    
    # 1. Iterate over single-dataset outputs
    for dataset_path in results_dir.iterdir():
        if not dataset_path.is_dir():
            continue
            
        raw_csv = dataset_path / "raw_results.csv"
        if not raw_csv.exists():
            continue
            
        # 2. Extract fold-level data and average it to a dataset-level scalar
        try:
            df = pd.read_csv(raw_csv)
            if primary_metric not in df.columns:
                print(f"Skipping {dataset_path.name}: primary_metric '{primary_metric}' not found in raw_results.csv")
                continue
            
            mean_scores = df.groupby('model')[primary_metric].mean().reset_index()
            mean_scores['dataset'] = dataset_path.name
            dfs.append(mean_scores)
        except Exception as e:
            print(f"Error reading {raw_csv}: {e}")
            continue
            
    if not dfs:
        print(f"No valid datasets found in {results_dir} containing metric {primary_metric}.")
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # 3. Pivot to strict Model vs Dataset matrix
    matrix = combined.pivot(index='dataset', columns='model', values=primary_metric)
    
    # 4. Strict Intersection (Drop models with missing dataset results)
    dropped_models = matrix.columns[matrix.isna().any()].tolist()
    if dropped_models:
        print(f"WARNING: Dropping models due to missing dataset scores (Strict Intersection): {dropped_models}")
        
    matrix = matrix.dropna(axis=1, how='any')
    
    return matrix

def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple single-dataset benchmarks into a global significance report.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing subdirectories of single-dataset benchmark outputs."
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="test_score",
        help="The metric column in raw_results.csv to aggregate on (default: test_score)."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save the aggregated artifacts. Defaults to <results_dir>/aggregated_report."
    )
    
    args = parser.parse_args()
    results_dir = args.results_dir
    
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Results directory {results_dir} does not exist.")
        sys.exit(1)
        
    out_dir = args.out_dir if args.out_dir else results_dir / "aggregated_report"
    
    print(f"Aggregating datasets from {results_dir} using metric '{args.primary_metric}'...")
    matrix = build_multidataset_matrix(results_dir, args.primary_metric)
    
    if matrix.empty:
        print("Aggregation failed. No matrix could be built.")
        sys.exit(1)
        
    print(f"Successfully built matrix with {len(matrix.index)} datasets and {len(matrix.columns)} models.")
    
    try:
        analysis = global_significance_analysis(matrix.values, model_names=matrix.columns.tolist())
        artifacts = write_significance_artifacts(analysis, out_dir=out_dir)
        print(f"Aggregated reports successfully generated at: {out_dir}")
        for k, v in artifacts.items():
            print(f"  - {k}: {v}")
    except Exception as e:
        print(f"Error generating significance report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
