import os
import json
import logging
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_runtime_and_bottlenecks(rows, features, classes, models, folds):
    total_fits = len(models) * folds
    
    svm_time_per_fit = (rows / 1000) ** 2.5 * (features / 10)
    rf_time_per_fit = (rows / 1000) * (features / 10) * 2
    gb_time_per_fit = (rows / 1000) * (features / 10) * 3
    other_time_per_fit = (rows / 1000) * (features / 10) * 0.5
    
    model_times = []
    has_svm = False
    for m in models:
        if m == 'SVM':
            model_times.append(svm_time_per_fit)
            has_svm = True
        elif m == 'RandomForest':
            model_times.append(rf_time_per_fit)
        elif m == 'GradientBoosting':
            model_times.append(gb_time_per_fit)
        else:
            model_times.append(other_time_per_fit)
            
    total_time_seconds = sum(model_times) * folds
    
    # Classifications
    if total_time_seconds < 60:
        cost_class = "Safe to rerun now"
        expected_runtime = "< 1 min"
    elif total_time_seconds < 600:
        cost_class = "High compute cost"
        expected_runtime = f"~ {int(total_time_seconds // 60)} mins"
    else:
        cost_class = "Very high compute cost"
        expected_runtime = f"~ {int(total_time_seconds // 60)} mins"
        
    bottlenecks = []
    failure_points = []
    if has_svm and rows > 10000:
        bottlenecks.append("SVM bottlenecks (O(N^2) to O(N^3) scaling)")
        failure_points.append("SVM could time out or appear hung")
    if rows * features > 500000:
        bottlenecks.append("Memory bottlenecks (Dataset scale)")
        failure_points.append("OOM during Parallel CV or SVM dense matrix allocation")
    
    if not bottlenecks:
        bottlenecks.append("None")
    if not failure_points:
        failure_points.append("Low risk")
        
    return {
        'Expected Runtime': expected_runtime,
        'Cost Classification': cost_class,
        'SVM Bottlenecks': has_svm and rows > 10000,
        'Bottlenecks': "; ".join(bottlenecks),
        'Failure Points': "; ".join(failure_points),
        'Raw Score': total_time_seconds
    }


def main():
    logger.info("Generating benchmark revalidation manifest...")
    
    results_dir = PROJECT_ROOT / 'results'
    datasets_info = []
    
    for item in results_dir.iterdir():
        if item.is_dir() and (item / 'metadata.json').exists():
            ds_name = item.name
            logger.info(f"Processing {ds_name}...")
            try:
                with open(item / 'metadata.json', 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                rows = meta.get('dataset_size', 0)
                features = meta.get('feature_count', 0)
                classes = len(meta.get('class_distribution', {}))
                
                models = meta.get('model_names', [])
                folds = meta.get('validation_strategy', {}).get('n_splits', 5)
                test_size = meta.get('validation_strategy', {}).get('train_test_split', {}).get('test_size', 0.3)
                fits = len(models) * folds
                
                estimates = estimate_runtime_and_bottlenecks(rows, features, classes, models, folds)
                
                datasets_info.append({
                    'Dataset': ds_name,
                    'Models': ", ".join(models),
                    'Config': f"test_size={test_size}, cv_folds={folds}, prep=standard+onehot",
                    'Rows': rows,
                    'Features': features,
                    'Classes': classes,
                    'CV Folds': folds,
                    'Model Fits': fits,
                    'Expected Runtime': estimates['Expected Runtime'],
                    'Cost Classification': estimates['Cost Classification'],
                    'Bottlenecks': estimates['Bottlenecks'],
                    'Failure Points': estimates['Failure Points'],
                    'Raw Score': estimates['Raw Score']
                })
            except Exception as e:
                logger.error(f"Failed to process {ds_name}: {e}")
                
    if not datasets_info:
        logger.error("No dataset metadata found!")
        return

    # Sort by Raw Score for execution order recommendation
    datasets_info.sort(key=lambda x: x['Raw Score'] if isinstance(x['Raw Score'], (int, float)) else 999999)
    
    # Save CSV
    df_manifest = pd.DataFrame(datasets_info)
    cols_to_drop = ['Raw Score']
    df_manifest_out = df_manifest.drop(columns=cols_to_drop)
    csv_path = PROJECT_ROOT / 'project_artifacts/audits/benchmark_revalidation_manifest.csv'
    df_manifest_out.to_csv(csv_path, index=False)
    
    # Save MD
    md_path = PROJECT_ROOT / 'project_artifacts/audits/benchmark_revalidation_manifest.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Revalidation Manifest\n\n")
        
        f.write("## 1. Execution Order Recommendation (Safest to Riskiest)\n\n")
        f.write("We recommend the following execution order to minimize risk and avoid blocking on slow datasets:\n\n")
        for i, row in enumerate(datasets_info, 1):
            f.write(f"{i}. **{row['Dataset']}** ({row['Cost Classification']} — {row['Expected Runtime']})\n")
        f.write("\n")
        
        f.write("## 2. Dataset Overview & Estimates\n\n")
        f.write(df_manifest_out[['Dataset', 'Rows', 'Features', 'Classes', 'CV Folds', 'Model Fits', 'Expected Runtime']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Configuration & Models\n\n")
        # Collect all unique models
        all_models = set()
        for d in datasets_info:
            all_models.update([m.strip() for m in d['Models'].split(',')])
            
        for m in sorted(all_models):
            if m:
                f.write(f"- **{m}**\n")
        f.write("\n")
        
        f.write("## 4. Risk Analysis & Classifications\n\n")
        f.write(df_manifest_out[['Dataset', 'Cost Classification', 'Bottlenecks', 'Failure Points']].to_markdown(index=False))
        f.write("\n")

    logger.info("Manifests generated successfully.")

if __name__ == '__main__':
    main()
