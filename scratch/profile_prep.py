import time
from pathlib import Path
import pandas as pd
import numpy as np

# Set up project path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import register_dataset
from preprocessing.data_preparation import prepare_dataset

def main():
    print("Registering Financial dataset...")
    register_dataset('Financial', 'datasets/Financial.csv', 'Payment_Behaviour')
    
    print("Profiling dataset preparation stages...")
    t0 = time.time()
    
    # 1. Step: Load raw CSV
    t_load_start = time.time()
    filepath = PROJECT_ROOT / 'datasets' / 'Financial.csv'
    df = pd.read_csv(filepath)
    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    print(f"  Raw CSV load time: {load_time:.4f}s")
    
    # 2. Step: Remediation / Sanitization
    t_sanitize_start = time.time()
    from datasets.sanitizer import DatasetSanitizer
    sanitizer = DatasetSanitizer()
    result = sanitizer.remediate_financial(df, target_col='Payment_Behaviour')
    df_clean = result['df']
    t_sanitize_end = time.time()
    sanitize_time = t_sanitize_end - t_sanitize_start
    print(f"  Sanitization / Remediation time: {sanitize_time:.4f}s")
    
    # 3. Step: Split and Preprocess
    t_split_prep_start = time.time()
    X_train, X_test, y_train, y_test, meta = prepare_dataset('Financial', test_size=0.3)
    t_split_prep_end = time.time()
    split_prep_time = t_split_prep_end - t_split_prep_start
    print(f"  Split & preprocessing pipeline fit/transform: {split_prep_time:.4f}s")
    
    total_prep_time = time.time() - t0
    print(f"Total Preprocessing stage runtime: {total_prep_time:.4f}s")

if __name__ == '__main__':
    main()
