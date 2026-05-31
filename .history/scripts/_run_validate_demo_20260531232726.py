from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from analysis.cbs_validation import run_validation
art = run_validation('results/demo_dataset_storage', out_dir='results/demo_dataset_storage/cbs_validation')
print('Report generated')
