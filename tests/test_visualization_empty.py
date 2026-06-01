import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.visualization import generate_plots_for_dataset


def test_generate_plots_with_empty_input():
    plots = generate_plots_for_dataset('empty_demo', {}, out_dir='results')
    assert plots == {}


if __name__ == '__main__':
    test_generate_plots_with_empty_input()
    print('Visualization empty-input test passed')
