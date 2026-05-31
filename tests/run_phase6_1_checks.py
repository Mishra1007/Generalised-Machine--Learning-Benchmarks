import sys
sys.path.insert(0, '.')
from tests.validate_phase6_1 import test_metadata_validation_rejects, test_config_snapshot_immutable, test_environment_snapshot_complete, test_deterministic_mode_identical_runs, test_fold_persistence_reconstructs

def main():
    print('metadata_validation=', test_metadata_validation_rejects())
    print('config_snapshot=', test_config_snapshot_immutable())
    print('environment_snapshot=', test_environment_snapshot_complete())
    print('deterministic_runs(3)=', test_deterministic_mode_identical_runs(3))
    print('fold_persistence=', test_fold_persistence_reconstructs())

if __name__ == "__main__":
    main()
