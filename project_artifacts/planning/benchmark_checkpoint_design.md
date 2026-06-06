# Benchmark Checkpoint Design

This document details the architectural design and implementation of the fault-tolerant checkpointing mechanism for the ML benchmarking framework.

---

## 1. Architecture Overview

To support resilience and prevent loss of compute during long-running benchmarks, we implemented a dual-granularity checkpointing system:

1. **Model-Level Checkpoint**: Saved immediately after a model finishes its entire cross-validation suite (5 folds × 3 repetitions).
2. **Fold-Level Checkpoint**: Saved immediately after each individual cross-validation fold completes.

Checkpoint data is written atomically using a tempfile-replace pattern to prevent corruption from partial writes if an interruption occurs mid-checkpointing.

---

## 2. Checkpoint State Schema

The checkpoint is saved as a structured JSON file at `results/<dataset_name>/checkpoint.json`. The schema is defined as:

```json
{
  "experiment_id": "string (UUID)",
  "run_id": "string (UUID)",
  "completed_models": {
    "ModelName": {
      "model_name": "string",
      "dataset_name": "string",
      "random_state": "int",
      "fold_results": [
        {
          "repetition_id": "int",
          "fold_id": "int",
          "metrics": {"accuracy": "float", "f1": "float", ...},
          "train_size": "int",
          "test_size": "int",
          "train_time": "float",
          "eval_time": "float",
          "timestamp": "string (ISO)",
          "y_test": "list or null",
          "y_pred": "list or null",
          "y_pred_proba": "list or null",
          "test_indices": "list or null"
        }
      ]
    }
  },
  "in_progress_model": "string or null",
  "in_progress_folds": [
    {
      "repetition_id": "int",
      "fold_id": "int",
      ...
    }
  ]
}
```

---

## 3. Core Component Integrations

### 1. `CrossValidator` (`validation/cross_validator.py`)
* **Signature Expansion**: Added `on_fold_complete` callback and `results` pre-population parameters to `validate()`.
* **State Check**: Inside the fold generation loop, `CrossValidator` checks if a matching fold result (same `repetition_id` and `fold_id`) already exists in `results.fold_results`. If found, training and inference are skipped, restoring the state from the checkpoint.
* **Notification**: Calls `on_fold_complete(fold_result)` after each fold (whether newly computed or skipped) to trigger state persistency.

### 2. `ExperimentExecutor` (`experiments/executor.py`)
* **Serialization**: Implemented `_serialize_fold_result`, `_deserialize_fold_result`, `_serialize_validation_results`, and `_deserialize_validation_results` helpers to convert dataclass structures containing numpy arrays to/from JSON-safe structures.
* **Atomicity**: Uses `_atomic_checkpoint_dump` to write the checkpoint file. It creates a temporary file in the target directory, flushes and syncs to disk, and uses `os.replace` to perform an atomic write.
* **Cleanup**: On successful completion of all models and atomic output generation, the executor automatically deletes the `checkpoint.json` file.
