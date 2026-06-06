# Repository Cleanup & Organization Report

This report documents the organization and relocation of repository planning, audit, remediation, and experimental artifacts into a structured `project_artifacts/` directory.

---

## 1. Moved Files & Artifacts

The following files were relocated from the repository root to prevent cluttering:

### Audits (`project_artifacts/audits/`)
* `benchmark_revalidation_manifest.csv`
* `benchmark_revalidation_manifest.md`
* `CBS_CROSS_DOMAIN_VALIDATION.md`
* `CBS_VALIDATION_REPORT.md`
* `MATRIX_ORIENTATION_AUDIT.md`
* `VALIDATION_AUDIT_REPORT.md`
* `dataset_audit_report.md`
* `dataset_audit.json`
* `duplicate_audit_report.md`
* `group_leakage_validation_report.md`
* `group_leakage_validation.json`
* `identifier_report.md`
* `financial_runtime_profile.md`
* `financial_optimization_report.md`
* `benchmark_resume_report.md`
* `checkpoint_integrity_report.md`
* `FRAMEWORK_STABILIZATION_REPORT.md`
* `PROJECT_READINESS_REPORT.md`
* `PHASE6_READINESS.md`
* `NEMENYI_VALIDATION.md`

### Remediation (`project_artifacts/remediation/`)
* `PHASE1_REMEDIATION_REPORT.md`
* `PHASE2_REMEDIATION_REPORT.md`
* `PHASE3_REMEDIATION_REPORT.md`
* `PHASE4_REMEDIATION_REPORT.md`
* `SMOKE_TEST_CI_FIX_REPORT.md`
* `financial_remediation_report.md`
* `heart_remediation_report.md`
* `remediation_verification_report.md`

### Planning (`project_artifacts/planning/`)
* `DELIVERABLES_CHECKLIST.md`
* `IMPLEMENTATION_SUMMARY.md`
* `benchmark_checkpoint_design.md`

### Experimental Tests (`project_artifacts/experimental_tests/`)
* `verify_resume.py` (copied from `scratch/`)
* `audit_checkpoint_integrity.py` (copied from `scratch/`)

---

## 2. Files Intentionally Left in Root

The following files have been intentionally left in the repository root for production execution, installation, and general orientation:

* **Production Code**: `main.py`, `launcher.py`, `runtime_bootstrap.py`
* **Dependency Files**: `requirements.txt`, `requirements-lock.txt`, `.env.example`, `.gitignore`
* **Documentation**: `README.md`, `ARCHITECTURE.md`, `DATA_PIPELINE_GUIDE.md`, `METRICS_ENGINE_GUIDE.md`, `MODELS_ENGINE_GUIDE.md`, `VALIDATION_FRAMEWORK_GUIDE.md`
* **Framework Tests**: `test_metrics_quick.py`

---

## 3. Broken References Fixed

The following internal script references were updated to reflect the new file locations:
* Updated reference `"group_leakage_validation_report.md"` to `"project_artifacts/audits/group_leakage_validation_report.md"` in `scripts/run_stage3_group_audit.py`
* Updated reference `"group_leakage_validation.json"` to `"project_artifacts/audits/group_leakage_validation.json"` in `scripts/run_stage3_group_audit.py`
* Updated reference `group_leakage_validation_report.md` to `project_artifacts/audits/group_leakage_validation_report.md` in `scripts/run_stage3_group_audit.py`
* Updated reference `"financial_remediation_report.md"` to `"project_artifacts/remediation/financial_remediation_report.md"` in `scripts/run_stage2_reports.py`
* Updated reference `"heart_remediation_report.md"` to `"project_artifacts/remediation/heart_remediation_report.md"` in `scripts/run_stage2_reports.py`
* Updated reference `duplicate_audit_report.md` to `project_artifacts/audits/duplicate_audit_report.md` in `scripts/run_stage2_reports.py`
* Updated reference `"dataset_audit_report.md"` to `"project_artifacts/audits/dataset_audit_report.md"` in `scripts/run_stage1.py`
* Updated reference `"identifier_report.md"` to `"project_artifacts/audits/identifier_report.md"` in `scripts/run_stage1.py`
* Updated reference `"dataset_audit_report.md"` to `"project_artifacts/audits/dataset_audit_report.md"` in `scripts/run_dataset_audit.py`
* Updated reference `"dataset_audit.json"` to `"project_artifacts/audits/dataset_audit.json"` in `scripts/run_dataset_audit.py`
* Updated reference `'CBS_CROSS_DOMAIN_VALIDATION.md'` to `'project_artifacts/audits/CBS_CROSS_DOMAIN_VALIDATION.md'` in `scripts/run_cross_domain_validation.py`
* Updated reference `'benchmark_revalidation_manifest.md'` to `'project_artifacts/audits/benchmark_revalidation_manifest.md'` in `scripts/generate_benchmark_manifest.py`
* Updated reference `'benchmark_revalidation_manifest.csv'` to `'project_artifacts/audits/benchmark_revalidation_manifest.csv'` in `scripts/generate_benchmark_manifest.py`

---

## 4. Final Repository Structure

```
Generalised Machine learning Benchmarks/
├── project_artifacts/
│   ├── audits/               # Dataset, matrix, leakage, and validation audits
│   ├── remediation/          # Remediation & stabilization reports
│   ├── planning/             # Design & checklists
│   ├── experimental_tests/    # Verification and resume audit scripts
│   └── cleanup_report.md     # Relocation registry (this file)
├── analysis/                 # Analysis and statistical modules
├── configs/                  # Benchmark configurations
├── datasets/                 # Local CSV cache/loaders
├── docs/                     # Guides and internal documentation
├── experiments/              # Benchmark executors and metadata
├── logs/                     # Application logs
├── metrics/                  # Score calculations and normalization
├── preprocessing/            # Pipeline structures
├── reproducibility/          # Environments, VCS, and report builders
├── results/                  # Execution outputs and logs per dataset
├── scripts/                  # Generation & validation runners
├── tests/                    # Unit & regression tests
├── validation/               # Cross-validation & results storage
├── ARCHITECTURE.md           # System architecture overview
├── README.md                 # Project README
├── main.py                   # Main pipeline orchestrator
└── launcher.py               # Framework CLI launcher
```
