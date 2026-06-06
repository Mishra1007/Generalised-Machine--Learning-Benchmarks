import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define directories to create
dirs_to_create = [
    PROJECT_ROOT / 'project_artifacts',
    PROJECT_ROOT / 'project_artifacts' / 'audits',
    PROJECT_ROOT / 'project_artifacts' / 'remediation',
    PROJECT_ROOT / 'project_artifacts' / 'planning',
    PROJECT_ROOT / 'project_artifacts' / 'experimental_tests'
]

# File move mapping: filename -> destination folder
moves_map = {
    # Audits
    'benchmark_revalidation_manifest.csv': 'audits',
    'benchmark_revalidation_manifest.md': 'audits',
    'CBS_CROSS_DOMAIN_VALIDATION.md': 'audits',
    'CBS_VALIDATION_REPORT.md': 'audits',
    'MATRIX_ORIENTATION_AUDIT.md': 'audits',
    'VALIDATION_AUDIT_REPORT.md': 'audits',
    'dataset_audit_report.md': 'audits',
    'dataset_audit.json': 'audits',
    'duplicate_audit_report.md': 'audits',
    'group_leakage_validation_report.md': 'audits',
    'group_leakage_validation.json': 'audits',
    'identifier_report.md': 'audits',
    'financial_runtime_profile.md': 'audits',
    'financial_optimization_report.md': 'audits',
    'benchmark_resume_report.md': 'audits',
    'checkpoint_integrity_report.md': 'audits',
    'FRAMEWORK_STABILIZATION_REPORT.md': 'audits',
    'PROJECT_READINESS_REPORT.md': 'audits',
    'PHASE6_READINESS.md': 'audits',
    'NEMENYI_VALIDATION.md': 'audits',
    
    # Remediation
    'PHASE1_REMEDIATION_REPORT.md': 'remediation',
    'PHASE2_REMEDIATION_REPORT.md': 'remediation',
    'PHASE3_REMEDIATION_REPORT.md': 'remediation',
    'PHASE4_REMEDIATION_REPORT.md': 'remediation',
    'SMOKE_TEST_CI_FIX_REPORT.md': 'remediation',
    'financial_remediation_report.md': 'remediation',
    'heart_remediation_report.md': 'remediation',
    'remediation_verification_report.md': 'remediation',
    
    # Planning
    'DELIVERABLES_CHECKLIST.md': 'planning',
    'IMPLEMENTATION_SUMMARY.md': 'planning',
    'benchmark_checkpoint_design.md': 'planning'
}

def main():
    print("Creating project_artifacts directories...")
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        
    files_moved = []
    
    # Perform moves
    for filename, dest_dir in moves_map.items():
        src_path = PROJECT_ROOT / filename
        if src_path.exists():
            dest_path = PROJECT_ROOT / 'project_artifacts' / dest_dir / filename
            shutil.move(str(src_path), str(dest_path))
            files_moved.append(f"Moved `{filename}` to `project_artifacts/{dest_dir}/`")
            print(f"  Moved {filename} -> project_artifacts/{dest_dir}/")
        else:
            print(f"  Warning: {filename} does not exist in root, skipping.")
            
    # Move experimental test files from scratch
    scratch_tests = ['verify_resume.py', 'audit_checkpoint_integrity.py']
    for test_file in scratch_tests:
        src_test = PROJECT_ROOT / 'scratch' / test_file
        if src_test.exists():
            dest_test = PROJECT_ROOT / 'project_artifacts' / 'experimental_tests' / test_file
            shutil.copy2(str(src_test), str(dest_test))
            files_moved.append(f"Copied `{test_file}` from `scratch/` to `project_artifacts/experimental_tests/`")
            print(f"  Copied {test_file} from scratch -> project_artifacts/experimental_tests/")

    # Update script references
    updates = [
        # scripts/run_stage3_group_audit.py
        ('scripts/run_stage3_group_audit.py', [
            ('"group_leakage_validation_report.md"', '"project_artifacts/audits/group_leakage_validation_report.md"'),
            ('"group_leakage_validation.json"', '"project_artifacts/audits/group_leakage_validation.json"'),
            ('group_leakage_validation_report.md', 'project_artifacts/audits/group_leakage_validation_report.md')
        ]),
        # scripts/run_stage2_reports.py
        ('scripts/run_stage2_reports.py', [
            ('"financial_remediation_report.md"', '"project_artifacts/remediation/financial_remediation_report.md"'),
            ('"heart_remediation_report.md"', '"project_artifacts/remediation/heart_remediation_report.md"'),
            ('duplicate_audit_report.md', 'project_artifacts/audits/duplicate_audit_report.md')
        ]),
        # scripts/run_stage1.py
        ('scripts/run_stage1.py', [
            ('"dataset_audit_report.md"', '"project_artifacts/audits/dataset_audit_report.md"'),
            ('"identifier_report.md"', '"project_artifacts/audits/identifier_report.md"')
        ]),
        # scripts/run_dataset_audit.py
        ('scripts/run_dataset_audit.py', [
            ('"dataset_audit_report.md"', '"project_artifacts/audits/dataset_audit_report.md"'),
            ('"dataset_audit.json"', '"project_artifacts/audits/dataset_audit.json"')
        ]),
        # scripts/run_cross_domain_validation.py
        ('scripts/run_cross_domain_validation.py', [
            ("'CBS_CROSS_DOMAIN_VALIDATION.md'", "'project_artifacts/audits/CBS_CROSS_DOMAIN_VALIDATION.md'")
        ]),
        # scripts/generate_benchmark_manifest.py
        ('scripts/generate_benchmark_manifest.py', [
            ("'benchmark_revalidation_manifest.md'", "'project_artifacts/audits/benchmark_revalidation_manifest.md'"),
            ("'benchmark_revalidation_manifest.csv'", "'project_artifacts/audits/benchmark_revalidation_manifest.csv'")
        ])
    ]
    
    fixed_references = []
    
    print("\nUpdating script references...")
    for script_rel, replacements in updates:
        script_path = PROJECT_ROOT / script_rel
        if script_path.exists():
            content = script_path.read_text(encoding='utf-8')
            modified = False
            for target, replacement in replacements:
                if target in content:
                    content = content.replace(target, replacement)
                    modified = True
                    fixed_references.append(f"Updated reference `{target}` to `{replacement}` in `{script_rel}`")
                    print(f"  Fixed ref: {target} -> {replacement} in {script_rel}")
            if modified:
                script_path.write_text(content, encoding='utf-8')
        else:
            print(f"  Warning: Script {script_rel} not found.")

    # Generate cleanup report
    cleanup_report_path = PROJECT_ROOT / 'project_artifacts' / 'cleanup_report.md'
    
    report_content = f"""# Repository Cleanup & Organization Report

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
"""
    for ref in fixed_references:
        report_content += f"* {ref}\n"
        
    report_content += """
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
"""
    cleanup_report_path.write_text(report_content, encoding='utf-8')
    print("\nCleanup report written successfully.")

if __name__ == '__main__':
    main()
