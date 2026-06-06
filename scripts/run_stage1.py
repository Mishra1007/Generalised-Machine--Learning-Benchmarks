"""
Script to execute Stage 1 Auditing.

Registers all standard datasets, runs the quality/leakage audit and identifier
detection, and saves the results in:
- dataset_audit.json
- dataset_audit_report.md
- identifier_report.md
"""

import logging
import json
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import register_dataset, DatasetAudit, IdentifierDetection

DATASETS_DIR = PROJECT_ROOT / "datasets"


def main():
    logger.info("Registering all benchmark datasets...")
    
    # Register the standard datasets in the global registry
    datasets_to_register = {
        'Edu-Primary': ('Edu-Primary.csv', 'G3'),
        'Edu-xApi': ('Edu-xApi.csv', 'Class'),
        'Financial': ('Financial.csv', 'Payment_Behaviour'),
        'german_credit_data': ('german_credit_data.csv', 'kredit'),
        'diabetes': ('diabetes.csv', 'Outcome'),
        'heart': ('heart.csv', 'target'),
    }

    for name, (filename, target_col) in datasets_to_register.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            logger.warning(f"File not found for registration: {filepath}")
            continue
        logger.info(f"Registering {name} (target={target_col})")
        register_dataset(
            name=name,
            filepath=str(filepath),
            target_column=target_col,
            description=f"{name} benchmark dataset"
        )

    logger.info("Running Dataset Integrity and Leakage Audit...")
    audit = DatasetAudit()
    audit_results = audit.audit_all_registered()

    # 1. Generate and save dataset_audit.json
    json_path = PROJECT_ROOT / "dataset_audit.json"
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Saved audit JSON to {json_path}")

    # 2. Generate and save dataset_audit_report.md
    report_md = audit.generate_markdown_report(audit_results)
    report_path = PROJECT_ROOT / "project_artifacts/audits/dataset_audit_report.md"
    with open(report_path, "w", encoding="utf8") as f:
        f.write(report_md)
    logger.info(f"Saved audit Markdown report to {report_path}")

    # 3. Generate and save identifier_report.md
    detector = IdentifierDetection()
    id_report_md = detector.generate_identifier_report(audit_results)
    id_report_path = PROJECT_ROOT / "project_artifacts/audits/identifier_report.md"
    with open(id_report_path, "w", encoding="utf8") as f:
        f.write(id_report_md)
    logger.info(f"Saved identifier report to {id_report_path}")

    logger.info("Stage 1 Audit complete successfully.")


if __name__ == "__main__":
    main()
