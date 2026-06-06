"""
Generate Financial and Heart remediation reports.

This script produces:
- financial_remediation_report.md
- heart_remediation_report.md

It loads each dataset, runs the DatasetSanitizer remediation, and writes
before/after statistics into structured markdown reports.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets.sanitizer import DatasetSanitizer

DATASETS_DIR = PROJECT_ROOT / "datasets"


def generate_financial_report():
    filepath = DATASETS_DIR / "Financial.csv"
    if not filepath.exists():
        logger.warning(f"Financial.csv not found at {filepath}")
        return

    logger.info("Loading Financial dataset for remediation report...")
    df = pd.read_csv(filepath)
    sanitizer = DatasetSanitizer()
    result = sanitizer.remediate_financial(df, target_col="Payment_Behaviour")
    stats = result["stats"]
    clean = result["df"]

    md = []
    md.append("# Financial Dataset Remediation Report\n")
    md.append("This report documents the remediations applied to the Financial dataset (`datasets/Financial.csv`) ")
    md.append("via `DatasetSanitizer.remediate_financial()`.\n")

    md.append("## Before / After Summary\n")
    md.append("| Metric | Before | After |")
    md.append("|---|---|---|")
    md.append(f"| Total rows | {stats['before_rows']:,} | {stats['after_rows']:,} |")
    md.append(f"| Total columns | {stats['before_cols']} | {stats['after_cols']} |")
    md.append(f"| Target classes | {stats['before_target_classes']} | {stats['after_target_classes']} |")
    md.append("")

    md.append("## Remediations Applied\n")

    md.append("### 1. Corrupted Target Label Removal\n")
    md.append(f"- **Corrupted label**: `!@9#%8`")
    md.append(f"- **Rows dropped**: {stats['corrupted_target_rows_dropped']:,}")
    md.append(f"- **Remaining target classes**: {stats['after_target_classes']}")
    md.append("")

    md.append("### 2. Identifier Columns Dropped\n")
    md.append("The following identifier columns were removed from the feature set to prevent target leakage:\n")
    for col in stats["identifier_columns_dropped"]:
        md.append(f"- `{col}`")
    md.append("")

    md.append("### 3. Placeholder Normalization\n")
    md.append("The following dirty placeholder strings were converted to `NaN`:\n")
    md.append("- `_______`, `---`, `#F%$D@*&8`, `_`, `nan`, `NAN`, `NaN`")
    md.append("- Regex patterns: `__<value>__`, `**<value>**`")
    md.append("")

    md.append("### 4. Numeric Column Coercion\n")
    md.append("The following columns were coerced from string/object dtype to numeric after cleaning trailing underscores:\n")
    for col in stats["numeric_columns_coerced"]:
        md.append(f"- `{col}`")
    md.append("")

    md.append("## Post-Remediation Target Distribution\n")
    md.append("| Class | Count |")
    md.append("|---|---|")
    for cls, count in clean["Payment_Behaviour"].value_counts().items():
        md.append(f"| {cls} | {count:,} |")
    md.append("")

    md.append("## Integration\n")
    md.append("These remediations are automatically applied when loading the Financial dataset via:")
    md.append("```python")
    md.append('X, y, metadata = load_dataset("datasets/Financial.csv", target_column="Payment_Behaviour")')
    md.append("```")
    md.append("The logic is implemented in `DatasetSanitizer.remediate_financial()` and invoked by `DatasetLoader.load_csv()`.")
    md.append("")

    report_path = PROJECT_ROOT / "financial_remediation_report.md"
    report_path.write_text("\n".join(md), encoding="utf8")
    logger.info(f"Saved {report_path}")


def generate_heart_report():
    filepath = DATASETS_DIR / "heart.csv"
    if not filepath.exists():
        logger.warning(f"heart.csv not found at {filepath}")
        return

    logger.info("Loading Heart dataset for remediation report...")
    df = pd.read_csv(filepath)
    sanitizer = DatasetSanitizer()
    result = sanitizer.remediate_heart(df, target_col="target")
    stats = result["stats"]

    md = []
    md.append("# Heart Dataset Remediation Report\n")
    md.append("This report documents the deduplication remediation applied to the Heart dataset (`datasets/heart.csv`) ")
    md.append("via `DatasetSanitizer.remediate_heart()`.\n")

    md.append("## Before / After Summary\n")
    md.append("| Metric | Before | After |")
    md.append("|---|---|---|")
    md.append(f"| Total rows | {stats['before_rows']:,} | {stats['after_rows']} |")
    md.append(f"| Unique rows | {stats['before_unique_rows']} | {stats['after_rows']} |")
    md.append(f"| Duplicate rows | {stats['duplicate_rows_dropped']} | 0 |")
    md.append(f"| Duplication rate | {stats['duplicate_percentage']:.2f}% | 0.00% |")
    md.append("")

    md.append("## Remediation Applied\n")
    md.append("### Row Deduplication\n")
    md.append(f"- **Total rows before**: {stats['before_rows']:,}")
    md.append(f"- **Exact duplicate rows detected**: {stats['duplicate_rows_dropped']} ({stats['duplicate_percentage']:.2f}%)")
    md.append(f"- **Unique rows retained**: {stats['after_rows']}")
    md.append(f"- The original Cleveland heart disease dataset contains 303 records; this dataset had 302 unique rows ")
    md.append(f"  (one record from the original may have been dropped during prior curation).")
    md.append("")

    md.append("## Class Distribution\n")
    md.append("### Before Deduplication\n")
    md.append("| Target Class | Count |")
    md.append("|---|---|")
    for cls, count in sorted(stats["before_class_distribution"].items()):
        md.append(f"| {cls} | {count:,} |")
    md.append("")

    md.append("### After Deduplication\n")
    md.append("| Target Class | Count |")
    md.append("|---|---|")
    for cls, count in sorted(stats["after_class_distribution"].items()):
        md.append(f"| {cls} | {count} |")
    md.append("")

    md.append("## Cross-Validation Leakage Impact\n")
    md.append("Prior to deduplication, a standard 5-fold CV on the duplicated dataset showed **~97.56% fold contamination** ")
    md.append("(see `duplicate_audit_report.md`). After deduplication, each fold will contain genuinely unseen samples, ")
    md.append("producing valid generalization estimates.\n")

    md.append("## Integration\n")
    md.append("Deduplication is automatically applied when loading the Heart dataset via:")
    md.append("```python")
    md.append('X, y, metadata = load_dataset("datasets/heart.csv", target_column="target")')
    md.append("```")
    md.append("The logic is implemented in `DatasetSanitizer.remediate_heart()` and invoked by `DatasetLoader.load_csv()`.")
    md.append("")

    report_path = PROJECT_ROOT / "heart_remediation_report.md"
    report_path.write_text("\n".join(md), encoding="utf8")
    logger.info(f"Saved {report_path}")


if __name__ == "__main__":
    generate_financial_report()
    generate_heart_report()
    logger.info("All remediation reports generated.")
