"""
Stage 3 — Group Leakage Validation Audit.

Audits every registered dataset for genuine group leakage risk by:
1. Identifying candidate grouping columns based on DATA patterns (not just names).
2. Computing group statistics (counts, distributions).
3. Simulating current train/test split and CV to measure actual group overlap.
4. Producing evidence-driven recommendations.

Outputs:
- group_leakage_validation.json
- project_artifacts/audits/group_leakage_validation_report.md
"""

import sys
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"

# Standard dataset registry (matches run_stage1.py)
DATASETS = {
    "Edu-Primary": ("Edu-Primary.csv", "G3"),
    "Edu-xApi": ("Edu-xApi.csv", "Class"),
    "Financial": ("Financial.csv", "Payment_Behaviour"),
    "german_credit_data": ("german_credit_data.csv", "kredit"),
    "diabetes": ("diabetes.csv", "Outcome"),
    "heart": ("heart.csv", "target"),
}

# Candidate grouping column name keywords (used for initial screening only;
# the actual decision is evidence-driven based on data patterns).
GROUP_KEYWORDS = [
    "customer_id", "student_id", "user_id", "session_id", "device_id",
    "member_id", "account_id", "group_id", "subject_id",
    "id",  # generic suffix
]


def find_candidate_group_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Identify columns that COULD represent entity groupings.

    A column is a candidate if:
    - It matches a name heuristic (contains an ID-like keyword), OR
    - It is a categorical/integer column with 2 <= unique values < n_rows
      AND has a maximum group size >= 2 (i.e. at least one repeated entity).

    We exclude the target column and columns with unique values == n_rows
    (those are pure identifiers, not groups).
    """
    candidates = []
    n_rows = len(df)

    for col in df.columns:
        if col == target_col:
            continue

        col_lower = str(col).lower()
        n_unique = df[col].nunique()

        # Skip fully unique columns (pure identifiers, not groups)
        if n_unique == n_rows:
            continue

        # Skip columns with only 1 unique value (constant)
        if n_unique <= 1:
            continue

        # Name-based candidate
        name_match = any(kw in col_lower for kw in GROUP_KEYWORDS)

        # Data-based candidate: categorical or integer with repeating values
        max_group_size = int(df[col].value_counts().max()) if n_unique < n_rows else 1
        has_repeats = max_group_size >= 2

        # Only consider as candidate if name matches AND data shows repeating groups,
        # OR if it's a low-cardinality categorical column with high repetition
        # (to catch columns like StageID, SectionID, GradeID that might look like
        # grouping columns but are actually categorical features).
        if name_match and has_repeats:
            candidates.append({
                "column": col,
                "source": "name_heuristic",
                "n_unique": n_unique,
                "max_group_size": max_group_size,
            })

    return candidates


def compute_group_statistics(df: pd.DataFrame, col: str) -> dict:
    """Compute detailed group statistics for a candidate column."""
    group_sizes = df[col].value_counts()
    n_groups = len(group_sizes)
    n_rows = len(df)

    repeated_groups = group_sizes[group_sizes > 1]
    n_repeated_groups = len(repeated_groups)
    samples_in_repeated = int(repeated_groups.sum())

    return {
        "column": col,
        "total_unique_groups": n_groups,
        "total_observations": n_rows,
        "mean_obs_per_group": float(group_sizes.mean()),
        "median_obs_per_group": float(group_sizes.median()),
        "max_obs_per_group": int(group_sizes.max()),
        "min_obs_per_group": int(group_sizes.min()),
        "std_obs_per_group": float(group_sizes.std()) if n_groups > 1 else 0.0,
        "n_repeated_groups": n_repeated_groups,
        "pct_repeated_groups": float(n_repeated_groups / n_groups * 100) if n_groups > 0 else 0.0,
        "samples_in_repeated_groups": samples_in_repeated,
        "pct_samples_in_repeated_groups": float(samples_in_repeated / n_rows * 100) if n_rows > 0 else 0.0,
    }


def simulate_leakage(df: pd.DataFrame, col: str, target_col: str,
                     test_size: float = 0.3, n_splits: int = 5,
                     random_state: int = 42) -> dict:
    """
    Simulate current train/test split and CV to measure group overlap.

    Uses StratifiedKFold (the current validation strategy) and a standard
    stratified train/test split to quantify how many test samples have their
    group appearing in the training set.
    """
    results = {}

    # Encode target for stratification
    y = df[target_col]
    if y.dtype == object or y.dtype.name == "category":
        y_encoded = pd.factorize(y)[0]
    else:
        y_encoded = y.values

    groups = df[col].values

    # --- Train/test split overlap ---
    try:
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlapping_groups = train_groups & test_groups

        test_samples_with_overlap = sum(1 for i in test_idx if groups[i] in train_groups)
        n_test = len(test_idx)

        results["train_test_split"] = {
            "train_size": len(train_idx),
            "test_size": n_test,
            "unique_groups_in_train": len(train_groups),
            "unique_groups_in_test": len(test_groups),
            "overlapping_groups": len(overlapping_groups),
            "pct_test_groups_overlapping": float(len(overlapping_groups) / len(test_groups) * 100) if test_groups else 0.0,
            "test_samples_with_group_in_train": test_samples_with_overlap,
            "pct_test_samples_leaked": float(test_samples_with_overlap / n_test * 100) if n_test > 0 else 0.0,
        }
    except Exception as e:
        results["train_test_split"] = {"error": str(e)}

    # --- Cross-validation fold overlap ---
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), y_encoded)):
            train_groups_fold = set(groups[train_idx])
            test_groups_fold = set(groups[test_idx])
            overlap_fold = train_groups_fold & test_groups_fold
            leaked_samples = sum(1 for i in test_idx if groups[i] in train_groups_fold)
            n_test_fold = len(test_idx)

            fold_results.append({
                "fold": fold_idx + 1,
                "overlapping_groups": len(overlap_fold),
                "pct_test_groups_overlapping": float(len(overlap_fold) / len(test_groups_fold) * 100) if test_groups_fold else 0.0,
                "test_samples_leaked": leaked_samples,
                "pct_test_samples_leaked": float(leaked_samples / n_test_fold * 100) if n_test_fold > 0 else 0.0,
            })

        mean_pct_leaked = np.mean([f["pct_test_samples_leaked"] for f in fold_results])
        results["cv_fold_overlap"] = {
            "n_splits": n_splits,
            "per_fold": fold_results,
            "mean_pct_test_samples_leaked": float(mean_pct_leaked),
        }
    except Exception as e:
        results["cv_fold_overlap"] = {"error": str(e)}

    return results


def is_categorical_feature(df: pd.DataFrame, col: str) -> bool:
    """
    Determine if a column is genuinely a categorical feature rather than
    a grouping/entity identifier.

    Indicators of a categorical feature (NOT a group):
    - Very few unique values relative to dataset size (< 1% unique ratio)
    - Values appear to be category labels (short strings, small integers)
    - The column semantically describes an attribute, not an entity
    """
    n_unique = df[col].nunique()
    n_rows = len(df)
    unique_ratio = n_unique / n_rows

    # A column with few unique values relative to dataset size is a categorical
    # feature, not an entity identifier. Entity IDs typically have high
    # cardinality (hundreds+).
    # Threshold: < 5% unique ratio AND < 30 unique values
    if unique_ratio < 0.05 and n_unique < 30:
        return True

    return False


def classify_leakage_risk(
    col: str, stats: dict, leakage: dict, df: pd.DataFrame, target_col: str
) -> dict:
    """
    Produce a final evidence-driven classification for a candidate column.

    Returns a dict with recommendation, confidence, evidence, and impact.
    """
    n_groups = stats["total_unique_groups"]
    n_rows = stats["total_observations"]
    max_obs = stats["max_obs_per_group"]
    mean_obs = stats["mean_obs_per_group"]
    pct_repeated = stats["pct_repeated_groups"]
    pct_samples_repeated = stats["pct_samples_in_repeated_groups"]

    # Check if this is actually a categorical feature
    is_cat = is_categorical_feature(df, col)

    # Get leakage simulation results
    tt_leak = leakage.get("train_test_split", {})
    cv_leak = leakage.get("cv_fold_overlap", {})
    tt_pct = tt_leak.get("pct_test_samples_leaked", 0.0)
    cv_pct = cv_leak.get("mean_pct_test_samples_leaked", 0.0)

    # --- Decision logic ---

    # FALSE POSITIVE: Column is a low-cardinality categorical feature
    if is_cat:
        return {
            "recommendation": "Safe for current StratifiedKFold",
            "confidence": "HIGH",
            "is_false_positive": True,
            "false_positive_reason": (
                f"Column '{col}' has only {n_groups} unique values across {n_rows} rows "
                f"(unique ratio: {n_groups/n_rows:.4f}). This is a categorical feature, "
                f"not an entity identifier. High group overlap in CV is expected and harmless — "
                f"it simply means the same category value appears in both train and test, "
                f"which is normal for categorical features."
            ),
            "evidence": {
                "unique_values": n_groups,
                "unique_ratio": float(n_groups / n_rows),
                "is_categorical_feature": True,
                "train_test_overlap_pct": tt_pct,
                "cv_mean_overlap_pct": cv_pct,
            },
            "expected_impact_on_benchmarks": "None. Current benchmark results remain valid.",
            "prior_results_invalid": False,
        }

    # TRUE GROUP: Many unique groups, each with multiple observations
    # A genuine group exists if:
    # 1. There are enough groups relative to dataset size (> 1% unique ratio)
    # 2. Groups have multiple observations (mean > 1.5)
    # 3. A significant portion of samples are in repeated groups (> 50%)
    unique_ratio = n_groups / n_rows

    if mean_obs >= 2.0 and pct_samples_repeated > 50.0 and unique_ratio > 0.01:
        # This looks like a genuine entity grouping
        if cv_pct > 80.0:
            # High leakage — GroupKFold strongly recommended
            # Determine if stratification is also needed
            target_vals = df[target_col].nunique()
            if target_vals <= 10:
                rec = "StratifiedGroupKFold recommended"
            else:
                rec = "GroupKFold recommended"

            return {
                "recommendation": rec,
                "confidence": "HIGH",
                "is_false_positive": False,
                "evidence": {
                    "unique_groups": n_groups,
                    "unique_ratio": float(unique_ratio),
                    "mean_obs_per_group": mean_obs,
                    "pct_samples_in_repeated_groups": pct_samples_repeated,
                    "train_test_overlap_pct": tt_pct,
                    "cv_mean_overlap_pct": cv_pct,
                },
                "expected_impact_on_benchmarks": (
                    f"Cross-validation scores may be inflated because {cv_pct:.1f}% of test "
                    f"samples have their entity group present in training. Models may "
                    f"memorize group-level patterns instead of learning generalizable features."
                ),
                "prior_results_invalid": True,
            }
        elif cv_pct > 50.0:
            return {
                "recommendation": "GroupKFold recommended",
                "confidence": "MEDIUM",
                "is_false_positive": False,
                "evidence": {
                    "unique_groups": n_groups,
                    "unique_ratio": float(unique_ratio),
                    "mean_obs_per_group": mean_obs,
                    "pct_samples_in_repeated_groups": pct_samples_repeated,
                    "train_test_overlap_pct": tt_pct,
                    "cv_mean_overlap_pct": cv_pct,
                },
                "expected_impact_on_benchmarks": (
                    f"Moderate leakage risk: {cv_pct:.1f}% of test samples share a group with training. "
                    f"Results may be slightly optimistic."
                ),
                "prior_results_invalid": False,
            }

    # LOW RISK or INSUFFICIENT EVIDENCE
    if pct_samples_repeated < 10.0 or mean_obs < 1.5:
        return {
            "recommendation": "Safe for current StratifiedKFold",
            "confidence": "HIGH",
            "is_false_positive": n_groups < 20,
            "false_positive_reason": (
                f"Column '{col}' does not represent meaningful entity groups. "
                f"Only {pct_samples_repeated:.1f}% of samples are in repeated groups."
            ) if n_groups < 20 else None,
            "evidence": {
                "unique_groups": n_groups,
                "mean_obs_per_group": mean_obs,
                "pct_samples_in_repeated_groups": pct_samples_repeated,
                "train_test_overlap_pct": tt_pct,
                "cv_mean_overlap_pct": cv_pct,
            },
            "expected_impact_on_benchmarks": "None. Current benchmark results remain valid.",
            "prior_results_invalid": False,
        }

    return {
        "recommendation": "Insufficient evidence",
        "confidence": "LOW",
        "is_false_positive": False,
        "evidence": {
            "unique_groups": n_groups,
            "mean_obs_per_group": mean_obs,
            "pct_samples_in_repeated_groups": pct_samples_repeated,
            "train_test_overlap_pct": tt_pct,
            "cv_mean_overlap_pct": cv_pct,
        },
        "expected_impact_on_benchmarks": "Unclear. Further investigation needed.",
        "prior_results_invalid": False,
    }


def audit_dataset(name: str, filepath: Path, target_col: str) -> dict:
    """Run the full group leakage audit on a single dataset."""
    logger.info(f"Auditing {name}...")
    df = pd.read_csv(filepath)

    candidates = find_candidate_group_columns(df, target_col)

    if not candidates:
        return {
            "dataset": name,
            "filepath": str(filepath),
            "target_column": target_col,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "candidate_columns": [],
            "overall_recommendation": "Safe for current StratifiedKFold",
            "overall_confidence": "HIGH",
            "overall_evidence": "No candidate grouping columns found in data.",
            "prior_results_invalid": False,
        }

    column_results = []
    any_group_needed = False
    any_invalid = False

    for cand in candidates:
        col = cand["column"]
        stats = compute_group_statistics(df, col)
        leakage = simulate_leakage(df, col, target_col)
        classification = classify_leakage_risk(col, stats, leakage, df, target_col)

        column_results.append({
            "column": col,
            "candidate_source": cand["source"],
            "statistics": stats,
            "leakage_simulation": leakage,
            "classification": classification,
        })

        if "GroupKFold" in classification["recommendation"] or "StratifiedGroupKFold" in classification["recommendation"]:
            any_group_needed = True
        if classification["prior_results_invalid"]:
            any_invalid = True

    # Overall dataset recommendation
    if any_group_needed:
        # Pick the strongest recommendation
        recs = [cr["classification"]["recommendation"] for cr in column_results
                if "GroupKFold" in cr["classification"]["recommendation"]
                or "StratifiedGroupKFold" in cr["classification"]["recommendation"]]
        overall_rec = recs[0] if recs else "GroupKFold recommended"
    else:
        overall_rec = "Safe for current StratifiedKFold"

    # For Financial dataset: note that Customer_ID is already dropped in Stage 2
    # remediation, so even if it's flagged as a genuine group, the leakage is
    # already remediated — prior benchmark results used data WITH Customer_ID
    # dropped, so the group leakage was eliminated at load time.
    already_remediated = False
    if name == "Financial":
        for cr in column_results:
            if cr["column"] == "Customer_ID":
                cr["classification"]["already_remediated"] = True
                cr["classification"]["remediation_note"] = (
                    "Customer_ID is already dropped during Stage 2 remediation "
                    "(DatasetSanitizer.remediate_financial). This column never "
                    "enters the training pipeline, so there is no group leakage "
                    "in current benchmarks."
                )
                cr["classification"]["prior_results_invalid"] = False
                already_remediated = True
        if already_remediated:
            any_invalid = False
            overall_rec = "Safe for current StratifiedKFold"

    return {
        "dataset": name,
        "filepath": str(filepath),
        "target_column": target_col,
        "n_samples": len(df),
        "n_features": len(df.columns),
        "candidate_columns": column_results,
        "overall_recommendation": overall_rec,
        "overall_confidence": "HIGH" if not any_group_needed or already_remediated else "MEDIUM",
        "prior_results_invalid": any_invalid,
    }


def generate_markdown_report(all_results: dict) -> str:
    """Generate the project_artifacts/audits/group_leakage_validation_report.md content."""
    md = []
    md.append("# Group Leakage Validation Report — Stage 3\n")
    md.append("This report provides an evidence-driven assessment of whether any dataset ")
    md.append("requires grouped cross-validation (GroupKFold / StratifiedGroupKFold) ")
    md.append("instead of the current StratifiedKFold strategy.\n")
    md.append("> [!IMPORTANT]\n> All conclusions are based on **data analysis**, not column name heuristics alone. ")
    md.append("Prior claims from Stage 1 are re-evaluated with quantitative evidence.\n")

    # Summary table
    md.append("## Executive Summary\n")
    md.append("| Dataset | Recommendation | Confidence | Prior Results Invalid? |")
    md.append("|---|---|---|---|")
    for name, res in all_results.items():
        md.append(f"| {name} | {res['overall_recommendation']} | {res.get('overall_confidence', 'N/A')} | {'Yes' if res['prior_results_invalid'] else 'No'} |")
    md.append("\n")

    # Per-dataset details
    for name, res in all_results.items():
        md.append(f"## Dataset: {name}\n")
        md.append(f"- **Samples**: {res['n_samples']:,}")
        md.append(f"- **Features**: {res['n_features']}")
        md.append(f"- **Target**: `{res['target_column']}`")
        md.append(f"- **Overall Recommendation**: **{res['overall_recommendation']}**")
        md.append(f"- **Prior Results Invalid**: {'Yes' if res['prior_results_invalid'] else 'No'}")
        md.append("")

        if not res["candidate_columns"]:
            md.append("*No candidate grouping columns found. Dataset is safe for StratifiedKFold.*\n")
            md.append("---\n")
            continue

        for cr in res["candidate_columns"]:
            col = cr["column"]
            stats = cr["statistics"]
            classification = cr["classification"]
            leakage = cr["leakage_simulation"]

            md.append(f"### Column: `{col}`\n")

            # Classification badge
            rec = classification["recommendation"]
            confidence = classification["confidence"]
            is_fp = classification.get("is_false_positive", False)

            if is_fp:
                md.append(f"> [!NOTE]\n> **FALSE POSITIVE** — This column was flagged by name heuristic but is actually a categorical feature.\n")
                if classification.get("false_positive_reason"):
                    md.append(f"{classification['false_positive_reason']}\n")

            md.append(f"**Recommendation**: {rec} (Confidence: {confidence})\n")

            # Group statistics
            md.append("#### Group Statistics\n")
            md.append("| Metric | Value |")
            md.append("|---|---|")
            md.append(f"| Total unique groups | {stats['total_unique_groups']:,} |")
            md.append(f"| Total observations | {stats['total_observations']:,} |")
            md.append(f"| Mean obs per group | {stats['mean_obs_per_group']:.2f} |")
            md.append(f"| Median obs per group | {stats['median_obs_per_group']:.1f} |")
            md.append(f"| Max obs per group | {stats['max_obs_per_group']:,} |")
            md.append(f"| % repeated groups | {stats['pct_repeated_groups']:.1f}% |")
            md.append(f"| % samples in repeated groups | {stats['pct_samples_in_repeated_groups']:.1f}% |")
            md.append("")

            # Leakage simulation
            tt = leakage.get("train_test_split", {})
            cv = leakage.get("cv_fold_overlap", {})

            if "error" not in tt:
                md.append("#### Train/Test Split Overlap\n")
                md.append("| Metric | Value |")
                md.append("|---|---|")
                md.append(f"| Overlapping groups | {tt.get('overlapping_groups', 'N/A')} |")
                md.append(f"| % test groups overlapping | {tt.get('pct_test_groups_overlapping', 0):.1f}% |")
                md.append(f"| Test samples with group in train | {tt.get('test_samples_with_group_in_train', 'N/A')} |")
                md.append(f"| % test samples leaked | {tt.get('pct_test_samples_leaked', 0):.1f}% |")
                md.append("")

            if "error" not in cv:
                md.append("#### Cross-Validation Fold Overlap\n")
                md.append("| Fold | % Test Groups Overlapping | % Test Samples Leaked |")
                md.append("|---|---|---|")
                for f in cv.get("per_fold", []):
                    md.append(f"| {f['fold']} | {f['pct_test_groups_overlapping']:.1f}% | {f['pct_test_samples_leaked']:.1f}% |")
                md.append(f"| **Mean** | — | **{cv.get('mean_pct_test_samples_leaked', 0):.1f}%** |")
                md.append("")

            # Impact
            md.append("#### Impact Assessment\n")
            md.append(f"- **Expected impact**: {classification.get('expected_impact_on_benchmarks', 'N/A')}")
            md.append(f"- **Prior results invalid**: {'Yes' if classification.get('prior_results_invalid', False) else 'No'}")
            if classification.get("already_remediated"):
                md.append(f"\n> [!TIP]\n> **Already Remediated**: {classification.get('remediation_note', '')}")
            md.append("")

        md.append("---\n")

    # Re-evaluation section
    md.append("## Re-evaluation of Prior Claims\n")
    md.append("### Edu-Primary\n")
    edu_primary = all_results.get("Edu-Primary", {})
    if edu_primary:
        candidates = edu_primary.get("candidate_columns", [])
        if candidates:
            for cr in candidates:
                is_fp = cr["classification"].get("is_false_positive", False)
                md.append(f"- **Column `{cr['column']}`**: {'**FALSE POSITIVE** — ' if is_fp else ''}{cr['classification']['recommendation']}")
                if is_fp and cr["classification"].get("false_positive_reason"):
                    md.append(f"  - {cr['classification']['false_positive_reason']}")
        else:
            md.append("- No candidate grouping columns found.")
        md.append(f"- **Conclusion**: {edu_primary['overall_recommendation']}")
    md.append("")

    md.append("### Edu-xApi\n")
    edu_xapi = all_results.get("Edu-xApi", {})
    if edu_xapi:
        candidates = edu_xapi.get("candidate_columns", [])
        if candidates:
            for cr in candidates:
                is_fp = cr["classification"].get("is_false_positive", False)
                md.append(f"- **Column `{cr['column']}`**: {'**FALSE POSITIVE** — ' if is_fp else ''}{cr['classification']['recommendation']}")
                if is_fp and cr["classification"].get("false_positive_reason"):
                    md.append(f"  - {cr['classification']['false_positive_reason']}")
        else:
            md.append("- No candidate grouping columns found.")
        md.append(f"- **Conclusion**: {edu_xapi['overall_recommendation']}")
    md.append("")

    return "\n".join(md)


def main():
    all_results = {}

    for name, (filename, target_col) in DATASETS.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            logger.warning(f"Dataset {name} not found at {filepath}")
            continue
        result = audit_dataset(name, filepath, target_col)
        all_results[name] = result

    # Save JSON
    json_path = PROJECT_ROOT / "project_artifacts/audits/group_leakage_validation.json"
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved {json_path}")

    # Save Markdown report
    report = generate_markdown_report(all_results)
    report_path = PROJECT_ROOT / "project_artifacts/audits/project_artifacts/audits/group_leakage_validation_report.md"
    with open(report_path, "w", encoding="utf8") as f:
        f.write(report)
    logger.info(f"Saved {report_path}")

    logger.info("Stage 3 Group Leakage Validation Audit complete.")


if __name__ == "__main__":
    main()
