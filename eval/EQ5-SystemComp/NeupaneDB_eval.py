# python -m python.typosquat.eval.EQ3-FPR.FPR

import os
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from python.typosquat.utils import FPVerifier
from python.typosquat.service import DatabaseManager

db_manager = DatabaseManager()


def evaluate_pkg_pairs(input_csv: str, output_dir: str, openai_api_key: str, eval_target: str):
    """
    Evaluate package pairs for false positives and save results to CSV.

    Args:
        input_csv: Path to input CSV containing package pairs to evaluate
        output_dir: Directory to save results
        openai_api_key: OpenAI API key for the verifier
    """
    # Initialize FP verifier
    verifier = FPVerifier(openai_api_key)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"fp_evaluation_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Read input CSV
    df = pd.read_csv(input_csv)

    # Filter for TP cases if evaluating true positives
    if eval_target == 'tp':
        df = df[df['confusion'] == 'TP']

    # Limit to first 5 packages for testing
    # df = df.head(5)
    logger.info(f"Loaded {len(df)} package pairs to evaluate")

    # Initialize results list
    results = []

    # Process each package pair
    for idx, row in df.iterrows():
        try:
            confusion = None  # Initialize confusion with default value

            if eval_target == 'fp':
                adversarial_pkg = row['Adversarial pkg']
                original_pkg = row['Original pkg']
                ecosystem = row['Ecosystem'].lower()
            elif eval_target == 'tp':
                adversarial_pkg = row['typosquat_pkg']
                original_pkg = row['legitimate_pkg']
                ecosystem = row['registry']
                confusion = row['confusion']

            # Skip processing for unknown cases in 'tp' evaluation
            if eval_target == 'tp' and confusion == "UNK":
                logger.debug(f"Skipping UNK confusion case: {adversarial_pkg} -> {original_pkg}")
                continue

            logger.debug(f"Evaluating {adversarial_pkg} -> {original_pkg} in {ecosystem}")

            # Verify if it's a false positive
            # Map ecosystem names to standardized format
            ecosystem = {'pip': 'pypi', 'RubyGems': 'ruby'}.get(ecosystem, ecosystem)
            typo_doc = db_manager.get_pkg_metadata(adversarial_pkg, ecosystem)
            legit_doc = db_manager.get_pkg_metadata(original_pkg, ecosystem)

            if typo_doc is None or legit_doc is None:
                logger.warning(f"No metadata found for {adversarial_pkg} or {original_pkg} in {ecosystem}")
                continue

            is_fp, metrics, explanation, FP_category = verifier.verify(typo_doc, legit_doc, ecosystem)

            # Prepare result row (exclude ground truth columns)
            result = {
                'package_name': adversarial_pkg,
                'neighbor': original_pkg,
                'label': 'false_positive' if is_fp else 'true_positive',
                'similarity': None,  # Add if available in your data
                'typo_category': None,  # Add if available in your data
                'FP_category': FP_category,
                'obvious_not_typosquat': metrics.get('obvious_not_typosquat', None),
                'is_adversarial_name': metrics.get('is_adversarial_name',   None),
                'is_fork': metrics.get('is_fork', None),
                'has_distinct_purpose': metrics.get('has_distinct_purpose', None),
                'is_test': metrics.get('is_test', None),
                'is_known_maintainer': metrics.get('is_known_maintainer', None),
                'no_readme': metrics.get('no_readme', None),
                'has_suspicious_intent': metrics.get('has_suspicious_intent', None),
                'is_relocated_package': metrics.get('is_relocated_package', None),
                'overlapped_maintainers': metrics.get('overlapped_maintainers', None),
                'comprehensive_metadata': metrics.get('comprehensive_metadata', None),
                'active_development': metrics.get('active_development', None),
                'explanation': explanation
            }

            results.append(result)

            # Log progress and save results after each pair
            logger.info(f"Processed {idx + 1}/{len(df)} pairs")
            save_results(results, output_path)

        except Exception as e:
            logger.error(f"Error processing pair {adversarial_pkg} -> {original_pkg}: {str(e)}")

    logger.info(f"Evaluation complete. Results saved to {output_path}")


def save_results(results: List[Dict], output_path: Path):
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to save the results
    """
    # Define column order
    columns = [
        'package_name', 'neighbor', 'label', 'similarity', 'typo_category',
        'is_fp_gt', 'fp_categories',  # Ground truth columns
        'obvious_not_typosquat', 'is_adversarial_name', 'is_fork',
        'has_distinct_purpose', 'is_test', 'is_known_maintainer', 'no_readme',
        'has_suspicious_intent', 'is_relocated_package', 'overlapped_maintainers',
        'comprehensive_metadata', 'active_development', 'explanation'
    ]

    # Load ground truth data - use absolute path from input_csv
    gt_path = Path("./datasets/NeupaneDB_no_malware.csv")
    gt_df = pd.read_csv(gt_path)
    gt_df = gt_df.rename(columns={
        'Adversarial pkg': 'package_name',
        'Original pkg': 'neighbor',
        'is_FP?': 'is_fp_gt',
        'FP Categories (Rules)': 'fp_categories'
    })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Merge with ground truth
    df = pd.merge(
        results_df,
        gt_df[['package_name', 'neighbor', 'is_fp_gt', 'fp_categories']],
        on=['package_name', 'neighbor'],
        how='left'
    )

    # Reorder columns
    df = df[columns]

    # Save to CSV
    output_file = output_path / 'fp_evaluation_results.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results)} results to {output_file}")

def analyze_results(results_csv: str):
    """
    Analyze the false positive verification results against ground truth data.

    Args:
        results_csv: Path to results CSV file from the verifier
    """
    # Load verifier's results
    df = pd.read_csv(results_csv)
    logger.info(f"Results columns: {df.columns.tolist()}")

    # Verify that required columns exist
    required_columns = ['is_fp_gt', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in the results data")

    # Filter out rows with unknown ground truth values
    df_known = df.copy()
    df_known = df_known[df_known['is_fp_gt'].notna()]

    # Convert is_fp_gt to string to avoid TypeError with string operations
    df_known['is_fp_gt'] = df_known['is_fp_gt'].astype(str)

    # Filter out "unknown" values (including misspelled "Unkonwn")
    unknown_patterns = ['unk', 'unknown', 'unkonwn']
    mask = ~df_known['is_fp_gt'].str.lower().str.contains('|'.join(unknown_patterns), na=False)
    df_known = df_known[mask]

    # Further filter to only yes/no values
    df_known = df_known[df_known['is_fp_gt'].str.lower().isin(['yes', 'no'])]

    # Log how many items had unknown ground truth
    unknown_count = len(df) - len(df_known)
    if unknown_count > 0:
        logger.info(f"Excluded {unknown_count} samples with unknown ground truth values")

    # Create ground truth labels
    df_known['label_gt'] = df_known['is_fp_gt'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Create predicted labels
    df_known['label_pred'] = df_known['label'].map(lambda x: 1 if x.strip().lower() == 'false_positive' else 0)

    # Check if we have any samples to evaluate
    if len(df_known) == 0:
        logger.warning("No samples with known ground truth to evaluate")
        return

    # Calculate metrics with zero_division parameter
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_known['label_gt'],
        df_known['label_pred'],
        average='binary',
        zero_division=0  # Explicitly handle zero division case
    )

    logger.info("=== False-Positive Classification Metrics ===")
    logger.info(f"Number of samples with known ground truth: {len(df_known)}")
    logger.info(f"Number of ground truth positives: {df_known['label_gt'].sum()}")
    logger.info(f"Number of predicted positives: {df_known['label_pred'].sum()}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall   : {recall:.3f}")
    logger.info(f"F1 score : {f1:.3f}")

    # Generate plots
    plot_analysis(df_known)

    # Analyze category matches
    analyze_category_matches(df_known)



def plot_analysis(df_merged: pd.DataFrame):
    """Generate analysis plots."""
    # Ground Truth FP Categories frequency
    plt.figure(figsize=(6,4))
    if not df_merged['fp_categories'].empty and df_merged['fp_categories'].notna().any():
        df_merged['fp_categories'].value_counts().plot(kind='bar')
        plt.title("Frequency of FP Categories (Ground Truth)")
        plt.xlabel("FP Category (Rules)")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, 'No FP categories data available',
                horizontalalignment='center',
                verticalalignment='center')
    plt.tight_layout()
    plt.savefig('fp_categories_freq.png')
    plt.close()

    # Verifier metrics frequency
    metrics_cols = [
        'obvious_not_typosquat', 'is_adversarial_name', 'is_fork',
        'has_distinct_purpose', 'is_test', 'is_known_maintainer', 'no_readme',
        'has_suspicious_intent', 'is_relocated_package', 'overlapped_maintainers',
        'comprehensive_metadata', 'active_development', 'is_relocated_package'
    ]

    plt.figure(figsize=(6,4))
    if not df_merged[metrics_cols].empty:
        df_merged[metrics_cols].sum().plot(kind='bar')
        plt.title("Frequency of True Values for Verifier Metrics")
        plt.xlabel("Verifier Metric")
        plt.ylabel("Count (rows where metric=True)")
    else:
        plt.text(0.5, 0.5, 'No metrics data available',
                horizontalalignment='center',
                verticalalignment='center')
    plt.tight_layout()
    plt.savefig('verifier_metrics_freq.png')
    plt.close()

def analyze_category_matches(df_merged: pd.DataFrame):
    """Analyze matches between ground truth categories and verifier metrics."""
    # Define mapping with explicit indication of inverse relationships
    fp_category_map = {
        'Distinct purpose': ('has_distinct_purpose', False),  # (metric_name, is_inverse)
        'Fork': ('is_fork', False),
        'Obvious name difference': ('is_adversarial_name', True),  # Inverse
        'Missing README': ('no_readme', True),  # Inverse
        'Well-known maintainer': ('is_known_maintainer', False),
        'Suspicious': ('has_suspicious_intent', False),
        'Experiment/test package': ('is_test', False),
        'Overlapped maintainers': ('overlapped_maintainers', False),
        'Comprehensive metadata': ('comprehensive_metadata', False),
        'Active development': ('active_development', False),
        'Relocated package': ('is_relocated_package', False)
    }

    matches = []
    for _, row in df_merged.iterrows():
        if pd.isna(row['fp_categories']):
            continue
        row_cats = [c.strip() for c in row['fp_categories'].split(',') if c.strip()]
        cat_to_metric_match = []
        for cat in row_cats:
            if cat in fp_category_map:
                metric_col, is_inverse = fp_category_map[cat]
                metric_val = row.get(metric_col, False)
                # If inverse relationship, flip the metric value for matching
                matches_category = metric_val if not is_inverse else not metric_val
                cat_to_metric_match.append((cat, matches_category))
        matches.append(cat_to_metric_match)

    total_cats = sum(len(m) for m in matches)
    matched_cats = sum(1 for mlist in matches for (_,val) in mlist if val)

    logger.info("\n=== Category-to-Metric Matching ===")
    logger.info(f"Total ground-truth FP categories found: {total_cats}")
    logger.info(f"Matched (metric=True) for those categories: {matched_cats}")
    if total_cats > 0:
        logger.info(f"Match Accuracy: {matched_cats/total_cats:.2f}")

if __name__ == "__main__":
    # Configure logging
    logger.add("./eval/fp_evaluation.log", rotation="1 day")

    # Set your paths and API key
    input_fp_csv = "./datasets/NeupaneDB_no_malware.csv"
    input_tp_csv = "./datasets/NeupaneDB_real_malware.csv"
    output_dir = "./eval/EQ3-FPR/results"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Run evaluation
    evaluate_pkg_pairs(input_fp_csv, output_dir, openai_api_key, 'fp')
    evaluate_pkg_pairs(input_tp_csv, output_dir, openai_api_key, 'tp')
    # Get the latest output path from the results directory
    output_paths = sorted(Path(output_dir).glob("fp_evaluation_*"), reverse=True)
    if not output_paths:
        raise ValueError(f"No evaluation results found in {output_dir}")
    output_path = output_paths[0]

    # # Add analysis after evaluation
    results_file = output_path / 'fp_evaluation_results.csv'
    analyze_results(results_file)
