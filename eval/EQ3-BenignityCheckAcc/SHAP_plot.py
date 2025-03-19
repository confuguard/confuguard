import os
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def setup_logger():
    """Configure logging for the analyzer."""
    log_path = Path("./eval/analysis_logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(log_path / "result_analysis.log", rotation="1 day")


def load_results(results_path):
    """
    Load results from CSV file.

    Args:
        results_path: Path to the results CSV file

    Returns:
        pandas DataFrame with the loaded results
    """
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)
    logger.info(f"Loaded results from {results_path} with {len(df)} entries")
    return df


def calculate_metrics(df):
    """
    Calculate classification metrics based on results.

    Args:
        df: DataFrame with results including ground truth

    Returns:
        Dict of metrics and DataFrame with known ground truth values
    """
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

    # Create predicted labels based on the "label" column
    df_known['label_pred'] = df_known['label'].map(lambda x: 1 if x.strip().lower() == 'false_positive' else 0)

    # Initialize metrics with default values
    metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "gt_positives": 0,
        "pred_positives": 0,
        "total_samples": len(df_known)
    }

    # Check if we have any samples to evaluate
    if len(df_known) == 0:
        logger.warning("No samples with known ground truth to evaluate")
        return metrics, df_known

    # Calculate metrics with zero_division parameter
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_known['label_gt'],
        df_known['label_pred'],
        average='binary',
        zero_division=0  # Explicitly handle zero division case
    )

    metrics.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_positives": df_known['label_gt'].sum(),
        "pred_positives": df_known['label_pred'].sum(),
    })

    return metrics, df_known


def plot_category_frequency(df, output_dir):
    """Generate plot for ground truth FP categories frequency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if not df['fp_categories'].empty and df['fp_categories'].notna().any():
        # Extract all categories (they can be comma-separated)
        all_categories = []
        for cats in df['fp_categories'].dropna():
            all_categories.extend([c.strip() for c in cats.split(',')])

        # Count frequencies
        cat_counts = pd.Series(all_categories).value_counts()

        # Plot
        cat_counts.plot(kind='barh')
        plt.title("Frequency of False Positive Categories (Ground Truth)")
        plt.xlabel("Count")
        plt.ylabel("FP Category")
    else:
        plt.text(0.5, 0.5, 'No FP categories data available',
                horizontalalignment='center',
                verticalalignment='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_categories_freq.png')
    plt.close()
    logger.info(f"Saved category frequency plot to {output_dir / 'fp_categories_freq.png'}")


def plot_verifier_metrics(df, output_dir):
    """Generate plot for verifier metrics frequency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_cols = [
        'obvious_not_typosquat', 'is_adversarial_name', 'is_fork',
        'has_distinct_purpose', 'is_test', 'is_known_maintainer',
        'no_readme', 'has_suspicious_intent', 'is_relocated_package',
        'overlapped_maintainers', 'comprehensive_metadata', 'active_development'
    ]

    # Filter to metrics columns that exist in the DataFrame
    metrics_cols = [col for col in metrics_cols if col in df.columns]

    if not metrics_cols:
        logger.warning("No verifier metrics columns found in the DataFrame")
        return

    plt.figure(figsize=(10, 6))
    if not df[metrics_cols].empty:
        # Count True values for each metric
        metrics_counts = df[metrics_cols].sum().sort_values(ascending=False)

        # Plot
        metrics_counts.plot(kind='barh')
        plt.title("Frequency of True Values for Verifier Metrics")
        plt.xlabel("Count (rows where metric=True)")
        plt.ylabel("Verifier Metric")
    else:
        plt.text(0.5, 0.5, 'No metrics data available',
                horizontalalignment='center',
                verticalalignment='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'verifier_metrics_freq.png')
    plt.close()
    logger.info(f"Saved verifier metrics plot to {output_dir / 'verifier_metrics_freq.png'}")


def plot_confusion_matrix(df, output_dir):
    """Generate confusion matrix plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'label_gt' not in df.columns or 'label_pred' not in df.columns or len(df) == 0:
        logger.warning("Required columns for confusion matrix not found or DataFrame is empty")
        # Create an empty plot with a message instead of trying to plot an empty confusion matrix
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No data available for confusion matrix',
                horizontalalignment='center',
                verticalalignment='center')
        plt.title('Confusion Matrix (No Data)')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        logger.info(f"Saved empty confusion matrix plot to {output_dir / 'confusion_matrix.png'}")
        return

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df['label_gt'], df['label_pred'])

    # Create more descriptive labels
    labels = ['True Positive', 'False Positive']

    # Plot confusion matrix with seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_dir / 'confusion_matrix.png'}")


def analyze_category_matches(df, output_dir):
    """
    Analyze matches between ground truth categories and verifier metrics.

    Args:
        df: DataFrame with ground truth categories and verifier metrics
        output_dir: Directory to save analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define mapping with explicit indication of inverse relationships
    fp_category_map = {
        'Distinct purpose': ('has_distinct_purpose', False),  # (metric_name, is_inverse)
        'Fork': ('is_fork', False),
        'Obvious name difference': ('is_adversarial_name', True),  # Inverse
        'Missing README': ('no_readme', True),  # Inverse
        'Well-known maintainer': ('is_known_maintainer', False),
        'Suspicious': ('has_suspicious_intent', False),
        'Experiment/test package': ('is_test', False),
        'Overlapped maintainers': ('is_relocated_package', False),
        'Inactively maintained': ('has_distinct_purpose', True),  # Inverse
    }

    # Filter metrics that exist in the DataFrame
    available_metrics = set(df.columns)
    filtered_map = {k: v for k, v in fp_category_map.items() if v[0] in available_metrics}

    if not filtered_map:
        logger.warning("No matching metrics found for category analysis")
        return

    # Initialize category match tracking
    category_matches = {}
    for category in filtered_map:
        category_matches[category] = {"total": 0, "matched": 0}

    for _, row in df.iterrows():
        if pd.isna(row.get('fp_categories')):
            continue

        row_cats = [c.strip() for c in row['fp_categories'].split(',') if c.strip()]

        for cat in row_cats:
            if cat in filtered_map:
                metric_col, is_inverse = filtered_map[cat]

                # Skip if metric not available for this row
                if metric_col not in row or pd.isna(row[metric_col]):
                    continue

                metric_val = row[metric_col]
                # If inverse relationship, flip the metric value for matching
                matches_category = metric_val if not is_inverse else not metric_val

                category_matches[cat]["total"] += 1
                if matches_category:
                    category_matches[cat]["matched"] += 1

    # Calculate match percentages and prepare for plotting
    categories = []
    match_percentages = []
    match_counts = []
    total_counts = []

    for cat, counts in category_matches.items():
        if counts["total"] > 0:
            categories.append(cat)
            percent = (counts["matched"] / counts["total"]) * 100
            match_percentages.append(percent)
            match_counts.append(counts["matched"])
            total_counts.append(counts["total"])

    # Sort by match percentage
    sorted_indices = np.argsort(match_percentages)
    categories = [categories[i] for i in sorted_indices]
    match_percentages = [match_percentages[i] for i in sorted_indices]
    match_counts = [match_counts[i] for i in sorted_indices]
    total_counts = [total_counts[i] for i in sorted_indices]

    # Plot category match percentages
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(categories))

    plt.barh(y_pos, match_percentages)
    plt.yticks(y_pos, categories)
    plt.xlabel('Match Percentage (%)')
    plt.title('Category to Metric Match Percentage')

    # Add count labels
    for i, (m, t) in enumerate(zip(match_counts, total_counts)):
        plt.text(max(5, match_percentages[i] + 2), i, f"{m}/{t}", va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'category_match_percentage.png')
    plt.close()

    # Save match stats to CSV
    match_df = pd.DataFrame({
        'Category': categories,
        'Total_Count': total_counts,
        'Matched_Count': match_counts,
        'Match_Percentage': match_percentages
    })
    match_df.to_csv(output_dir / 'category_match_stats.csv', index=False)

    # Calculate overall match rate
    total_all = sum(total_counts)
    matched_all = sum(match_counts)
    overall_match_rate = (matched_all / total_all * 100) if total_all > 0 else 0

    logger.info("\n=== Category-to-Metric Matching ===")
    logger.info(f"Total ground-truth FP categories found: {total_all}")
    logger.info(f"Matched (metric=True) for those categories: {matched_all}")
    logger.info(f"Overall Match Rate: {overall_match_rate:.2f}%")

    # Write summary to text file
    with open(output_dir / 'category_match_summary.txt', 'w') as f:
        f.write("=== Category-to-Metric Matching ===\n")
        f.write(f"Total ground-truth FP categories found: {total_all}\n")
        f.write(f"Matched (metric=True) for those categories: {matched_all}\n")
        f.write(f"Overall Match Rate: {overall_match_rate:.2f}%\n\n")
        f.write("Category-wise Match Statistics:\n")
        for i, cat in enumerate(categories):
            f.write(f"{cat}: {match_counts[i]}/{total_counts[i]} = {match_percentages[i]:.2f}%\n")


def calculate_TP_metrics(df):
    """
    Calculate metrics for true positive verification results.
    This function is used when all packages in the dataset are known to be real threats.

    Args:
        df: DataFrame with results

    Returns:
        Dict of metrics and DataFrame with ground truth values
    """
    # Verify that required columns exist
    if 'label' not in df.columns:
        raise KeyError(f"Required column 'label' not found in the results data")

    # Create a copy of the dataframe
    df_known = df.copy()

    # For TP evaluation, all packages are real threats (label_gt=0)
    df_known['label_gt'] = 0

    # Create predicted labels based on the "label" column
    # For TP evaluation, we want to identify real threats (not false positives)
    # So label_pred=1 when the model says it's a false positive, 0 when true positive
    df_known['label_pred'] = df_known['label'].map(lambda x: 1 if x.strip().lower() == 'false_positive' else 0)

    # Initialize metrics with default values
    metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "gt_positives": 0,  # No false positives in ground truth
        "pred_positives": df_known['label_pred'].sum(),  # Count predicted false positives
        "total_samples": len(df_known)
    }

    # Check if we have any samples to evaluate
    if len(df_known) == 0:
        logger.warning("No samples to evaluate")
        return metrics, df_known

    # Calculate metrics with zero_division parameter
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_known['label_gt'],
        df_known['label_pred'],
        average='binary',
        zero_division=0
    )

    metrics.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    return metrics, df_known


def learn_metric_weights(df):
    """
    Learn weights for each verifier metric using logistic regression with cross-validation.

    Args:
        df: DataFrame with ground truth labels ('label_gt') and verifier metrics.
             Expected metrics include:
             'obvious_not_typosquat', 'is_adversarial_name', 'is_fork',
             'has_distinct_purpose', 'is_test', 'is_known_maintainer',
             'no_readme', 'has_suspicious_intent', 'is_relocated_package',
             'overlapped_maintainers', 'comprehensive_metadata', 'active_development'

    Returns:
        weights: dict mapping metric names to learned weight values
        model: trained logistic regression model
        cv_scores: cross-validation scores
    """
    # List of metrics used as features
    feature_cols = ['obvious_not_typosquat', 'is_adversarial_name', 'is_fork',
                    'has_distinct_purpose', 'is_test', 'is_known_maintainer',
                    'no_readme', 'has_suspicious_intent', 'is_relocated_package',
                    'overlapped_maintainers', 'comprehensive_metadata', 'active_development']
    # Keep only the available columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Check if we have enough data
    if len(df) < 10 or 'label_gt' not in df.columns:
        logger.warning("Not enough data for learning metric weights")
        # Return empty results
        return {}, None, {}

    # Convert boolean metrics to integers (if not already numeric)
    X = df[feature_cols].fillna(False).astype(int)
    y = df['label_gt']

    # Check if all samples belong to the same class
    if len(y.unique()) == 1:
        logger.warning(f"All samples belong to the same class ({y.iloc[0]}). Skipping cross-validation.")
        # Create a simple model that always predicts the single class
        class SingleClassModel:
            def __init__(self, class_value):
                self.class_value = class_value
                self.coef_ = np.zeros((1, len(feature_cols)))

            def predict(self, X):
                return np.full(len(X), self.class_value)

            def predict_proba(self, X):
                probs = np.zeros((len(X), 2))
                if self.class_value == 0:
                    probs[:, 0] = 1.0
                else:
                    probs[:, 1] = 1.0
                return probs

            def fit(self, X, y):
                pass

        model = SingleClassModel(y.iloc[0])
        weights = dict(zip(feature_cols, model.coef_[0]))

        # Create dummy CV scores
        cv_scores = {
            'accuracy': {'mean': 1.0, 'std': 0.0, 'values': [1.0]},
            'precision': {'mean': 1.0, 'std': 0.0, 'values': [1.0]},
            'recall': {'mean': 1.0, 'std': 0.0, 'values': [1.0]},
            'f1': {'mean': 1.0, 'std': 0.0, 'values': [1.0]}
        }

        logger.info("Single class detected. Model will always predict this class.")
        logger.info(f"Dummy metrics (all 1.0) created for visualization purposes.")

        return weights, model, cv_scores

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Perform cross-validation
    cv_scores = {}
    if len(df) >= 20:  # Only do cross-validation if we have enough samples
        try:
            # Calculate cross-validation scores for accuracy, precision, recall, and F1
            cv_accuracy = cross_val_score(model, X, y, cv=min(5, len(df) // 4), scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=min(5, len(df) // 4), scoring='precision')
            cv_recall = cross_val_score(model, X, y, cv=min(5, len(df) // 4), scoring='recall')
            cv_f1 = cross_val_score(model, X, y, cv=min(5, len(df) // 4), scoring='f1')

            cv_scores = {
                'accuracy': {
                    'mean': cv_accuracy.mean(),
                    'std': cv_accuracy.std(),
                    'values': cv_accuracy.tolist()
                },
                'precision': {
                    'mean': cv_precision.mean(),
                    'std': cv_precision.std(),
                    'values': cv_precision.tolist()
                },
                'recall': {
                    'mean': cv_recall.mean(),
                    'std': cv_recall.std(),
                    'values': cv_recall.tolist()
                },
                'f1': {
                    'mean': cv_f1.mean(),
                    'std': cv_f1.std(),
                    'values': cv_f1.tolist()
                }
            }

            logger.info(f"Cross-validation results:")
            logger.info(f"  Accuracy: {cv_scores['accuracy']['mean']:.3f} ± {cv_scores['accuracy']['std']:.3f}")
            logger.info(f"  Precision: {cv_scores['precision']['mean']:.3f} ± {cv_scores['precision']['std']:.3f}")
            logger.info(f"  Recall: {cv_scores['recall']['mean']:.3f} ± {cv_scores['recall']['std']:.3f}")
            logger.info(f"  F1: {cv_scores['f1']['mean']:.3f} ± {cv_scores['f1']['std']:.3f}")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            # Create empty CV scores
            cv_scores = {}

    # Train the final model on all data
    try:
        model.fit(X, y)
        weights = dict(zip(X.columns, model.coef_[0]))
        logger.info(f"Learned metric weights: {weights}")
        return weights, model, cv_scores
    except Exception as e:
        logger.warning(f"Model training failed: {str(e)}")
        return {}, None, cv_scores


def plot_learned_weights(weights, cv_scores, df_features, model, output_dir):
    """
    Plot the learned weights as a horizontal bar chart, cross-validation results,
    and compute SHAP value analysis for feature importance.

    Args:
        weights: dict mapping metric names to learned weight values.
        cv_scores: cross-validation scores.
        df_features: DataFrame used to train the model (with feature columns).
        model: trained logistic regression model.
        output_dir: Directory to save the plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set larger font sizes for all plots
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    # Skip if no weights
    if not weights:
        logger.warning("No weights to plot")
        return

    # Define human-readable feature names that match the table
    feature_display_names = {
        "obvious_not_typosquat": "R1: Obviously unconfusing",
        "has_distinct_purpose": "R2: Distinct Purpose",
        "is_fork": "R3: Fork Package",
        "active_development": "R4: Active Development/Maintained",
        "comprehensive_metadata": "R5: Comprehensive Metadata",
        "overlapped_maintainers": "R6: Overlapped Maintainers",
        "is_adversarial_name": "R7: Adversarial Package Name",
        "is_known_maintainer": "R8: Well-known Maintainers",
        "no_readme": "R9: Clear Description",
        "has_suspicious_intent": "R10: Has Malicious Intent",
        "is_test": "R11: Experiment/test package",
        "is_relocated_package": "R12: Package Relocation"

    }

    # Create a mapping from feature name to R number for sorting
    def get_r_number(name):
        if name.startswith('R') and ':' in name:
            try:
                return int(name.split(':')[0][1:])
            except ValueError:
                return 999
        return 999

    # Plot raw learned weights
    plt.figure(figsize=(12, 8))  # Increased figure size
    metrics_list = list(weights.keys())
    w_values = list(weights.values())

    # Create display names for all metrics
    display_names = [feature_display_names.get(metric, metric) for metric in metrics_list]

    # Create a list of (display_name, weight, original_metric) tuples
    combined_data = list(zip(display_names, w_values, metrics_list))

    # Sort strictly by R number first, then by absolute weight value for non-R items
    combined_data.sort(key=lambda x: (get_r_number(x[0]), -abs(x[1])))

    # Unpack the sorted data
    sorted_display_names, sorted_values, sorted_metrics = zip(*combined_data)

    # Reverse the order for horizontal bar chart (so R1 is at the top)
    sorted_display_names = list(reversed(sorted_display_names))
    sorted_values = list(reversed(sorted_values))

    bars = plt.barh(sorted_display_names, sorted_values)
    for i, bar in enumerate(bars):
        if sorted_values[i] < 0:
            bar.set_color('r')
    plt.title("Learned Weights for Verifier Metrics", fontsize=18)
    plt.xlabel("Weight", fontsize=16)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout(pad=2.0)  # Add more padding
    plt.savefig(output_dir / 'learned_weights.png', dpi=300)  # Higher resolution
    plt.close()
    logger.info(f"Saved learned weights plot to {output_dir / 'learned_weights.png'}")

    # Plot cross-validation results if available
    if cv_scores:
        plt.figure(figsize=(10, 8))  # Increased figure size
        cv_metrics = ['accuracy', 'precision', 'recall', 'f1']
        means = [cv_scores[m]['mean'] for m in cv_metrics]
        stds = [cv_scores[m]['std'] for m in cv_metrics]
        bars = plt.bar(cv_metrics, means, yerr=stds, capsize=10)
        plt.ylim(0, 1.1)
        plt.title("Cross-Validation Performance Metrics", fontsize=18)
        plt.ylabel("Score", fontsize=16)
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=14)
        plt.tight_layout(pad=2.0)  # Add more padding
        plt.savefig(output_dir / 'cross_validation_scores.png', dpi=300)  # Higher resolution
        plt.close()
        with open(output_dir / 'cross_validation_scores.txt', 'w') as f:
            f.write("=== Cross-Validation Performance Metrics ===\n")
            for metric in cv_metrics:
                f.write(f"{metric.capitalize()}: {cv_scores[metric]['mean']:.3f} ± {cv_scores[metric]['std']:.3f}\n")
                f.write(f"  Individual fold scores: {cv_scores[metric]['values']}\n")

    # Add SHAP value analysis
    try:
        import shap
        feature_cols = list(weights.keys())
        X = df_features[feature_cols].fillna(False).astype(int)
        if model is not None and hasattr(model, 'predict'):
            explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)

            # For feature importance bar chart, order by R number
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_display_list = [feature_display_names.get(feat, feat) for feat in feature_cols]
            combined_shap_data = list(zip(feature_display_list, mean_abs_shap, feature_cols))
            combined_shap_data.sort(key=lambda x: get_r_number(x[0]))
            sorted_display_features, sorted_shap, _ = zip(*combined_shap_data)
            # Reverse the order so that R1 is at the top
            sorted_display_features = list(reversed(sorted_display_features))
            sorted_shap = list(reversed(sorted_shap))

            plt.figure(figsize=(12, 8))
            plt.barh(sorted_display_features, sorted_shap, color='skyblue')
            plt.xlabel("Mean Absolute SHAP Value", fontsize=16)
            plt.title("Feature Importance based on SHAP values", fontsize=18)
            plt.tight_layout(pad=2.0)
            plt.savefig(output_dir / 'shap_feature_importance.png', dpi=300)
            plt.close()
            logger.info(f"Saved SHAP feature importance plot to {output_dir / 'shap_feature_importance.png'}")

                        # For SHAP summary plot, order features by R number (R1 to R11)
            feature_name_mapping = {feat: feature_display_names.get(feat, feat) for feat in feature_cols}
            sorted_features = sorted(feature_cols,
                                       key=lambda x: get_r_number(feature_name_mapping[x]))
            X_sorted = X[sorted_features]
            shap_values_sorted = np.array([shap_values[:, feature_cols.index(feat)] for feat in sorted_features]).T
            sorted_feature_names = [feature_name_mapping[feat] for feat in sorted_features]

            # Set a larger font size for the SHAP summary plot
            # Update rcParams for larger fonts across titles, labels, and ticks
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 22,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18
            })

            # Create the SHAP summary plot with custom ordering and larger fonts.
            shap.summary_plot(shap_values_sorted, X_sorted,
                              feature_names=sorted_feature_names,
                              show=False, plot_size=(8, 6), sort=False)
            plt.tight_layout(pad=2.0)
            plt.savefig(output_dir / 'shap_summary_plot.pdf', dpi=300)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {output_dir / 'shap_summary_plot.pdf'}")

    except ImportError:
        logger.warning("SHAP package not installed. Skipping SHAP analysis.")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {str(e)}")

    # Reset rcParams to default after we're done
    plt.rcParams.update(plt.rcParamsDefault)


def plot_risk_scores(df, weights, model, output_dir):
    """
    Compute a learned risk score for each sample as a weighted sum of metrics and plot its distribution.

    Args:
        df: DataFrame with verifier metrics.
        weights: dict mapping metric names to learned weights.
        model: Trained logistic regression model.
        output_dir: Directory to save the plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights or model is None:
        logger.warning("No weights or model for risk score calculation")
        return

    feature_cols = list(weights.keys())
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(False).astype(int)
    df['learned_risk_score'] = df[feature_cols].dot(pd.Series(weights))
    X = df[feature_cols]
    try:
        df['probability_score'] = model.predict_proba(X)[:, 1]
    except (IndexError, AttributeError) as e:
        logger.warning(f"Could not compute probability scores: {str(e)}")
        df['probability_score'] = 0.0

    plt.figure(figsize=(10, 6))
    if 'label_gt' in df.columns:
        unique_classes = df['label_gt'].unique()
        if len(unique_classes) == 1:
            class_value = unique_classes[0]
            class_name = "Real Threats" if class_value == 0 else "False Positives"
            plt.hist(df['learned_risk_score'], bins=20, alpha=0.7,
                     color='red' if class_value == 0 else 'blue',
                     label=class_name)
            plt.legend()
        else:
            real_threats = df[df['label_gt'] == 0]['learned_risk_score']
            false_positives = df[df['label_gt'] == 1]['learned_risk_score']
            plt.hist(real_threats, bins=20, alpha=0.7, color='red', label='Real Threats')
            plt.hist(false_positives, bins=20, alpha=0.7, color='blue', label='False Positives')
            plt.legend()
    else:
        plt.hist(df['learned_risk_score'], bins=20)
    plt.title("Distribution of Learned Risk Scores")
    plt.xlabel("Learned Risk Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_risk_scores.png')
    plt.close()
    logger.info(f"Saved learned risk scores plot to {output_dir / 'learned_risk_scores.png'}")

    plt.figure(figsize=(10, 6))
    if 'label_gt' in df.columns:
        unique_classes = df['label_gt'].unique()
        if len(unique_classes) == 1:
            class_value = unique_classes[0]
            class_name = "Real Threats" if class_value == 0 else "False Positives"
            plt.hist(df['probability_score'], bins=20, alpha=0.7,
                     color='red' if class_value == 0 else 'blue',
                     label=class_name)
            plt.legend()
        else:
            real_threats = df[df['label_gt'] == 0]['probability_score']
            false_positives = df[df['label_gt'] == 1]['probability_score']
            plt.hist(real_threats, bins=20, alpha=0.7, color='red', label='Real Threats')
            plt.hist(false_positives, bins=20, alpha=0.7, color='blue', label='False Positives')
            plt.legend()
    else:
        plt.hist(df['probability_score'], bins=20)
    plt.title("Distribution of Probability Scores")
    plt.xlabel("Probability of Being a False Positive")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_scores.png')
    plt.close()
    logger.info(f"Saved probability scores plot to {output_dir / 'probability_scores.png'}")

    if 'label_gt' in df.columns and len(df) > 0:
        from sklearn.metrics import roc_curve, auc
        if len(df['label_gt'].unique()) < 2:
            logger.warning("Cannot create ROC curve: need samples from at least two classes")
        else:
            try:
                fpr, tpr, _ = roc_curve(df['label_gt'], df['probability_score'])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(output_dir / 'roc_curve.png')
                plt.close()
                logger.info(f"Saved ROC curve plot to {output_dir / 'roc_curve.png'}")

                # SHAP force plots for individual samples
                try:
                    import shap
                    if model is not None and hasattr(model, 'predict'):
                        sample_size = min(50, len(df))
                        sample_indices = np.random.choice(len(df), sample_size, replace=False)
                        X_sample = X.iloc[sample_indices]
                        explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
                        shap_values = explainer.shap_values(X_sample)
                        for i in range(min(5, len(X_sample))):
                            plt.figure(figsize=(12, 3))
                            shap.force_plot(explainer.expected_value, shap_values[i],
                                            X_sample.iloc[i], feature_names=feature_cols,
                                            matplotlib=True, show=False)
                            plt.tight_layout()
                            plt.savefig(output_dir / f'shap_force_plot_sample_{i}.png')
                            plt.close()
                        logger.info("Saved SHAP force plots for individual samples")
                except ImportError:
                    logger.warning("SHAP package not installed. Skipping individual SHAP analysis.")
                except Exception as e:
                    logger.warning(f"Individual SHAP analysis failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Failed to create ROC curve: {str(e)}")


def analyze_fp_results(results_path, output_dir=None, is_tp_evaluation=False):
    """
    Analyze false positive verification results and generate visualizations.

    Args:
        results_path: Path to the CSV file with results
        output_dir: Directory to save analysis outputs (default: same directory as results)
        is_tp_evaluation: If True, treat all packages as real threats (TP evaluation)
    """
    setup_logger()

    if output_dir is None:
        output_dir = Path(results_path).parent / "analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = load_results(results_path)

    if is_tp_evaluation:
        metrics, df_known = calculate_TP_metrics(df)
        eval_type = "True-Positive"
    else:
        metrics, df_known = calculate_metrics(df)
        eval_type = "False-Positive"

    logger.info(f"=== {eval_type} Classification Metrics ===")
    logger.info(f"Number of samples with known ground truth: {metrics['total_samples']}")
    logger.info(f"Number of ground truth positives: {metrics['gt_positives']}")
    logger.info(f"Number of predicted positives: {metrics['pred_positives']}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall   : {metrics['recall']:.3f}")
    logger.info(f"F1 score : {metrics['f1']:.3f}")

    with open(Path(output_dir) / "metrics_summary.txt", "w") as f:
        f.write(f"=== {eval_type} Classification Metrics ===\n")
        f.write(f"Number of samples with known ground truth: {metrics['total_samples']}\n")
        f.write(f"Number of ground truth positives: {metrics['gt_positives']}\n")
        f.write(f"Number of predicted positives: {metrics['pred_positives']}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall   : {metrics['recall']:.3f}\n")
        f.write(f"F1 score : {metrics['f1']:.3f}\n")

    plot_category_frequency(df, output_dir)
    plot_verifier_metrics(df, output_dir)
    plot_confusion_matrix(df_known, output_dir)
    analyze_category_matches(df, output_dir)

    # Learn weights and then pass the DataFrame (df_known) for SHAP analysis
    weights, model, cv_scores = learn_metric_weights(df_known)
    plot_learned_weights(weights, cv_scores, df_known, model, output_dir)
    plot_risk_scores(df_known, weights, model, output_dir)

    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return metrics


def analyze_multiple_results(result_folders, output_dir=None, tp_evaluation_folders=None):
    """
    Compare results from multiple evaluation runs.

    Args:
        result_folders: List of paths to result folders for FP evaluation
        output_dir: Directory to save the comparison
        tp_evaluation_folders: List of paths to result folders for TP evaluation
    """
    setup_logger()

    if output_dir is None:
        output_dir = Path("./eval/result_comparison")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = []

    if result_folders:
        for folder in result_folders:
            folder_path = Path(folder)
            result_files = list(folder_path.glob("fp_evaluation_results.csv"))
            if not result_files:
                logger.warning(f"No results found in {folder}")
                continue
            result_file = result_files[0]
            folder_name = folder_path.name
            folder_metrics = analyze_fp_results(result_file, folder_path / "analysis", is_tp_evaluation=False)
            folder_metrics["folder"] = folder_name
            folder_metrics["evaluation_type"] = "FP"
            all_metrics.append(folder_metrics)

    if tp_evaluation_folders:
        for folder in tp_evaluation_folders:
            folder_path = Path(folder)
            result_files = list(folder_path.glob("fp_evaluation_results.csv"))
            if not result_files:
                logger.warning(f"No results found in {folder}")
                continue
            result_file = result_files[0]
            folder_name = folder_path.name
            folder_metrics = analyze_fp_results(result_file, folder_path / "analysis", is_tp_evaluation=True)
            folder_metrics["folder"] = folder_name
            folder_metrics["evaluation_type"] = "TP"
            all_metrics.append(folder_metrics)

    if not all_metrics:
        logger.warning("No metrics collected from the provided folders")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[["folder", "evaluation_type", "precision", "recall", "f1",
                           "gt_positives", "pred_positives", "total_samples"]]
    comparison_file = Path(output_dir) / "metrics_comparison.csv"
    metrics_df.to_csv(comparison_file, index=False)

    plt.figure(figsize=(12, 6))
    metrics_df['combined_label'] = metrics_df['folder'] + ' (' + metrics_df['evaluation_type'] + ')'
    fp_metrics = metrics_df[metrics_df['evaluation_type'] == 'FP']
    tp_metrics = metrics_df[metrics_df['evaluation_type'] == 'TP']
    bar_width = 0.35
    index = np.arange(len(metrics_df['combined_label'].unique()))

    if not fp_metrics.empty:
        plt.bar(index - bar_width/2, fp_metrics['precision'], bar_width, label='Precision (FP)', color='blue')
        plt.bar(index - bar_width/2, fp_metrics['recall'], bar_width, bottom=fp_metrics['precision'],
                label='Recall (FP)', color='lightblue')
        plt.bar(index - bar_width/2, fp_metrics['f1'], bar_width,
                bottom=fp_metrics['precision'] + fp_metrics['recall'], label='F1 (FP)', color='darkblue')
    if not tp_metrics.empty:
        plt.bar(index + bar_width/2, tp_metrics['precision'], bar_width, label='Precision (TP)', color='red')
        plt.bar(index + bar_width/2, tp_metrics['recall'], bar_width, bottom=tp_metrics['precision'],
                label='Recall (TP)', color='lightcoral')
        plt.bar(index + bar_width/2, tp_metrics['f1'], bar_width,
                bottom=tp_metrics['precision'] + tp_metrics['recall'], label='F1 (TP)', color='darkred')

    plt.title("Comparison of Metrics Across Runs")
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.xticks(index, metrics_df['combined_label'].unique(), rotation=45, ha='right')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "metrics_comparison.png")
    plt.close()

    logger.info(f"Multi-run comparison complete. Results saved to {output_dir}")


def merge_and_analyze_datasets(fp_result_paths, tp_result_paths, output_dir=None):
    """
    Merge datasets from FP and TP evaluations and perform combined analysis.

    Args:
        fp_result_paths: List of paths to FP evaluation result CSV files
        tp_result_paths: List of paths to TP evaluation result CSV files
        output_dir: Directory to save the merged analysis results

    Returns:
        DataFrame with the merged dataset
    """
    setup_logger()

    if output_dir is None:
        output_dir = Path("./merged_analysis")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    all_dfs = []
    fp_dfs = []
    tp_dfs = []

    for result_path in fp_result_paths:
        try:
            df = load_results(result_path)
            df['data_source'] = Path(result_path).parent.name
            df['evaluation_type'] = 'FP'
            if 'is_fp_gt' in df.columns:
                df = df.copy()
                df['is_fp_gt'] = df['is_fp_gt'].astype(str)
                unknown_patterns = ['unk', 'unknown', 'unkonwn']
                mask = ~df['is_fp_gt'].str.lower().str.contains('|'.join(unknown_patterns), na=False)
                df = df[mask]
                df = df[df['is_fp_gt'].str.lower().isin(['yes', 'no'])]
                df['label_gt'] = df['is_fp_gt'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
                df['label_pred'] = df['label'].map(lambda x: 1 if x.strip().lower() == 'false_positive' else 0)
                all_dfs.append(df)
                fp_dfs.append(df)
            else:
                logger.warning(f"No ground truth column found in {result_path}")
        except Exception as e:
            logger.error(f"Error processing {result_path}: {str(e)}")

    for result_path in tp_result_paths:
        try:
            df = load_results(result_path)
            df['data_source'] = Path(result_path).parent.name
            df['evaluation_type'] = 'TP'
            df['label_gt'] = 0
            df['label_pred'] = df['label'].map(lambda x: 1 if x.strip().lower() == 'false_positive' else 0)
            all_dfs.append(df)
            tp_dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing {result_path}: {str(e)}")

    if not all_dfs:
        logger.error("No valid data found in the provided paths")
        return None
    for df in all_dfs:
        print(df.head())

    merged_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Merged dataset contains {len(merged_df)} samples from {len(all_dfs)} sources")
    merged_df.to_csv(Path(output_dir) / "merged_dataset.csv", index=False)
    logger.info(f"Saved merged dataset to {Path(output_dir) / 'merged_dataset.csv'}")

    # Calculate metrics for the entire merged dataset
    calculate_dataset_metrics(merged_df, "Overall", output_dir)

    # Calculate separate metrics for FP and TP datasets
    if fp_dfs:
        fp_merged = pd.concat(fp_dfs, ignore_index=True)
        calculate_dataset_metrics(fp_merged, "FP", output_dir)

    if tp_dfs:
        tp_merged = pd.concat(tp_dfs, ignore_index=True)
        calculate_dataset_metrics(tp_merged, "TP", output_dir)

    plot_confusion_matrix(merged_df, output_dir)
    plot_category_frequency(merged_df, output_dir)
    plot_verifier_metrics(merged_df, output_dir)
    analyze_category_matches(merged_df, output_dir)

    # Learn weights and then pass merged_df for SHAP analysis
    weights, model, cv_scores = learn_metric_weights(merged_df)
    plot_learned_weights(weights, cv_scores, merged_df, model, output_dir)
    plot_risk_scores(merged_df, weights, model, output_dir)

    plt.figure(figsize=(10, 6))
    source_counts = merged_df['data_source'].value_counts()
    source_counts.plot(kind='barh')
    plt.title("Number of Samples by Data Source")
    plt.xlabel("Count")
    plt.ylabel("Data Source")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'data_source_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    eval_counts = merged_df['evaluation_type'].value_counts()
    plt.pie(eval_counts, labels=eval_counts.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title("Distribution of Evaluation Types")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'evaluation_type_distribution.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    metrics_by_source = merged_df.groupby('data_source').apply(
        lambda x: pd.Series({
            'precision': precision_recall_fscore_support(
                x['label_gt'], x['label_pred'], average='binary', zero_division=0
            )[0],
            'recall': precision_recall_fscore_support(
                x['label_gt'], x['label_pred'], average='binary', zero_division=0
            )[1],
            'f1': precision_recall_fscore_support(
                x['label_gt'], x['label_pred'], average='binary', zero_division=0
            )[2],
            'count': len(x)
        })
    )
    ax = metrics_by_source[['precision', 'recall', 'f1']].plot(kind='bar', figsize=(12, 6))
    plt.title("Performance Metrics by Data Source")
    plt.ylabel("Score")
    plt.xlabel("Data Source")
    plt.xticks(rotation=45, ha='right')
    for i, count in enumerate(metrics_by_source['count']):
        plt.text(i, 0.05, f"n={count}", ha='center', va='bottom', color='black')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'metrics_by_source.png')
    plt.close()

    logger.info(f"Merged analysis complete. Results saved to {output_dir}")
    return merged_df


def calculate_dataset_metrics(df, dataset_type, output_dir):
    """
    Calculate classification metrics for a specific dataset.

    Args:
        df: DataFrame with results including ground truth
        dataset_type: String identifier for the dataset (e.g., "Overall", "FP", "TP")
        output_dir: Directory to save the metrics summary
    """
    metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "gt_positives": 0,
        "pred_positives": 0,
        "total_samples": len(df)
    }

    if 'label_gt' in df.columns and 'label_pred' in df.columns:
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['label_gt'],
            df['label_pred'],
            average='binary',
            zero_division=0
        )
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_positives": df['label_gt'].sum(),
            "pred_positives": df['label_pred'].sum(),
        })

    # Calculate confusion matrix values
    if 'label_gt' in df.columns and 'label_pred' in df.columns and len(df) > 0:
        cm = confusion_matrix(df['label_gt'], df['label_pred'])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp
            })

    logger.info(f"=== {dataset_type} Dataset Metrics ===")
    logger.info(f"Number of samples: {metrics['total_samples']}")
    logger.info(f"Number of ground truth positives (FP): {metrics['gt_positives']}")
    logger.info(f"Number of predicted positives (FP): {metrics['pred_positives']}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall   : {metrics['recall']:.3f}")
    logger.info(f"F1 score : {metrics['f1']:.3f}")

    if 'true_positives' in metrics:
        logger.info(f"True Positives: {metrics['true_positives']}")
        logger.info(f"False Positives: {metrics['false_positives']}")
        logger.info(f"True Negatives: {metrics['true_negatives']}")
        logger.info(f"False Negatives: {metrics['false_negatives']}")

    with open(Path(output_dir) / f"{dataset_type.lower()}_metrics_summary.txt", "w") as f:
        f.write(f"=== {dataset_type} Dataset Metrics ===\n")
        f.write(f"Number of samples: {metrics['total_samples']}\n")
        f.write(f"Number of ground truth positives (FP): {metrics['gt_positives']}\n")
        f.write(f"Number of predicted positives (FP): {metrics['pred_positives']}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall   : {metrics['recall']:.3f}\n")
        f.write(f"F1 score : {metrics['f1']:.3f}\n")

        if 'true_positives' in metrics:
            f.write(f"\nConfusion Matrix Details:\n")
            f.write(f"True Positives: {metrics['true_positives']}\n")
            f.write(f"False Positives: {metrics['false_positives']}\n")
            f.write(f"True Negatives: {metrics['true_negatives']}\n")
            f.write(f"False Negatives: {metrics['false_negatives']}\n")

    return metrics



if __name__ == "__main__":
    # Example usage:
    fp_result_paths = [
        # "./results/fp_evaluation_20250313_214328/fp_evaluation_results.csv",
        "./results/fp_evaluation_20250315_013958/fp_evaluation_results.csv",
    ]
    tp_result_paths = [
        "./results/fp_evaluation_20250314_013926/fp_evaluation_results.csv",
    ]
    merge_and_analyze_datasets(fp_result_paths, tp_result_paths)
    logger.info("Run this script with specific result paths to perform analysis")
