#!/usr/bin/env python3
# python -m python.typosquat.eval.EQ3-ThreatFeed.analyze_CG_benchmark_metrics

import pandas as pd
import json
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Dict, List
from statistics import mean, median

@dataclass
class BenchmarkMetrics:
    """Class to store benchmark metrics for a tool"""
    total_time: float = 0.0
    total_packages: int = 0
    total_pairs: int = 0
    latencies: List[float] = field(default_factory=list)
    ecosystem_metrics: Dict[str, Dict] = field(default_factory=dict)
    typosquats_identified: int = 0  # Count of packages identified as typosquats
    correct_target_matches: int = 0  # Count of correctly identified legitimate targets

    # Confusion matrix metrics
    true_positives: int = 0  # Correctly identified as typosquats
    false_positives: int = 0  # Incorrectly identified as typosquats
    true_negatives: int = 0  # Correctly identified as not typosquats
    false_negatives: int = 0  # Missed typosquats

    # Subset metrics for benign, stealthy, and active packages
    benign_metrics: Dict = field(default_factory=lambda: {
        "true_positives": 0, "false_positives": 0,
        "true_negatives": 0, "false_negatives": 0
    })
    stealthy_metrics: Dict = field(default_factory=lambda: {
        "true_positives": 0, "false_positives": 0,
        "true_negatives": 0, "false_negatives": 0
    })
    active_metrics: Dict = field(default_factory=lambda: {
        "true_positives": 0, "false_positives": 0,
        "true_negatives": 0, "false_negatives": 0
    })

    @property
    def precision(self) -> float:
        """Calculate overall precision: TP / (TP + FP)"""
        return self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0

    @property
    def recall(self) -> float:
        """Calculate overall recall: TP / (TP + FN)"""
        return self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0

    @property
    def f1_score(self) -> float:
        """Calculate overall F1 score: 2 * (precision * recall) / (precision + recall)"""
        if (self.precision + self.recall) > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return 0

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy: (TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0

    def calculate_subset_metrics(self, subset_name):
        """Calculate precision, recall, F1, and accuracy for a subset"""
        metrics = getattr(self, f"{subset_name}_metrics")

        if (metrics["true_positives"] + metrics["false_positives"]) > 0:
            precision = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"])
        else:
            precision = 0

        if (metrics["true_positives"] + metrics["false_negatives"]) > 0:
            recall = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_negatives"])
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        total = sum(metrics.values())
        accuracy = (metrics["true_positives"] + metrics["true_negatives"]) / total if total > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            **metrics
        }

    def get_metrics_dict(self):
        """Get all metrics in dictionary format"""
        return {
            'overall': {
                'total_packages': self.total_packages,
                'total_pairs': self.total_pairs,
                'typosquats_identified': self.typosquats_identified,
                'correct_target_matches': self.correct_target_matches,
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'accuracy': self.accuracy
            },
            'benign': self.calculate_subset_metrics('benign'),
            'stealthy': self.calculate_subset_metrics('stealthy'),
            'active': self.calculate_subset_metrics('active')
        }

    def print_metrics(self):
        """Print the benchmark metrics"""
        metrics = self.get_metrics_dict()

        print("\n=== Confuguard Metrics ===")
        print(f"Total packages processed: {metrics['overall']['total_packages']}")
        print(f"Total pairs evaluated: {metrics['overall']['total_pairs']}")

        print("\n=== Detection Metrics ===")
        print(f"Packages identified as typosquats: {metrics['overall']['typosquats_identified']}")

        print("\n=== Confusion Matrix ===")
        print(f"True Positives: {metrics['overall']['true_positives']} (correctly identified typosquats)")
        print(f"False Positives: {metrics['overall']['false_positives']} (incorrectly flagged as typosquats)")
        print(f"True Negatives: {metrics['overall']['true_negatives']} (correctly identified non-typosquats)")
        print(f"False Negatives: {metrics['overall']['false_negatives']} (missed typosquats)")

        print("\n=== Performance Metrics ===")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall: {metrics['overall']['recall']:.4f}")
        print(f"F1 Score: {metrics['overall']['f1_score']:.4f}")
        print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")

        for subset in ['benign', 'stealthy', 'active']:
            print(f"\n=== {subset.capitalize()} Subset Metrics ===")
            print(f"True Positives: {metrics[subset]['true_positives']}")
            print(f"False Positives: {metrics[subset]['false_positives']}")
            print(f"True Negatives: {metrics[subset]['true_negatives']}")
            print(f"False Negatives: {metrics[subset]['false_negatives']}")
            print(f"Precision: {metrics[subset]['precision']:.4f}")
            print(f"Recall: {metrics[subset]['recall']:.4f}")
            print(f"F1 Score: {metrics[subset]['f1_score']:.4f}")
            print(f"Accuracy: {metrics[subset]['accuracy']:.4f}")


def analyze_results(results_file_path, ground_truth_file=None):
    """
    Analyze the benchmark results from the CSV file.

    Args:
        results_file_path: Path to the threatfeed_results.csv file.
        ground_truth_file: Optional path to ground truth data.

    Returns:
        BenchmarkMetrics object with calculated metrics.
    """
    # Load the confuguard results
    df = pd.read_csv(results_file_path)

    # Build ground truth mapping if provided.
    # The key is constructed as: "<type>:<name>[::<namespace>]" using ground truth's columns.
    ground_truth_map = {}
    if ground_truth_file:
        ground_truth = pd.read_csv(ground_truth_file)
        for _, row in ground_truth.iterrows():
            # Use 'type' and 'name'; add namespace if present and non-empty.
            key = f"{row['type'].strip().lower()}:{row['name'].strip().lower()}"
            if 'namespace' in row and pd.notna(row['namespace']) and row['namespace'].strip():
                key += f":{row['namespace'].strip().lower()}"
            threat_type = str(row['threat_type']).strip().lower()
            if threat_type == 'false_positive':
                category = 'benign'
            elif threat_type == 'typosquat':
                category = 'stealthy'
            else:
                category = 'active'
            ground_truth_map[key] = {
                'category': category,
                'is_typosquat': category in ['stealthy', 'active']
            }

    # Inverse mapping: output's "ecosystem" value to ground truth "type"
    inverse_ecosystem_map = {
        'ruby': 'gem',
        'npm': 'npm',
        'pypi': 'pypi',
        'maven': 'maven',
        'golang': 'go'
    }

    metrics = BenchmarkMetrics()
    metrics.total_pairs = len(df)

    # Group by package_name to consolidate neighbors for the same package
    package_groups = df.groupby('package_name')
    metrics.total_packages = len(package_groups)

    for package_name, group in package_groups:
        # Determine if this package is flagged as a typosquat:
        # If any neighbor has is_false_positive == False, then the package is flagged.
        is_typosquat = group['is_false_positive'].eq(False).any()

        # Use the output "ecosystem" column (if available) to convert back to ground truth type.
        if 'ecosystem' in group.columns:
            ecosystem = group['ecosystem'].iloc[0].strip().lower()
            package_type = inverse_ecosystem_map.get(ecosystem, ecosystem)
        else:
            package_type = "unknown"

        # Check for namespace in the results if available (output CSV may not have it)
        namespace = group['namespace'].iloc[0].strip().lower() if 'namespace' in group.columns and pd.notna(group['namespace'].iloc[0]) else None

        # Build key for ground truth lookup
        gt_key = f"{package_type}:{package_name.strip().lower()}"
        if namespace:
            gt_key += f":{namespace}"

        if ground_truth_map and gt_key in ground_truth_map:
            true_is_typosquat = ground_truth_map[gt_key]['is_typosquat']
            category = ground_truth_map[gt_key]['category']
        else:
            # Fallback: use fp_category inspection.
            is_benign = any(group['fp_category'].str.contains('benign', case=False, na=False))
            is_stealthy = any(group['fp_category'].str.contains('typosquat', case=False, na=False))
            is_active = any(group['fp_category'].str.contains('others', case=False, na=False))
            if not any([is_benign, is_stealthy, is_active]):
                is_benign = True
            true_is_typosquat = is_stealthy or is_active
            category = 'benign' if is_benign else ('stealthy' if is_stealthy else 'active')

        # Update overall metrics
        if is_typosquat:
            metrics.typosquats_identified += 1
            if true_is_typosquat:  # Correct detection
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1
        else:
            if true_is_typosquat:
                metrics.false_negatives += 1
            else:
                metrics.true_negatives += 1

        # Update subset-specific metrics.
        # According to instructions: treat stealthy and active as positives, benign as negative.
        if category == 'benign':
            if is_typosquat:
                metrics.benign_metrics["false_positives"] += 1
            else:
                metrics.benign_metrics["true_negatives"] += 1
        elif category in ['stealthy', 'active']:
            if is_typosquat:
                if category == 'stealthy':
                    metrics.stealthy_metrics["true_positives"] += 1
                else:
                    metrics.active_metrics["true_positives"] += 1
            else:
                if category == 'stealthy':
                    metrics.stealthy_metrics["false_negatives"] += 1
                else:
                    metrics.active_metrics["false_negatives"] += 1

    return metrics


def main():
    """Main function to run the analysis."""
    results_file = "./eval/EQ3-ThreatFeed/results/20250314_000142/confuguard_evaluation_20250314_000142/threatfeed_results.csv"
    ground_truth_file = "./datasets/NeupaneDB_real_malware/data/threatfeed-data-processed.csv"

    if not Path(results_file).exists():
        print(f"Error: Results file not found at {results_file}")
        sys.exit(1)

    if not Path(ground_truth_file).exists():
        print(f"Warning: Ground truth file not found at {ground_truth_file}")
        print("Proceeding with evaluation using only the results file categories.")
        ground_truth_file = None

    print(f"Analyzing results from: {results_file}")
    if ground_truth_file:
        print(f"Using ground truth data from: {ground_truth_file}")

    metrics = analyze_results(results_file, ground_truth_file)
    metrics.print_metrics()

    output_path = Path(results_file).parent / 'confuguard_metrics_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(metrics.get_metrics_dict(), f, indent=2)

    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
