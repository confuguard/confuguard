#!/usr/bin/env python3
"""
analyze_fpr_metrics.py

This script calculates evaluation metrics from FP evaluation results produced by your tool.
It supports two dataset types:

1. FPR Dataset (using FP_ground_truth.csv):
   - Uses the column "Adversarial pkg" as the candidate package name.
   - Uses "Original pkg" as the legitimate package.
   - Uses "is_FP?" to determine the label:
         "Yes" → benign (negative)
         "No"  → stealthy (positive)

2. Active Dataset (using dataset.csv):
   - Filters out rows where the "confusion" column equals "UNK".
   - Uses the column "typosquat_pkg" as the package name.
   - Uses "legitimate_pkg" as the legitimate package.
   - All remaining packages are active (positive).

All ground truth packages are used for evaluation. However, only candidates for which the tool output exists
(i.e. there is a row for the package) are taken into account.
For each candidate, we check the tool’s predicted label (from the "label" column):
  - We interpret a package as predicted positive if its label (case-insensitive) is not "false_positive".
  - Otherwise, it is predicted negative.

Then the confusion matrix is computed as follows:
  - If ground truth is positive (stealthy/active) and predicted positive → TP,
  - If ground truth is positive and predicted negative → FN,
  - If ground truth is benign and predicted negative → TN,
  - If ground truth is benign and predicted positive → FP.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class BenchmarkMetrics:
    total_time: float = 0.0
    total_packages: int = 0      # total number of ground truth candidates (that appear in the output)
    total_pairs: int = 0         # equals total_packages
    typosquats_identified: int = 0  # number of candidates for which there is a tool output row
    correct_target_matches: int = 0  # (if applicable)

    # Overall confusion matrix counts
    true_positives: int = 0    # ground truth positive and predicted positive
    false_positives: int = 0   # ground truth benign and predicted positive
    true_negatives: int = 0    # ground truth benign and predicted negative
    false_negatives: int = 0   # ground truth positive and predicted negative

    # Subset metrics for benign, stealthy, and active
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
        return (self.true_positives / (self.true_positives + self.false_positives)
                if (self.true_positives + self.false_positives) > 0 else 0)

    @property
    def recall(self) -> float:
        return (self.true_positives / (self.true_positives + self.false_negatives)
                if (self.true_positives + self.false_negatives) > 0 else 0)

    @property
    def f1_score(self) -> float:
        if (self.precision + self.recall) > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return 0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return ((self.true_positives + self.true_negatives) / total if total > 0 else 0)

    def calculate_subset_metrics(self, subset_name: str):
        metrics = getattr(self, f"{subset_name}_metrics")
        precision = (metrics["true_positives"] /
                     (metrics["true_positives"] + metrics["false_positives"])
                     if (metrics["true_positives"] + metrics["false_positives"]) > 0 else 0)
        recall = (metrics["true_positives"] /
                  (metrics["true_positives"] + metrics["false_negatives"])
                  if (metrics["true_positives"] + metrics["false_negatives"]) > 0 else 0)
        f1 = (2 * (precision * recall) / (precision + recall)
              if (precision + recall) > 0 else 0)
        total = sum(metrics.values())
        accuracy = ((metrics["true_positives"] + metrics["true_negatives"]) / total
                    if total > 0 else 0)
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            **metrics
        }

    def get_metrics_dict(self):
        return {
            "overall": {
                "total_packages": self.total_packages,
                "total_pairs": self.total_pairs,
                "typosquats_identified": self.typosquats_identified,
                "correct_target_matches": self.correct_target_matches,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "accuracy": self.accuracy
            },
            "benign": self.calculate_subset_metrics("benign"),
            "stealthy": self.calculate_subset_metrics("stealthy"),
            "active": self.calculate_subset_metrics("active")
        }

    def print_metrics(self):
        metrics = self.get_metrics_dict()
        print("\n=== Evaluation Metrics ===")
        print("\n--- Overall Metrics ---")
        for key, value in metrics["overall"].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        for subset in ["benign", "stealthy", "active"]:
            print(f"\n--- {subset.capitalize()} Subset Metrics ---")
            for key, value in metrics[subset].items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")


def build_ground_truth(ground_truth_file: str, dataset_type: str) -> Dict:
    """
    Build a ground truth mapping (keys are lowercased package names) based on dataset type.

    For dataset_type "FPR":
      - Use the FP_ground_truth.csv file.
      - Use the column "Adversarial pkg" as the candidate package name.
      - Use the column "Original pkg" as the legitimate package.
      - Use the column "is_FP?" to determine the label:
            "Yes" → benign (negative)
            "No"  → stealthy (positive)

    For dataset_type "active":
      - Use dataset.csv.
      - Filter out rows where the "confusion" column equals "UNK".
      - Use the column "typosquat_pkg" as the package name.
      - Use "legitimate_pkg" as the legitimate package.
      - All remaining packages are active (positive).
    """
    gt_df = pd.read_csv(ground_truth_file)
    mapping = {}
    if dataset_type.upper() == "FPR":
        for _, row in gt_df.iterrows():
            pkg = str(row["Adversarial pkg"]).strip().lower()
            is_fp = str(row["is_FP?"]).strip().lower()
            if is_fp == "yes":
                category = "benign"
            elif is_fp == "no":
                category = "stealthy"
            else:
                continue
            mapping[pkg] = {
                "target": str(row.get("Original pkg", "")).strip().lower(),
                "is_true_typosquat": (category == "stealthy"),
                "category": category
            }
    elif dataset_type.upper() == "ACTIVE":
        gt_df = gt_df[gt_df["confusion"].str.upper() != "UNK"]
        for _, row in gt_df.iterrows():
            pkg = str(row["typosquat_pkg"]).strip().lower()
            mapping[pkg] = {
                "target": str(row.get("legitimate_pkg", "")).strip().lower(),
                "is_true_typosquat": True,
                "category": "active"
            }
    else:
        print(f"Unknown dataset_type: {dataset_type}")
        sys.exit(1)
    return mapping


def parse_tool_output(output_file: str) -> Dict:
    """
    Parse the tool's output CSV file and return a dictionary mapping package names to predicted labels.
    Assumes the CSV has columns "package_name" and "label".
    """
    df = pd.read_csv(output_file)
    # Build a mapping from lowercased package_name to lowercased label.
    mapping = {}
    for _, row in df.iterrows():
        pkg = str(row["package_name"]).strip().lower()
        label = str(row["label"]).strip().lower()
        mapping[pkg] = label
    return mapping


def analyze_metrics(ground_truth_file: str, output_file: str, dataset_type: str) -> BenchmarkMetrics:
    """
    Calculate evaluation metrics by comparing full ground truth and tool output.
    For each ground truth candidate:
      - If the tool output contains a row for that package, use its predicted label to decide whether
        it is predicted positive (if label != "false_positive") or negative (if label == "false_positive").
      - Otherwise, ignore that candidate.
    Then compute the confusion matrix based on these candidates.
    """
    gt_mapping = build_ground_truth(ground_truth_file, dataset_type)
    tool_mapping = parse_tool_output(output_file)

    # Only consider ground truth candidates that appear in the tool output.
    valid_candidates = {pkg: gt_mapping[pkg] for pkg in gt_mapping if pkg in tool_mapping}

    metrics = BenchmarkMetrics()
    metrics.total_packages = len(valid_candidates)
    metrics.total_pairs = metrics.total_packages
    metrics.typosquats_identified = len(valid_candidates)

    for pkg, gt_info in valid_candidates.items():
        category = gt_info["category"]
        is_positive = gt_info["is_true_typosquat"]
        # Get the predicted label from tool output.
        predicted_label = tool_mapping[pkg]
        # Interpret prediction: predicted positive if label is not "false_positive".
        predicted_positive = (predicted_label != "false_positive")

        if predicted_positive:
            if is_positive:
                metrics.true_positives += 1
                if category == "stealthy":
                    metrics.stealthy_metrics["true_positives"] += 1
                elif category == "active":
                    metrics.active_metrics["true_positives"] += 1
            else:
                metrics.false_positives += 1
                if category == "benign":
                    metrics.benign_metrics["false_positives"] += 1
        else:
            if is_positive:
                metrics.false_negatives += 1
                if category == "stealthy":
                    metrics.stealthy_metrics["false_negatives"] += 1
                elif category == "active":
                    metrics.active_metrics["false_negatives"] += 1
            else:
                metrics.true_negatives += 1
                if category == "benign":
                    metrics.benign_metrics["true_negatives"] += 1

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Calculate FP Evaluation Metrics for EQ3-FPR datasets")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth CSV file (FP_ground_truth.csv for FPR or dataset.csv for active)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the tool output CSV file (fp_evaluation_results.csv)")
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Dataset type: 'FPR' or 'active'")
    args = parser.parse_args()

    start_time = time.time()
    metrics = analyze_metrics(args.ground_truth, args.output_file, args.dataset_type)
    metrics.total_time = time.time() - start_time

    metrics.print_metrics()

    output_path = Path(args.output_file).parent / "fp_metrics_analysis.json"
    with open(output_path, "w") as f:
        json.dump(metrics.get_metrics_dict(), f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
