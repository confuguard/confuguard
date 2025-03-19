#!/usr/bin/env python3
"""
analyze_typomind_metrics.py

This script analyzes Typomind output files from a given results directory and computes
overall as well as subset metrics (for benign, stealthy, and active packages).

For Typomind, if a package appears in any of the output files (i.e. a package pair is reported),
it is considered as positive (an attack). The total number of evaluation candidates is set equal
to the total number of ground-truth packages.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import pandas as pd


@dataclass
class BenchmarkMetrics:
    total_time: float = 0.0
    total_packages: int = 0      # number of ground truth packages
    total_pairs: int = 0         # for Typomind, equals total_packages
    typosquats_identified: int = 0  # count of packages flagged as positive
    correct_target_matches: int = 0  # count of packages with correct target match

    # Overall confusion matrix counts
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

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
        return (self.true_positives /
                (self.true_positives + self.false_positives)
                if (self.true_positives + self.false_positives) > 0 else 0)

    @property
    def recall(self) -> float:
        return (self.true_positives /
                (self.true_positives + self.false_negatives)
                if (self.true_positives + self.false_negatives) > 0 else 0)

    @property
    def f1_score(self) -> float:
        if (self.precision + self.recall) > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return 0

    @property
    def accuracy(self) -> float:
        total = (self.true_positives + self.true_negatives +
                 self.false_positives + self.false_negatives)
        return ((self.true_positives + self.true_negatives) / total
                if total > 0 else 0)

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
        print("\n=== Typomind Metrics ===")
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


def parse_typomind_output_file(file_path: str):
    """
    Parse a Typomind output file and return a set of detected packages and a dict
    for correct target match per package.
    Expected line format:
      ('target', 'input'): {'detection_info'}, timing
    For example:
      ('tfs_graph', 'aws-graph'): {'aws-graph': '1-step D-L dist'}, 0:00:00.000503
    """
    from pathlib import Path
    detected = set()
    correct_matches = {}  # key: package, value: True if any detection had a correct target match

    if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:
        print(f"Warning: Output file {file_path} does not exist or is empty")
        return detected, correct_matches

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split(": ", 1)
            if len(parts) != 2:
                print(f"Warning: Unexpected format in line: {line}")
                continue
            pair_str = parts[0].strip()
            if not (pair_str.startswith("(") and pair_str.endswith(")")):
                print(f"Warning: Invalid package pair format: {pair_str}")
                continue
            pair_content = pair_str[1:-1].split(", ")
            if len(pair_content) != 2:
                print(f"Warning: Invalid package pair content: {pair_content}")
                continue
            target_package = pair_content[0].strip("'\"").lower()
            input_package = pair_content[1].strip("'\"").lower()

            # Mark the input package as detected.
            detected.add(input_package)

            # Parse detection info (expected as a dict-like string).
            remainder = parts[1].strip()
            detection_timing = remainder.split(", ")
            if len(detection_timing) < 2:
                print(f"Warning: Invalid detection and timing format: {remainder}")
                continue
            detection_info_str = ", ".join(detection_timing[:-1])
            try:
                import ast
                detection_info = ast.literal_eval(detection_info_str)
                # If the detection info contains the input_package with a non-empty value, mark it as a correct match.
                if input_package in detection_info and detection_info[input_package]:
                    correct_matches[input_package] = True
            except Exception as e:
                print(f"Warning: Could not parse detection info: {detection_info_str}, error: {e}")
        except Exception as e:
            print(f"Warning: Failed to parse line: {line}, error: {e}")

    return detected, correct_matches


def analyze_typomind_metrics(results_dir: str, ground_truth_file: str) -> BenchmarkMetrics:
    """
    Analyze Typomind output metrics from all files in the given results directory.
    Expects files matching the pattern:
        typomind_results_{ecosystem}.txt
    For Typomind, every ground truth package is a candidate.
    """
    # Build ground truth mapping from CSV.
    gt_df = pd.read_csv(ground_truth_file)
    legitimate_targets = {}
    for _, row in gt_df.iterrows():
        name = str(row["name"]).strip().lower()
        threat_type = str(row["threat_type"]).strip().lower()
        if threat_type == "false_positive":
            category = "benign"
        elif threat_type == "typosquat":
            category = "stealthy"
        else:
            category = "active"
        target = str(row["legitimate_package"]).strip().lower() if pd.notna(row["legitimate_package"]) else ""
        legitimate_targets[name] = {
            "target": target,
            "is_true_typosquat": threat_type != "false_positive",
            "category": category
        }

    # Initialize metrics.
    metrics = BenchmarkMetrics()
    # Total candidates equals number of ground truth packages.
    metrics.total_packages = len(legitimate_targets)
    metrics.total_pairs = metrics.total_packages

    detected_packages_all = set()
    package_correct_match = {}

    results_path = Path(results_dir)
    output_files = list(results_path.glob("typomind_results_*.txt"))
    if not output_files:
        print(f"No typomind_results_*.txt files found in {results_dir}")
        sys.exit(1)

    start_time = time.time()
    for file in output_files:
        print(f"Processing file: {file}")
        detected, correct_matches = parse_typomind_output_file(str(file))
        detected_packages_all.update(detected)
        for pkg, correct in correct_matches.items():
            if correct:
                package_correct_match[pkg] = True
    metrics.total_time = time.time() - start_time

    # For Typomind, each unique detected package counts as a positive.
    metrics.typosquats_identified = len(detected_packages_all)

    # Update overall confusion matrix and subset-specific metrics.
    for pkg, gt_info in legitimate_targets.items():
        category = gt_info["category"]
        is_true = gt_info["is_true_typosquat"]
        detected = pkg in detected_packages_all

        # Overall counts.
        if detected:
            if is_true:
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1
        else:
            if is_true:
                metrics.false_negatives += 1
            else:
                metrics.true_negatives += 1

        # Subset counts.
        if category == "benign":
            if detected:
                metrics.benign_metrics["false_positives"] += 1
            else:
                metrics.benign_metrics["true_negatives"] += 1
        elif category == "stealthy":
            if detected:
                metrics.stealthy_metrics["true_positives"] += 1
            else:
                metrics.stealthy_metrics["false_negatives"] += 1
        elif category == "active":
            if detected:
                metrics.active_metrics["true_positives"] += 1
            else:
                metrics.active_metrics["false_negatives"] += 1

        # Update correct target matches (at most one per package).
        if detected and pkg in package_correct_match and package_correct_match[pkg]:
            metrics.correct_target_matches += 1

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze Typomind Metrics from results directory")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to directory containing typomind_results_{ecosystem}.txt files")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth CSV file")
    args = parser.parse_args()

    metrics = analyze_typomind_metrics(args.results_dir, args.ground_truth)
    metrics.print_metrics()

    output_path = Path(args.results_dir) / "typomind_metrics_analysis.json"
    with open(output_path, "w") as f:
        json.dump(metrics.get_metrics_dict(), f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
