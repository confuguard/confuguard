import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
import subprocess
from typing import Dict, List
import json
from datetime import datetime
import time
from dataclasses import dataclass, field
from statistics import mean, median
from loguru import logger
from itertools import product

from Part4.confusion_search import DatabaseManager
from config import TYPOSQUAT_BEARER_TOKEN
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import Timeout
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from datetime import datetime
import gc
import logging
import argparse
from multiprocessing import Pool
import concurrent.futures

sys.path.append("submodules/typomind-release")
from core.detectors import classify_typosquat

TYPOSQUAT_URL = "http://localhost:5555"

MAX_RETRIES = 1   # Maximum number of API call retries
TIMEOUT = 60

# Constants for typomind
TIME_LIMIT = 120  # seconds (2 minutes)
MEMORY_LIMIT_MB = 10 * 1024  # 10GB in MB
MAX_CONCURRENT_JOBS = 32

curr_pop_pkgs = []

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
    false_negatives: int = 0  # Incorrectly identified as not typosquats (missed typosquats)

    @property
    def throughput(self) -> float:
        """Calculate throughput in packages per second"""
        return self.total_packages / self.total_time if self.total_time > 0 else 0

    @property
    def avg_latency(self) -> float:
        """Calculate average latency in milliseconds per package"""
        return mean(self.latencies) * 1000 if self.latencies else 0

    @property
    def median_latency(self) -> float:
        """Calculate median latency in milliseconds per package"""
        return median(self.latencies) * 1000 if self.latencies else 0

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)"""
        return self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)"""
        return self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
        if (self.precision + self.recall) > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return 0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0

    def add_ecosystem_metric(self, ecosystem: str, packages: int, time_taken: float, latencies: List[float]):
        """Add metrics for a specific ecosystem"""
        self.ecosystem_metrics[ecosystem] = {
            'total_packages': packages,
            'total_time': time_taken,
            'latencies': latencies,
            'throughput': packages / time_taken if time_taken > 0 else 0,
            'avg_latency': mean(latencies) * 1000 if latencies else 0,
            'median_latency': median(latencies) * 1000 if latencies else 0
        }

    def get_metrics_dict(self):
        """Get all metrics in dictionary format"""
        return {
            'overall': {
                'total_time': self.total_time,
                'total_packages': self.total_packages,
                'total_pairs': self.total_pairs,
                'throughput': self.throughput,
                'avg_latency': self.avg_latency,
                'median_latency': self.median_latency,
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
            'per_ecosystem': self.ecosystem_metrics
        }

    def log_metrics(self, tool_name: str):
        """Log the benchmark metrics"""
        metrics = self.get_metrics_dict()

        # Log overall metrics
        logger.info(f"\n=== {tool_name} Overall Metrics ===")
        logger.info(f"Total time: {metrics['overall']['total_time']:.2f} seconds")
        logger.info(f"Total packages processed: {metrics['overall']['total_packages']}")
        logger.info(f"Total pairs evaluated: {metrics['overall']['total_pairs']}")
        logger.info(f"Overall throughput: {metrics['overall']['throughput']:.2f} packages/second")
        logger.info(f"Overall average latency: {metrics['overall']['avg_latency']:.2f} milliseconds/package")
        logger.info(f"Overall median latency: {metrics['overall']['median_latency']:.2f} milliseconds/package")

        # Log detection metrics
        logger.info(f"\n=== {tool_name} Detection Metrics ===")
        logger.info(f"Packages identified as typosquats: {metrics['overall']['typosquats_identified']}")
        logger.info(f"Correctly identified legitimate targets: {metrics['overall']['correct_target_matches']}")

        # Log confusion matrix metrics
        logger.info(f"\n=== {tool_name} Confusion Matrix ===")
        logger.info(f"True Positives: {metrics['overall']['true_positives']} (correctly identified typosquats)")
        logger.info(f"False Positives: {metrics['overall']['false_positives']} (incorrectly flagged as typosquats)")
        logger.info(f"True Negatives: {metrics['overall']['true_negatives']} (correctly identified non-typosquats)")
        logger.info(f"False Negatives: {metrics['overall']['false_negatives']} (missed typosquats)")

        # Log derived metrics
        logger.info(f"\n=== {tool_name} Performance Metrics ===")
        logger.info(f"Precision: {metrics['overall']['precision']:.4f}")
        logger.info(f"Recall: {metrics['overall']['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['overall']['f1_score']:.4f}")
        logger.info(f"Accuracy: {metrics['overall']['accuracy']:.4f}")

        # Log per-ecosystem metrics
        logger.info(f"\n=== {tool_name} Per-Ecosystem Metrics ===")
        for ecosystem, eco_metrics in metrics['per_ecosystem'].items():
            logger.info(f"\n  {ecosystem}:")
            logger.info(f"  Packages processed: {eco_metrics['total_packages']}")
            logger.info(f"  Time taken: {eco_metrics['total_time']:.2f} seconds")
            logger.info(f"  Throughput: {eco_metrics['throughput']:.2f} packages/second")
            logger.info(f"  Average latency: {eco_metrics['avg_latency']:.2f} milliseconds/package")
            logger.info(f"  Median latency: {eco_metrics['median_latency']:.2f} milliseconds/package")

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Timeout, requests.exceptions.RequestException))
)
def get_neighbors_from_api(pkg, registry):
    """Get valid neighbors using the REST API with retry."""
    headers = {
        'Authorization': f'Bearer {TYPOSQUAT_BEARER_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        'package_name': pkg,
        'registry': registry
    }

    try:
        response = requests.post(
            f"{TYPOSQUAT_URL}/get_neighbors",
            json=data,
            headers=headers,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get('valid_neighbors', [])
    except Timeout:
        logger.warning(f"Timeout while getting neighbors for {pkg}, retrying...")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for get_neighbors: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Timeout, requests.exceptions.RequestException))
)
def verify_fp_from_api(pkg, registry, neighbor):
    """Verify false positive using the REST API endpoint with retry."""
    headers = {
        'Authorization': f'Bearer {TYPOSQUAT_BEARER_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        'package_name': pkg,
        'registry': registry,
        'neighbor': neighbor
    }

    try:
        response = requests.post(
            f"{TYPOSQUAT_URL}/verify_fp",
            json=data,
            headers=headers,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except Timeout:
        logger.warning(f"Timeout while verifying FP for {pkg} -> {neighbor['package_name']}, retrying...")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for verify_fp: {str(e)}")
        raise

def evaluate_confuguard(df: pd.DataFrame, output_dir: str, openai_api_key: str):
    """
    Evaluate package pairs using the REST API endpoints.
    """
    start_time = time.time()
    metrics = BenchmarkMetrics()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"confuguard_evaluation_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating {len(df)} packages with Confuguard")

    # Initialize results list
    results = []
    packages_with_neighbors = 0
    false_positives = 0

    # Ecosystem mapping
    ecosystem_map = {
        'gem': 'ruby',
        'npm': 'npm',
        'pypi': 'pypi',
        'maven': 'maven',
        'go': 'golang'
    }

    # Process each package
    for idx, row in df.iterrows():
        try:
            metrics.total_packages += 1
            package_name = row['name']
            raw_ecosystem = row['type'].lower()
            legitimate_name = row.get('legitimate_name', '')  # Get the legitimate name from the input

            # Get the ground truth for this package
            is_true_typosquat = row.get('threat_type', '').lower() != 'false_positive'

            # Handle npm packages with namespaces
            if raw_ecosystem == 'npm' and pd.notna(row['namespace']) and row['namespace']:
                package_name = f"{row['namespace']}/{package_name}"
            elif raw_ecosystem == 'maven' and pd.notna(row['namespace']) and row['namespace']:
                package_name = f"{row['namespace']}:{package_name}"

            # Map ecosystem names for consistency
            ecosystem = ecosystem_map.get(raw_ecosystem, raw_ecosystem)

            # Get neighbors from API with timing
            neighbor_start_time = time.time()
            neighbors = get_neighbors_from_api(package_name, ecosystem)
            neighbor_time = time.time() - neighbor_start_time
            logger.debug(f"get_neighbors time for {package_name} ({ecosystem}): {neighbor_time:.2f}s")

            if not neighbors:
                logger.debug(f"No neighbors found for {package_name}")
                # If no neighbors found and it's actually a typosquat, it's a false negative
                if is_true_typosquat:
                    metrics.false_negatives += 1
                else:
                    metrics.true_negatives += 1
                continue

            packages_with_neighbors += 1

            found_true_positive = False
            fp_start_time = time.time()
            package_classified_as_typosquat = False

            for idx, neighbor in enumerate(neighbors):
                if idx > 2:  # Only process up to 3 neighbors
                    break

                # Skip the second neighbor (idx=1) if a true positive was detected
                if idx == 1 and found_true_positive:
                    logger.info(f"Skipping second neighbor for {package_name} because a true positive was detected")
                    continue

                pair_start_time = time.time()
                metrics.total_pairs += 1

                # Verify if it's a false positive using API
                fp_result = verify_fp_from_api(package_name, ecosystem, neighbor)

                # Record latency for this pair
                pair_latency = time.time() - pair_start_time
                metrics.latencies.append(pair_latency)

                is_fp = fp_result.get('is_false_positive', True)
                neighbor_name = neighbor['package_name']

                # Check if this is a true positive and update our flag
                if not is_fp:
                    found_true_positive = True
                    false_positives += 1
                    metrics.typosquats_identified += 1
                    package_classified_as_typosquat = True

                    # Check if the identified legitimate target matches the expected one
                    if legitimate_name and fp_result.get('legitimate_target') == legitimate_name:
                        metrics.correct_target_matches += 1

                # Store result
                result = {
                    'package_name': package_name,
                    'neighbor': neighbor['package_name'],
                    'ecosystem': ecosystem,
                    'is_false_positive': is_fp,
                    'fp_category': fp_result.get('FP_category', ''),
                    'metrics': fp_result.get('metrics', {}),
                    'explanation': fp_result.get('explanation', '')
                }
                results.append(result)

            # Track total false positive verification time
            fp_time = time.time() - fp_start_time
            logger.debug(f"FP verification time for {package_name}: {fp_time:.2f}s")

            # Update confusion matrix metrics after processing all neighbors for this package
            if package_classified_as_typosquat:
                if is_true_typosquat:
                    metrics.true_positives += 1
                else:
                    metrics.false_positives += 1
            else:
                if is_true_typosquat:
                    metrics.false_negatives += 1
                else:
                    metrics.true_negatives += 1

            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} packages")
                save_results(results, output_path)

        except Exception as e:
            logger.error(f"Error processing {package_name}: {str(e)}")

    # Save final results
    save_results(results, output_path)

    # Log summary statistics
    logger.info(f"Total packages processed: {len(df)}")
    logger.info(f"Packages with neighbors: {packages_with_neighbors}")
    logger.info(f"False positives found: {false_positives}")

    metrics.total_time = time.time() - start_time
    metrics.log_metrics("Confuguard")

    return output_path, metrics



################################################################################
# Typomind
################################################################################
def enforce_memory_limit():
    """Enforce memory limit using ulimit (Unix/Mac) or psutil (cross-platform)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_MB * 1024 * 1024,
                                                  MEMORY_LIMIT_MB * 1024 * 1024))
    except ImportError:
        process = psutil.Process(os.getpid())
        process.rlimit(psutil.RLIMIT_AS, (MEMORY_LIMIT_MB * 1024 * 1024,
                                          MEMORY_LIMIT_MB * 1024 * 1024))

def run_cli(base_file, adv_file, out_file):
    """
    Runs the Typomind detector CLI command for a given pair of package files,
    applying a timeout and memory limit.
    Returns a tuple:
      (base_file, adv_file, out_file, status, latency, stdout, stderr)
    """
    command = [
        "python", "submodules/typomind-release/__main__.py",
        "--base_file", base_file,
        "--adv_file", adv_file,
        "--outfile_path", out_file
    ]
    start = time.time()
    try:
        enforce_memory_limit()
        result = subprocess.run(command, timeout=TIME_LIMIT,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        latency = time.time() - start
        return (base_file, adv_file, out_file, "SUCCESS", latency, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        latency = time.time() - start
        return (base_file, adv_file, out_file, "TIMEOUT", latency, "", "Process exceeded time limit")
    except Exception as e:
        latency = time.time() - start
        return (base_file, adv_file, out_file, "ERROR", latency, "", str(e))


def update_pop_pkgs(mapped_ecosystem: str, legitimate_packages: list, num_base_packages: int):
    """
    Load base packages from file and combine with legitimate packages,
    slicing the result to num_base_packages.
    """
    global curr_pop_pkgs
    ecosystem_map = {
        'gem': 'ruby',
        'npm': 'npm',
        'pypi': 'pypi',
        'maven': 'maven',
        'go': 'golang'
    }
    mapped_ecosystem = ecosystem_map.get(mapped_ecosystem.lower(), mapped_ecosystem.lower())
    popular_file = Path("./legit_packages") / f"{mapped_ecosystem}_legit_packages.csv"
    if not popular_file.exists():
        print(f"Warning: No popular packages file found for {mapped_ecosystem}")
        return [str(pkg) for pkg in legitimate_packages[:num_base_packages if num_base_packages is not None else None]]
    pop_df = pd.read_csv(popular_file, usecols=['package_name'])
    popular_packages = pop_df['package_name'].tolist()

    # Use OrderedDict to maintain insertion order while removing duplicates
    from collections import OrderedDict
    # First add popular packages, then legitimate packages
    combined_packages = list(OrderedDict.fromkeys([str(pkg) for pkg in popular_packages + legitimate_packages]))

    # Only slice if num_base_packages is not None
    if num_base_packages is not None:
        curr_pop_pkgs = combined_packages[:num_base_packages]
    else:
        curr_pop_pkgs = combined_packages

    return curr_pop_pkgs

def cli_main(tasks):
    """Orchestrates execution of multiple CLI calls concurrently."""
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = {executor.submit(run_cli, t[0], t[1], t[2]): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def evaluate_typomind(df: pd.DataFrame, output_path: Path, num_base_packages: int, num_adv_packages: int) -> BenchmarkMetrics:
    """
    Evaluate typomind detection using the CLI calls (via run_cli) per ecosystem.
    For each ecosystem, base and adversarial package lists are created (with namespace handling),
    temporary files are written, and tasks are submitted concurrently.
    The latency and CLI results are logged, and the combined output is parsed to update benchmark metrics.
    """
    start_time = time.time()
    metrics = BenchmarkMetrics()
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)
    ecosystem_map = {'gem': 'ruby', 'npm': 'npm', 'pypi': 'pypi', 'maven': 'maven', 'go': 'golang'}

    for ecosystem in df['type'].unique():
        mapped_ecosystem = ecosystem_map.get(ecosystem.lower(), ecosystem.lower())
        group = df[df['type'].str.lower() == ecosystem.lower()]
        # Build legitimate (base) package list from false positives and optional legitimate_package column
        legit_packages = group[group['threat_type'].str.lower() == 'false_positive']['name'].tolist()
        if 'legitimate_package' in group.columns:
            legit_packages += group['legitimate_package'].dropna().tolist()
        legit_packages = list(set(legit_packages))

        base_pkgs = update_pop_pkgs(ecosystem, legit_packages, num_base_packages)
        base_file = temp_dir / f"tmp_{mapped_ecosystem}_base.txt"
        with open(base_file, "w") as f:
            f.write("\n".join(base_pkgs))

        # Build adversarial package list (handle namespace for npm/maven)
        adv_pkgs = []
        for _, row in group.iterrows():
            pkg_name = row['name']
            if mapped_ecosystem == 'npm' and pd.notna(row.get('namespace')) and row.get('namespace'):
                pkg_name = f"{row['namespace']}/{pkg_name}"
            elif mapped_ecosystem == 'maven' and pd.notna(row.get('namespace')) and row.get('namespace'):
                pkg_name = f"{row['namespace']}:{pkg_name}"
            adv_pkgs.append(pkg_name)
        adv_pkgs = list(set(adv_pkgs))
        if num_adv_packages is not None:
            adv_pkgs = adv_pkgs[:num_adv_packages]

        tasks = []
        adv_files = []
        for i, adv_pkg in enumerate(adv_pkgs):
            adv_file = temp_dir / f"tmp_{mapped_ecosystem}_adv_{i}.txt"
            with open(adv_file, "w") as f:
                f.write(adv_pkg)
            out_file = temp_dir / f"typomind_results_{mapped_ecosystem}_{i}.txt"
            tasks.append((str(base_file), str(adv_file), str(out_file)))
            adv_files.append(out_file)

        ecosystem_latencies = []
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
                future_to_task = {executor.submit(run_cli, t[0], t[1], t[2]): t for t in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    t = future_to_task[future]
                    result = future.result()  # (base_file, adv_file, out_file, status, latency, stdout, stderr)
                    status = result[3]
                    latency = result[4]
                    ecosystem_latencies.append(latency)
                    # Add latency to the main metrics object as well
                    metrics.latencies.append(latency)
                    logger.info(f"Task {t}: Status {status}, Latency {latency:.2f}s")
        # Combine individual output files into one for parsing
        ecosystem_output_file = temp_dir / f"typomind_results_{mapped_ecosystem}.txt"
        with open(ecosystem_output_file, "w") as outfile:
            for fpath in adv_files:
                if Path(fpath).exists():
                    with open(fpath, "r") as infile:
                        outfile.write(infile.read())
        # Build legitimate targets mapping for this ecosystem (if provided)
        legitimate_targets = {}
        for _, row in group.iterrows():
            if 'legitimate_package' in row and pd.notna(row['legitimate_package']):
                legitimate_targets[row['name']] = {
                    'target': row['legitimate_package'],
                    'is_true_typosquat': row.get('threat_type', '').lower() != 'false_positive'
                }
        detections_per_package = {}
        parse_typomind_output(ecosystem_output_file, metrics, legitimate_targets, detections_per_package)
        ecosystem_pairs = len(base_pkgs) * len(adv_pkgs)
        metrics.total_pairs += ecosystem_pairs
        metrics.total_packages += len(adv_pkgs)  # Count adversarial packages, not pairs
        ecosystem_time = time.time() - start_time
        metrics.add_ecosystem_metric(mapped_ecosystem, len(adv_pkgs), ecosystem_time, ecosystem_latencies)  # Use adv_pkgs count instead of pairs
        gc.collect()

    metrics.total_time = time.time() - start_time
    metrics.log_metrics("Typomind")
    return metrics


def parse_typomind_output(output_file, metrics, legitimate_targets, detections_per_package):
    """
    Parse the typomind output file and update metrics.
    The output format appears to be:
      target_package,input_package,detection_info,timing
    """
    import ast
    from pathlib import Path
    import pandas as pd

    if not Path(output_file).exists() or Path(output_file).stat().st_size == 0:
        logger.warning(f"Output file {output_file} does not exist or is empty")
        return []

    # Fallback to the older (non-CSV) format
    detection_results = []
    detection_list = []
    detected_packages_set = set()

    with open(output_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Expecting a format like: ('target', 'input'): {'detection_info'}, timing
            parts = line.split(': ', 1)
            if len(parts) != 2:
                logger.warning(f"Unexpected format in line: {line}")
                continue

            # Extract and validate package pair
            pair_str = parts[0].strip()
            if not (pair_str.startswith('(') and pair_str.endswith(')')):
                logger.warning(f"Invalid package pair format: {pair_str}")
                continue

            pair_content = pair_str[1:-1].split(', ')
            if len(pair_content) != 2:
                logger.warning(f"Invalid package pair content: {pair_content}")
                continue

            # Remove quotes to get the package names
            target_package = pair_content[0].strip("'\"")
            input_package = pair_content[1].strip("'\"")

            # The remainder contains detection_info and timing
            remainder = parts[1].strip()
            detection_timing = remainder.split(', ')
            if len(detection_timing) < 2:
                logger.warning(f"Invalid detection and timing format: {remainder}")
                continue

            timing = detection_timing[-1]
            detection_info = ', '.join(detection_timing[:-1])

            detection_results.append({
                'target_package': target_package,
                'input_package': input_package,
                'detection_info': detection_info,
                'timing': timing
            })

            # Update counts per detection instance
            detections_per_package[input_package] = detections_per_package.get(input_package, 0) + 1
            detection_list.append(input_package)
            detected_packages_set.add(input_package)

            # Check if we correctly identified the legitimate target
            if input_package in legitimate_targets and target_package == legitimate_targets[input_package]['target']:
                metrics.correct_target_matches += 1

        except Exception as e:
            logger.warning(f"Failed to parse line: {line}, error: {str(e)}")

    metrics.typosquats_identified += len(detection_list)

    # Update confusion matrix metrics
    for pkg, pkg_info in legitimate_targets.items():
        is_true_typosquat = pkg_info['is_true_typosquat']
        was_detected = pkg in detected_packages_set

        if was_detected:
            if is_true_typosquat:
                # Correctly identified as a typosquat
                metrics.true_positives += 1
            else:
                # Incorrectly identified as a typosquat (it's actually a false positive)
                metrics.false_positives += 1
        else:
            if is_true_typosquat:
                # Failed to identify a true typosquat
                metrics.false_negatives += 1
            else:
                # Correctly did not flag a non-typosquat
                metrics.true_negatives += 1

    # Handle packages that were detected but aren't in our legitimate_targets mapping
    for pkg in detected_packages_set:
        if pkg not in legitimate_targets:
            # We don't know the ground truth for these packages, so we can't classify them
            logger.warning(f"Package {pkg} was detected but has no ground truth in legitimate_targets")

    return detection_results




def evaluate_ossgadget(df: pd.DataFrame, output_path: Path):
    """
    Run OSS Find Squats tool on package list.
    """
    import re, ast
    start_time = time.time()
    metrics = BenchmarkMetrics()

    logger.info(f"Evaluating {len(df)} packages with OSS Gadget")

    results = []
    output_file = output_path / 'ossgadget_results.csv'
    pattern = r"INFO\s+-\s+(\S+)\s+package exists\. Potential squat\. (.+)"

    for idx, row in df.iterrows():
        try:
            package_name = row['name']
            ecosystem = row['type'].lower()

            # Skip Maven and Golang packages
            if ecosystem in ['maven', 'golang', 'go']:
                logger.info(f"Skipping {ecosystem} package {package_name} - too slow for OSS Gadget")
                continue

            pkg_start_time = time.time()
            metrics.total_packages += 1

            legitimate_name = str(row.get('legitimate_package', '')) if pd.notna(row.get('legitimate_package', '')) else ''
            threat_type = row.get('threat_type', '')
            is_true_typosquat = str(threat_type).lower() != 'false_positive' if pd.notna(threat_type) else True

            # Construct purl
            if ecosystem == 'npm' and 'namespace' in row and pd.notna(row['namespace']) and row['namespace']:
                purl = f"pkg:npm/{row['namespace']}/{package_name}"
            elif ecosystem == 'pypi':
                purl = f"pkg:pypi/{package_name}"
            elif ecosystem == 'npm':
                purl = f"pkg:npm/{package_name}"
            elif ecosystem in ['rubygems', 'gem', 'ruby']:
                purl = f"pkg:gem/{package_name}"
            else:
                logger.warning(f"Unsupported ecosystem: {ecosystem}, skipping {package_name}")
                continue

            logger.debug(f"Running oss-find-squats for {purl}")

            try:
                # Run oss-find-squats
                cmd_result = subprocess.run(
                    ["oss-find-squats", purl],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=60
                )

                pkg_latency = time.time() - pkg_start_time
                metrics.latencies.append(pkg_latency)

                output_lines = cmd_result.stdout.splitlines()
                squats = []
                has_potential_squats = False
                legit_found = None

                for line in output_lines:
                    match = re.search(pattern, line)
                    if match:
                        squat_name = match.group(1)
                        classification_str = match.group(2)
                        try:
                            classifications = ast.literal_eval(classification_str)
                        except Exception as e:
                            logger.error(f"Error parsing classifications: {line}, {str(e)}")
                            classifications = []

                        metrics.total_pairs += 1
                        squats.append({'squat_name': squat_name, 'classifications': classifications})
                        has_potential_squats = True

                        # Check legitimate match
                        if legitimate_name and squat_name.lower() == legitimate_name.lower():
                            legit_found = squat_name

                if has_potential_squats:
                    metrics.typosquats_identified += 1
                    if is_true_typosquat:
                        metrics.true_positives += 1
                    else:
                        metrics.false_positives += 1
                else:
                    if is_true_typosquat:
                        metrics.false_negatives += 1
                    else:
                        metrics.true_negatives += 1

                result = {
                    'package_name': package_name,
                    'ecosystem': ecosystem,
                    'purl': purl,
                    'is_typosquat': has_potential_squats,
                    'legitimate_package_identified': legit_found is not None,
                    'legitimate_package': legit_found,
                    'squats_found': len(squats),
                    'squats': squats,
                    'return_code': cmd_result.returncode,
                    'output': cmd_result.stdout,
                    'timed_out': False
                }

            except subprocess.TimeoutExpired:
                logger.warning(f"OSS Gadget timed out for {purl}")
                result = {
                    'package_name': package_name,
                    'ecosystem': ecosystem,
                    'purl': purl,
                    'is_typosquat': False,
                    'legitimate_package_identified': False,
                    'legitimate_package': None,
                    'squats_found': 0,
                    'squats': [],
                    'return_code': None,
                    'output': "TIMEOUT: 60s limit",
                    'timed_out': True
                }

                if is_true_typosquat:
                    metrics.false_negatives += 1
                else:
                    metrics.true_negatives += 1

            results.append(result)

            # Periodically save progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} packages with OSS Gadget")
                pd.DataFrame(results).to_csv(output_file, index=False)

        except Exception as e:
            logger.error(f"Error processing {package_name} with oss-find-squats: {str(e)}")

    # Save final results
    pd.DataFrame(results).to_csv(output_file, index=False)
    metrics.total_time = time.time() - start_time
    metrics.log_metrics("OSS Find Squats")

    return metrics


def save_results(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    output_file = output_path / 'threatfeed_results.csv'
    df.to_csv(output_file, index=False)


def save_metrics_to_file(metrics: BenchmarkMetrics, tool_name: str, output_path: Path):
    """Save tool-specific metrics to a separate JSON file."""
    metrics_file = output_path / f'{tool_name.lower()}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics.get_metrics_dict(), f, indent=2)
    logger.info(f"Saved {tool_name} metrics to {metrics_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark typosquat detection tools")
    parser.add_argument("--num-adv-packages", type=int, default=None,
                        help="Number of packages to evaluate (default: all packages)")
    parser.add_argument("--num-base-packages", type=int, default=None,
                        help="Number of packages to evaluate (default: all packages)")
    args = parser.parse_args()

    num_adv_packages = args.num_adv_packages
    num_base_packages = args.num_base_packages

    # Configure logging
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[3]  # Go up 4 levels to reach project root

    logger.add(script_dir / "benchmark.log", rotation="1 day")

    # Set paths and API key
    input_csv = project_root / "./datasets/ConfuDB.csv"
    output_dir = script_dir / "results"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Read input data
    df = pd.read_csv(input_csv)

    # Limit the number of packages if specified
    if num_adv_packages is not None:
        logger.info(f"Limiting benchmark to {num_adv_packages} adversarial packages")
        df = df.head(num_adv_packages)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running evaluations on {len(df)} packages")
    output_path = output_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(exist_ok=True)
    logger.info(f"Saving results to {output_path}")

    typomind_metrics = evaluate_typomind(df, output_path, num_base_packages, num_adv_packages)
    save_metrics_to_file(typomind_metrics, "Typomind", output_path)

    # ossgadget_metrics = evaluate_ossgadget(df, output_path)
    # save_metrics_to_file(ossgadget_metrics, "OSSGadget", output_path)

    # confuguard_output, fp_metrics = evaluate_confuguard(df, str(output_path), openai_api_key)
    # save_metrics_to_file(fp_metrics, "Confuguard", output_path)


    # Example if you wanted to combine metrics:
    benchmark_results = {
        # 'confuguard': fp_metrics.get_metrics_dict(),
        'typomind': typomind_metrics.get_metrics_dict(),
        # 'ossgadget': ossgadget_metrics.get_metrics_dict()
    }
    with open(output_path / 'benchmark_metrics.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Benchmark complete. Results saved to {output_path}")
