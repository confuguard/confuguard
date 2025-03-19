import subprocess
import concurrent.futures
import time
import os
import psutil
from functools import partial
import argparse
import pandas as pd
from pathlib import Path
from itertools import product

# Constants
TIME_LIMIT = 120  # Seconds (2 minutes)
MEMORY_LIMIT_MB = 10 * 1024  # 10GB in MB
MAX_CONCURRENT_JOBS = 32

def enforce_memory_limit():
    """Enforce memory limit using ulimit (Linux/Mac) or psutil (cross-platform)."""
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
    """
    command = [
        "python", "submodules/typomind-release/__main__.py",
        "--base_file", base_file,
        "--adv_file", adv_file,
        "--outfile_path", out_file
    ]
    try:
        enforce_memory_limit()
        result = subprocess.run(
            command,
            timeout=TIME_LIMIT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )

        return (base_file, adv_file, out_file, "SUCCESS", result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (base_file, adv_file, out_file, "TIMEOUT", "", "Process exceeded time limit")
    except Exception as e:
        return (base_file, adv_file, out_file, "ERROR", "", str(e))

def update_pop_pkgs(mapped_ecosystem: str, legitimate_packages: list, num_base_packages: int):
    """
    Load base packages from file and combine with legitimate packages,
    slicing the result to num_base_packages.
    """
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
        return [str(pkg) for pkg in legitimate_packages[:num_base_packages]]
    pop_df = pd.read_csv(popular_file, usecols=['package_name'])
    popular_packages = pop_df['package_name'].tolist()
    # Combine popular and legitimate packages and slice the result
    popular_set = set([str(pkg) for pkg in popular_packages + legitimate_packages])
    return list(popular_set)[:num_base_packages]

def main(tasks):
    """Orchestrates execution of multiple CLI calls with concurrency control."""
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        futures = {executor.submit(run_cli, task[0], task[1], task[2]): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Typomind Detector concurrently for package pairs (per ecosystem)"
    )
    parser.add_argument("--num-adv-packages", type=int, default=None,
                        help="Number of adversarial packages to evaluate per ecosystem (default: all)")
    parser.add_argument("--num-base-packages", type=int, default=None,
                        help="Number of base packages to evaluate per ecosystem (default: all)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parents[3]
    input_csv = project_root / "./datasets/ConfuDB.csv"
    df = pd.read_csv(input_csv)

    # Ecosystem mapping (used for naming and namespace handling)
    ecosystem_map = {
        'gem': 'ruby',
        'npm': 'npm',
        'pypi': 'pypi',
        'maven': 'maven',
        'go': 'golang'
    }

    # Create temporary directory for files
    tmp_dir = script_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    tasks = []
    # Group legitimate packages by ecosystem (used as base packages)
    grouped_legit_packages = df[df['threat_type'] == 'false_positive'].groupby('type')['name'].apply(list).to_dict()
    for ecosystem, legit_packages in grouped_legit_packages.items():
        mapped_ecosystem = ecosystem_map.get(ecosystem.lower(), ecosystem.lower())
        num_base = args.num_base_packages if args.num_base_packages is not None else len(legit_packages)
        base_pkgs = update_pop_pkgs(ecosystem, legit_packages, num_base)
        # Write base packages to a temporary file (one file per ecosystem)
        base_file = tmp_dir / f"tmp_{mapped_ecosystem}_base.txt"
        with open(base_file, "w") as f:
            f.write("\n".join(base_pkgs))

        # Filter adversarial packages for the same ecosystem (handle namespace for npm/maven)
        adv_group = df[df['type'].str.lower() == ecosystem.lower()]
        adv_pkgs = []
        for idx, row in adv_group.iterrows():
            pkg_name = row['name']
            if mapped_ecosystem == 'npm' and pd.notna(row.get('namespace')) and row.get('namespace'):
                pkg_name = f"{row['namespace']}/{pkg_name}"
            elif mapped_ecosystem == 'maven' and pd.notna(row.get('namespace')) and row.get('namespace'):
                pkg_name = f"{row['namespace']}:{pkg_name}"
            adv_pkgs.append(pkg_name)
        adv_pkgs = list(set(adv_pkgs))
        if args.num_adv_packages is not None:
            adv_pkgs = adv_pkgs[:args.num_adv_packages]

        # For each adversarial package in this ecosystem, create a task with its own file
        for i, adv_pkg in enumerate(adv_pkgs):
            adv_file = tmp_dir / f"tmp_{mapped_ecosystem}_adv_{i}.txt"
            with open(adv_file, "w") as f:
                f.write(adv_pkg)
            # Define output file for this task
            out_file = tmp_dir / f"typomind_results_{mapped_ecosystem}_{i}.txt"
            tasks.append((str(base_file), str(adv_file), str(out_file)))

    start_time = time.time()
    output = main(tasks)
    end_time = time.time()
    print(f"Completed {len(output)} tasks in {end_time - start_time:.2f} seconds.")

    # Optionally, log all results
    with open("cli_results.log", "w") as f:
        for entry in output:
            f.write(f"{entry}\n")
