import csv
import os
from collections import defaultdict

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read the threatfeed data and extract legitimate package names
def extract_legitimate_packages(input_file):
    packages_by_type = defaultdict(set)

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['legitimate_package'] and row['legitimate_package'] != 'None':
                package_type = row['type']
                packages_by_type[package_type].add(row['legitimate_package'])

    return packages_by_type

# Extract legitimate packages from FP_ground_truth.csv
def extract_from_fp_ground_truth(input_file):
    packages_by_type = defaultdict(set)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Original pkg' in row and row['Original pkg'] and 'Ecosystem' in row:
                    package_type = row['Ecosystem']
                    packages_by_type[package_type].add(row['Original pkg'])
    except FileNotFoundError:
        print(f"Warning: File {input_file} not found. Skipping.")

    return packages_by_type

# Extract legitimate packages from dataset.csv
def extract_from_dataset(input_file):
    packages_by_type = defaultdict(set)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if ('legitimate_pkg' in row and row['legitimate_pkg'] and 'registry' in row and
                    'confusion' in row and row['confusion'] != 'UNK'):
                    package_type = row['registry']
                    packages_by_type[package_type].add(row['legitimate_pkg'])
    except FileNotFoundError:
        print(f"Warning: File {input_file} not found. Skipping.")

    return packages_by_type

# Write legitimate packages to their respective files
def write_legitimate_packages(packages_by_type, output_dir):
    ensure_dir(output_dir)

    # Map ecosystem names to file prefixes
    type_mapping = {
        'npm': 'npm',
        'pypi': 'pypi',
        'pip': 'pypi',
        'maven': 'maven',
        'golang': 'golang',
        'gem': 'ruby',
        'rubygems': 'ruby'
    }

    # Reorganize packages by normalized type
    normalized_packages = defaultdict(set)
    for package_type, packages in packages_by_type.items():
        normalized_type = type_mapping.get(package_type.lower(), package_type.lower())
        normalized_packages[normalized_type].update(packages)

    for normalized_type, packages in normalized_packages.items():
        legit_packages_dir = "./legit_packages"
        if os.path.exists(legit_packages_dir):
            output_file = os.path.join(legit_packages_dir, f"{normalized_type}_legit_packages.csv")
        else:
            output_file = os.path.join(output_dir, f"{normalized_type}_legit_packages.csv")

        # Read existing packages if file exists
        existing_packages = {}
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'package_name' in row:
                        existing_packages[row['package_name']] = row.get('popularity', '0')

        # Identify new packages
        new_packages = set(packages) - set(existing_packages.keys())

        # Prepare data for writing
        all_packages_data = []

        # Add new packages at the top with appropriate popularity value
        for package in sorted(new_packages):
            # Use 0.1 for maven and golang, 9999999 for others
            popularity = '0.1' if normalized_type in ['maven', 'golang'] else '9999999'
            all_packages_data.append({'package_name': package, 'popularity': popularity})

        # Add existing packages
        for package, popularity in existing_packages.items():
            all_packages_data.append({'package_name': package, 'popularity': popularity})

        # Write all packages to file
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['package_name', 'popularity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for data in all_packages_data:
                writer.writerow(data)

        print(f"Added {len(new_packages)} new legitimate packages to {output_file}")
        print(f"Total packages in {output_file}: {len(all_packages_data)}")

def main():
    threatfeed_file = "./datasets/NeupaneDB_real_malware/data/threatfeed-data-processed.csv"
    fp_ground_truth_file = "./datasets/ConfuDB.csv"
    dataset_file = "./datasets/NeupaneDB_real_malware.csv"
    output_dir = "./legit_packages"

    # Extract packages from all sources
    packages_by_type_threatfeed = extract_legitimate_packages(threatfeed_file)
    packages_by_type_fp = extract_from_fp_ground_truth(fp_ground_truth_file)
    packages_by_type_dataset = extract_from_dataset(dataset_file)

    # Combine all packages
    all_packages_by_type = defaultdict(set)

    # Add packages from threatfeed
    for pkg_type, packages in packages_by_type_threatfeed.items():
        all_packages_by_type[pkg_type].update(packages)

    # Add packages from FP ground truth
    for pkg_type, packages in packages_by_type_fp.items():
        all_packages_by_type[pkg_type].update(packages)

    # Add packages from dataset
    for pkg_type, packages in packages_by_type_dataset.items():
        all_packages_by_type[pkg_type].update(packages)

    # Write all packages to their respective files
    write_legitimate_packages(all_packages_by_type, output_dir)

    # Print summary of normalized packages
    type_mapping = {
        'npm': 'npm',
        'pypi': 'pypi',
        'pip': 'pypi',
        'maven': 'maven',
        'golang': 'golang',
        'gem': 'ruby',
        'rubygems': 'ruby'
    }

    normalized_packages = defaultdict(set)
    for package_type, packages in all_packages_by_type.items():
        normalized_type = type_mapping.get(package_type.lower(), package_type.lower())
        normalized_packages[normalized_type].update(packages)

    print("\nSummary of all sources (normalized):")
    for normalized_type, packages in normalized_packages.items():
        print(f"{normalized_type}: {len(packages)} legitimate packages")

if __name__ == "__main__":
    main()
