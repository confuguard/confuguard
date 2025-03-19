import os
import pandas as pd
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Registry mapping
REGISTRY_MAPPING = {
    'gem': 'ruby',
    'pypi': 'pypi',
    'npm': 'npm',
    'maven': 'maven',
    'go': 'golang'
}

def call_detection_service(package_name, registry):
    """Call the typosquat detection service."""
    url = "http://localhost:5444/detect"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TYPOSQUAT_BEARER_TOKEN')}"
    }

    # Map the registry to the correct format
    mapped_registry = REGISTRY_MAPPING.get(registry, registry)

    data = {
        "package_name": package_name,
        "registry": mapped_registry
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error calling detection service for {package_name} ({mapped_registry}): {e}")
        return None

def process_package(row):
    """Process a single package from the threat feed."""
    try:
        artifact_data = json.loads(row['artifact'])
        registry = artifact_data['type'].lower()

        # Format package name according to registry
        if registry == 'maven':
            namespace = artifact_data.get('namespace', '')
            name = artifact_data.get('name', '')
            if not namespace or not name:
                logger.warning(f"Missing namespace or name for Maven package: {artifact_data}")
                return None
            package_name = f"{namespace}:{name}"
        else:
            package_name = artifact_data['name']

        # Skip unsupported registries
        if registry not in REGISTRY_MAPPING:
            logger.warning(f"Unsupported registry {registry} for package {package_name}")
            return None

        # Call detection service
        result = call_detection_service(package_name, registry)

        detection_status = "FP"  # Default to False Positive
        typo_category = None
        missing_metadata = False

        if result:
            if result.get("metadata_missing"):
                detection_status = "TP (Missing Metadata)"
                missing_metadata = True
                if result.get("typo_results"):
                    typo_category = result["typo_results"][0].get("typo_category")
            elif result.get("typo_results"):
                detection_status = "TP"
                typo_category = result["typo_results"][0].get("typo_category")

        return {
            'package_name': package_name,
            'registry': registry,
            'detection_status': detection_status,
            'typo_category': typo_category,
            'missing_metadata': missing_metadata,
            'original_confidence': row.get('confidence'),
            'needs_review': row.get('needsHumanReview'),
            'risk_level': row.get('riskLevel'),
            'is_false_positive': row.get('is_false_positive', False)
        }
    except Exception as e:
        logger.error(f"Error processing row: {e}")
        logger.exception("Full traceback:")  # Add full stack trace for debugging
        return None

def create_visualizations(results_df, output_dir='evaluation_results'):
    """Create and save visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True

    # Split datasets
    fp_dataset = results_df[results_df['is_false_positive']]
    tp_dataset = results_df[~results_df['is_false_positive']]

    # Create visualizations for each dataset
    for dataset, name in [(fp_dataset, 'false_positives'), (tp_dataset, 'true_positives')]:
        # Detection Status Distribution
        plt.figure(figsize=(8, 6))
        status_counts = dataset['detection_status'].value_counts()
        colors = ['green' if x == 'FP' else 'red' for x in status_counts.index]
        sns.barplot(x=status_counts.index, y=status_counts.values, palette=colors)
        plt.title(f'Detection Status Distribution ({name.replace("_", " ").title()} Dataset)')
        plt.xlabel('Detection Status')
        plt.ylabel('Count')
        for i, v in enumerate(status_counts):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{name}_analysis.png')
        plt.close()

        # Registry Distribution (if needed)
        plt.figure(figsize=(8, 6))
        registry_counts = dataset['registry'].value_counts()
        sns.barplot(x=registry_counts.index, y=registry_counts.values)
        plt.title(f'Distribution by Registry ({name.replace("_", " ").title()} Dataset)')
        plt.xticks(rotation=45)
        for i, v in enumerate(registry_counts):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{name}_registry_distribution.png')
        plt.close()

    # Save summary statistics
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        # False Positives Dataset Statistics
        f.write("False Positives Dataset Analysis\n")
        f.write("==============================\n\n")
        total_fp = len(fp_dataset)
        true_negatives = len(fp_dataset[fp_dataset['detection_status'] == 'FP'])
        false_positives = len(fp_dataset[fp_dataset['detection_status'].str.startswith('TP')])
        
        specificity = true_negatives / total_fp if total_fp > 0 else 0
        fp_rate = false_positives / total_fp if total_fp > 0 else 0

        f.write(f"Total packages analyzed: {total_fp}\n")
        f.write(f"True Negatives (correct non-typosquat identification): {true_negatives}\n")
        f.write(f"False Positives (incorrect typosquat flags): {false_positives}\n")
        f.write(f"Specificity (True Negative Rate): {specificity:.2%}\n")
        f.write(f"False Positive Rate: {fp_rate:.2%}\n\n")

        # True Positives Dataset Statistics
        f.write("\nTrue Positives Dataset Analysis\n")
        f.write("============================\n\n")
        total_tp = len(tp_dataset)
        true_positives = len(tp_dataset[tp_dataset['detection_status'].str.startswith('TP')])
        false_negatives = len(tp_dataset[tp_dataset['detection_status'] == 'FP'])
        
        sensitivity = true_positives / total_tp if total_tp > 0 else 0
        fn_rate = false_negatives / total_tp if total_tp > 0 else 0

        f.write(f"Total packages analyzed: {total_tp}\n")
        f.write(f"True Positives (correct typosquat identification): {true_positives}\n")
        f.write(f"False Negatives (missed typosquats): {false_negatives}\n")
        f.write(f"Sensitivity (True Positive Rate): {sensitivity:.2%}\n")
        f.write(f"False Negative Rate: {fn_rate:.2%}\n\n")

        # Combined Performance Metrics
        f.write("\nCombined Performance Metrics\n")
        f.write("==========================\n\n")
        total = len(results_df)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = sensitivity  # Same as sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.2%}\n")
        f.write(f"Recall: {recall:.2%}\n")
        f.write(f"F1 Score: {f1:.2%}\n")

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', action='store_true', help='Restart analysis from scratch')
    args = parser.parse_args()

    # Check for API token
    if not os.getenv('TYPOSQUAT_BEARER_TOKEN'):
        logger.error("TYPOSQUAT_BEARER_TOKEN environment variable not set")
        exit(1)

    # Find most recent results directory
    base_dir = './datasets/NeupaneDB_real_malware/data'
    existing_results = None
    processed_packages = set()  # Create a set to track processed packages
    
    if not args.restart:
        eval_dirs = [d for d in os.listdir(base_dir) if d.startswith('evaluation_results_')]
        if eval_dirs:
            latest_dir = max(eval_dirs)
            results_file = f'{base_dir}/{latest_dir}/threat_feed_evaluation_results.csv'
            if os.path.exists(results_file):
                existing_results = pd.read_csv(results_file)
                # Extract processed package names, considering both normal and Maven format
                processed_packages = set(existing_results['package_name'])
                logger.info(f"Resuming from existing results in {latest_dir}")

    # Read only false positives dataset first
    fp_df = pd.read_csv('./datasets/ConfuDB.csv')
    fp_df['is_false_positive'] = True
    combined_df = fp_df  # Only use false positives dataset

    # Filter out already processed packages more robustly
    def get_package_id(artifact_str):
        try:
            artifact = json.loads(artifact_str)
            registry = artifact['type'].lower()
            if registry == 'maven':
                namespace = artifact.get('namespace', '')
                name = artifact.get('name', '')
                return f"{namespace}:{name}" if namespace and name else None
            return artifact['name']
        except:
            return None

    if existing_results is not None:
        # Filter using the package ID function
        combined_df['package_id'] = combined_df['artifact'].apply(get_package_id)
        combined_df = combined_df[~combined_df['package_id'].isin(processed_packages)]
        combined_df = combined_df.drop('package_id', axis=1)  # Remove temporary column
        results = existing_results.to_dict('records')
        logger.info(f"Found {len(results)} existing results, {len(combined_df)} packages remaining")
    else:
        results = []

    total_packages = len(combined_df)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_package, row): row for _, row in combined_df.iterrows()}

        with tqdm(total=total_packages, desc="Processing packages") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Now process the npm true positives dataset
    tp_df = pd.read_csv('./datasets/NeupaneDB_real_malware.csv')
    tp_results = []
    
    for _, row in tp_df.iterrows():
        package_name = row['Package Name']
        result = call_detection_service(package_name, 'npm')
        
        detection_status = "FP"  # Default to False Positive
        typo_category = None
        missing_metadata = False

        if result:
            if result.get("metadata_missing"):
                detection_status = "TP (Missing Metadata)"
                missing_metadata = True
                if result.get("typo_results"):
                    typo_category = result["typo_results"][0].get("typo_category")
            elif result.get("typo_results"):
                detection_status = "TP"
                typo_category = result["typo_results"][0].get("typo_category")

        tp_results.append({
            'package_name': package_name,
            'registry': 'npm',
            'detection_status': detection_status,
            'typo_category': typo_category,
            'missing_metadata': missing_metadata,
            'is_false_positive': False
        })

    # Combine results
    tp_results_df = pd.DataFrame(tp_results)
    results_df = pd.concat([results_df, tp_results_df], ignore_index=True)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./evaluation_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/threat_feed_evaluation_results.csv', index=False)

    # Create visualizations and print results
    create_visualizations(results_df, output_dir)

    # Split datasets for printing results
    fp_dataset = results_df[results_df['is_false_positive']]
    tp_dataset = results_df[~results_df['is_false_positive']]

    # Calculate and print metrics for False Positives dataset
    print("\nFalse Positives Dataset Evaluation Results:")
    print("=========================================")
    total_fp = len(fp_dataset)
    true_negatives = len(fp_dataset[fp_dataset['detection_status'] == 'FP'])
    false_positives = len(fp_dataset[fp_dataset['detection_status'].str.startswith('TP')])
    
    specificity = true_negatives / total_fp if total_fp > 0 else 0
    fp_rate = false_positives / total_fp if total_fp > 0 else 0

    print(f"Total packages processed: {total_fp}")
    print(f"True Negatives (correctly identified non-typosquats): {true_negatives}")
    print(f"False Positives (incorrectly flagged as typosquats): {false_positives}")
    print(f"Specificity (True Negative Rate): {specificity:.2%}")
    print(f"False Positive Rate: {fp_rate:.2%}")

    # Calculate and print metrics for True Positives dataset
    print("\nTrue Positives Dataset Evaluation Results:")
    print("=======================================")
    total_tp = len(tp_dataset)
    true_positives = len(tp_dataset[tp_dataset['detection_status'].str.startswith('TP')])
    false_negatives = len(tp_dataset[tp_dataset['detection_status'] == 'FP'])
    
    sensitivity = true_positives / total_tp if total_tp > 0 else 0
    fn_rate = false_negatives / total_tp if total_tp > 0 else 0

    print(f"Total packages processed: {total_tp}")
    print(f"True Positives (correct typosquat identification): {true_positives}")
    print(f"False Negatives (missed typosquats): {false_negatives}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2%}")
    print(f"False Negative Rate: {fn_rate:.2%}")

    # Print Combined Metrics
    print("\nCombined Performance Metrics:")
    print("===========================")
    total = len(results_df)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = sensitivity  # Same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    print(f"\nDetailed results and visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    main()
