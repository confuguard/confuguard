import pandas as pd
from loguru import logger

DATA_PATH = "typosquat-data/typosquat-lfs/all_pkgs"

def concatenate_package_names(all_ecosystems=False):
    package_names = []
    ecosystems = ['npm', 'pypi', 'ruby', 'maven', 'hf', 'nuget']
    ecosystem_list = []  # Track ecosystems separately from the loop

    logger.debug(f"Processing the {len(ecosystems)} ecosystems: {ecosystems}")

    for idx, ecosystem in enumerate(ecosystems):
        logger.info(f"Processing the {idx + 1}th ecosystem: {ecosystem}")
        file = f"{DATA_PATH}/{ecosystem}_packages.csv"

        try:
            df = pd.read_csv(file)
        except FileNotFoundError:
            logger.error(f"File not found: {file}")
            continue

        if ecosystem == 'maven':
            # For Maven packages, concatenate group_id and artifact_id
            df['package_name'] = df['group_id'] + ':' + df['artifact_id']
        elif ecosystem == 'hf':
            # For HF, package_name should be context_id
            df['package_name'] = df['context_id']

        package_names.extend(df['package_name'].tolist())
        ecosystem_list.extend([ecosystem] * len(df))

        logger.success(f"Processed {file} for ecosystem {ecosystem}")

    ecosystems_df = pd.DataFrame({'package_name': package_names, 'Ecosystem': ecosystem_list})
    ecosystems_df.to_csv('./Part3/ecosystem_packages.csv', index=False)
    logger.info(f"Saved {len(ecosystems_df)} package names with their ecosystems")

if __name__ == "__main__":
    concatenate_package_names()
