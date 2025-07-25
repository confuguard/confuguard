# ConfuGuard Artifacts

> This is the artifact for the ICSE'26 submission: `ConfuGuard: Using Metadata to Detect Active and Stealthy Package Confusion Attacks Accurately and at Scale`.

**Update**: We made some changes to the artifact as part of the ICSE'26 Revision.
- We fixed some refactoring issues in the artifact. (Thanks to Reviewer A!)
- We added the missing `org_allowlist.json` file to the artifact.
- We added the `prompt_v1.py` and `prompt_v2.py` scripts to the artifact.
- We added the required external resources and extra setup instructions to the artifact.

## Table of Contents

| Item | Description | Corresponding content in the paper | Path |
|------|-------------|---------------------| ---------|
| Empirical Analysis | The empirical analysis scripts and data | $4, Figure 2, Table3| [attack_analysis](./attack_analysis) |
| Implementation of `Confuguard` | The implementation of ConfuGuard, including 6 parts | $5, Figure 3| [confuguard](./confuguard) |
| Prompt | The prompt we used for Step 5 Benignity Check | Listing 1| [prompt](./confuguard/Part5/prompt_v1.py) |
| Dataset | We provide the dataset we used, including `NeupaneDB` (1840 packages) and `ConfuDB` (1561 packages)| $6.1| [datasets](./datasets) |
| Evaluation | The evaluation scripts and results | $6.2-$6.6, Table 4-6, Figure 4-5| [eval](./eval) |





- [Installation](#installation)
- [Legit Packages Update](#legit-packages-update)
- [Update Popular Packages](#update-popular-packages)
- [HTTP Service](#http-service)
  - [Start the Service](#start-the-service)
  - [Functionalities](#functionalities)
- [Example Requests](#example-requests)
  - [Detect](#detect)
  - [Get Neighbors](#get-neighbors)
  - [Add Package](#add-package)
  - [Similarity](#similarity)
- [Example Outputs](#example-outputs)
- [Supported Registries](#supported-registries)

# Installation

```
pip install -r requirements.txt
```

## External Dependencies Required

This project requires the following external setup:

**OpenAI API:**
- Set `OPENAI_API_KEY` environment variable with your API key

**Google Cloud PostgreSQL (for cloud deployment):**
- Set `GCP_PROJECT_ID` environment variable
- Set `GCP_REGION` environment variable  
- Set `GCP_INSTANCE_NAME` environment variable
- Set `DB_USER` environment variable
- Set `DB_PASS` environment variable
- Set `DB_NAME` environment variable

**Bearer Token:**
- Set `TYPOSQUAT_BEARER_TOKEN` environment variable for API authentication

**Package Metadata Database:**
- As noted in the paper, ConfuGuard uses the package metadata database to get the package metadata. The metadata database should be installed and running before running the ConfuGuard.
- ConfuGuard implementation supports two kinds of metadata databases:
  - Google Cloud PostgreSQL Database (for cloud deployment)
  - Local PostgreSQL with pgvector extension (for local testing)
  - Local SQLite database (for local testing)



# Run the full script

The full script includes the following steps:

1. Legit Packages Update
2. Update Popular Packages
3. Start HTTP Service

All these steps are run in the `entrypoint.sh` script so you can just run it by running the following command:

```
./entrypoint.sh
```

# Run or test the individual steps locally

## 1. Legit Packages Update

Run the `get_legit_packages.py` script.

```
python confuguard/Part2/get_legit_packages.py
```

This will save all the legit packages from each ecosystem into `legit_packages/{ecosystem}_legit_packages.json` file.

```
python confuguard/Part2/get_legit_packages.py --push_to_postgres
```

This will push all the legit packages from each ecosystem into postgres table named `{ecosystem}_pop_packages`.

## 2. Update Popular Packages

```
python confuguard/Part2/update_pop_pkgs.py
```

This will update the popular packages in the database in a weekly basis.

## 3. HTTP Service

### Start

```
python confuguard/app.py
```

## Functionalities

Endpoints:

- **`/detect`**: Detects confusing packages by comparing a package against its neighbors. **NOTE: This is the main endpoint that is used in the pipeline. The rest are mainly for testing purposes.**
- `/get_neighbors`: Retrieves neighboring packages based on vector and name-based similarity.
- `/add_package`: Adds or updates package details in the PostgreSQL database.
- `/similarity`: Computes cosine similarity between two package embeddings.

# Example Requests

## Detect

```
curl -X POST http://localhost:5444/detect \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $BEARER_TOKEN" \
-d '{"package_name": "matplotlip", "registry": "pypi"}'
```

## Get Neighbors

```
curl -X POST http://localhost:5444/get_neighbors \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $BEARER_TOKEN" \
-d '{"package_name": "dotenv", "registry": "pypi"}'
```

## Add Package

```
curl -X POST http://localhost:5444/add_package \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $BEARER_TOKEN" \
-d '{"package_name": "lodash", "registry": "npm"}'
```

## Similarity

```
curl -X POST http://localhost:5444/similarity \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $BEARER_TOKEN" \
-d '{"package_name1": "matplotlib", "package_name2": "catplotlib", "registry": "pypi"}'
```

# Example Outputs

```
{
  typo_results: [
    {
      explanation: "The package name 'matplobblib' is very similar to 'matplotlib', with minor character changes that could confuse users. The description of 'matplobblib' is vague and does not indicate a distinct purpose. The maintainer 'Ackrome' is not recognized as a known maintainer in the community. These factors suggest it could be a package confusion attack.",
      metadata_missing: false,
      package_name: 'matplotlib',
      typo_category: '1-step D-L dist'
    }
  ]
}
```

# Supported registries names:

```
{
  "npm",
  "pypi",
  "ruby",
  "maven",
  "golang",
  "hf"
}
```

