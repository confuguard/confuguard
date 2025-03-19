# Configuration
import os
import re
import json
from dotenv import load_dotenv
from google.cloud import secretmanager

# Load environment variables from .env file
load_dotenv('.env')

ECOSYSTEMS_MAPPING = {
    "npm": "npm",
    "pypi": "pypi.org",
    "maven": "repo1.maven.org",
    "golang": "proxy.golang.org",
    "ruby": "rubygems.org",
    "hf": "huggingface.co",
    "nuget": "nuget"
}

POP_THRESHOLD = {
    "npm": 5000,
    "pypi": 5000,
    "maven": 10,
    "golang": 4,
    "ruby": 5000,
    "hf": 1000,
    "nuget": 5000
}

# Metadata database configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_INSTANCE_NAME = os.getenv("GCP_INSTANCE_NAME")
DB_USER = 'postgres'
DB_PORT = 5433
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# Huggingface database configuration
HF_GCP_PROJECT_ID = os.getenv("HF_GCP_PROJECT_ID")
HF_GCP_REGION = os.getenv("HF_GCP_REGION")
HF_GCP_INSTANCE_NAME = os.getenv("HF_GCP_INSTANCE_NAME")
HF_DB_USER = os.getenv("HF_DB_USER")
HF_DB_PASS = os.getenv("HF_DB_PASS")
HF_DB_NAME = os.getenv("HF_DB_NAME")
HF_DB_PORT = 5434

IS_SQLITE = False
IS_LOCAL = True
IS_OPENAI = False

# Constants
DELIMITERS = ('-', '_', ' ', '.', '~', '@', '/', ':')
DELIMITER_PATTERN = re.compile(f'[{"".join(DELIMITERS)}]+')

TYPOSQUAT_URL = 'http://localhost:5444'

TYPOSQUAT_BEARER_TOKEN = os.getenv("TYPOSQUAT_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TYPOSQUAT_MODELS_BASE_PATH = os.getenv("TYPOSQUAT_MODELS_BASE_PATH")

# Allowlist
with open('./org_allowlist.json') as f:
    NAMESPACE_ALLOWLIST = json.load(f)

USE_INDEXING = False

MODEL_QUANTIZATION_FORMAT = 'float32' # 'float16', 'float32', or 'int8'
if MODEL_QUANTIZATION_FORMAT == 'float32':
    MODEL_PATH = f"{TYPOSQUAT_MODELS_BASE_PATH}/fasttext_all_pkgs.bin"
else:
    MODEL_PATH = f"{TYPOSQUAT_MODELS_BASE_PATH}/fasttext_all_pkgs_{MODEL_QUANTIZATION_FORMAT}.bin"

REGISTRIES = ['npm', 'pypi', 'maven', 'golang', 'ruby', 'nuget'] # This is only used in service initialization; Remove hf in production

