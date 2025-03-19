import gensim
from gensim.models import FastText
import re
import os
import numpy as np
import os.path as osp
import pandas as pd
import wordsegment as ws
from tqdm import tqdm
from collections import defaultdict
import psutil
import compress_pickle
import time
from scipy.spatial.distance import cosine
from loguru import logger
import requests
import zipfile
import argparse
import fasttext.util

# ==========================
# Configuration and Setup
# ==========================

# Initialize word segmentation
ws.load()

# Set up logging with rotation to prevent oversized log files
logger.add('embedding_log.txt', rotation='10 MB', level="INFO")

# Define default paths
DEFAULT_DATA_PATH = "typosquat-data/typosquat-lfs"
DEFAULT_PRETRAINED_MODEL_PATH = osp.join(DEFAULT_DATA_PATH, 'cc.en.300.bin') # Original model path
DEFAULT_FASTTEXT_MODEL_PATH = osp.join(DEFAULT_DATA_PATH, 'fasttext_all_pkgs.bin') # This is the output path
DEFAULT_PACKAGES_CSV = 'Part3/train/ecosystem_packages.csv'

# Define delimiters for tokenization
DELIMITERS = ('-', '_', ' ', '.', '~', '@', '/', ':')
DELIMITER_PATTERN = re.compile(f'[{"".join(DELIMITERS)}]+')

# ==========================
# Helper Classes and Functions
# ==========================

class Timer:
    """Context manager for timing code execution."""
    def __enter__(self):
        self._timer = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._timer

def replace_delimiters(target: str, replacement: str) -> str:
    target = target.lower()
    delim_pass = re.sub(DELIMITER_PATTERN, replacement, target)
    num_pass = re.sub(r'([0-9]+)', r' \1 ', delim_pass)
    return num_pass

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = ' '.join(text.split())
    return text

def to_sequence(target: str) -> list:
    preprocessed = replace_delimiters(target, ' ')
    tokens = preprocessed.strip().split()
    return tokens

def get_mem_usage(precision: int = 2) -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1_000_000, precision)

def get_packages(csv_path: str) -> set:
    try:
        df = pd.read_csv(csv_path).dropna()
        if 'package_name' in df.columns:
            packages = set(df['package_name'].astype(str))
        elif 'context_id' in df.columns:
            # For HuggingFace packages
            packages = set(df['context_id'].astype(str))
        elif 'group_id' in df.columns and 'artifact_id' in df.columns:
            # For Maven packages
            packages = set(df['group_id'].astype(str) + ':' + df['artifact_id'].astype(str))
        else:
            logger.error(f"Expected 'package_name' or 'group_id' and 'artifact_id' in CSV columns.")
            raise KeyError(f"Expected 'package_name' or 'group_id' and 'artifact_id' in CSV columns.")
        logger.info(f"Loaded {len(packages)} unique package names from '{csv_path}'.")
        return packages
    except Exception as e:
        logger.error(f"Error reading package data from '{csv_path}': {e}")
        raise

def preprocess_pkg_names(packages: set) -> list:
    logger.info("Preprocessing package names...")
    sentences = [to_sequence(pkg) for pkg in packages]
    logger.info("Package names preprocessed successfully.")
    return sentences

def get_pkg_vector(name: str, model: FastText) -> np.ndarray:
    words = to_sequence(name)
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
        else:
            # Fallback to subword vector
            word_vectors.append(model.wv.get_vector(word))
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        # Use subword embeddings if all words are unknown
        return model.wv.get_vector(name, norm=True)

def compute_similarity(name1: str, name2: str, model: FastText) -> float:
    vec1 = get_pkg_vector(name1, model)
    vec2 = get_pkg_vector(name2, model)
    similarity = 1 - cosine(vec1, vec2)
    return similarity

def check_file_integrity(file_path: str) -> bool:
    if not osp.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False

    if osp.getsize(file_path) == 0:
        logger.warning(f"File is empty: {file_path}")
        return False

    logger.info(f"File exists and is not empty: {file_path}")
    logger.info(f"File size: {osp.getsize(file_path) / (1024 * 1024):.2f} MB")

    return True

def load_or_finetune_model(sentences: list, model_path: str, pretrained_model_path: str, download_url: str = None):
    # Check pre-trained model file integrity
    if not check_file_integrity(pretrained_model_path):
        logger.warning(f"Pre-trained model not found at '{pretrained_model_path}'. Downloading...")
        # Use fasttext.util to download the model
        fasttext.util.download_model('en', if_exists='ignore')
        # Move the downloaded model to the desired location
        downloaded_path = 'cc.en.300.bin'
        if os.path.exists(downloaded_path):
            os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
            os.rename(downloaded_path, pretrained_model_path)
            logger.success(f"Model downloaded and moved to '{pretrained_model_path}'")

    # Load pre-trained FastText model twice: one as original, one to fine-tune
    logger.info(f"Loading pre-trained FastText model from '{pretrained_model_path}'...")
    try:
        original_model = FastText.load_fasttext_format(pretrained_model_path)
        fine_tuned_model = FastText.load_fasttext_format(pretrained_model_path)
        logger.success("Pre-trained FastText models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading pre-trained FastText model from '{pretrained_model_path}': {e}")
        raise

    # Update subword parameters on the fine-tuned model
    fine_tuned_model.min_n = 2
    fine_tuned_model.max_n = 5

    # Update model vocabulary with new words on the fine-tuned model
    logger.info("Updating model vocabulary for fine-tuning...")
    fine_tuned_model.build_vocab(sentences, update=True)
    logger.success("Model vocabulary updated successfully.")

    # Fine-tune the model
    logger.info("Starting FastText model fine-tuning...")
    with Timer() as finetuning_timer:
        fine_tuned_model.train(
            corpus_iterable=sentences,
            total_examples=len(sentences),
            epochs=5,
            report_delay=1
        )
    logger.success(f"FastText model fine-tuning completed in {finetuning_timer.elapsed:.2f} seconds.")

    # Save the fine-tuned model
    logger.info(f"Saving fine-tuned FastText model to '{model_path}'...")
    fine_tuned_model.save(model_path)

    logger.success(f"Fine-tuned FastText model saved successfully at '{model_path}'.")

    return original_model, fine_tuned_model

def main(args):
    data_path = args.data_path
    pretrained_model_path = args.pretrained_model_path or osp.join(data_path, 'cc.en.300.bin')
    fasttext_model_path = args.fasttext_model_path or osp.join(data_path, 'fasttext_all_pkgs.bin')
    test_packages_csv = args.test_packages_csv or DEFAULT_PACKAGES_CSV

    try:
        # Step 1: Load and Preprocess Package Names
        logger.info(f"Loading package names from '{test_packages_csv}'...")
        packages = get_packages(test_packages_csv)
        preprocessed_packages = set(preprocess_text(pkg) for pkg in packages)
        logger.info(f"Loaded and preprocessed {len(preprocessed_packages)} unique package names.")

        # Step 2: Preprocess Packages into Sentences
        sentences = preprocess_pkg_names(preprocessed_packages)

        # Step 3: Load or Fine-Tune FastText Model
        original_model, fine_tuned_model = load_or_finetune_model(sentences, fasttext_model_path, pretrained_model_path, args.download_url)

        # Step 4: Define Test Cases for Similarity Computation
        test_cases = [
            ('facebook', 'meta'),
            ('Facebook/SDK', 'meta-sdk'),
            ('facebook-sdk', 'meta-sdk'),
            ('facebook-sdk', 'facebook-sdk'),
            ('facebook-sdk', 'meta-sdk'),
            ('facebook-sdk', 'facebook-sdk-test'),
            ('facebook-sdk', 'meta-sdk-test'),
            ('facebook-sdk', 'facebook-sdk-test-test'),
            ('facebook-sdk', 'meta-sdk-test-test'),
            # Additional test cases
            ('Facebook', 'facebook'),
            ('FACEBOOK', 'facebook'),
            ('face/book', 'facebook'),
            ('face.book', 'facebook'),
            ('face_book', 'facebook'),
            ('react-native', 'react_native'),
            ('ReactNative', 'react-native'),
            ('python-requests', 'requests'),
            ('numpy', 'scikit-learn'),
            ('matplotlib', 'matplotlib-pyplot'),
            ('matplotlib', 'matblotlip'),
            ('matplotlib', 'catplotlib'),
            ('numpy', 'scipy')
        ]

        # Compute similarities with the original (pre-trained) model
        logger.info("Computing similarities with the ORIGINAL pre-trained model...")
        start_time = time.time()
        for pkg_name1, pkg_name2 in test_cases:
            try:
                sim_original = compute_similarity(pkg_name1, pkg_name2, original_model)
                print(f"[Original] Similarity between '{pkg_name1}' and '{pkg_name2}': {sim_original:.4f}")
            except Exception as e:
                logger.error(f"Error computing similarity (original) for '{pkg_name1}' and '{pkg_name2}': {e}")
        elapsed_time = time.time() - start_time
        print(f"Time taken to compute similarities (original): {elapsed_time:.2f} seconds")

        # Compute similarities with the fine-tuned model
        logger.info("Computing similarities with the FINE-TUNED model...")
        start_time = time.time()
        for pkg_name1, pkg_name2 in test_cases:
            try:
                sim_finetuned = compute_similarity(pkg_name1, pkg_name2, fine_tuned_model)
                print(f"[Fine-tuned] Similarity between '{pkg_name1}' and '{pkg_name2}': {sim_finetuned:.4f}")
            except Exception as e:
                logger.error(f"Error computing similarity (fine-tuned) for '{pkg_name1}' and '{pkg_name2}': {e}")
        elapsed_time = time.time() - start_time
        print(f"Time taken to compute similarities (fine-tuned): {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        logger.exception("Traceback:")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FastText Model for Package Names Similarity')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to the data directory')
    parser.add_argument('--pretrained_model_path', type=str, default=DEFAULT_PRETRAINED_MODEL_PATH,
                        help='Path to the pre-trained word vectors')
    parser.add_argument('--fasttext_model_path', type=str, default=DEFAULT_FASTTEXT_MODEL_PATH,
                        help='Path to save/load the FastText model')
    parser.add_argument('--test_packages_csv', type=str, default=DEFAULT_PACKAGES_CSV,
                        help='Path to the CSV file containing package names')
    parser.add_argument('--download_url', type=str, default=None,
                        help='URL to download the pre-trained model if not present')
    args = parser.parse_args()

    main(args)
