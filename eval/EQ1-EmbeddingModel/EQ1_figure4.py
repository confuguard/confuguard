import os
import sys
import csv
import random
import numpy as np
from typing import List, Tuple, Set
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import re
from contextlib import redirect_stdout
import time
from collections import defaultdict
import psutil
import copy
from tqdm import tqdm

from loguru import logger

# For plotting
import matplotlib.pyplot as plt

# Gensim-based imports
from gensim.models import FastText

# Get the absolute path to the project root (3 levels up from current file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from scipy.spatial.distance import cosine
from typosquat.create_embedding import Preprocessor

# 1. Config
DATASET_CSV = "data/NeupaneDB_real_malware.csv"         
ALL_PACKAGES_FILE = "all_packages.txt"

FINE_TUNED_MODEL_PATH = "fasttext_all_pkgs.bin"
OUTPUT_DIR = "./"
EDIT_DISTANCE_THRESHOLDS = [2, 3, 4]
EMBEDDING_SIM_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
NUM_NEGATIVE_SAMPLES = 1239

DELIMITERS = ('-', '_', ' ', '.', '~', '@', '/', ':')
DELIMITER_PATTERN = re.compile(f'[{"".join(DELIMITERS)}]+')
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# 2. Tokenization & Similarity
# ---------------------------------------------------------------------
def replace_delimiters(target: str, replacement: str) -> str:
    """Lowercase, replace delimiters, then separate out digits."""
    target = target.lower()
    delim_pass = re.sub(DELIMITER_PATTERN, replacement, target)
    num_pass = re.sub(r'([0-9]+)', r' \1 ', delim_pass)
    return num_pass

def to_sequence(target: str) -> List[str]:
    """Tokenize exactly like your new code, returning a list of tokens."""
    preprocessed = replace_delimiters(target, ' ')
    tokens = preprocessed.strip().split()
    return tokens

def get_pkg_vector(name: str, model: FastText) -> np.ndarray:
    """
    Get package vector using the same Preprocessor as the training code.
    This uses main and subword embeddings as appropriate.
    """
    preprocessor = Preprocessor(model)
    return preprocessor.get_embedding(name)

def embedding_similarity(model: FastText, pkgA: str, pkgB: str) -> float:
    """
    Compute similarity using cosine.
    If the model was int8-quantized, apply scale factors before normalizing.
    """
    vecA = get_pkg_vector(pkgA, model)
    vecB = get_pkg_vector(pkgB, model)
    
    # Dequantize if needed
    if hasattr(model, 'scale'):
        vecA = vecA * model.scale
        vecB = vecB * model.scale
    # If subword is scaled separately:
    if hasattr(model, 'scale_syn'):
        vecA = vecA * model.scale_syn
        vecB = vecB * model.scale_syn

    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0.0 or normB == 0.0:
        return 0.0
    
    sim = np.dot(vecA, vecB) / (normA * normB)
    return float(np.clip(sim, -1.0, 1.0))

def edit_distance_similarity(a: str, b: str) -> float:
    """Normalized Damerau-Levenshtein distance to similarity in [0..1]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return float(a == b)
    dist = damerau_levenshtein_distance(a, b)
    return 1.0 - dist / max_len

def jaccard_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity of token sets."""
    tokens_a = set(to_sequence(a))
    tokens_b = set(to_sequence(b))
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union) if union else 0.0

def hybrid_similarity(pkgA: str, pkgB: str, model: FastText, weights=(0.5, 0.5)) -> float:
    """
    Weighted combination of edit-distance-sim and embedding-sim.
    Extend or adapt if you want a third metric like jaccard.
    """
    edit_sim = edit_distance_similarity(pkgA, pkgB)
    embed_sim = embedding_similarity(model, pkgA, pkgB)
    return weights[0] * edit_sim + weights[1] * embed_sim

def classify_hybrid(pairs: List[Tuple[str, str]],
                    model: FastText,
                    threshold=0.7,
                    weights=(0.5, 0.5)) -> List[int]:
    """Hybrid classification using 2 metrics (extend if you have more)."""
    y_pred = []
    for (a, b) in pairs:
        total_sim = weights[0]*edit_distance_similarity(a,b) \
                    + weights[1]*embedding_similarity(model,a,b)
        y_pred.append(1 if total_sim >= threshold else 0)
    return y_pred

# ---------------------------------------------------------------------
# 3. Data Loading
# ---------------------------------------------------------------------
def load_positive_pairs_from_dataset(csv_file: str) -> List[Tuple[str, str]]:
    """
    Loads (typosquat_pkg, legitimate_pkg) from dataset.csv,
    skipping rows with missing fields or hallucinated==true, etc.
    """
    pairs = []
    if not os.path.isfile(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        sys.exit(1)
    logger.info(f"Loading positive pairs from {csv_file}")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_pkg = row.get("typosquat_pkg", "").strip()
            l_pkg = row.get("legitimate_pkg", "").strip()
            source = row.get("source", "").strip()
            halluc = row.get("hallucinated", "false").lower().strip()
            if not t_pkg or not l_pkg or not source:
                continue
            if halluc == "true":
                continue
            pairs.append((t_pkg, l_pkg))
    return pairs

def load_all_packages(txt_file: str) -> Set[str]:
    """Load all known package names from a text file, one per line."""
    packages = set()
    if not os.path.isfile(txt_file):
        print(f"[ERROR] File not found: {txt_file}")
        sys.exit(1)

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            pkg = line.strip()
            if pkg:
                packages.add(pkg)
    return packages

def generate_negative_pairs(
    all_pkgs: Set[str],
    pos_pairs: List[Tuple[str, str]],
    limit=NUM_NEGATIVE_SAMPLES
) -> List[Tuple[str, str]]:
    """
    Randomly pick package pairs that are not in pos_pairs
    and ensure they have edit-distance-sim < ~0.3 to be "true negatives."
    """
    random.seed(RANDOM_SEED)
    negative_pairs = []
    positive_set = set(pos_pairs) | set((b,a) for (a,b) in pos_pairs)

    all_pkgs_list = list(all_pkgs)
    random.shuffle(all_pkgs_list)

    trials = 0
    max_trials = 50000

    while len(negative_pairs) < limit and trials < max_trials:
        trials += 1
        pkgA = random.choice(all_pkgs_list)
        pkgB = random.choice(all_pkgs_list)
        if pkgA == pkgB:
            continue
        if (pkgA, pkgB) in positive_set:
            continue

        ed_sim = edit_distance_similarity(pkgA, pkgB)
        if ed_sim > 0.2:
            continue

        negative_pairs.append((pkgA, pkgB))

    if len(negative_pairs) < limit:
        print(f"WARNING: only generated {len(negative_pairs)} negative pairs.")
    return negative_pairs

# ---------------------------------------------------------------------
# 4. Classification & Evaluation
# ---------------------------------------------------------------------
def classify_edit_distance(pairs: List[Tuple[str, str]], threshold=2) -> List[int]:
    """Predict 1 if edit distance <= threshold, else 0."""
    y_pred = []
    for (a, b) in pairs:
        dist = damerau_levenshtein_distance(a, b)
        y_pred.append(1 if dist <= threshold else 0)
    return y_pred

def classify_embedding(pairs: List[Tuple[str, str]], model: FastText, threshold=0.5) -> List[int]:
    """Predict 1 if embedding_similarity >= threshold, else 0."""
    y_pred = []
    for a, b in tqdm(pairs, desc="Computing similarities"):
        sim = embedding_similarity(model, a, b)
        y_pred.append(1 if sim >= threshold else 0)
    return y_pred

def eval_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, F1 using sklearn, ignoring zero_division warnings."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    return prec, rec, f1_

def eval_metrics_by_class(y_true: List[int], y_pred: List[int], pos_len: int) -> Tuple[dict, dict]:
    """
    Compute metrics separately for positive (first pos_len) and negative (rest).
    For negative pairs, we invert the predictions to measure correct identification of "non-typosquats."
    """
    y_pred_pos = y_pred[:pos_len]
    y_pred_neg = y_pred[pos_len:]
    y_true_pos = y_true[:pos_len]
    y_true_neg = y_true[pos_len:]
    
    pos_metrics = {
        'precision': precision_score(y_true_pos, y_pred_pos, zero_division=0),
        'recall': recall_score(y_true_pos, y_pred_pos, zero_division=0),
        'f1': f1_score(y_true_pos, y_pred_pos, zero_division=0)
    }
    
    # Invert negative labels
    y_true_neg_inv = [1 if y == 0 else 0 for y in y_true_neg]
    y_pred_neg_inv = [1 if y == 0 else 0 for y in y_pred_neg]
    
    neg_metrics = {
        'precision': precision_score(y_true_neg_inv, y_pred_neg_inv, zero_division=0),
        'recall': recall_score(y_true_neg_inv, y_pred_neg_inv, zero_division=0),
        'f1': f1_score(y_true_neg_inv, y_pred_neg_inv, zero_division=0)
    }
    
    return pos_metrics, neg_metrics

# ---------------------------------------------------------------------
# 5. Model Quantization
# ---------------------------------------------------------------------
def quantize_model_float16(model: FastText) -> FastText:
    """Simulate float16 quantization by downcasting vectors to float16."""
    quantized_model = copy.deepcopy(model)
    quantized_model.wv.vectors = quantized_model.wv.vectors.astype(np.float16)
    if hasattr(quantized_model.wv, 'syn1neg'):
        quantized_model.wv.syn1neg = quantized_model.wv.syn1neg.astype(np.float16)
    print("[INFO] Model quantized to float16.")
    return quantized_model

def quantize_model_int8(model: FastText) -> FastText:
    """
    Quantize model to int8 format. We store a scale for dequantization.
    If you also want subword vectors scaled, store scale_syn similarly.
    """
    quantized_model = copy.deepcopy(model)
    
    # Scale factor for main vectors
    scale = np.max(np.abs(quantized_model.wv.vectors)) / 127
    quantized_model.wv.vectors = np.round(quantized_model.wv.vectors / scale).astype(np.int8)
    quantized_model.scale = scale  # store for dequantization
    
    # Subword vectors
    if hasattr(quantized_model.wv, 'syn1neg'):
        scale_syn = np.max(np.abs(quantized_model.wv.syn1neg)) / 127
        quantized_model.wv.syn1neg = np.round(quantized_model.wv.syn1neg / scale_syn).astype(np.int8)
        quantized_model.scale_syn = scale_syn

    print("[INFO] Model quantized to int8.")
    return quantized_model

def measure_performance(model: FastText, pairs: List[Tuple[str, str]], num_runs: int = 3) -> Tuple[float, float]:
    """Measures throughput (pairs/sec) and avg latency (ms/pair) with multiple runs."""
    throughputs = []
    latencies = []
    
    for _ in range(num_runs):
        start_time = time.time()
        for a, b in pairs:
            _ = embedding_similarity(model, a, b)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughputs.append(len(pairs) / total_time if total_time > 0 else float('inf'))
        latencies.append((total_time / len(pairs)) * 1000 if len(pairs) > 0 else 0.0)
    
    return np.mean(throughputs), np.mean(latencies)

def get_model_memory_usage(model: FastText) -> float:
    """Approximate memory usage of model in MB."""
    # Consider adding other model components
    total_bytes = 0
    total_bytes += model.wv.vectors.nbytes
    if hasattr(model.wv, 'syn1neg'):
        total_bytes += model.wv.syn1neg.nbytes
    # Add vocabulary size estimation
    if hasattr(model.wv, 'key_to_index'):
        total_bytes += len(model.wv.key_to_index) * 42  # Approximate string overhead
    return total_bytes / (1024 * 1024)

# ---------------------------------------------------------------------
# 6. Main Table Printing
# ---------------------------------------------------------------------
def print_latex_table(results):
    """
    Print results in a LaTeX tabular format (unchanged from your snippet).
    """
    print("{")
    print("\\renewcommand*{\\arraystretch}{0.5}")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\footnotesize")
    print("\\caption{Evaluation Metrics (EQ1) Across Thresholds for Positive and Negative Pairs.}")
    print("\\label{tab:EQ1_eval}")
    print("\\begin{tabular}{lllccccccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Quantization} & \\textbf{Threshold} & \\multicolumn{3}{c}{\\textbf{Positive Pairs}} & \\multicolumn{3}{c}{\\textbf{Negative Pairs}} & \\textbf{Overall Score} \\\\")
    print("\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-10}")
    print("& & & Precision & Recall & F1 Score & Precision & Recall & F1 Score & \\\\")
    print("\\midrule")

    # Edit Distance block
    print("\\multirow{3}{*}{Edit Distance} & \\multirow{3}{*}{N/A}")
    ed_thresholds = [2, 3, 4]
    for idx, threshold in enumerate(ed_thresholds):
        metrics = results["Edit Distance"][f"Edit_Distance_{threshold}"]
        overall = (metrics['F1_Positive'] + metrics['F1_Negative']) / 2
        if idx == 0:
            print(f" & {threshold} & {metrics['Precision_Positive']:.2f} & {metrics['Recall_Positive']:.2f} & {metrics['F1_Positive']:.2f} & {metrics['Precision_Negative']:.2f} & {metrics['Recall_Negative']:.2f} & {metrics['F1_Negative']:.2f} & {overall:.2f} \\\\")
        else:
            print(f" & & {threshold} & {metrics['Precision_Positive']:.2f} & {metrics['Recall_Positive']:.2f} & {metrics['F1_Positive']:.2f} & {metrics['Precision_Negative']:.2f} & {metrics['Recall_Negative']:.2f} & {metrics['F1_Negative']:.2f} & {overall:.2f} \\\\")
    print("\\midrule")

    # Embedding models
    models = ["Pretrained", "Fine-Tuned"]  
    quants = ["float32", "float16", "int8"]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for model_idx, model_key in enumerate(models):
        rows_for_this_model = len(quants) * len(thresholds)
        print(f"\\multirow{{{rows_for_this_model}}}{{*}}{{{model_key}}}")

        for q_idx, quant in enumerate(quants):
            print(f" & \\multirow{{{len(thresholds)}}}{{*}}{{{quant}}} ")
            for t_idx, tval in enumerate(thresholds):
                metrics = results[f"{model_key} ({quant})"][f"Embedding_Sim_{tval}"]
                overall = (metrics['F1_Positive'] + metrics['F1_Negative']) / 2
                if t_idx == 0:
                    print(
                        f"    & {tval:.1f} & "
                        f"{metrics['Precision_Positive']:.2f} & {metrics['Recall_Positive']:.2f} & {metrics['F1_Positive']:.2f} & "
                        f"{metrics['Precision_Negative']:.2f} & {metrics['Recall_Negative']:.2f} & {metrics['F1_Negative']:.2f} & {overall:.2f} \\\\"
                    )
                else:
                    print(
                        f" & & {tval:.1f} & "
                        f"{metrics['Precision_Positive']:.2f} & {metrics['Recall_Positive']:.2f} & {metrics['F1_Positive']:.2f} & "
                        f"{metrics['Precision_Negative']:.2f} & {metrics['Recall_Negative']:.2f} & {metrics['F1_Negative']:.2f} & {overall:.2f} \\\\"
                    )
                # Add a \cmidrule after each quant group except last
                if t_idx == len(thresholds) - 1 and q_idx < len(quants) - 1:
                    print("\\cmidrule{2-10}")
            if q_idx == len(quants) - 1 and model_idx < len(models) - 1:
                print("\\midrule")

    print("\\midrule")

    # Hybrid results
    print("\\multirow{5}{*}{Hybrid} & \\multirow{5}{*}{N/A}")
    hybrid_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for idx, thr in enumerate(hybrid_thresholds):
        prec, rec, f1_ = results["Hybrid Similarity"][f"Threshold {thr}"]
        if idx == 0:
            print(
                f" & {thr:.1f} & {prec:.2f} & {rec:.2f} & {f1_:.2f} & "
                f"{prec:.2f} & {rec:.2f} & {f1_:.2f} & {f1_:.2f} \\\\"
            )
        else:
            print(
                f" & & {thr:.1f} & {prec:.2f} & {rec:.2f} & {f1_:.2f} & "
                f"{prec:.2f} & {rec:.2f} & {f1_:.2f} & {f1_:.2f} \\\\"
            )

    print("\\bottomrule")
    print("\\end{table*}")
    print("}")

def grid_search_weights(
    X: List[Tuple[str, str]], 
    y_true: List[int], 
    model: FastText,
    threshold: float = 0.7,
    weight_step: float = 0.1
) -> Tuple[Tuple[float, float], float]:
    """
    Grid search to find best (w1,w2) for hybrid_similarity (2 metrics).
    If you truly have 3 metrics, expand accordingly.
    """
    best_weights = (0.5, 0.5)
    best_f1 = 0.0
    
    # Generate weight combinations that sum to 1.0
    candidates = []
    for w1 in np.arange(0, 1.01, weight_step):
        w2 = 1 - w1
        candidates.append((w1, w2))

    print(f"\nPerforming grid search with {len(candidates)} weight combinations...")

    for w1, w2 in candidates:
        y_pred = []
        for pkgA, pkgB in X:
            sim = hybrid_similarity(pkgA, pkgB, model, weights=(w1, w2))
            y_pred.append(1 if sim >= threshold else 0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (w1, w2)
    
    print(f"Best weights found: {best_weights} (F1: {best_f1:.3f})")
    return best_weights, best_f1

# ---------------------------------------------------------------------
# 7. PLOTTING FUNCTIONS FOR ROC AND THRESHOLD ACCURACY
# ---------------------------------------------------------------------
def plot_roc_curve(y_true, y_scores, title="ROC Curve", output_file="roc_curve.pdf", highlight_threshold=0.8):
    """
    Creates and saves a ROC curve given true labels and similarity scores.
    For similarity scores:
    - Higher similarity (closer to 1.0) indicates positive class (1)
    - Lower similarity (closer to 0.0) indicates negative class (0)
    """
    plt.rcParams.update({
        'font.size': 24,  # Increased base font size
        'axes.labelweight': 'bold',  # Only make axis labels bold
        'axes.titleweight': 'bold',  # Only make title bold
    })
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=4, label=f"ROC curve (AUC = {roc_auc:.2f})")  # Thicker line
    plt.plot([0, 1], [0, 1], color="navy", lw=4, linestyle="--")  # Thicker line
    
    threshold_idx = np.abs(thresholds - highlight_threshold).argmin()
    plt.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=12,  # Larger marker
            label=f'Similarity Threshold = {highlight_threshold}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=24)
    plt.ylabel("True Positive Rate", fontsize=24)
    plt.title(title, fontsize=24)
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2)  # Thicker grid
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved ROC curve as {output_file}")

def plot_threshold_accuracy_curve(y_true, y_scores, title="Threshold-Accuracy Curve", 
                                output_file="threshold_accuracy.pdf", highlight_threshold=0.8):
    """
    Plots accuracy vs. similarity threshold.
    For each threshold:
    - Scores >= threshold are classified as positive (1)
    - Scores < threshold are classified as negative (0)
    """
    plt.rcParams.update({
        'font.size': 24,  # Increased base font size
        'axes.labelweight': 'bold',  # Only make axis labels bold
        'axes.titleweight': 'bold',  # Only make title bold
    })
    
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracies, color='darkorange', lw=4, label="Accuracy")  # Thicker line
    
    threshold_idx = np.abs(thresholds - highlight_threshold).argmin()
    plt.plot(highlight_threshold, accuracies[threshold_idx], 'ro', markersize=12,  # Larger marker
            label=f'Similarity Threshold = {highlight_threshold}')
    
    plt.xlabel("Similarity Threshold", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.title(title, fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2)  # Thicker grid
    plt.legend(loc="best", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved threshold-accuracy curve as {output_file}")

# ---------------------------------------------------------------------
# 8. Main Script
# ---------------------------------------------------------------------
def find_optimal_threshold(y_true: List[int], y_scores: List[float], 
                         start: float = 0.0, end: float = 1.0, step: float = 0.01) -> Tuple[float, dict]:
    """
    Find optimal similarity threshold that maximizes F1 score.
    For each threshold:
    - Scores >= threshold are classified as positive (1)
    - Scores < threshold are classified as negative (0)
    """
    best_threshold = 0
    best_metrics = {
        'f1': 0,
        'precision': 0,
        'recall': 0,
        'accuracy': 0
    }
    
    thresholds = np.arange(start, end + step, step)
    
    for threshold in tqdm(thresholds, desc="Searching thresholds"):
        # Convert similarity scores to binary predictions
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        
        if f1 > best_metrics['f1']:
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
    
    return best_threshold, best_metrics

if __name__ == "__main__":
    output_file = "./EQ1_similarity_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f, redirect_stdout(f):
        logger.info("Starting EQ1 Effectiveness Evaluation")
        # 1) Positive pairs
        pos_pairs = load_positive_pairs_from_dataset(DATASET_CSV)
        logger.info(f"Loaded {len(pos_pairs)} positive pairs from {DATASET_CSV}.")

        # 2) All packages
        logger.info(f"Loading all packages from {ALL_PACKAGES_FILE} ...")
        all_pkgs = load_all_packages(ALL_PACKAGES_FILE)
        logger.info(f"Loaded {len(all_pkgs)} total packages from {ALL_PACKAGES_FILE}.")

        # 3) Load fine-tuned model
        finetuned_model = FastText.load(FINE_TUNED_MODEL_PATH)
        finetuned_model.min_n = 2
        finetuned_model.max_n = 5
        logger.info("Done loading fine-tuned.\n")

        # 4) Generate negative pairs
        logger.info(f"Generating up to {NUM_NEGATIVE_SAMPLES} negative pairs...")
        neg_pairs = generate_negative_pairs(all_pkgs, pos_pairs, limit=NUM_NEGATIVE_SAMPLES)
        logger.info(f"Generated {len(neg_pairs)} negative pairs.\n")

        X = pos_pairs + neg_pairs
        y_true = [1]*len(pos_pairs) + [0]*len(neg_pairs)

        # Generate ROC and Threshold-Accuracy curves for fine-tuned model
        print("\n[INFO] Generating ROC and Threshold-Accuracy curves for Fine-tuned model...")
        y_scores_finetuned = [embedding_similarity(finetuned_model, a, b) for (a, b) in tqdm(X, desc="Computing similarities")]

        # Plot the ROC
        plot_roc_curve(
            y_true, 
            y_scores_finetuned, 
            title="ROC Curve (Fine-tuned)", 
            output_file=OUTPUT_DIR+"finetuned_roc_curve.pdf",
            highlight_threshold=0.9
        )

        # Plot the threshold-accuracy curve
        plot_threshold_accuracy_curve(
            y_true, 
            y_scores_finetuned, 
            title="Threshold vs Accuracy (Fine-tuned)",
            output_file=OUTPUT_DIR+"finetuned_threshold_accuracy.pdf",
            highlight_threshold=0.9
        )

        print("\n[INFO] Performing grid search for optimal threshold...")
        best_threshold, best_metrics = find_optimal_threshold(y_true, y_scores_finetuned)
        print(f"\nOptimal threshold found: {best_threshold:.3f}")
        print(f"Best metrics:")
        print(f"  F1 Score:  {best_metrics['f1']:.3f}")
        print(f"  Precision: {best_metrics['precision']:.3f}")
        print(f"  Recall:    {best_metrics['recall']:.3f}")

        # Highlight the optimal threshold in the plots
        plot_roc_curve(
            y_true, 
            y_scores_finetuned, 
            title="ROC Curve (Fine-tuned)", 
            output_file=OUTPUT_DIR+"finetuned_roc_curve.pdf",
            highlight_threshold=best_threshold
        )

        plot_threshold_accuracy_curve(
            y_true, 
            y_scores_finetuned, 
            title="Threshold vs Accuracy (Fine-tuned)",
            output_file=OUTPUT_DIR+"finetuned_threshold_accuracy.pdf",
            highlight_threshold=best_threshold
        )

        print("\nDone.\n")

