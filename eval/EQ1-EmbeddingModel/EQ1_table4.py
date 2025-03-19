#!/usr/bin/env python3
# python -m python.typosquat.eval.EQ1-effectiveness.EQ1_similarity_table

import os
import sys
import csv
import random
import numpy as np
from typing import List, Tuple, Set
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from contextlib import redirect_stdout
import time
from collections import defaultdict
import psutil
import copy

# Gensim-based imports
from gensim.models import FastText

# Get the absolute path to the project root (3 levels up from current file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from scipy.spatial.distance import cosine
from create_embedding import Preprocessor

# 1. Config
DATASET_CSV = "./datasets/NeupaneDB_real_malware.csv"
ALL_PACKAGES_FILE = "./all_packages.txt"

PRETRAINED_MODEL_PATH = "./cc.en.300.bin"
FINE_TUNED_MODEL_PATH = "./fasttext_all_pkgs.bin"

EDIT_DISTANCE_THRESHOLDS = [2, 3, 4]
EMBEDDING_SIM_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
NUM_NEGATIVE_SAMPLES = 200

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
    # If subword is scaled separately, you might also do something like:
    if hasattr(model, 'scale_syn'):
        # *Only* if your get_embedding actually merges subword vectors
        # which also were scaled in the same manner:
        # This might be more complicated, but for illustration:
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
    If you want a third metric like jaccard, expand weights to (0.4, 0.3, 0.3).
    """
    edit_sim = edit_distance_similarity(pkgA, pkgB)
    embed_sim = embedding_similarity(model, pkgA, pkgB)
    return weights[0] * edit_sim + weights[1] * embed_sim

def classify_hybrid(pairs: List[Tuple[str, str]],
                    model: FastText,
                    threshold=0.7,
                    weights=(0.5, 0.3, 0.2)) -> List[int]:
    """
    Hybrid classification if you want more than 2 metrics.
    For now, if you are only using 2 metrics, adapt accordingly.
    """
    y_pred = []
    for (a, b) in pairs:
        # If you have 3 metrics, e.g. edit, embed, jaccard:
        # total_sim = weights[0]*edit_distance_similarity(a,b) \
        #           + weights[1]*embedding_similarity(model,a,b) \
        #           + weights[2]*jaccard_similarity(a,b)
        # If you only have 2:
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

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_pkg = row.get("typosquat_pkg", "").strip()
            l_pkg = row.get("legitimate_pkg", "").strip()
            source = row.get("source", "").strip()
            halluc = row.get("hallucianated", "false").lower().strip()
            if not t_pkg or not l_pkg or not source:
                continue
            if halluc == "true":
                continue
            pairs.append((t_pkg, l_pkg))
    return pairs

def load_all_packages(txt_file: str) -> Set[str]:
    """
    Load all known package names from a text file, one per line.
    If file doesn't exist, concatenate CSVs from typosquat-data/typosquat-lfs/all_pkgs.
    """
    packages = set()

    if not os.path.isfile(txt_file):
        print(f"[INFO] File not found: {txt_file}. Concatenating CSV files...")
        csv_dir = "typosquat-data/typosquat-lfs/all_pkgs"

        if not os.path.exists(csv_dir):
            print(f"[ERROR] Directory not found: {csv_dir}")
            sys.exit(1)

        # Get all CSV files in directory
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"[ERROR] No CSV files found in {csv_dir}")
            sys.exit(1)

        # Read and combine all CSVs
        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header if exists
                    for row in reader:
                        if row and row[0].strip():  # Assuming package name is first column
                            packages.add(row[0].strip())
            except Exception as e:
                print(f"[WARNING] Error reading {csv_file}: {e}")

        # Save combined packages to txt_file
        output_dir = os.path.dirname(txt_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(txt_file, 'w', encoding='utf-8') as f:
            for pkg in sorted(packages):
                f.write(f"{pkg}\n")

        print(f"[INFO] Created {txt_file} with {len(packages)} packages")

    else:
        # Original functionality
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
    and ensure they have edit-distance-sim <~0.3 to be "true negatives."
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

def classify_embedding(pairs: List[Tuple[str, str]],
                       model: FastText,
                       threshold=0.5) -> List[int]:
    """Predict 1 if embedding_similarity >= threshold, else 0."""
    y_pred = []
    for (a, b) in pairs:
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

def measure_performance(model: FastText, pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    """Measures throughput (pairs/sec) and avg latency (ms/pair)."""
    start_time = time.time()
    for a, b in pairs:
        _ = embedding_similarity(model, a, b)
    end_time = time.time()

    total_time = end_time - start_time
    throughput = len(pairs) / total_time if total_time > 0 else float('inf')
    latency = (total_time / len(pairs)) * 1000 if len(pairs) > 0 else 0.0
    return throughput, latency

def get_model_memory_usage(model: FastText) -> float:
    """Approximate memory usage of model in MB."""
    total_bytes = 0
    total_bytes += model.wv.vectors.nbytes
    if hasattr(model.wv, 'syn1neg'):
        total_bytes += model.wv.syn1neg.nbytes
    return total_bytes / (1024 * 1024)

# ---------------------------------------------------------------------
# 6. Main Script
# ---------------------------------------------------------------------
def print_latex_table(results):
    """
    Print results in a LaTeX tabular format.
    (Left largely unchanged; you can keep your existing table logic.)
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
    models = ["Pretrained", "Fine-Tuned"]  # or your desired order
    quants = ["float32", "float16", "int8"]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for model_idx, model_key in enumerate(models):
        # We have 3 quant types => 3 * len(thresholds) rows
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
            # midrule between different base models
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
    print("\\end{tabular}")
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
    If you truly have 3 metrics, expand to (w1, w2, w3).
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

if __name__ == "__main__":
    output_file = "./eval/EQ1-effectiveness/EQ1_similarity_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f, redirect_stdout(f):

        # 1) Positive pairs
        pos_pairs = load_positive_pairs_from_dataset(DATASET_CSV)
        print(f"Loaded {len(pos_pairs)} positive pairs from {DATASET_CSV}.")

        # 2) All packages
        all_pkgs = load_all_packages(ALL_PACKAGES_FILE)
        print(f"Loaded {len(all_pkgs)} total packages from {ALL_PACKAGES_FILE}.")

        # 3) Load pretrained model (official fastText .bin)
        print(f"\nLoading PRETRAINED model from {PRETRAINED_MODEL_PATH} ...")
        pretrained_model = FastText.load_fasttext_format(PRETRAINED_MODEL_PATH)
        print("Done loading pretrained.\n")

        # 4) Load fine-tuned model
        # If your fine-tuned model is also a fastText .bin:
        #   finetuned_model = FastText.load_fasttext_format(FINE_TUNED_MODEL_PATH)
        # Else if it's a native Gensim model:
        finetuned_model = FastText.load(FINE_TUNED_MODEL_PATH)
        finetuned_model.min_n = 2
        finetuned_model.max_n = 5
        print("Done loading fine-tuned.\n")

        # 5) Generate negative pairs
        print(f"Generating up to {NUM_NEGATIVE_SAMPLES} negative pairs...")
        neg_pairs = generate_negative_pairs(all_pkgs, pos_pairs, limit=NUM_NEGATIVE_SAMPLES)
        print(f"Generated {len(neg_pairs)} negative pairs.\n")

        # 6) Example similarities
        print("=== EXAMPLE SIMILARITIES ===")
        example_pos = pos_pairs[:3]
        example_neg = random.sample(neg_pairs, 3) if len(neg_pairs) >= 3 else neg_pairs

        print("\nPositive Pairs:")
        for pkgA, pkgB in example_pos:
            ed_sim = edit_distance_similarity(pkgA, pkgB)
            pt_sim = embedding_similarity(pretrained_model, pkgA, pkgB)
            ft_sim = embedding_similarity(finetuned_model, pkgA, pkgB)
            print(f"({pkgA} | {pkgB}) => ED-Sim={ed_sim:.3f}, Pretrained={pt_sim:.3f}, FineTuned={ft_sim:.3f}")

        if example_neg:
            print("\nNegative Pairs:")
            for pkgA, pkgB in example_neg:
                ed_sim = edit_distance_similarity(pkgA, pkgB)
                pt_sim = embedding_similarity(pretrained_model, pkgA, pkgB)
                ft_sim = embedding_similarity(finetuned_model, pkgA, pkgB)
                print(f"({pkgA} | {pkgB}) => ED-Sim={ed_sim:.3f}, Pretrained={pt_sim:.3f}, FineTuned={ft_sim:.3f}")

        # 7) Evaluate with multiple thresholds + quantized models
        X = pos_pairs + neg_pairs
        y_true = [1]*len(pos_pairs) + [0]*len(neg_pairs)
        pos_len = len(pos_pairs)

        base_models = {
            "Pretrained (float32)": pretrained_model,
            "Fine-Tuned (float32)": finetuned_model
        }

        # Quantize each base model
        quantized_models = {}
        for name, m in base_models.items():
            bname = name.split(" (")[0]  # "Pretrained" or "Fine-Tuned"
            # Float16
            float16_model = quantize_model_float16(m)
            quantized_models[f"{bname} (float16)"] = float16_model
            # Int8
            int8_model = quantize_model_int8(m)
            quantized_models[f"{bname} (int8)"] = int8_model

        # Combine
        all_models = {**base_models, **quantized_models}

        # Initialize results structure
        results = {
            "Edit Distance": {},
            "Hybrid Similarity": {},
            "Pretrained (float32)": {},
            "Fine-Tuned (float32)": {},
            "Pretrained (float16)": {},
            "Fine-Tuned (float16)": {},
            "Pretrained (int8)": {},
            "Fine-Tuned (int8)": {}
        }

        # Evaluate Edit Distance
        print("\n[EVALUATING] Edit Distance")
        for ed_t in EDIT_DISTANCE_THRESHOLDS:
            y_pred_ed = classify_edit_distance(X, ed_t)
            pos_metrics, neg_metrics = eval_metrics_by_class(y_true, y_pred_ed, pos_len)
            results["Edit Distance"][f"Edit_Distance_{ed_t}"] = {
                "Precision_Positive": pos_metrics['precision'],
                "Recall_Positive": pos_metrics['recall'],
                "F1_Positive": pos_metrics['f1'],
                "Precision_Negative": neg_metrics['precision'],
                "Recall_Negative": neg_metrics['recall'],
                "F1_Negative": neg_metrics['f1']
            }

        # Evaluate all embedding models
        for model_name, model_inst in all_models.items():
            print(f"\n[EVALUATING] {model_name}")
            for sim_threshold in EMBEDDING_SIM_THRESHOLDS:
                y_pred_emb = classify_embedding(X, model_inst, sim_threshold)
                pos_metrics, neg_metrics = eval_metrics_by_class(y_true, y_pred_emb, pos_len)
                results[model_name][f"Embedding_Sim_{sim_threshold}"] = {
                    "Precision_Positive": pos_metrics['precision'],
                    "Recall_Positive": pos_metrics['recall'],
                    "F1_Positive": pos_metrics['f1'],
                    "Precision_Negative": neg_metrics['precision'],
                    "Recall_Negative": neg_metrics['recall'],
                    "F1_Negative": neg_metrics['f1']
                }

            # Performance
            tput, lat = measure_performance(model_inst, X)
            results[model_name]["Performance"] = {
                "Throughput (pairs/sec)": tput,
                "Latency (ms/pair)": lat
            }
            mem_usage = get_model_memory_usage(model_inst)
            print(f"Memory Usage: {mem_usage:.2f} MB")
            print(f"Throughput: {tput:.2f} pairs/sec")
            print(f"Latency: {lat:.2f} ms/pair")

        # Hybrid thresholds
        hybrid_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        # If only 2 metrics => weights=(0.5, 0.5)
        # If 3 => (0.4, 0.4, 0.2), etc.
        for thr in hybrid_thresholds:
            y_pred_h = classify_hybrid(X, pretrained_model, thr, (0.4, 0.6))
            prec, rec, f1_ = eval_metrics(y_true, y_pred_h)
            results["Hybrid Similarity"][f"Threshold {thr}"] = (prec, rec, f1_)

        # Print table
        print_latex_table(results)

        # Test specific pairs
        print("\n=== TESTING SPECIFIC PACKAGE PAIRS ===")
        test_cases = [
            ('facebook', 'meta'),
            ('facebook-sdk', 'meta-sdk'),
            ('facebook-sdk', 'facebook-sdk'),
            ('FACEBOOK', 'facebook'),
            ('react-native', 'react_native'),
            ('python-requests', 'requests'),
            ('numpy', 'scikit-learn'),
            ('matplotlib', 'catplotlib'),
            # etc...
        ]
        for pkgA, pkgB in test_cases:
            try:
                sim_pre = embedding_similarity(pretrained_model, pkgA, pkgB)
                print(f"[Pretrained] '{pkgA}' vs '{pkgB}': {sim_pre:.4f}")
            except Exception as e:
                print(f"Error computing similarity (pretrained) for '{pkgA}' and '{pkgB}': {e}")

        # Grid search for best hybrid weights
        print("\n=== GRID SEARCH FOR HYBRID WEIGHTS ===")
        thresholds_gs = [0.6, 0.7, 0.8]
        best_global = (None, None, 0.0)
        for thr in thresholds_gs:
            print(f"\nTesting threshold: {thr}")
            best_w, best_f1_ = grid_search_weights(X, y_true, pretrained_model, threshold=thr)
            if best_f1_ > best_global[2]:
                best_global = (best_w, thr, best_f1_)
        print(f"\nBest overall: weights={best_global[0]}, threshold={best_global[1]}, F1={best_global[2]:.3f}")

        # Store in results
        results["Hybrid Similarity"]["Optimized"] = {
            "Weights": best_global[0],
            "Threshold": best_global[1],
            "F1": best_global[2]
        }

        print("\nDone.\n")
