#!/usr/bin/env python3
"""
EQ1-visualization.py

Generates three separate plots:
  1. Top-K Packages
  2. Positive Pairs
  3. Negative Pairs

Each plot displays package names and includes a clear legend.

Usage:
  python python/typosquat/eval/EQ1-embedding/EQ1_visualization.py
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from matplotlib.lines import Line2D
from adjustText import adjust_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from gensim.models import FastText

# Ensure EQ1_similarity.py is in the same directory or adjust the path accordingly
import EQ1_similarity as eq1  # Make sure to rename EQ1-similarity.py to EQ1_similarity.py

def get_embeddings(model, packages):
    """
    Computes the embeddings for a list of package names using the provided model.
    """
    pkg_vectors = [eq1.get_pkg_vector(pkg, model) for pkg in packages]
    return np.array(pkg_vectors)

def perform_tsne(embeddings, perplexity=30, learning_rate=200, n_iter=1000):
    """
    Reduces dimensionality of embeddings to 2D using t-SNE.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter, init='pca', verbose=1)
    return tsne.fit_transform(embeddings)

def perform_umap(embeddings, n_neighbors=10, min_dist=0.2, metric='cosine'):
    """
    Reduces dimensionality of embeddings to 2D using UMAP.
    """
    reducer = umap.UMAP(n_components=2,
                        random_state=42,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        verbose=True,
                        low_memory=True)
    return reducer.fit_transform(embeddings)

def plot_topk_packages(embeddings_2d, topk_packages, output_file):
    """
    Plots the Top-K packages with their names.
    """
    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='black', alpha=0.7, s=50, label='Top-K Packages')

    # Annotate package names
    texts = []
    for i, pkg in enumerate(topk_packages):
        texts.append(plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], pkg, fontsize=9, alpha=0.9))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top-K Packages',
               markerfacecolor='black', markersize=10)
    ]

    plt.legend(handles=legend_elements, loc='best')
    plt.title("UMAP Visualization of Top-K Package Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[INFO] Top-K Packages plot saved to '{output_file}'")

def plot_pairs(pairs, pair_type, output_file, model, max_pairs=500):
    """
    Plots either positive or negative pairs with their names and connections.
    Processes data in smaller batches to avoid memory issues.
    """
    plt.figure(figsize=(14, 10))

    if pair_type == 'Positive':
        color = 'green'
        marker = '^'
        label = 'Positive Pairs'
    elif pair_type == 'Negative':
        color = 'red'
        marker = 'x'
        label = 'Negative Pairs'
    else:
        raise ValueError("pair_type must be 'Positive' or 'Negative'")

    # Limit the number of pairs to plot
    original_pair_count = len(pairs)
    if original_pair_count > max_pairs:
        pairs = random.sample(pairs, max_pairs)
        print(f"[DEBUG] Sampling {max_pairs} out of {original_pair_count} {pair_type} pairs for visualization")

    # Collect unique packages in pairs
    unique_pkgs = list(set(pkg for pair in pairs for pkg in pair))
    unique_pkgs = sorted(unique_pkgs)

    # Process embeddings in batches
    batch_size = 1000  # Adjust based on available memory
    pkg_embeddings = []

    for i in range(0, len(unique_pkgs), batch_size):
        batch_pkgs = unique_pkgs[i:i + batch_size]
        batch_embeddings = get_embeddings(model, batch_pkgs)
        pkg_embeddings.append(batch_embeddings)

    pkg_embeddings = np.vstack(pkg_embeddings)

    # Perform UMAP with reduced n_neighbors
    emb_2d = perform_umap(pkg_embeddings, n_neighbors=min(10, len(unique_pkgs) - 1))

    # Plot packages
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], color=color, alpha=0.7, s=100, marker=marker, label=label)

    # Annotate package names
    texts = []
    for i, pkg in enumerate(unique_pkgs):
        texts.append(plt.text(emb_2d[i, 0], emb_2d[i, 1], pkg, fontsize=9, alpha=0.9))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    # Draw lines between pairs
    for pkgA, pkgB in pairs:
        try:
            idxA = unique_pkgs.index(pkgA)
            idxB = unique_pkgs.index(pkgB)
            plt.plot([emb_2d[idxA, 0], emb_2d[idxB, 0]],
                     [emb_2d[idxA, 1], emb_2d[idxB, 1]],
                     color='gray', linestyle='--', linewidth=1, alpha=0.5)
        except ValueError:
            print(f"[WARNING] Package not found in unique_pkgs: {pkgA}, {pkgB}")
            continue  # Skip if package not found

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker=marker, color='w', label=label,
               markerfacecolor=color, markersize=10),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Pair Connections')
    ]

    plt.legend(handles=legend_elements, loc='best')
    plt.title(f"UMAP Visualization of {pair_type} Pairs")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[INFO] {pair_type} Pairs plot saved to '{output_file}'")

def find_low_similarity_pairs(pairs, model, threshold=0.5, batch_size=100):
    """
    Finds pairs with similarity scores below the threshold.
    Process pairs in batches to avoid memory issues.
    """
    low_sim_pairs = []
    total_pairs = len(pairs)

    for i in range(0, total_pairs, batch_size):
        batch_pairs = pairs[i:i + batch_size]
        for pkgA, pkgB in batch_pairs:
            try:
                vecA = eq1.get_pkg_vector(pkgA, model)
                vecB = eq1.get_pkg_vector(pkgB, model)
                sim = cosine_similarity(vecA.reshape(1, -1), vecB.reshape(1, -1))[0][0]
                if sim < threshold:
                    low_sim_pairs.append((pkgA, pkgB, sim))
            except Exception as e:
                print(f"[WARNING] Error processing pair ({pkgA}, {pkgB}): {e}")

    return sorted(low_sim_pairs, key=lambda x: x[2])

def plot_similarity_distributions(pos_pairs, neg_pairs, model, output_file, pretrained_model):
    """
    Plots the distribution of different similarity metrics with threshold lines.
    """
    # Ensure balanced pairs by taking min length of both sets
    n_pairs = min(len(pos_pairs), len(neg_pairs))
    pos_pairs = random.sample(pos_pairs, n_pairs)
    neg_pairs = random.sample(neg_pairs, n_pairs)
    
    print(f"[INFO] Using {n_pairs} pairs each for positive and negative samples")

    # Calculate similarities for positive pairs
    pos_edit_sims = [eq1.edit_distance_similarity(a, b) for a, b in pos_pairs]
    pos_embed_sims = [eq1.embedding_similarity(model, a, b) for a, b in pos_pairs]
    pos_pretrained_sims = [eq1.embedding_similarity(pretrained_model, a, b) for a, b in pos_pairs]
    pos_hybrid_sims = [eq1.hybrid_similarity(a, b, model, weights=(0.4, 0.4, 0.2)) for a, b in pos_pairs]

    # Calculate similarities for negative pairs
    neg_edit_sims = [eq1.edit_distance_similarity(a, b) for a, b in neg_pairs]
    neg_embed_sims = [eq1.embedding_similarity(model, a, b) for a, b in neg_pairs]
    neg_pretrained_sims = [eq1.embedding_similarity(pretrained_model, a, b) for a, b in neg_pairs]
    neg_hybrid_sims = [eq1.hybrid_similarity(a, b, model, weights=(0.4, 0.4, 0.2)) for a, b in neg_pairs]

    # Find optimal thresholds using F1 score
    def find_best_threshold(pos_sims, neg_sims):
        all_sims = np.concatenate([pos_sims, neg_sims])
        y_true = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
        best_f1 = 0
        best_threshold = 0
        
        for threshold in np.arange(0.1, 1.0, 0.05):
            y_pred = (all_sims >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1

    edit_threshold, edit_f1 = find_best_threshold(pos_edit_sims, neg_edit_sims)
    embed_threshold, embed_f1 = find_best_threshold(pos_embed_sims, neg_embed_sims)
    pretrained_threshold, pretrained_f1 = find_best_threshold(pos_pretrained_sims, neg_pretrained_sims)
    hybrid_threshold, hybrid_f1 = find_best_threshold(pos_hybrid_sims, neg_hybrid_sims)

    print(f"[INFO] Optimal thresholds (F1 scores):")
    print(f"Edit Distance: {edit_threshold:.2f} ({edit_f1:.3f})")
    print(f"Embedding: {embed_threshold:.2f} ({embed_f1:.3f})")
    print(f"Pretrained: {pretrained_threshold:.2f} ({pretrained_f1:.3f})")
    print(f"Hybrid: {hybrid_threshold:.2f} ({hybrid_f1:.3f})")

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Edit Distance Similarities
    ax1.hist(pos_edit_sims, bins=30, alpha=0.5, color='green', label='Positive Pairs')
    ax1.hist(neg_edit_sims, bins=30, alpha=0.5, color='red', label='Negative Pairs')
    ax1.axvline(x=edit_threshold, color='blue', linestyle='--')
    ax1.text(edit_threshold + 0.02, ax1.get_ylim()[1]*0.9, f'threshold={edit_threshold}', 
             rotation=90, verticalalignment='top')
    ax1.set_title('Edit Distance Similarity Distribution')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Plot Fine-tuned Embedding Similarities
    ax2.hist(pos_embed_sims, bins=30, alpha=0.5, color='green', label='Positive Pairs')
    ax2.hist(neg_embed_sims, bins=30, alpha=0.5, color='red', label='Negative Pairs')
    ax2.axvline(x=embed_threshold, color='blue', linestyle='--')
    ax2.text(embed_threshold + 0.02, ax2.get_ylim()[1]*0.9, f'threshold={embed_threshold}', 
             rotation=90, verticalalignment='top')
    ax2.set_title('Fine-tuned Embedding Similarity Distribution')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Plot Pretrained FastText Similarities
    ax3.hist(pos_pretrained_sims, bins=30, alpha=0.5, color='green', label='Positive Pairs')
    ax3.hist(neg_pretrained_sims, bins=30, alpha=0.5, color='red', label='Negative Pairs')
    ax3.axvline(x=pretrained_threshold, color='blue', linestyle='--')
    ax3.text(pretrained_threshold + 0.02, ax3.get_ylim()[1]*0.9, f'threshold={pretrained_threshold}', 
             rotation=90, verticalalignment='top')
    ax3.set_title('Pretrained FastText Similarity Distribution')
    ax3.set_xlabel('Similarity Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # Plot Optimized Hybrid Similarities
    ax4.hist(pos_hybrid_sims, bins=30, alpha=0.5, color='green', label='Positive Pairs')
    ax4.hist(neg_hybrid_sims, bins=30, alpha=0.5, color='red', label='Negative Pairs')
    ax4.axvline(x=hybrid_threshold, color='blue', linestyle='--')
    ax4.text(hybrid_threshold + 0.02, ax4.get_ylim()[1]*0.9, f'threshold={hybrid_threshold}', 
             rotation=90, verticalalignment='top')
    ax4.set_title('Optimized Hybrid Similarity Distribution')
    ax4.set_xlabel('Similarity Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Similarity distributions plot saved to '{output_file}'")

def plot_similarity_heatmap(pairs, model, output_file, max_pairs=100):
    """
    Creates a heatmap comparing different similarity metrics for a subset of pairs.
    """
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    # Calculate similarities
    similarities = []
    for a, b in pairs:
        edit_sim = eq1.edit_distance_similarity(a, b)
        embed_sim = eq1.embedding_similarity(model, a, b)
        jaccard_sim = eq1.jaccard_similarity(a, b)
        hybrid_sim = eq1.hybrid_similarity(a, b, model)
        similarities.append([edit_sim, embed_sim, jaccard_sim, hybrid_sim])

    similarities = np.array(similarities)

    plt.figure(figsize=(10, 8))
    plt.imshow(similarities.T, aspect='auto', cmap='YlOrRd')
    
    plt.yticks(range(4), ['Edit', 'Embedding', 'Jaccard', 'Hybrid'])
    plt.colorbar(label='Similarity Score')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity Metric')
    plt.title('Similarity Metrics Comparison')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[INFO] Similarity heatmap saved to '{output_file}'")

def hybrid_similarity(pkgA: str, pkgB: str, model: FastText, weights=(0.4, 0.4, 0.2)) -> float:
    """
    Optimized hybrid similarity function that combines:
    - Edit distance similarity (40%)
    - Fine-tuned embedding similarity (40%)
    - Pretrained FastText similarity (20%)
    """
    edit_sim = eq1.edit_distance_similarity(pkgA, pkgB)
    embed_sim = eq1.embedding_similarity(model, pkgA, pkgB)
    
    # Use pretrained model for the third component
    pretrained_model = eq1.FastText.load_fasttext_format(eq1.PRETRAINED_MODEL_PATH)
    pretrained_sim = eq1.embedding_similarity(pretrained_model, pkgA, pkgB)
    
    # Apply non-linear transformation to increase separation
    edit_sim = np.tanh(2 * edit_sim)  # Steeper curve around 0.5
    embed_sim = np.tanh(2 * embed_sim)
    pretrained_sim = np.tanh(2 * pretrained_sim)
    
    return (weights[0] * edit_sim) + (weights[1] * embed_sim) + (weights[2] * pretrained_sim)

def plot_grid_search_results(grid_results, output_file):
    """
    Visualizes only the top 3 best performing weight combinations.
    """
    plt.figure(figsize=(10, 6))
    
    # Extract and sort all results by F1 score
    all_results = []
    for threshold, results in grid_results.items():
        for result in results:
            all_results.append({
                'threshold': threshold,
                'weights': result['weights'],
                'f1': result['f1']
            })
    
    # Sort by F1 score and get top 3
    top_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:3]
    
    # Plot bar chart for top 3 results
    x_pos = np.arange(len(top_results))
    f1_scores = [r['f1'] for r in top_results]
    
    plt.bar(x_pos, f1_scores)
    plt.ylabel('F1 Score')
    plt.title('Top 3 Weight Combinations')
    
    # Add annotations for weights
    for i, result in enumerate(top_results):
        w1, w2, w3 = result['weights']
        plt.annotate(f'Weights: ({w1:.2f}, {w2:.2f}, {w3:.2f})\nThreshold: {result["threshold"]}',
                    xy=(i, f1_scores[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to generate three separate plots:
      1. Top-K Packages
      2. Positive Pairs
      3. Negative Pairs
    """
    try:
        # Convert set to list before slicing
        pos_pairs = list(eq1.load_positive_pairs_from_dataset(eq1.DATASET_CSV))[:1500]
        print(f"[INFO] Loaded {len(pos_pairs)} positive pairs.")

        # Convert set to list before slicing
        all_pkgs = list(eq1.load_all_packages(eq1.ALL_PACKAGES_FILE))[:500]
        print(f"[INFO] Loaded {len(all_pkgs)} total packages.")

        # Load fine-tuned model
        finetuned_model = eq1.FastText.load(eq1.FINE_TUNED_MODEL_PATH)
        finetuned_model.min_n = 2
        finetuned_model.max_n = 5
        print("[INFO] Fine-tuned model loaded successfully.")

        pretrained_model = eq1.FastText.load_fasttext_format(eq1.PRETRAINED_MODEL_PATH)
        print("[INFO] Pretrained model loaded successfully.")

        # Generate Negative Pairs
        neg_pairs = list(eq1.generate_negative_pairs(all_pkgs, pos_pairs, finetuned_model, limit=5000))
        print(f"[INFO] Generated {len(neg_pairs)} negative pairs.")

        # Select Top-K Packages
        topk_packages = sorted(all_pkgs)[:50]

        # Compute embeddings for Top-K Packages
        topk_embeddings = get_embeddings(finetuned_model, topk_packages)
        topk_emb_2d = perform_umap(topk_embeddings)

        # Plot Top-K Packages
        plot_topk_packages(topk_emb_2d, topk_packages, "TopK_Packages_Visualization.png")

        # Plot Positive Pairs
        plot_pairs(pos_pairs, 'Positive', "Positive_Pairs_Visualization.png", finetuned_model)

        # Plot Negative Pairs
        plot_pairs(neg_pairs, 'Negative', "Negative_Pairs_Visualization.png", finetuned_model)

        # Find and print low similarity positive pairs
        low_sim_pairs = find_low_similarity_pairs(pos_pairs, finetuned_model, threshold=0.5)
        if low_sim_pairs:
            print("\nPositive pairs with low similarity:")
            for pkgA, pkgB, sim in low_sim_pairs:
                print(f"{pkgA}, {pkgB}: {sim:.4f}")

        # Add new visualizations for similarity metrics
        print("[INFO] Generating similarity distribution plots...")
        plot_similarity_distributions(
            pos_pairs[:500],
            neg_pairs[:500],
            finetuned_model,
            "Similarity_Distributions.png",
            pretrained_model
        )

        print("[INFO] Generating similarity heatmap...")
        plot_similarity_heatmap(
            pos_pairs,  # Will be sampled inside the function
            finetuned_model,
            "Similarity_Heatmap_Positive.png"
        )
        plot_similarity_heatmap(
            neg_pairs,
            finetuned_model,
            "Similarity_Heatmap_Negative.png"
        )

        print("[INFO] Performing grid search for hybrid weights...")
        
        # Reduce data size for grid search
        sample_size = 1000  # Reduce from full dataset
        pos_sample = random.sample(pos_pairs, min(len(pos_pairs), sample_size//2))
        neg_sample = random.sample(neg_pairs, min(len(neg_pairs), sample_size//2))
        
        X = pos_sample + neg_sample
        y_true = [1]*len(pos_sample) + [0]*len(neg_sample)
        
        # Reduce parameter space
        thresholds = [0.6, 0.7, 0.8]
        weight_steps = 0.2  # Increase step size from 0.1
        
        grid_results = {}
        for threshold in thresholds:
            print(f"\nTesting threshold: {threshold}")
            results = []
            
            # Reduced weight combinations
            for w1 in np.arange(0, 0.9, weight_steps):
                for w2 in np.arange(0, 0.9-w1, weight_steps):
                    w3 = round(1 - w1 - w2, 2)  # Ensure weights sum to 1
                    if w3 < 0: continue
                    
                    weights = (w1, w2, w3)
                    # ... rest of the grid search code ...

        # Visualize grid search results
        plot_grid_search_results(grid_results, "Grid_Search_Results.png")

        print("[INFO] All visualizations generated successfully.")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
