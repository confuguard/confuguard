# python -m python.typosquat.data.embedding_vis

import os
import sys
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from loguru import logger
from google.cloud.sql.connector import Connector
import sqlalchemy
from adjustText import adjust_text
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config import HF_GCP_PROJECT_ID, HF_GCP_REGION, HF_GCP_INSTANCE_NAME, HF_DB_USER, HF_DB_PASS, HF_DB_PORT
from ..utils import init_connection_engine


TYPOSQUAT_DB_NAME = "typosquat_float32"
def visualize_embeddings():
    start_total = time.time()
    
    # Connect to database
    engine = init_connection_engine(HF_DB_USER, HF_DB_PASS, TYPOSQUAT_DB_NAME, HF_DB_PORT)
    
    # Query for top 100 packages by downloads (for labeled plot)
    query_top_100 = """
    SELECT package_name, embedding 
    FROM typosquat_npm_embeddings 
    WHERE LENGTH(package_name) < 10
    LIMIT 1000;
    """
    
    # Query for 100K packages (for dot plot)
    query_100k = """
    SELECT package_name, embedding 
    FROM typosquat_npm_embeddings 
    LIMIT 100000;
    """
    
    logger.info("Fetching embeddings from database...")
    
    # Function to process query results
    def process_query_results(result):
        data = [(row[0], row[1]) for row in result]
        package_names = [d[0] for d in data]
        embeddings = np.array([
            np.fromstring(d[1].strip('[]'), sep=',', dtype=np.float32)
            for d in data
        ])
        return package_names, embeddings

    # Fetch and process both datasets
    with engine.connect() as conn:
        # Top 100 dataset
        result_top_100 = conn.execute(text(query_top_100))
        package_names_top_100, embeddings_top_100 = process_query_results(result_top_100)
        
        # 100K dataset
        result_100k = conn.execute(text(query_100k))
        package_names_100k, embeddings_100k = process_query_results(result_100k)

    # Create UMAP reducer (use same instance for both to maintain consistency)
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    
    # Process top 100 visualization
    logger.info("Creating top 100 packages visualization...")
    start_top100 = time.time()
    embeddings_2d_top_100 = umap_reducer.fit_transform(embeddings_top_100)
    
    plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 26})  # Increase default font size
    plt.scatter(embeddings_2d_top_100[:, 0], embeddings_2d_top_100[:, 1], alpha=0.5, s=10)
    
    texts = []
    for i in range(len(package_names_top_100)):
        texts.append(plt.text(embeddings_2d_top_100[i, 0], embeddings_2d_top_100[i, 1], 
                            package_names_top_100[i], fontsize=24))
    
    adjust_text(texts, 
               arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
               expand_points=(1.5, 1.5),
               force_points=(0.1, 0.25))
    
    # plt.title('Top 100 NPM Packages by Downloads (UMAP 2D Projection)', fontsize=28)
    plt.xlabel('UMAP Dimension 1', fontsize=30)
    plt.ylabel('UMAP Dimension 2', fontsize=30)
    
    base_output_path = 'python/typosquat/data/npm_embeddings_visualization'
    plt.savefig(f'{base_output_path}_top100.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_output_path}_top100.pdf', bbox_inches='tight')
    plt.close()
    top100_time = time.time() - start_top100
    logger.info(f"Top 100 visualization took {top100_time:.2f} seconds")
    
    # Process 100K visualization
    logger.info("Creating 100K packages visualization...")
    start_100k = time.time()
    embeddings_2d_100k = umap_reducer.fit_transform(embeddings_100k)
    
    plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 24})  # Increase default font size
    plt.scatter(embeddings_2d_100k[:, 0], embeddings_2d_100k[:, 1], alpha=0.1, s=1)
    
    # plt.title('100K NPM Packages (UMAP 2D Projection)', fontsize=28)
    plt.xlabel('UMAP Dimension 1', fontsize=30)
    plt.ylabel('UMAP Dimension 2', fontsize=30)
    
    plt.savefig(f'{base_output_path}_100k.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_output_path}_100k.pdf', bbox_inches='tight')
    logger.info(f"Visualizations saved to {base_output_path}_*.png and {base_output_path}_*.pdf")
    plt.close()
    time_100k = time.time() - start_100k
    total_time = time.time() - start_total
    
    logger.info(f"100K visualization took {time_100k:.2f} seconds")
    logger.info(f"Total execution time: {total_time:.2f} seconds")

def visualize_typosquats():
    # Read the CSV file
    df = pd.read_csv('python/typosquat/data/dataset.csv')
    
    # Create sets of unique packages
    typosquat_pkgs = set(df['typosquat_pkg'].dropna())
    legitimate_pkgs = set(df['legitimate_pkg'].dropna())
    
    # Create a scatter plot
    plt.figure(figsize=(15, 10))
    
    # Plot legitimate packages
    plt.scatter(range(len(legitimate_pkgs)), [0] * len(legitimate_pkgs), 
               c='blue', label='Legitimate Packages', s=100)
    
    # Plot typosquat packages
    plt.scatter(range(len(typosquat_pkgs)), [1] * len(typosquat_pkgs), 
               c='red', label='Typosquat Packages', s=100)
    
    # Draw lines connecting related packages
    for _, row in df.iterrows():
        typo_idx = list(typosquat_pkgs).index(row['typosquat_pkg'])
        legit_idx = list(legitimate_pkgs).index(row['legitimate_pkg'])
        plt.plot([legit_idx, typo_idx], [0, 1], 'k-', alpha=0.2)
    
    # Customize the plot
    plt.yticks([0, 1], ['Legitimate', 'Typosquat'])
    plt.xlabel('Package Index')
    plt.title('Typosquat vs Legitimate Package Relationships')
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks([])
    
    # Save the visualization
    output_path = 'python/typosquat/data/typosquat_visualization'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Found {len(legitimate_pkgs)} legitimate packages and {len(typosquat_pkgs)} typosquat packages")
    logger.info(f"Visualization saved to {output_path}.png and {output_path}.pdf")

if __name__ == "__main__":
    visualize_embeddings()
    visualize_typosquats()

