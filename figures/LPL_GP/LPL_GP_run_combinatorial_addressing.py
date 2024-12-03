#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Configure paths - adjusted for your specific structure
script_dir = Path('/mnt/LaCIE/annaM/heart_Xenium')
tools_dir = Path('tools')  # relative to current working directory
data_dir = Path('data/heart_Xenium')  # relative to current working directory

sys.path.append(str(tools_dir))

# Set matplotlib backend before other imports
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

# Standard imports
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Custom imports
import combinatorial_addressing
from combinatorial_addressing import (
    get_genes_list, 
    genes_pathway,
    normalize_and_filter,
    silh_pathway_bootstrap_single_gpu,
    silh_pathway_bootstrap_multiple_gpus,
    plot_silhouette_scores,
    plot_calinski_harabasz_index,
    plot_davies_bouldin_index,
    plot_elbow,
    randomize_gene_expression,
    make_bootstrap_df,
    silhouette_zscore,
    calinski_harabasz_evaluation,
    davies_bouldin_evaluation,
    elbow_method_evaluation
)

def setup_environment():
    """Configure scanpy settings and create output directory"""
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(
        dpi=180,
        color_map='magma_r',
        dpi_save=300,
        vector_friendly=True,
        format='png'
    )
    
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_prepare_data(adata_path):
    """Load and prepare the initial dataset"""
    adata = sc.read_h5ad(adata_path)
    
    # Extract pathway genes
    gp = pd.DataFrame(adata.uns['nichecompass_gp_summary'])
    LPL_GP = gp[gp['gp_name'] == 'LPL_ligand_receptor_target_gene_GP']
    source_genes = LPL_GP['gp_source_genes'].values[0]
    target_genes = LPL_GP['gp_target_genes'].values[0]
    pathway_genes = [source_genes] + target_genes.split(', ')
    
    return adata, pathway_genes

def process_real_data(adata, pathway_genes, output_dir, output_prefix):
    """Process the real (non-randomized) dataset"""
    adata_filtered, pathway_genes, gene_quantiles, scaler = normalize_and_filter(
        adata, pathway_genes, sat_val=0.99, min_genes_on=1, min_expr=0.01
    )
    
    df = pd.DataFrame(
        adata_filtered.X.toarray(),
        index=adata_filtered.obs_names,
        columns=adata_filtered.var_names
    )
    
    results, final_data_frame = silh_pathway_bootstrap_multiple_gpus(
        df, pathway_genes, k_max=100, dist_metric='cosine',
        n_boots=50, pct_boots=0.9, batch_size=1000
    )
    
    # Generate and save plots
    plot_silhouette_scores(results, k_max=100, output_dir=output_dir, output_prefix=output_prefix)
    ch_scores = plot_calinski_harabasz_index(final_data_frame, k_max=100, output_dir=output_dir, output_prefix=output_prefix)
    db_scores = plot_davies_bouldin_index(final_data_frame, k_max=100, output_dir=output_dir, output_prefix=output_prefix)
    inertias = plot_elbow(final_data_frame, k_max=100, output_dir=output_dir, output_prefix=output_prefix)
    
    return results, final_data_frame, ch_scores, db_scores, inertias

def process_randomized_data(adata, pathway_genes, output_dir, output_prefix):
    """Process the randomized dataset"""
    randomized_adata = randomize_gene_expression(adata, pathway_genes)
    
    randomized_adata_filtered, pathway_genes, gene_quantiles, scaler = normalize_and_filter(
        randomized_adata, pathway_genes, sat_val=0.99, min_genes_on=1, min_expr=0.01
    )
    
    randomized_df = pd.DataFrame(
        randomized_adata_filtered.X.toarray(),
        index=randomized_adata_filtered.obs_names,
        columns=randomized_adata_filtered.var_names
    )
    
    randomized_results, randomized_final_data_frame = silh_pathway_bootstrap_multiple_gpus(
        randomized_df, pathway_genes, k_max=100, dist_metric='cosine',
        n_boots=50, pct_boots=0.9, batch_size=1000
    )
    
    # Generate and save plots for randomized data
    plot_silhouette_scores(randomized_results, k_max=100, output_dir=output_dir, 
                          output_prefix=f"{output_prefix}_randomized")
    randomized_ch_scores = plot_calinski_harabasz_index(randomized_final_data_frame, k_max=100, 
                                                       output_dir=output_dir, output_prefix=f"{output_prefix}_randomized")
    randomized_db_scores = plot_davies_bouldin_index(randomized_final_data_frame, k_max=100, 
                                                    output_dir=output_dir, output_prefix=f"{output_prefix}_randomized")
    randomized_inertias = plot_elbow(randomized_final_data_frame, k_max=100, 
                                    output_dir=output_dir, output_prefix=f"{output_prefix}_randomized")
    
    return (randomized_results, randomized_final_data_frame, 
            randomized_ch_scores, randomized_db_scores, randomized_inertias)

def save_results(results, randomized_results, output_dir, output_prefix):
    """Save bootstrap results to CSV files"""
    boot_df = make_bootstrap_df(results)
    boot_df.to_csv(f"{output_dir}/{output_prefix}_boot_data_frame.csv")
    
    randomized_boot_df = make_bootstrap_df(randomized_results)
    randomized_boot_df.to_csv(f"{output_dir}/{output_prefix}_randomized_boot_data_frame.csv")
    
    return boot_df, randomized_boot_df

def generate_evaluation_plots(boot_df, randomized_boot_df, ch_scores, randomized_ch_scores,
                            db_scores, randomized_db_scores, inertias, randomized_inertias,
                            output_dir, output_prefix):
    """Generate and save evaluation plots"""
    silh_result = (None, boot_df, randomized_boot_df)
    
    fig, z_scores = silhouette_zscore(silh_result, min_expr=0.2, x_offset=1, min_y=0.1, 
                                    max_y=0.7, k_max=100, file_name=f"{output_dir}/{output_prefix}_z_score.png")
    
    fig, optimal_k = calinski_harabasz_evaluation(ch_scores, randomized_ch_scores, output_dir, output_prefix)
    fig, optimal_k = davies_bouldin_evaluation(db_scores, randomized_db_scores, output_dir, output_prefix)
    fig, optimal_k = elbow_method_evaluation(inertias, randomized_inertias, output_dir, output_prefix)

def main():
    # Setup
    output_dir = setup_environment()
    output_prefix = 'LPL_ligand_receptor_target_gene_GP'
    
    # Load data
    adata_path = data_dir / 'adata_results.h5ad'
    adata, pathway_genes = load_and_prepare_data(adata_path)
    
    # Process real data
    results, final_data_frame, ch_scores, db_scores, inertias = process_real_data(
        adata, pathway_genes, output_dir, output_prefix
    )
    
    # Process randomized data
    (randomized_results, randomized_final_data_frame, 
     randomized_ch_scores, randomized_db_scores, 
     randomized_inertias) = process_randomized_data(
        adata, pathway_genes, output_dir, output_prefix
    )
    
    # Save results
    boot_df, randomized_boot_df = save_results(
        results, randomized_results, output_dir, output_prefix
    )
    
    # Generate evaluation plots
    generate_evaluation_plots(
        boot_df, randomized_boot_df,
        ch_scores, randomized_ch_scores,
        db_scores, randomized_db_scores,
        inertias, randomized_inertias,
        output_dir, output_prefix
    )

if __name__ == '__main__':
    main()