import pandas as pd
import numpy as np
import scipy
import anndata
from anndata import AnnData

import scipy.sparse as sp
from scipy import stats

import os
import faiss
import torch
import warnings
import multiprocessing as mp

from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.metrics import silhouette_score, pairwise_distances, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pySankey import sankey

output_dir="."
output_prefix="."

def get_genes_list(pathway_df=pd.DataFrame(), pathway_name="Notch"):
    """
    This function extracts the list with genes that belongs to indicated pathway.
    """
    if pathway_name not in pathway_df["pathway"].values:
        raise ValueError("We don't know this pathway! Add the genes in this pathway to our list using the `add_genes` function!")
    
    return pathway_df.loc[pathway_df["pathway"] == pathway_name, "gene"].tolist()

def genes_pathway(pathway_name="Notch", pathway_df=pd.DataFrame(), adata=anndata.AnnData()):
    """
    This function checks which genes from gene list also present in anndata object, and leaves only those which are present in both objects.
    """
    pathway_genes = get_genes_list(pathway_df=pathway_df, pathway_name=pathway_name)
    
    if adata is not None:
        var_names = adata.var_names.tolist()  # Convert Index to list
        pathway_genes = [gene for gene in pathway_genes if gene in var_names]
    
    return list(set(pathway_genes))

def normalize_and_filter(adata, pathway_genes, gene_quantiles=None, scaler=None, sat_val=0.99, min_genes_on=2, min_expr=0.2):
    """
    Process gene expression data for single-cell analysis, focusing on specific pathway genes.
    This function assumes the data is already filtered and square root normalized.
    
    Parameters:
    - adata: AnnData object containing pre-filtered and square root normalized gene expression data
    - pathway_genes: List of genes from pathway
    - gene_quantiles: Pre-computed quantiles for each gene (optional)
    - scaler: Pre-fitted MinMaxScaler (optional)
    - sat_val: Saturation value for upper quantile normalization (default: 0.99, used if gene_quantiles not provided)
    - min_genes_on: Minimum number of genes that should be "ON" in a cell
    - min_expr: Minimum expression value for a gene to be considered "ON"
    
    Returns:
    - adata: Processed AnnData object
    - pathway_genes: Updated list of pathway genes
    - gene_quantiles: Computed or provided gene quantiles
    - scaler: Fitted or provided MinMaxScaler
    """
    # Ensure all pathway genes are in adata
    if not set(pathway_genes).issubset(set(adata.var_names)):
        raise ValueError("Some pathway genes are not in adata.var_names")
    
    # Subset to pathway genes
    adata = adata[:, pathway_genes]
    
    # Convert to dense array if sparse
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    
    # Compute or use provided gene quantiles
    if gene_quantiles is None:
        gene_quantiles = np.quantile(X, sat_val, axis=0)
    
    # Saturation (clipping)
    X_clipped = np.clip(X, a_min=None, a_max=gene_quantiles)
    
    # Scaling
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_clipped)
    else:
        X_scaled = scaler.transform(X_clipped)
    
    # Now filter cells based on normalized values
    genes_on = (X_scaled > min_expr).sum(axis=1)
    mask = genes_on >= min_genes_on
    
    # Apply the filter
    adata = adata[mask, :]
    X_scaled = X_scaled[mask, :]
    
    # Update the AnnData object
    adata.X = X_scaled
    
    return adata, list(adata.var_names), gene_quantiles, scaler


def pairwise_distances_batch(x, y, metric='cosine'):
    """
    Calculate pairwise distances between two sets of vectors.
    
    Args:
    x, y: torch.Tensor, input vectors
    metric: str, distance metric ('cosine' or 'euclidean')
    
    Returns:
    torch.Tensor: Pairwise distances
    """
    if metric == 'cosine':
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)
        y_normalized = y / (y_norm + 1e-8)
        return 1 - torch.mm(x_normalized, y_normalized.t())
    elif metric == 'euclidean':
        return torch.cdist(x, y, p=2)
    else:
        raise ValueError("Unsupported metric")

def batched_silhouette_score_torch(X, labels, metric='cosine', batch_size=1000):
    """
    Calculate the silhouette score using batched processing on GPU.
    
    Args:
    X: torch.Tensor, input data
    labels: array-like, cluster labels
    metric: str, distance metric
    batch_size: int, size of batches for processing
    
    Returns:
    float: Silhouette score
    """
    device = X.device
    n_samples = X.shape[0]
    labels = torch.tensor(labels, device=device)
    unique_labels = torch.unique(labels)
    
    silhouette_scores = []
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = X[i:batch_end]
        batch_labels = labels[i:batch_end]
        
        distances = pairwise_distances_batch(batch_X, X, metric)
        
        a = torch.zeros(batch_end - i, device=device)
        b = torch.full((batch_end - i,), float('inf'), device=device)
        
        for label in unique_labels:
            mask = batch_labels == label
            cluster_size = torch.sum(labels == label).item()
            
            if cluster_size > 1:
                cluster_distances = distances[mask][:, labels == label]
                a[mask] = cluster_distances.sum(dim=1) / (cluster_size - 1)
            else:
                a[mask] = 0
            
            other_mask = labels != label
            other_distances = distances[:, other_mask]
            mean_other_distances = other_distances.mean(dim=1)
            b = torch.min(b, mean_other_distances)
        
        s = (b - a) / torch.max(a, b)
        silhouette_scores.append(s)
    
    return torch.cat(silhouette_scores).mean().item()

def silh_pathway_bootstrap_single_gpu(sub_adata_df, pathway_genes, k_max=10, dist_metric='cosine', n_boots=50, pct_boots=0.9, batch_size=1000):
    """
    Perform silhouette score bootstrap analysis using a single GPU.
    
    Args:
    sub_adata_df: pandas.DataFrame, input data
    pathway_genes: list, genes to use for analysis
    k_max: int, maximum number of clusters
    dist_metric: str, distance metric
    n_boots: int, number of bootstrap iterations
    pct_boots: float, percentage of data to use in each bootstrap
    batch_size: int, batch size for silhouette score calculation
    
    Returns:
    tuple: (boot_list, data_frame)
    """
    boot_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for _ in tqdm(range(n_boots), desc="Bootstrap iterations"):
        data_frame = sub_adata_df.sample(n=int(len(sub_adata_df) * pct_boots), replace=True)
        
        if data_frame.isnull().values.any() or np.isinf(data_frame.values).any():
            raise ValueError("Data contains NaN or infinite values, which cannot be processed.")
        if np.all(data_frame[pathway_genes] == 0, axis=1).any() and dist_metric == "cosine":
            raise ValueError("Data contains zero vectors, which cannot be used with cosine distance.")
        
        x = data_frame[pathway_genes].values
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        
        ss = []
        for k in tqdm(range(2, k_max + 1), desc="Clusters", leave=False):
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300)
            clustering = kmeans.fit_predict(x)
            score = batched_silhouette_score_torch(x_tensor, clustering, metric=dist_metric, batch_size=batch_size)
            ss.append(score)
        
        boot_list.append(ss)
    
    return boot_list, data_frame

def silh_pathway_bootstrap_single_gpu_wrapper(sub_adata_df, pathway_genes, k_max, dist_metric, n_boots, pct_boots, batch_size, gpu_id):
    """
    Wrapper function for silh_pathway_bootstrap_single_gpu that sets the GPU device.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return silh_pathway_bootstrap_single_gpu(sub_adata_df, pathway_genes, k_max, dist_metric, n_boots, pct_boots, batch_size)

def silh_pathway_bootstrap_multiple_gpus(sub_adata_df, pathway_genes, k_max=10, dist_metric='cosine', n_boots=50, pct_boots=0.9, batch_size=1000):
    """
    Perform silhouette score bootstrap analysis using multiple GPUs.
    
    Args:
    (same as silh_pathway_bootstrap_single_gpu)
    
    Returns:
    tuple: (results, sub_adata_df)
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise ValueError("This function requires at least 2 GPUs")
    
    boots_per_gpu = n_boots // n_gpus
    remainder = n_boots % n_gpus
    
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(n_gpus) as pool:
        args_list = [
            (sub_adata_df, pathway_genes, k_max, dist_metric, boots_per_gpu + (1 if i < remainder else 0), pct_boots, batch_size, i)
            for i in range(n_gpus)
        ]
        
        results = []
        with tqdm(total=n_boots, desc="Bootstrap iterations") as pbar:
            for result in pool.starmap(silh_pathway_bootstrap_single_gpu_wrapper, args_list):
                results.extend(result[0])  # Only extend the boot_list, not the data_frame
                pbar.update(len(result[0]))
    
    return results, sub_adata_df

def plot_silhouette_scores(boot_silhouette_scores, k_max, output_dir=None, output_prefix=None):
    """
    Plot the silhouette scores for different numbers of clusters.
    Args:
        boot_silhouette_scores: resulted silhouette scores from computation.
        k_max: number of clusters.
    """
    k_values = np.arange(2, k_max + 1)
    mean_sil = np.mean(boot_silhouette_scores, axis=0)
    ci_sil = stats.t.interval(0.95, len(boot_silhouette_scores)-1, loc=mean_sil, scale=stats.sem(boot_silhouette_scores, axis=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_sil, marker='o', color='blue', label='Mean Silhouette Score')
    plt.fill_between(k_values, ci_sil[0], ci_sil[1], color='blue', alpha=0.2, label='95% CI')
    for scores in boot_silhouette_scores:
        plt.plot(k_values, scores, color='grey', alpha=0.1, linewidth=0.5)
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True)
    
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_silhouette_scores.png")
    plt.show()

def plot_calinski_harabasz_index(X, k_max, output_dir=None, output_prefix=None):
    """
    Calculate and plot the Calinski-Harabasz Index for different numbers of clusters.
    
    This function computes the Calinski-Harabasz Index for k=2 to k_max clusters,
    plots the results, and optionally saves the plot to a file.
    
    Returns:
    - k_values: List of k values (number of clusters)
    - ch_scores: List of Calinski-Harabasz Index scores
    """
    k_values = np.arange(2, k_max + 1)
    ch_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(X)
        ch_scores.append(calinski_harabasz_score(X, labels))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, ch_scores, marker='o', color='green')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.grid(True)
    
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_calinski_harabasz_index.png")
    plt.show()
    
    return ch_scores

def plot_davies_bouldin_index(X, k_max, output_dir=None, output_prefix=None):
    """
    Calculate and plot the Davies-Bouldin Index for different numbers of clusters.
    
    This function computes the Davies-Bouldin Index for k=2 to k_max clusters,
    plots the results, and optionally saves the plot to a file.
    
    Returns:
    - k_values: List of k values (number of clusters)
    - db_scores: List of Davies-Bouldin Index scores
    """
    k_values = np.arange(2, k_max + 1)
    db_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(X)
        db_scores.append(davies_bouldin_score(X, labels))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, db_scores, marker='o', color='red')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.grid(True)
    
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_davies_bouldin_index.png")
    plt.show()
    
    return db_scores

def plot_elbow(X, k_max, output_dir=None, output_prefix=None):
    """
    Calculate and plot the Elbow curve for different numbers of clusters.
    
    This function computes the inertia (within-cluster sum of squares) for k=2 to k_max clusters,
    plots the results, and optionally saves the plot to a file.
    
    Returns:
    - k_values: List of k values (number of clusters)
    - inertias: List of inertia values
    """
    k_values = np.arange(2, k_max + 1)
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', color='purple')
    plt.title('Elbow Plot')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_elbow_plot.png")
    plt.show()
    
    return inertias

def randomize_gene_expression(adata: AnnData, pathway_genes: list[str]) -> AnnData:
    """
    Randomly permute gene expression values for specified genes in an AnnData object.
    
    This function serves as a negative control by removing gene-gene dependencies
    while maintaining the overall distribution of expression values.
    
    Args:
        adata (AnnData): The AnnData object containing gene expression data.
        pathway_genes (list[str]): List of gene names to be randomized.
    
    Returns:
        AnnData: The modified AnnData object with randomized gene expression.
    """
    # Find indices of genes present in the dataset
    gene_indices = [adata.var_names.get_loc(gene) for gene in pathway_genes if gene in adata.var_names]
    
    if len(gene_indices) == 0:
        print("No specified genes found in the dataset.")
        return adata
    
    # Check if adata.X is sparse or dense
    if scipy.sparse.issparse(adata.X):
        # For sparse matrix: convert to dense, permute, then back to sparse
        X_dense = adata.X[:, gene_indices].toarray()
        X_permuted = np.apply_along_axis(np.random.permutation, 0, X_dense)
        adata.X[:, gene_indices] = scipy.sparse.csr_matrix(X_permuted)
    else:
        # For dense matrix: permute directly
        adata.X[:, gene_indices] = np.apply_along_axis(np.random.permutation, 0, adata.X[:, gene_indices])
    
    # Print genes not found in the dataset
    missing_genes = set(pathway_genes) - set(adata.var_names[gene_indices])
    if missing_genes:
        print(f"The following genes were not found in the dataset: {', '.join(missing_genes)}")
    
    return adata

def make_bootstrap_df(s_boots):
    """
    Create a DataFrame from bootstrap results.
    
    Args:
    boots: list of arrays with silhouette scores and bootstrap results
    
    Returns:
    pandas.DataFrame: DataFrame with mean and standard deviation of bootstrap results
    """
    boot_mat = np.vstack(s_boots)
    means = np.mean(boot_mat, axis=0)
    sds = np.std(boot_mat, axis=0)
    boot_df = pd.DataFrame({'m': means, 's': sds})
    boot_df['k'] = np.arange(1, boot_df.shape[0] + 1)
    return boot_df

def perc_k_finder(z_score, percentile=0.9):
    """
    Find the optimal number of clusters based on the Z-score using a vectorized approach.
    """
    max_z = np.max(z_score)
    threshold = max_z * percentile
    above_threshold = z_score >= threshold
    if np.any(above_threshold):
        return np.where(above_threshold)[0][-1]
    else:
        return np.argmax(z_score)

def silhouette_zscore(silh_result, min_expr=0.2, x_offset=1, min_y=0.1, max_y=0.7, k_max=150, file_name=f"{output_dir}/{output_prefix}_z_score.png"):
    # Extract data from silh_result
    boot_df = silh_result[1]
    boot_df_control = silh_result[2]
    
    # Filter data frames
    boot_df = boot_df[boot_df['k'] > x_offset]
    boot_df_control = boot_df_control[boot_df_control['k'] > x_offset]
    
    # Compute the z-score
    diff = boot_df['m'] - boot_df_control['m']
    z_score = diff / boot_df_control['s']
    
    df = pd.DataFrame({'k': boot_df['k'], 'z': z_score})
    
    # Find optimal k
    optimal_k = df['k'].iloc[perc_k_finder(z_score.values, percentile=0.9)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['k'], df['z'], color='blue', linewidth=2)
    ax.axvline(x=optimal_k, linestyle='--', color='red', label=f'Optimal k = {optimal_k}')
    ax.set_ylabel('Z-score')
    ax.set_xlabel('Number of clusters')
    ax.set_title('Silhouette Score Evaluation')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    
    return fig, z_score

def davies_bouldin_evaluation(db_scores, randomized_db_scores, output_dir, output_prefix):
    """
    Evaluate clustering using Davies-Bouldin Index.
    Plot both normal and randomized scores.
    """
    k_values = np.arange(2, len(db_scores) + 2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, db_scores, 'bo-', label='Actual Data')
    ax.plot(k_values, randomized_db_scores, 'ro--', label='Randomized Data')
    
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index Evaluation')
    ax.legend()
    
    # Calculate the difference between actual and randomized scores
    score_diff = np.array(randomized_db_scores) - np.array(db_scores)
    
    # Find optimal k where the difference is maximum
    optimal_k = k_values[np.argmax(score_diff)]
    
    ax.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k = {optimal_k}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_db_evaluation.png")
    plt.show()
    
    return fig, optimal_k

def calinski_harabasz_evaluation(ch_scores, randomized_ch_scores, output_dir, output_prefix):
    """
    Evaluate clustering using Calinski-Harabasz Index.
    Plot both normal and randomized scores.
    """
    k_values = np.arange(2, len(ch_scores) + 2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, ch_scores, 'bo-', label='Actual Data')
    ax.plot(k_values, randomized_ch_scores, 'ro--', label='Randomized Data')
    
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Calinski-Harabasz Index Evaluation')
    ax.legend()
    
    # Calculate the ratio between actual and randomized scores
    score_ratio = np.array(ch_scores) / np.array(randomized_ch_scores)
    
    # Find optimal k where the ratio is maximum
    optimal_k = k_values[np.argmax(score_ratio)]
    
    ax.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k = {optimal_k}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_ch_evaluation.png")
    plt.show()
    
    return fig, optimal_k

def elbow_method_evaluation(inertias, randomized_inertias, output_dir, output_prefix):
    """
    Evaluate clustering using the Elbow method.
    """
    k_values = np.arange(2, len(inertias) + 2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, inertias, 'bo-', label='Actual Data')
    ax.plot(k_values, randomized_inertias, 'ro--', label='Randomized Data')
    
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    ax.legend()
    
    # Calculate the difference between actual and randomized inertias
    inertia_diff = np.array(inertias) - np.array(randomized_inertias)
    optimal_k = k_values[np.argmax(inertia_diff)]
    
    ax.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k = {optimal_k}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}_elbow_plot.png")
    plt.show()
    
    return fig, optimal_k

def perform_optimal_clustering(data_df, pathway_genes, optimal_k, dist_metric='cosine', random_state=42):
    """
    Perform clustering with the optimal number of clusters determined from previous analysis.
    
    Args:
    data_df: pandas.DataFrame, input data
    pathway_genes: list, genes to use for analysis
    optimal_k: int, optimal number of clusters determined from previous analysis
    dist_metric: str, distance metric ('cosine' or 'euclidean')
    random_state: int, random seed for reproducibility
    
    Returns:
    tuple: (cluster_labels, cluster_centers, cluster_sizes, silhouette_scores)
        - cluster_labels: array of cluster assignments for each cell
        - cluster_centers: array of cluster centroids
        - cluster_sizes: dictionary with cluster sizes
        - silhouette_scores: array of silhouette scores for each cell
    """
    # Input validation
    if data_df.isnull().values.any() or np.isinf(data_df.values).any():
        raise ValueError("Data contains NaN or infinite values")
    if np.all(data_df[pathway_genes] == 0, axis=1).any() and dist_metric == "cosine":
        raise ValueError("Data contains zero vectors, which cannot be used with cosine distance")
    
    # Prepare data
    X = data_df[pathway_genes].values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, max_iter=300, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate silhouette scores for each point
    batch_size = 1000  # Adjust based on your GPU memory
    silhouette_scores = []
    
    for i in range(0, len(X), batch_size):
        batch_end = min(i + batch_size, len(X))
        batch_X = X_tensor[i:batch_end]
        batch_labels = cluster_labels[i:batch_end]
        
        # Calculate distances
        distances = pairwise_distances_batch(batch_X, X_tensor, metric=dist_metric)
        
        # Calculate a (mean distance to same cluster) and b (mean distance to nearest cluster)
        a = torch.zeros(batch_end - i, device=device)
        b = torch.full((batch_end - i,), float('inf'), device=device)
        
        for label in range(optimal_k):
            mask = batch_labels == label
            cluster_size = np.sum(cluster_labels == label)
            
            if cluster_size > 1:
                cluster_distances = distances[mask][:, cluster_labels == label]
                a[mask] = cluster_distances.sum(dim=1) / (cluster_size - 1)
            
            other_mask = cluster_labels != label
            other_distances = distances[:, other_mask]
            mean_other_distances = other_distances.mean(dim=1)
            b = torch.min(b, mean_other_distances)
        
        s = (b - a) / torch.max(a, b)
        silhouette_scores.append(s.cpu().numpy())
    
    silhouette_scores = np.concatenate(silhouette_scores)
    
    # Calculate cluster sizes
    cluster_sizes = {i: np.sum(cluster_labels == i) for i in range(optimal_k)}
    
    # Add clustering results to the original dataframe
    data_df['cluster'] = cluster_labels
    data_df['silhouette_score'] = silhouette_scores
    
    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate summary statistics for each cluster
    cluster_stats = {}
    for cluster in range(optimal_k):
        cluster_mask = cluster_labels == cluster
        cluster_stats[cluster] = {
            'size': cluster_sizes[cluster],
            'mean_silhouette': np.mean(silhouette_scores[cluster_mask]),
            'std_silhouette': np.std(silhouette_scores[cluster_mask]),
            'median_silhouette': np.median(silhouette_scores[cluster_mask])
        }
    
    return {
        'data': data_df,  # Original data with cluster assignments and silhouette scores
        'labels': cluster_labels,  # Cluster assignments
        'centers': cluster_centers,  # Cluster centroids
        'sizes': cluster_sizes,  # Number of points in each cluster
        'silhouette_scores': silhouette_scores,  # Individual silhouette scores
        'cluster_stats': cluster_stats,  # Summary statistics for each cluster
    }

def visualize_clustering_results(results, pathway_genes, output_dir=None, output_prefix=None):
    """
    Create visualizations for the clustering results.
    
    Args:
    results: dict, output from perform_optimal_clustering
    pathway_genes: list, genes used for clustering
    output_dir: str, directory to save plots (optional)
    output_prefix: str, prefix for plot filenames (optional)
    """
    # 1. Silhouette score distribution by cluster
    plt.figure(figsize=(12, 6))
    plt.boxplot([results['silhouette_scores'][results['labels'] == i] 
                for i in range(len(results['sizes']))])
    plt.title('Silhouette Score Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_silhouette_distribution.png")
    plt.close()
    
    # 2. Cluster sizes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results['sizes'])), list(results['sizes'].values()))
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_cluster_sizes.png")
    plt.close()
    
    # 3. Gene expression heatmap of cluster centers
    plt.figure(figsize=(15, 8))
    sns.heatmap(results['centers'], 
                xticklabels=pathway_genes,
                yticklabels=[f'Cluster {i}' for i in range(len(results['sizes']))],
                cmap='magma_r',
                center=0.5)
    plt.title('Gene Expression Pattern by Cluster')
    plt.xticks(rotation=45, ha='right')
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}_expression_heatmap.png")
    plt.close()


def create_basic_sankey(adata, 
                       left_col, 
                       right_col, 
                       left_label=None,
                       right_label=None,
                       output_dir=None, 
                       output_prefix=None,
                       figsize=(15, 10),
                       fontsize=12,
                       aspect=20):
    """
    Create a Sankey plot showing relationship between two categorical variables.
    
    Args:
    adata: AnnData object
    left_col: str, name of the column for left side of Sankey
    right_col: str, name of the column for right side of Sankey
    left_label: str, optional label for left side (defaults to column name)
    right_label: str, optional label for right side (defaults to column name)
    output_dir: str, directory to save the plot
    output_prefix: str, prefix for the output file name
    figsize: tuple, figure size (width, height)
    fontsize: int, font size for labels
    aspect: int, aspect ratio of the plot
    """
    # Validate column names
    if left_col not in adata.obs.columns:
        raise ValueError(f"Column '{left_col}' not found in adata.obs")
    if right_col not in adata.obs.columns:
        raise ValueError(f"Column '{right_col}' not found in adata.obs")
    
    # Use column names as labels if not provided
    left_label = left_label or left_col
    right_label = right_label or right_col
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create Sankey plot
    sankey.sankey(
        left=adata.obs[left_col],
        right=adata.obs[right_col],
        aspect=aspect,
        fontsize=fontsize,
        colorDict=None  # Auto-generate colors
    )
    
    # Customize plot
    plt.title(f'{left_label} to {right_label}', fontsize=fontsize+2, pad=20)
    
    # Save if output directory is specified
    if output_dir and output_prefix:
        plt.savefig(f"{output_dir}/{output_prefix}.png", 
                   bbox_inches='tight', dpi=300)
    plt.close()

def create_group_specific_sankeys(adata, 
                                left_col, 
                                right_col, 
                                group_col,
                                left_label=None,
                                right_label=None,
                                group_label=None,
                                output_dir=None, 
                                output_prefix=None,
                                figsize=(15, 10),
                                fontsize=12,
                                aspect=20):
    """
    Create separate Sankey plots for each group showing relationship between two categorical variables.
    
    Args:
    adata: AnnData object
    left_col: str, name of the column for left side of Sankey
    right_col: str, name of the column for right side of Sankey
    group_col: str, name of the column to group by
    left_label: str, optional label for left side (defaults to column name)
    right_label: str, optional label for right side (defaults to column name)
    group_label: str, optional label for groups (defaults to column name)
    output_dir: str, directory to save the plots
    output_prefix: str, prefix for the output file names
    figsize: tuple, figure size (width, height)
    fontsize: int, font size for labels
    aspect: int, aspect ratio of the plot
    """
    # Validate column names
    for col in [left_col, right_col, group_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
    
    # Use column names as labels if not provided
    left_label = left_label or left_col
    right_label = right_label or right_col
    group_label = group_label or group_col
    
    # Get unique groups
    groups = adata.obs[group_col].unique()
    
    # Create a separate plot for each group
    for group in groups:
        # Subset data for this group
        group_data = adata[adata.obs[group_col] == group]
        
        # Skip if no data in this group
        if len(group_data) == 0:
            print(f"Warning: No data found for {group_label} {group}")
            continue
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create Sankey plot
        sankey.sankey(
            left=group_data.obs[left_col],
            right=group_data.obs[right_col],
            aspect=aspect,
            fontsize=fontsize,
            colorDict=None
        )
        
        # Customize plot
        plt.title(f'{left_label} to {right_label} in {group_label} {group}', 
                 fontsize=fontsize+2, pad=20)
        
        # Save if output directory is specified
        if output_dir and output_prefix:
            plt.savefig(
                f"{output_dir}/{output_prefix}_sankey_{left_col}_to_{right_col}_{group_col}_{group}.png",
                bbox_inches='tight', dpi=300
            )
        plt.close()