"""
Visualization Tools for Audio Embedding Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import os

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class EmbeddingVisualizer:
    """
    Comprehensive visualization tools for embedding analysis
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_similarity_matrices(self,
                                 embeddings_a: np.ndarray,
                                 embeddings_b: np.ndarray,
                                 labels: Optional[np.ndarray] = None,
                                 model_a_name: str = "Model A",
                                 model_b_name: str = "Model B",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot side-by-side similarity matrices
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            labels: Optional labels for sorting
            model_a_name: Name of model A
            model_b_name: Name of model B
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        from sklearn.preprocessing import normalize
        
        # Normalize and compute similarities
        emb_a_norm = normalize(embeddings_a)
        emb_b_norm = normalize(embeddings_b)
        
        sim_a = np.dot(emb_a_norm, emb_a_norm.T)
        sim_b = np.dot(emb_b_norm, emb_b_norm.T)
        
        # Sort by labels if provided
        if labels is not None:
            sort_idx = np.argsort(labels)
            sim_a = sim_a[sort_idx][:, sort_idx]
            sim_b = sim_b[sort_idx][:, sort_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Model A similarity
        im1 = axes[0].imshow(sim_a, cmap='viridis', aspect='auto')
        axes[0].set_title(f'{model_a_name} Similarity Matrix', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Sample Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Model B similarity
        im2 = axes[1].imshow(sim_b, cmap='viridis', aspect='auto')
        axes[1].set_title(f'{model_b_name} Similarity Matrix', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Sample Index')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = sim_a - sim_b
        im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[2].set_title('Difference (A - B)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Sample Index')
        axes[2].set_ylabel('Sample Index')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved similarity matrices to {save_path}")
        
        return fig
    
    def plot_dimensionality_reduction(self,
                                      embeddings_a: np.ndarray,
                                      embeddings_b: np.ndarray,
                                      labels: np.ndarray,
                                      method: str = "tsne",
                                      model_a_name: str = "Wav2Vec2",
                                      model_b_name: str = "Data2Vec",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot dimensionality reduction comparison
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            labels: Class labels
            method: Reduction method ("tsne", "umap", "pca")
            model_a_name: Name of model A
            model_b_name: Name of model B
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Apply dimensionality reduction
        if method == "tsne":
            reducer_a = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_a)-1))
            reducer_b = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_b)-1))
        elif method == "umap":
            if not UMAP_AVAILABLE:
                print("UMAP not available, using PCA instead")
                reducer_a = PCA(n_components=2)
                reducer_b = PCA(n_components=2)
            else:
                reducer_a = umap.UMAP(n_components=2, random_state=42)
                reducer_b = umap.UMAP(n_components=2, random_state=42)
        else:  # pca
            reducer_a = PCA(n_components=2)
            reducer_b = PCA(n_components=2)
        
        # Reduce dimensions
        emb_a_2d = reducer_a.fit_transform(embeddings_a)
        emb_b_2d = reducer_b.fit_transform(embeddings_b)
        
        # Plot model A
        scatter_a = axes[0].scatter(
            emb_a_2d[:, 0], emb_a_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.6, s=50
        )
        axes[0].set_title(f'{model_a_name} - {method.upper()}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter_a, ax=axes[0], label='Class')
        
        # Plot model B
        scatter_b = axes[1].scatter(
            emb_b_2d[:, 0], emb_b_2d[:, 1],
            c=labels, cmap='tab10', alpha=0.6, s=50
        )
        axes[1].set_title(f'{model_b_name} - {method.upper()}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter_b, ax=axes[1], label='Class')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dimensionality reduction plot to {save_path}")
        
        return fig
    
    def plot_layer_comparison(self,
                              embeddings_by_layer_a: Dict[int, np.ndarray],
                              embeddings_by_layer_b: Dict[int, np.ndarray],
                              labels: np.ndarray,
                              model_a_name: str = "Wav2Vec2",
                              model_b_name: str = "Data2Vec",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare embeddings across layers
        
        Args:
            embeddings_by_layer_a: Dict of embeddings per layer (model A)
            embeddings_by_layer_b: Dict of embeddings per layer (model B)
            labels: Class labels
            model_a_name: Name of model A
            model_b_name: Name of model B
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        layers_a = sorted(embeddings_by_layer_a.keys())
        layers_b = sorted(embeddings_by_layer_b.keys())
        
        n_layers = max(len(layers_a), len(layers_b))
        fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 10))
        
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, layer_idx in enumerate(layers_a):
            if layer_idx in embeddings_by_layer_a:
                emb = embeddings_by_layer_a[layer_idx]
                
                # Apply PCA for visualization
                pca = PCA(n_components=2)
                emb_2d = pca.fit_transform(emb)
                
                scatter = axes[0, i].scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    c=labels, cmap='tab10', alpha=0.6, s=30
                )
                axes[0, i].set_title(f'{model_a_name} - Layer {layer_idx}', fontsize=10)
                axes[0, i].set_xlabel('PC1')
                axes[0, i].set_ylabel('PC2')
        
        for i, layer_idx in enumerate(layers_b):
            if layer_idx in embeddings_by_layer_b:
                emb = embeddings_by_layer_b[layer_idx]
                
                # Apply PCA for visualization
                pca = PCA(n_components=2)
                emb_2d = pca.fit_transform(emb)
                
                scatter = axes[1, i].scatter(
                    emb_2d[:, 0], emb_2d[:, 1],
                    c=labels, cmap='tab10', alpha=0.6, s=30
                )
                axes[1, i].set_title(f'{model_b_name} - Layer {layer_idx}', fontsize=10)
                axes[1, i].set_xlabel('PC1')
                axes[1, i].set_ylabel('PC2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved layer comparison to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self,
                                results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of various metrics
        
        Args:
            results: Comparison results dictionary
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract metrics
        model_a_name = results['models']['model_a']
        model_b_name = results['models']['model_b']
        
        metrics_a = []
        metrics_b = []
        metric_names = []
        
        # Similarity metrics
        if 'similarity_comparison' in results:
            sim = results['similarity_comparison']
            metrics_a.append(sim['mean_sim_model_a'])
            metrics_b.append(sim['mean_sim_model_b'])
            metric_names.append('Mean Cosine\nSimilarity')
        
        # Clustering metrics
        if 'clustering_comparison' in results:
            clust = results['clustering_comparison']
            metrics_a.append(clust['silhouette_model_a'])
            metrics_b.append(clust['silhouette_model_b'])
            metric_names.append('Silhouette\nScore')
        
        # Classification metrics
        if 'classification_comparison' in results:
            clf = results['classification_comparison']
            metrics_a.append(clf['accuracy_model_a'])
            metrics_b.append(clf['accuracy_model_b'])
            metric_names.append('Classification\nAccuracy')
        
        # Plot bar chart
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, metrics_a, width, label=model_a_name, alpha=0.8)
        axes[0, 0].bar(x + width/2, metrics_b, width, label=model_b_name, alpha=0.8)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Retrieval metrics
        if 'retrieval_comparison' in results:
            ret = results['retrieval_comparison']
            k_values = []
            recall_a = []
            recall_b = []
            
            for k in [1, 5, 10]:
                key_a = f'recall@{k}_model_a'
                if key_a in ret:
                    k_values.append(f'R@{k}')
                    recall_a.append(ret[key_a])
                    recall_b.append(ret[f'recall@{k}_model_b'])
            
            x = np.arange(len(k_values))
            axes[0, 1].bar(x - width/2, recall_a, width, label=model_a_name, alpha=0.8)
            axes[0, 1].bar(x + width/2, recall_b, width, label=model_b_name, alpha=0.8)
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_title('Retrieval Performance', fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(k_values)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Similarity matrix correlation
        if 'similarity_comparison' in results:
            sim = results['similarity_comparison']
            axes[1, 0].bar(['Correlation'], [sim['similarity_correlation']], color='steelblue', alpha=0.8)
            axes[1, 0].set_ylabel('Correlation Coefficient')
            axes[1, 0].set_title('Similarity Matrix Correlation', fontweight='bold')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Compactness comparison
        if 'clustering_comparison' in results:
            clust = results['clustering_comparison']
            axes[1, 1].bar([model_a_name, model_b_name], 
                          [clust['compactness_model_a'], clust['compactness_model_b']],
                          color=['coral', 'skyblue'], alpha=0.8)
            axes[1, 1].set_ylabel('Intra-cluster Distance')
            axes[1, 1].set_title('Cluster Compactness', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        
        return fig
    
    def plot_embedding_distributions(self,
                                     embeddings_a: np.ndarray,
                                     embeddings_b: np.ndarray,
                                     model_a_name: str = "Wav2Vec2",
                                     model_b_name: str = "Data2Vec",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution statistics of embeddings
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Mean activation per dimension
        mean_a = embeddings_a.mean(axis=0)
        mean_b = embeddings_b.mean(axis=0)
        
        axes[0, 0].plot(mean_a, alpha=0.7, label=model_a_name)
        axes[0, 0].plot(mean_b, alpha=0.7, label=model_b_name)
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].set_title('Mean Activation per Dimension')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation per dimension
        std_a = embeddings_a.std(axis=0)
        std_b = embeddings_b.std(axis=0)
        
        axes[0, 1].plot(std_a, alpha=0.7, label=model_a_name)
        axes[0, 1].plot(std_b, alpha=0.7, label=model_b_name)
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Std Dev per Dimension')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of all values
        axes[0, 2].hist(embeddings_a.flatten(), bins=50, alpha=0.5, label=model_a_name, density=True)
        axes[0, 2].hist(embeddings_b.flatten(), bins=50, alpha=0.5, label=model_b_name, density=True)
        axes[0, 2].set_xlabel('Embedding Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Value Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # L2 norms
        norms_a = np.linalg.norm(embeddings_a, axis=1)
        norms_b = np.linalg.norm(embeddings_b, axis=1)
        
        axes[1, 0].hist(norms_a, bins=30, alpha=0.5, label=model_a_name, density=True)
        axes[1, 0].hist(norms_b, bins=30, alpha=0.5, label=model_b_name, density=True)
        axes[1, 0].set_xlabel('L2 Norm')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Embedding L2 Norm Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pairwise distances
        from scipy.spatial.distance import pdist
        
        dists_a = pdist(embeddings_a, metric='euclidean')
        dists_b = pdist(embeddings_b, metric='euclidean')
        
        axes[1, 1].hist(dists_a, bins=50, alpha=0.5, label=model_a_name, density=True)
        axes[1, 1].hist(dists_b, bins=50, alpha=0.5, label=model_b_name, density=True)
        axes[1, 1].set_xlabel('Pairwise Distance')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Pairwise Distance Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Variance explained by PCA
        pca_a = PCA()
        pca_b = PCA()
        
        pca_a.fit(embeddings_a)
        pca_b.fit(embeddings_b)
        
        cumvar_a = np.cumsum(pca_a.explained_variance_ratio_)
        cumvar_b = np.cumsum(pca_b.explained_variance_ratio_)
        
        axes[1, 2].plot(cumvar_a, alpha=0.7, label=model_a_name)
        axes[1, 2].plot(cumvar_b, alpha=0.7, label=model_b_name)
        axes[1, 2].set_xlabel('Number of Components')
        axes[1, 2].set_ylabel('Cumulative Variance Explained')
        axes[1, 2].set_title('PCA Variance Explained')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plots to {save_path}")
        
        return fig
    
    def plot_attention_heatmap(self,
                               attention_weights: np.ndarray,
                               audio_duration: float,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention weights as heatmap
        
        Args:
            attention_weights: Attention weights (layer, head, seq_len, seq_len)
            audio_duration: Duration of audio in seconds
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        n_layers = attention_weights.shape[0]
        n_heads = attention_weights.shape[1]
        
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(3*n_heads, 3*n_layers))
        
        if n_layers == 1 and n_heads == 1:
            axes = [[axes]]
        elif n_layers == 1:
            axes = [axes]
        elif n_heads == 1:
            axes = [[ax] for ax in axes]
        
        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attention_weights[layer, head]
                im = axes[layer][head].imshow(attn, cmap='viridis', aspect='auto')
                axes[layer][head].set_title(f'L{layer}H{head}', fontsize=8)
                axes[layer][head].set_xticks([])
                axes[layer][head].set_yticks([])
        
        plt.suptitle('Attention Patterns Across Layers and Heads', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")
        
        return fig
    
    def create_comparison_report(self,
                                 embeddings_a: np.ndarray,
                                 embeddings_b: np.ndarray,
                                 labels: np.ndarray,
                                 results: Dict,
                                 model_a_name: str = "Wav2Vec2",
                                 model_b_name: str = "Data2Vec",
                                 output_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive comparison report with multiple figures
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            labels: Class labels
            results: Comparison results dictionary
            model_a_name: Name of model A
            model_b_name: Name of model B
            output_dir: Directory to save all figures
        
        Returns:
            Path to report directory
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        report_dir = os.path.join(output_dir, 'comparison_report')
        os.makedirs(report_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE COMPARISON REPORT")
        print("="*70)
        
        # 1. Similarity matrices
        print("\n[1/5] Plotting similarity matrices...")
        self.plot_similarity_matrices(
            embeddings_a, embeddings_b, labels,
            model_a_name, model_b_name,
            save_path=os.path.join(report_dir, 'similarity_matrices.png')
        )
        
        # 2. Dimensionality reduction
        print("[2/5] Plotting dimensionality reduction (t-SNE)...")
        self.plot_dimensionality_reduction(
            embeddings_a, embeddings_b, labels, method='tsne',
            model_a_name=model_a_name, model_b_name=model_b_name,
            save_path=os.path.join(report_dir, 'tsne_comparison.png')
        )
        
        # 3. UMAP if available
        if UMAP_AVAILABLE:
            print("[3/5] Plotting dimensionality reduction (UMAP)...")
            self.plot_dimensionality_reduction(
                embeddings_a, embeddings_b, labels, method='umap',
                model_a_name=model_a_name, model_b_name=model_b_name,
                save_path=os.path.join(report_dir, 'umap_comparison.png')
            )
        
        # 4. Metrics comparison
        print("[4/5] Plotting metrics comparison...")
        self.plot_metrics_comparison(
            results,
            save_path=os.path.join(report_dir, 'metrics_comparison.png')
        )
        
        # 5. Embedding distributions
        print("[5/5] Plotting embedding distributions...")
        self.plot_embedding_distributions(
            embeddings_a, embeddings_b,
            model_a_name, model_b_name,
            save_path=os.path.join(report_dir, 'embedding_distributions.png')
        )
        
        print(f"\n✓ Report saved to: {report_dir}")
        print("="*70)
        
        return report_dir