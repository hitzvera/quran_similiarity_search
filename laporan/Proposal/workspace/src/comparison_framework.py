"""
Comparison Framework for Audio Embeddings
Provides comprehensive comparison between Wav2Vec2 and Data2Vec
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')


class EmbeddingComparator:
    """
    Comprehensive comparison framework for audio embeddings
    """
    
    def __init__(self, 
                 model_a_name: str,
                 model_b_name: str):
        """
        Initialize comparator
        
        Args:
            model_a_name: Name of first model (e.g., "wav2vec2")
            model_b_name: Name of second model (e.g., "data2vec")
        """
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.results_history = []
    
    def compare_pairwise_similarity(self,
                                    embeddings_a: np.ndarray,
                                    embeddings_b: np.ndarray,
                                    labels: Optional[List] = None) -> Dict:
        """
        Compare pairwise similarity matrices between two models
        
        Args:
            embeddings_a: Embeddings from model A (n_samples, dim_a)
            embeddings_b: Embeddings from model B (n_samples, dim_b)
            labels: Optional ground truth labels
        
        Returns:
            Comparison metrics
        """
        # Normalize embeddings
        emb_a_norm = normalize(embeddings_a)
        emb_b_norm = normalize(embeddings_b)
        
        # Compute similarity matrices
        sim_a = cosine_similarity(emb_a_norm)
        sim_b = cosine_similarity(emb_b_norm)
        
        # Matrix correlation
        upper_tri_a = sim_a[np.triu_indices_from(sim_a, k=1)]
        upper_tri_b = sim_b[np.triu_indices_from(sim_b, k=1)]
        
        correlation = np.corrcoef(upper_tri_a, upper_tri_b)[0, 1]
        
        # Frobenius norm difference
        frobenius_diff = np.linalg.norm(sim_a - sim_b, 'fro')
        
        # Relative difference
        mean_sim = (sim_a + sim_b) / 2
        relative_diff = np.linalg.norm(sim_a - sim_b, 'fro') / (np.linalg.norm(mean_sim, 'fro') + 1e-8)
        
        result = {
            "similarity_correlation": correlation,
            "frobenius_difference": frobenius_diff,
            "relative_difference": relative_diff,
            "mean_sim_model_a": upper_tri_a.mean(),
            "mean_sim_model_b": upper_tri_b.mean(),
            "std_sim_model_a": upper_tri_a.std(),
            "std_sim_model_b": upper_tri_b.std()
        }
        
        return result
    
    def compare_clustering_quality(self,
                                   embeddings_a: np.ndarray,
                                   embeddings_b: np.ndarray,
                                   n_clusters: int = 10,
                                   labels: Optional[np.ndarray] = None) -> Dict:
        """
        Compare clustering quality between models
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            n_clusters: Number of clusters
            labels: Ground truth labels (optional)
        
        Returns:
            Clustering comparison metrics
        """
        results = {}
        
        # K-means clustering on both embeddings
        kmeans_a = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_b = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels_a = kmeans_a.fit_predict(embeddings_a)
        labels_b = kmeans_b.fit_predict(embeddings_b)
        
        # Silhouette scores
        sil_a = silhouette_score(embeddings_a, labels_a)
        sil_b = silhouette_score(embeddings_b, labels_b)
        
        results["silhouette_model_a"] = sil_a
        results["silhouette_model_b"] = sil_b
        results["silhouette_difference"] = sil_a - sil_b
        
        # Cluster agreement (Adjusted Rand Index)
        ari = adjusted_rand_score(labels_a, labels_b)
        results["cluster_agreement_ari"] = ari
        
        # If ground truth labels provided
        if labels is not None:
            ari_a = adjusted_rand_score(labels, labels_a)
            ari_b = adjusted_rand_score(labels, labels_b)
            results["ari_vs_ground_truth_model_a"] = ari_a
            results["ari_vs_ground_truth_model_b"] = ari_b
        
        # Cluster compactness (intra-cluster distance)
        def compute_compactness(embeddings, cluster_labels, n_clusters):
            compactness = 0
            for k in range(n_clusters):
                cluster_points = embeddings[cluster_labels == k]
                if len(cluster_points) > 1:
                    centroid = cluster_points.mean(axis=0)
                    compactness += np.mean([euclidean(p, centroid) for p in cluster_points])
            return compactness / n_clusters
        
        compactness_a = compute_compactness(embeddings_a, labels_a, n_clusters)
        compactness_b = compute_compactness(embeddings_b, labels_b, n_clusters)
        
        results["compactness_model_a"] = compactness_a
        results["compactness_model_b"] = compactness_b
        
        return results
    
    def compare_discriminative_power(self,
                                     embeddings_a: np.ndarray,
                                     embeddings_b: np.ndarray,
                                     labels: np.ndarray,
                                     test_size: float = 0.2) -> Dict:
        """
        Compare discriminative power using a simple classifier
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            labels: Class labels
            test_size: Fraction of data for testing
        
        Returns:
            Classification performance metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        
        results = {}
        
        # Split data
        X_a_train, X_a_test, y_train, y_test = train_test_split(
            embeddings_a, labels, test_size=test_size, random_state=42, stratify=labels
        )
        X_b_train, X_b_test, _, _ = train_test_split(
            embeddings_b, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train classifiers
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        
        clf_a.fit(X_a_train, y_train)
        clf_b.fit(X_b_train, y_train)
        
        # Predict and evaluate
        y_pred_a = clf_a.predict(X_a_test)
        y_pred_b = clf_b.predict(X_b_test)
        
        results["accuracy_model_a"] = accuracy_score(y_test, y_pred_a)
        results["accuracy_model_b"] = accuracy_score(y_test, y_pred_b)
        results["f1_model_a"] = f1_score(y_test, y_pred_a, average='weighted')
        results["f1_model_b"] = f1_score(y_test, y_pred_b, average='weighted')
        results["accuracy_difference"] = results["accuracy_model_a"] - results["accuracy_model_b"]
        
        return results
    
    def compare_retrieval_performance(self,
                                      embeddings_a: np.ndarray,
                                      embeddings_b: np.ndarray,
                                      labels: np.ndarray,
                                      k_values: List[int] = [1, 5, 10]) -> Dict:
        """
        Compare information retrieval performance
        
        Args:
            embeddings_a: Embeddings from model A
            embeddings_b: Embeddings from model B
            labels: Ground truth labels
            k_values: List of k values for recall@k
        
        Returns:
            Retrieval metrics
        """
        results = {}
        
        # Normalize embeddings
        emb_a_norm = normalize(embeddings_a)
        emb_b_norm = normalize(embeddings_b)
        
        # Compute similarity matrices
        sim_a = cosine_similarity(emb_a_norm)
        sim_b = cosine_similarity(emb_b_norm)
        
        # Set diagonal to -inf to exclude self-similarity
        np.fill_diagonal(sim_a, -np.inf)
        np.fill_diagonal(sim_b, -np.inf)
        
        for k in k_values:
            recall_a = 0
            recall_b = 0
            mrr_a = 0
            mrr_b = 0
            
            for i in range(len(labels)):
                # Get top-k indices for model A
                top_k_a = np.argsort(sim_a[i])[-k:]
                correct_a = sum(labels[top_k_a] == labels[i])
                recall_a += correct_a / sum(labels == labels[i])
                
                # MRR for model A
                sorted_indices_a = np.argsort(sim_a[i])[::-1]
                for rank, idx in enumerate(sorted_indices_a, 1):
                    if labels[idx] == labels[i]:
                        mrr_a += 1.0 / rank
                        break
                
                # Get top-k indices for model B
                top_k_b = np.argsort(sim_b[i])[-k:]
                correct_b = sum(labels[top_k_b] == labels[i])
                recall_b += correct_b / sum(labels == labels[i])
                
                # MRR for model B
                sorted_indices_b = np.argsort(sim_b[i])[::-1]
                for rank, idx in enumerate(sorted_indices_b, 1):
                    if labels[idx] == labels[i]:
                        mrr_b += 1.0 / rank
                        break
            
            results[f"recall@{k}_model_a"] = recall_a / len(labels)
            results[f"recall@{k}_model_b"] = recall_b / len(labels)
            results[f"recall@{k}_difference"] = results[f"recall@{k}_model_a"] - results[f"recall@{k}_model_b"]
        
        results["mrr_model_a"] = mrr_a / len(labels)
        results["mrr_model_b"] = mrr_b / len(labels)
        results["mrr_difference"] = results["mrr_model_a"] - results["mrr_model_b"]
        
        return results
    
    def compare_layer_evolution(self,
                                embeddings_a_by_layer: Dict[int, np.ndarray],
                                embeddings_b_by_layer: Dict[int, np.ndarray]) -> Dict:
        """
        Compare how embeddings evolve through layers
        
        Args:
            embeddings_a_by_layer: Dict mapping layer index to embeddings (model A)
            embeddings_b_by_layer: Dict mapping layer index to embeddings (model B)
        
        Returns:
            Layer evolution metrics
        """
        results = {
            "layer_wise_correlation": [],
            "dimensional_variance": {"model_a": [], "model_b": []}
        }
        
        layers_a = sorted(embeddings_a_by_layer.keys())
        layers_b = sorted(embeddings_b_by_layer.keys())
        
        # Compute correlation between corresponding layers
        for la, lb in zip(layers_a, layers_b):
            if la in embeddings_a_by_layer and lb in embeddings_b_by_layer:
                emb_a = embeddings_a_by_layer[la]
                emb_b = embeddings_b_by_layer[lb]
                
                # Normalize
                emb_a_norm = normalize(emb_a)
                emb_b_norm = normalize(emb_b)
                
                # Compute similarity matrices
                sim_a = cosine_similarity(emb_a_norm)
                sim_b = cosine_similarity(emb_b_norm)
                
                # Correlation
                upper_a = sim_a[np.triu_indices_from(sim_a, k=1)]
                upper_b = sim_b[np.triu_indices_from(sim_b, k=1)]
                corr = np.corrcoef(upper_a, upper_b)[0, 1]
                
                results["layer_wise_correlation"].append({
                    "layer_a": la,
                    "layer_b": lb,
                    "correlation": corr
                })
                
                # Dimensional variance (explained variance ratio)
                from sklearn.decomposition import PCA
                
                pca_a = PCA(n_components=min(50, emb_a.shape[1]))
                pca_a.fit(emb_a)
                results["dimensional_variance"]["model_a"].append(
                    pca_a.explained_variance_ratio_.sum()
                )
                
                pca_b = PCA(n_components=min(50, emb_b.shape[1]))
                pca_b.fit(emb_b)
                results["dimensional_variance"]["model_b"].append(
                    pca_b.explained_variance_ratio_.sum()
                )
        
        return results
    
    def statistical_significance_test(self,
                                      metric_values_a: np.ndarray,
                                      metric_values_b: np.ndarray,
                                      test: str = "paired_ttest") -> Dict:
        """
        Test statistical significance of difference between models
        
        Args:
            metric_values_a: Metric values from model A (multiple runs)
            metric_values_b: Metric values from model B (multiple runs)
            test: Statistical test to use
        
        Returns:
            Statistical test results
        """
        from scipy import stats
        
        if test == "paired_ttest":
            t_stat, p_value = stats.ttest_rel(metric_values_a, metric_values_b)
        elif test == "wilcoxon":
            t_stat, p_value = stats.wilcoxon(metric_values_a, metric_values_b)
        elif test == "mann_whitney":
            t_stat, p_value = stats.mannwhitneyu(metric_values_a, metric_values_b)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            "test_name": test,
            "statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "mean_diff": np.mean(metric_values_a) - np.mean(metric_values_b),
            "std_diff": np.std(metric_values_a - metric_values_b)
        }
    
    def run_full_comparison(self,
                           embeddings_a: np.ndarray,
                           embeddings_b: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           audio_files: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive comparison
        
        Args:
            embeddings_a: All embeddings from model A
            embeddings_b: All embeddings from model B
            labels: Optional class labels
            audio_files: Optional list of audio file paths
        
        Returns:
            Complete comparison results
        """
        results = {
            "models": {
                "model_a": self.model_a_name,
                "model_b": self.model_b_name
            },
            "dataset_info": {
                "num_samples": len(embeddings_a),
                "embedding_dim_a": embeddings_a.shape[1],
                "embedding_dim_b": embeddings_b.shape[1]
            }
        }
        
        # 1. Pairwise similarity comparison
        print("Computing pairwise similarity...")
        results["similarity_comparison"] = self.compare_pairwise_similarity(
            embeddings_a, embeddings_b, labels
        )
        
        # 2. Clustering quality
        if labels is not None:
            n_clusters = len(np.unique(labels))
            print(f"Evaluating clustering (k={n_clusters})...")
            results["clustering_comparison"] = self.compare_clustering_quality(
                embeddings_a, embeddings_b, n_clusters, labels
            )
        
        # 3. Discriminative power
        if labels is not None:
            print("Evaluating discriminative power...")
            results["classification_comparison"] = self.compare_discriminative_power(
                embeddings_a, embeddings_b, labels
            )
        
        # 4. Retrieval performance
        if labels is not None:
            print("Evaluating retrieval performance...")
            results["retrieval_comparison"] = self.compare_retrieval_performance(
                embeddings_a, embeddings_b, labels
            )
        
        self.results_history.append(results)
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a formatted comparison report
        
        Args:
            results: Comparison results dictionary
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("AUDIO EMBEDDING COMPARISON REPORT")
        report.append("=" * 70)
        report.append(f"\nModel A: {results['models']['model_a']}")
        report.append(f"Model B: {results['models']['model_b']}")
        report.append(f"\nDataset: {results['dataset_info']['num_samples']} samples")
        
        # Similarity comparison
        if "similarity_comparison" in results:
            sim = results["similarity_comparison"]
            report.append("\n" + "-" * 70)
            report.append("PAIRWISE SIMILARITY COMPARISON")
            report.append("-" * 70)
            report.append(f"Similarity Matrix Correlation: {sim['similarity_correlation']:.4f}")
            report.append(f"Frobenius Norm Difference: {sim['frobenius_difference']:.4f}")
            report.append(f"Relative Difference: {sim['relative_difference']:.4f}")
            report.append(f"\nMean Similarity - {results['models']['model_a']}: {sim['mean_sim_model_a']:.4f}")
            report.append(f"Mean Similarity - {results['models']['model_b']}: {sim['mean_sim_model_b']:.4f}")
        
        # Clustering comparison
        if "clustering_comparison" in results:
            clust = results["clustering_comparison"]
            report.append("\n" + "-" * 70)
            report.append("CLUSTERING QUALITY COMPARISON")
            report.append("-" * 70)
            report.append(f"Silhouette Score - {results['models']['model_a']}: {clust['silhouette_model_a']:.4f}")
            report.append(f"Silhouette Score - {results['models']['model_b']}: {clust['silhouette_model_b']:.4f}")
            report.append(f"Cluster Agreement (ARI): {clust['cluster_agreement_ari']:.4f}")
            if "ari_vs_ground_truth_model_a" in clust:
                report.append(f"\nARI vs Ground Truth - {results['models']['model_a']}: {clust['ari_vs_ground_truth_model_a']:.4f}")
                report.append(f"ARI vs Ground Truth - {results['models']['model_b']}: {clust['ari_vs_ground_truth_model_b']:.4f}")
        
        # Classification comparison
        if "classification_comparison" in results:
            clf = results["classification_comparison"]
            report.append("\n" + "-" * 70)
            report.append("DISCRIMINATIVE POWER COMPARISON")
            report.append("-" * 70)
            report.append(f"Accuracy - {results['models']['model_a']}: {clf['accuracy_model_a']:.4f}")
            report.append(f"Accuracy - {results['models']['model_b']}: {clf['accuracy_model_b']:.4f}")
            report.append(f"Accuracy Difference: {clf['accuracy_difference']:+.4f}")
            report.append(f"\nF1 Score - {results['models']['model_a']}: {clf['f1_model_a']:.4f}")
            report.append(f"F1 Score - {results['models']['model_b']}: {clf['f1_model_b']:.4f}")
        
        # Retrieval comparison
        if "retrieval_comparison" in results:
            ret = results["retrieval_comparison"]
            report.append("\n" + "-" * 70)
            report.append("INFORMATION RETRIEVAL COMPARISON")
            report.append("-" * 70)
            for k in [1, 5, 10]:
                if f"recall@{k}_model_a" in ret:
                    report.append(f"Recall@{k} - {results['models']['model_a']}: {ret[f'recall@{k}_model_a']:.4f}")
                    report.append(f"Recall@{k} - {results['models']['model_b']}: {ret[f'recall@{k}_model_b']:.4f}")
                    report.append(f"Recall@{k} Difference: {ret[f'recall@{k}_difference']:+.4f}\n")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)