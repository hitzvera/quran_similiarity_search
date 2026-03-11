"""
Quick Start Example: Minimal Working Code
Compare Wav2Vec2 and Data2Vec on a small synthetic dataset
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wav2vec2_extractor import Wav2Vec2Extractor
from data2vec_extractor import Data2VecExtractor
from comparison_framework import EmbeddingComparator
from dataset_loader import AudioDatasetLoader
from visualization import EmbeddingVisualizer


def main():
    print("="*70)
    print("QUICK START: Wav2Vec2 vs Data2Vec Comparison")
    print("="*70)
    
    # Step 1: Create synthetic dataset
    print("\n[1/5] Creating synthetic dataset...")
    loader = AudioDatasetLoader(sample_rate=16000, max_duration=5.0)
    audio_list, labels, _ = loader.create_synthetic_dataset(
        n_samples=20,  # Small dataset for quick demo
        n_classes=5
    )
    labels = np.array(labels)
    print(f"Created {len(audio_list)} samples across {len(np.unique(labels))} classes")
    
    # Step 2: Load models
    print("\n[2/5] Loading models...")
    print("  Loading Wav2Vec2 (this may take a moment)...")
    wav2vec2 = Wav2Vec2Extractor(
        model_name="facebook/wav2vec2-base-960h"
    )
    
    print("  Loading Data2Vec (this may take a moment)...")
    try:
        data2vec = Data2VecExtractor(
            model_path="facebook/data2vec-audio-base"
        )
    except Exception as e:
        print(f"  Warning: Could not load Data2Vec: {e}")
        print("  Using mock embeddings for demonstration...")
        data2vec = None
    
    # Step 3: Extract embeddings
    print("\n[3/5] Extracting embeddings...")
    embeddings_w2v = []
    embeddings_d2v = []
    
    for i, audio in enumerate(audio_list):
        print(f"  Processing sample {i+1}/{len(audio_list)}...", end='\r')
        
        # Wav2Vec2
        result_w2v = wav2vec2.extract_embeddings(
            audio,
            layers=[-1],  # Last layer
            pooling="mean"
        )
        embeddings_w2v.append(result_w2v["embeddings"]["layer_-1"].squeeze().numpy())
        
        # Data2Vec
        if data2vec:
            result_d2v = data2vec.extract_embeddings(
                audio,
                layers=[-1],
                pooling="mean"
            )
            embeddings_d2v.append(result_d2v["embeddings"]["layer_-1"].squeeze().numpy())
        else:
            # Mock embeddings for demo
            embeddings_d2v.append(np.random.randn(768) * 0.1)
    
    print(f"  Processing sample {len(audio_list)}/{len(audio_list)}... Done!")
    
    embeddings_w2v = np.array(embeddings_w2v)
    embeddings_d2v = np.array(embeddings_d2v)
    
    print(f"\n  Wav2Vec2 embeddings shape: {embeddings_w2v.shape}")
    print(f"  Data2Vec embeddings shape: {embeddings_d2v.shape}")
    
    # Step 4: Compare embeddings
    print("\n[4/5] Running comparison analysis...")
    comparator = EmbeddingComparator("Wav2Vec2", "Data2Vec")
    results = comparator.run_full_comparison(
        embeddings_w2v,
        embeddings_d2v,
        labels=labels
    )
    
    # Step 5: Generate results
    print("\n[5/5] Generating report...")
    report = comparator.generate_report(results)
    print("\n" + report)
    
    # Create visualizations
    print("\nCreating visualizations...")
    os.makedirs("outputs/example", exist_ok=True)
    
    visualizer = EmbeddingVisualizer(output_dir="outputs/example")
    
    # Similarity matrices
    print("  - Similarity matrices...")
    visualizer.plot_similarity_matrices(
        embeddings_w2v, embeddings_d2v, labels,
        model_a_name="Wav2Vec2", model_b_name="Data2Vec",
        save_path="outputs/example/similarity_matrices.png"
    )
    
    # t-SNE
    print("  - t-SNE visualization...")
    visualizer.plot_dimensionality_reduction(
        embeddings_w2v, embeddings_d2v, labels,
        method="tsne",
        model_a_name="Wav2Vec2", model_b_name="Data2Vec",
        save_path="outputs/example/tsne_comparison.png"
    )
    
    # Metrics
    print("  - Metrics comparison...")
    visualizer.plot_metrics_comparison(
        results,
        save_path="outputs/example/metrics_comparison.png"
    )
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
    print("\nOutputs saved to: outputs/example/")
    print("  - similarity_matrices.png")
    print("  - tsne_comparison.png")
    print("  - metrics_comparison.png")
    print("\nTo run a full comparison, use: python compare_embeddings.py --help")
    print("="*70)


if __name__ == '__main__':
    main()