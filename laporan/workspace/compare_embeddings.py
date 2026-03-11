"""
Main Comparison Script for Wav2Vec2 vs Data2Vec
Complete pipeline for audio embedding comparison
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from src.wav2vec2_extractor import Wav2Vec2Extractor
from src.data2vec_extractor import Data2VecExtractor
from src.comparison_framework import EmbeddingComparator
from src.visualization import EmbeddingVisualizer
from src.dataset_loader import AudioDatasetLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare Wav2Vec2 and Data2Vec audio embeddings'
    )
    
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing audio files')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'librispeech', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--librispeech_root', type=str, default=None,
                       help='Root directory of LibriSpeech (if using librispeech)')
    parser.add_argument('--librispeech_subset', type=str, default='test-clean',
                       help='LibriSpeech subset to use')
    
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to process')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Sample rate (16kHz recommended)')
    parser.add_argument('--max_duration', type=float, default=10.0,
                       help='Maximum audio duration in seconds')
    
    parser.add_argument('--wav2vec2_model', type=str,
                       default='facebook/wav2vec2-base-960h',
                       help='Wav2Vec2 model name')
    parser.add_argument('--data2vec_model', type=str,
                       default='facebook/data2vec-audio-base',
                       help='Data2Vec model name/path')
    
    parser.add_argument('--layer', type=int, default=-1,
                       help='Which layer to extract (-1 for last)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'cls', 'last'],
                       help='Pooling strategy')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--save_embeddings', action='store_true',
                       help='Save extracted embeddings')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    return parser.parse_args()


def extract_embeddings_batch(extractor, audio_list, batch_size=8, layer=-1, pooling='mean'):
    """
    Extract embeddings in batches with progress bar
    
    Args:
        extractor: Model extractor (Wav2Vec2Extractor or Data2VecExtractor)
        audio_list: List of audio arrays
        batch_size: Batch size for processing
        layer: Layer index to extract
        pooling: Pooling strategy
    
    Returns:
        Array of embeddings
    """
    embeddings = []
    
    print(f"Extracting embeddings from {len(audio_list)} samples...")
    for i in tqdm(range(0, len(audio_list), batch_size), desc="Processing batches"):
        batch = audio_list[i:i+batch_size]
        
        batch_embeddings = []
        for audio in batch:
            try:
                result = extractor.extract_embeddings(
                    audio,
                    layers=[layer],
                    pooling=pooling
                )
                emb = result['embeddings'][f'layer_{layer}']
                batch_embeddings.append(emb.squeeze().numpy())
            except Exception as e:
                print(f"\nError processing sample: {e}")
                # Use zero embedding as fallback
                dim = getattr(extractor, 'hidden_size', 768)
                batch_embeddings.append(np.zeros(dim))
        
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("AUDIO EMBEDDING COMPARISON: Wav2Vec2 vs Data2Vec")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Device: {device}")
    print(f"  Output: {args.output_dir}")
    print(f"  Layer: {args.layer}")
    print(f"  Pooling: {args.pooling}")
    print("="*70)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_dir = os.path.join(args.output_dir, 'embeddings')
    figures_dir = os.path.join(args.output_dir, 'figures')
    results_dir = os.path.join(args.output_dir, 'results')
    
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load dataset
    print("\n" + "="*70)
    print("STEP 1: Loading Dataset")
    print("="*70)
    
    loader = AudioDatasetLoader(
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )
    
    if args.dataset == 'synthetic':
        audio_list, labels, _ = loader.create_synthetic_dataset(
            n_samples=args.max_samples // 5,
            n_classes=5
        )
        labels = np.array(labels)
    elif args.dataset == 'librispeech':
        if not args.librispeech_root:
            print("Error: --librispeech_root required for LibriSpeech dataset")
            sys.exit(1)
        audio_list, labels, _ = loader.load_librispeech(
            args.librispeech_root,
            subset=args.librispeech_subset,
            max_samples=args.max_samples
        )
        labels = np.array(labels)
    else:  # custom
        if not args.data_dir:
            print("Error: --data_dir required for custom dataset")
            sys.exit(1)
        audio_list, labels, _ = loader.load_custom_dataset(
            args.data_dir,
            max_samples=args.max_samples
        )
        labels = np.array(labels)
    
    if len(audio_list) == 0:
        print("Error: No audio files loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(audio_list)} samples")
    print(f"Unique labels: {len(np.unique(labels))}")
    
    # Step 2: Initialize extractors
    print("\n" + "="*70)
    print("STEP 2: Initializing Models")
    print("="*70)
    
    print("\nLoading Wav2Vec2...")
    wav2vec2_extractor = Wav2Vec2Extractor(
        model_name=args.wav2vec2_model,
        device=device,
        sample_rate=args.sample_rate
    )
    
    print("\nLoading Data2Vec...")
    data2vec_extractor = Data2VecExtractor(
        model_path=args.data2vec_model,
        device=device,
        sample_rate=args.sample_rate
    )
    
    # Step 3: Extract embeddings
    print("\n" + "="*70)
    print("STEP 3: Extracting Embeddings")
    print("="*70)
    
    print("\nExtracting Wav2Vec2 embeddings...")
    embeddings_wav2vec2 = extract_embeddings_batch(
        wav2vec2_extractor,
        audio_list,
        layer=args.layer,
        pooling=args.pooling
    )
    
    print("\nExtracting Data2Vec embeddings...")
    embeddings_data2vec = extract_embeddings_batch(
        data2vec_extractor,
        audio_list,
        layer=args.layer,
        pooling=args.pooling
    )
    
    print(f"\nWav2Vec2 embeddings shape: {embeddings_wav2vec2.shape}")
    print(f"Data2Vec embeddings shape: {embeddings_data2vec.shape}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        print("\nSaving embeddings...")
        np.savez(
            os.path.join(embeddings_dir, 'embeddings_wav2vec2.npz'),
            embeddings=embeddings_wav2vec2,
            labels=labels
        )
        np.savez(
            os.path.join(embeddings_dir, 'embeddings_data2vec.npz'),
            embeddings=embeddings_data2vec,
            labels=labels
        )
        print(f"  Saved to {embeddings_dir}")
    
    # Step 4: Run comparison
    print("\n" + "="*70)
    print("STEP 4: Running Comparison Analysis")
    print("="*70)
    
    comparator = EmbeddingComparator(
        model_a_name="Wav2Vec2",
        model_b_name="Data2Vec"
    )
    
    results = comparator.run_full_comparison(
        embeddings_wav2vec2,
        embeddings_data2vec,
        labels=labels
    )
    
    # Step 5: Generate report
    print("\n" + "="*70)
    print("STEP 5: Generating Report")
    print("="*70)
    
    report = comparator.generate_report(results)
    print(report)
    
    # Save report
    report_path = os.path.join(results_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save results as JSON
    results_path = os.path.join(results_dir, 'comparison_results.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Step 6: Create visualizations
    print("\n" + "="*70)
    print("STEP 6: Creating Visualizations")
    print("="*70)
    
    visualizer = EmbeddingVisualizer(output_dir=figures_dir)
    
    try:
        visualizer.create_comparison_report(
            embeddings_wav2vec2,
            embeddings_data2vec,
            labels,
            results,
            model_a_name="Wav2Vec2",
            model_b_name="Data2Vec",
            output_dir=figures_dir
        )
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {args.output_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Results: {results_dir}")
    print(f"  - Embeddings: {embeddings_dir}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()