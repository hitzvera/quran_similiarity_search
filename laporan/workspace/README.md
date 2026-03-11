# Audio Embedding Comparison: Wav2Vec2 vs Data2Vec

A comprehensive research framework for comparing Wav2Vec2 and Data2Vec audio embeddings. This tool enables deep analysis of self-supervised audio representations through multiple evaluation metrics and visualization techniques.

## Overview

This project provides a complete pipeline for comparing two state-of-the-art self-supervised audio models:

- **Wav2Vec2**: Contrastive learning approach using masked prediction
- **Data2Vec**: Unified self-supervised learning with teacher-student framework

### Key Research Questions Addressed

1. **Embedding Quality**: How do the embeddings cluster and represent audio semantics?
2. **Discriminative Power**: Which model produces more discriminative features for downstream tasks?
3. **Layer Evolution**: How do representations evolve through different layers?
4. **Similarity Structure**: How similar are the embedding spaces between models?
5. **Retrieval Performance**: Which model is better for information retrieval tasks?

## Features

### Model Extractors
- **Wav2Vec2**: Full support for HuggingFace transformers
- **Data2Vec**: Support for both Fairseq and HuggingFace implementations
- Multi-layer extraction with customizable pooling strategies
- Attention weight visualization (for supported models)

### Comparison Metrics
- **Similarity Analysis**: Matrix correlation, Frobenius distance
- **Clustering Quality**: Silhouette score, Adjusted Rand Index (ARI), compactness
- **Discriminative Power**: Classification accuracy, F1-score
- **Information Retrieval**: Recall@k, Mean Reciprocal Rank (MRR)
- **Statistical Tests**: Paired t-test, Wilcoxon, Mann-Whitney U

### Visualization Tools
- Similarity matrix heatmaps
- t-SNE and UMAP dimensionality reduction
- Embedding distribution analysis
- Layer-wise comparison plots
- Comprehensive report generation

### Dataset Support
- LibriSpeech (test-clean, test-other, dev-clean, dev-other)
- Custom datasets with flexible label support
- Synthetic dataset generation for controlled experiments

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone or navigate to the repository
cd audio-embedding-comparison

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install fairseq (for Data2Vec)
pip install fairseq
```

### Required Packages
```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
fairseq>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
librosa>=0.10.0
soundfile>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
umap-learn>=0.5.3
```

## Quick Start

### 1. Basic Comparison with Synthetic Data

```bash
python compare_embeddings.py \
    --dataset synthetic \
    --max_samples 100 \
    --output_dir outputs/test_run
```

### 2. Compare on LibriSpeech

```bash
python compare_embeddings.py \
    --dataset librispeech \
    --librispeech_root /path/to/LibriSpeech \
    --librispeech_subset test-clean \
    --max_samples 500 \
    --output_dir outputs/librispeech_comparison
```

### 3. Compare on Custom Dataset

```bash
python compare_embeddings.py \
    --dataset custom \
    --data_dir /path/to/your/audio \
    --max_samples 200 \
    --output_dir outputs/custom_comparison
```

### 4. Advanced Options

```bash
python compare_embeddings.py \
    --dataset librispeech \
    --librispeech_root /data/LibriSpeech \
    --wav2vec2_model facebook/wav2vec2-large-960h \
    --data2vec_model facebook/data2vec-audio-large \
    --layer -1 \
    --pooling mean \
    --device cuda \
    --save_embeddings \
    --output_dir outputs/advanced_comparison
```

## Usage Examples

### Python API

```python
from src.wav2vec2_extractor import Wav2Vec2Extractor
from src.data2vec_extractor import Data2VecExtractor
from src.comparison_framework import EmbeddingComparator
from src.dataset_loader import AudioDatasetLoader

# Initialize models
wav2vec2 = Wav2Vec2Extractor(model_name="facebook/wav2vec2-base-960h")
data2vec = Data2VecExtractor(model_path="facebook/data2vec-audio-base")

# Load data
loader = AudioDatasetLoader()
audio_list, labels, _ = loader.create_synthetic_dataset(n_samples=100, n_classes=5)

# Extract embeddings
embeddings_w2v = []
embeddings_d2v = []

for audio in audio_list:
    # Wav2Vec2
    result_w2v = wav2vec2.extract_embeddings(audio, layers=[-1], pooling="mean")
    embeddings_w2v.append(result_w2v["embeddings"]["layer_-1"].squeeze().numpy())
    
    # Data2Vec
    result_d2v = data2vec.extract_embeddings(audio, layers=[-1], pooling="mean")
    embeddings_d2v.append(result_d2v["embeddings"]["layer_-1"].squeeze().numpy())

embeddings_w2v = np.array(embeddings_w2v)
embeddings_d2v = np.array(embeddings_d2v)

# Compare
comparator = EmbeddingComparator("Wav2Vec2", "Data2Vec")
results = comparator.run_full_comparison(embeddings_w2v, embeddings_d2v, labels=np.array(labels))

# Generate report
report = comparator.generate_report(results)
print(report)
```

### Jupyter Notebook

See `notebooks/comparison_demo.ipynb` for an interactive walkthrough.

## Project Structure

```
audio-embedding-comparison/
├── src/
│   ├── __init__.py
│   ├── wav2vec2_extractor.py     # Wav2Vec2 model wrapper
│   ├── data2vec_extractor.py     # Data2Vec model wrapper
│   ├── comparison_framework.py   # Comparison metrics
│   ├── visualization.py          # Plotting tools
│   └── dataset_loader.py         # Data loading utilities
├── compare_embeddings.py         # Main script
├── requirements.txt              # Dependencies
├── config.yaml                   # Configuration template
└── notebooks/
    └── comparison_demo.ipynb     # Interactive demo
```

## Output Structure

After running a comparison, the output directory contains:

```
outputs/
├── embeddings/
│   ├── embeddings_wav2vec2.npz
│   └── embeddings_data2vec.npz
├── figures/
│   ├── comparison_report/
│   │   ├── similarity_matrices.png
│   │   ├── tsne_comparison.png
│   │   ├── umap_comparison.png
│   │   ├── metrics_comparison.png
│   │   └── embedding_distributions.png
│   └── ... (additional visualizations)
└── results/
    ├── comparison_report.txt
    └── comparison_results.json
```

## Research Methodology

### 1. Embedding Extraction
- Extract embeddings from different layers (0, middle, -1)
- Multiple pooling strategies (mean, CLS, last)
- Support for variable-length audio

### 2. Similarity Analysis
- Compute pairwise cosine similarity matrices
- Measure correlation between model similarity structures
- Analyze Frobenius norm differences

### 3. Clustering Evaluation
- K-means clustering on embeddings
- Silhouette score (intra-cluster cohesion vs inter-cluster separation)
- Adjusted Rand Index (cluster agreement)
- Compactness (intra-cluster distance)

### 4. Discriminative Power
- Train linear classifier (Logistic Regression) on embeddings
- Evaluate classification accuracy and F1-score
- Compare downstream task performance

### 5. Information Retrieval
- Recall@k metrics (k=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Evaluate embedding space for retrieval tasks

### 6. Statistical Significance
- Paired t-tests
- Wilcoxon signed-rank tests
- Mann-Whitney U tests
- Determine if differences are statistically significant

## Model Details

### Wav2Vec2
- **Architecture**: Transformer-based encoder
- **Training**: Contrastive task + diversity penalty
- **Objective**: Distinguish true from distractor quantized representations
- **Pre-trained**: 960h of LibriSpeech
- **Paper**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

### Data2Vec
- **Architecture**: Unified architecture (transformer + FFN)
- **Training**: Teacher-student framework with target representation
- **Objective**: Predict latent target representations
- **Pre-trained**: Multiple modalities (vision, NLP, speech)
- **Paper**: [Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)

## Key Differences

| Aspect | Wav2Vec2 | Data2Vec |
|--------|----------|----------|
| Objective | Contrastive learning | Regression of latent targets |
| Quantization | Required (VQ) | Not required |
| Architecture | Encoder only | Encoder + FFN |
| Modality | Speech-specific | Multi-modal (unified) |
| Training Signal | Discrete targets | Continuous representations |

## Troubleshooting

### Fairseq Import Error
If you encounter issues with fairseq:
```bash
pip install --upgrade fairseq
# Or build from source
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install -e .
```

### CUDA Out of Memory
Reduce batch size or use CPU:
```bash
python compare_embeddings.py --device cpu --max_samples 50
```

### Model Download Issues
Models are downloaded automatically on first run. If behind a proxy:
```python
import os
os.environ['HF_HOME'] = '/path/to/cache'
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in neural information processing systems},
  year={2020}
}

@article{baevski2022data2vec,
  title={Data2vec: A general framework for self-supervised learning in speech, vision and language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  journal={arXiv preprint arXiv:2202.03555},
  year={2022}
}
```

## License

This project is provided for research purposes. Please check individual model licenses:
- Wav2Vec2: Apache 2.0
- Data2Vec: Apache 2.0

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model support (HuBERT, WavLM, etc.)
- More visualization types
- Additional downstream tasks
- Benchmark datasets
- Performance optimizations

## Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This is a research framework designed for academic comparison studies. Results may vary based on dataset, model variants, and hyperparameters.