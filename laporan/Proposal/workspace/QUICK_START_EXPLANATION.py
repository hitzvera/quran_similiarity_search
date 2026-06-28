"""
DETAILED EXPLANATION: quick_start.py
Research Pipeline for Wav2Vec2 vs Data2Vec Comparison

This document explains every process in detail for academic presentation.
"""

# =============================================================================
# SECTION 1: IMPORTS & MODULE SETUP
# =============================================================================

"""
IMPORTS EXPLANATION:
-------------------

1. numpy (np)
   - Purpose: Numerical computing and array operations
   - Why needed: Handle audio arrays and embedding matrices efficiently
   - Key functions used: np.array(), np.unique(), np.random.randn()

2. sys
   - Purpose: System-specific parameters and functions
   - Why needed: Modify Python's module search path (sys.path)
   - Critical for: Importing custom modules from src/ directory

3. os
   - Purpose: Operating system interface
   - Why needed: File path operations and directory creation
   - Key functions: os.path.join(), os.makedirs(), os.path.dirname()

MODULE PATH SETUP:
-----------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

Explanation:
- __file__ : Path to the current script (quick_start.py)
- os.path.dirname(__file__) : Directory containing quick_start.py
- os.path.join(..., 'src') : Path to src/ directory
- sys.path.insert(0, ...) : Add src/ to beginning of Python's search path

Why this matters:
- Without this, Python cannot find modules in src/
- Alternative: Install package with "pip install -e ." (requires setup.py)
- The '0' ensures src/ is checked FIRST before other paths

CUSTOM MODULES IMPORTED:
-----------------------
1. Wav2Vec2Extractor (wav2vec2_extractor.py)
   - Wraps Facebook's Wav2Vec2 model
   - Handles model loading, preprocessing, embedding extraction
   
2. Data2VecExtractor (data2vec_extractor.py)
   - Wraps Facebook's Data2Vec model
   - Similar functionality to Wav2Vec2Extractor
   
3. EmbeddingComparator (comparison_framework.py)
   - Implements comparison metrics
   - Statistical analysis and significance testing
   
4. AudioDatasetLoader (dataset_loader.py)
   - Handles dataset loading and preprocessing
   - Supports synthetic, LibriSpeech, and custom datasets
   
5. EmbeddingVisualizer (visualization.py)
   - Creates publication-quality plots
   - t-SNE, UMAP, similarity matrices, metric comparisons
"""

# =============================================================================
# SECTION 2: STEP 1 - SYNTHETIC DATASET CREATION
# =============================================================================

"""
STEP 1: CREATE SYNTHETIC DATASET
--------------------------------

CODE:
    loader = AudioDatasetLoader(sample_rate=16000, max_duration=5.0)
    audio_list, labels, _ = loader.create_synthetic_dataset(
        n_samples=20,
        n_classes=5
    )
    labels = np.array(labels)

DETAILED EXPLANATION:
--------------------

1. AudioDatasetLoader Initialization
   ---------------------------------
   Parameters:
   - sample_rate=16000 : 
     * Both Wav2Vec2 and Data2Vec were trained on 16kHz audio
     * Higher sample rates would be downsampled
     * Lower sample rates would be upsampled (interpolation)
   
   - max_duration=5.0 :
     * Maximum audio length in seconds
     * 5 seconds = 80,000 samples at 16kHz (16000 × 5)
     * Audio longer than this is truncated
     * Audio shorter is zero-padded

2. create_synthetic_dataset() Method
   ----------------------------------
   Parameters:
   - n_samples=20 : Total number of audio clips to generate
   - n_classes=5 : Number of distinct categories (20 ÷ 5 = 4 samples per class)
   
   Returns:
   - audio_list : List of numpy arrays, each containing audio waveform
   - labels : List of integers (0-4) indicating class membership
   - _ : Descriptions (unused in this example)

3. Five Signal Types Generated
   ----------------------------
   The method creates 5 distinct audio patterns:
   
   Class 0: Low Frequency (200 Hz sine wave)
   - Mathematical: sin(2π × 200 × t)
   - Characteristics: Deep, bass-like tone
   
   Class 1: High Frequency (2000 Hz sine wave)
   - Mathematical: sin(2π × 2000 × t)
   - Characteristics: High-pitched whistle-like tone
   
   Class 2: Chirp Signal (Frequency sweep)
   - Mathematical: sin(2π × (200 + 1800×t/5) × t)
   - Characteristics: Frequency increases from 200Hz to 2000Hz
   
   Class 3: White Noise (Random signal)
   - Mathematical: np.random.randn(samples)
   - Characteristics: Contains all frequencies equally
   
   Class 4: Mixed Signal (Combination)
   - Mathematical: 500Hz sine + 1500Hz sine + 0.3×noise
   - Characteristics: Complex harmonic structure

4. Why Synthetic Data?
   -------------------
   ADVANTAGES:
   ✓ Reproducible: Same random seed produces identical results
   ✓ Fast: No file I/O, downloads, or preprocessing
   ✓ Controlled: Ground truth is known exactly
   ✓ No legal issues: No copyright or licensing concerns
   ✓ Small size: Easy to share and version control
   
   DISADVANTAGES:
   ✗ Not realistic: Real speech has more complexity
   ✗ Limited generalization: Results may not transfer to real audio
   
   FOR RESEARCH:
   - Used for quick testing and debugging
   - Real datasets (LibriSpeech) used for actual experiments

5. labels = np.array(labels)
   -------------------------
   - Converts Python list to numpy array
   - Enables efficient numerical operations
   - Shape: (20,) - 20 integer labels
   - Values: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]
"""

# =============================================================================
# SECTION 3: STEP 2 - MODEL LOADING
# =============================================================================

"""
STEP 2: LOAD PRE-TRAINED MODELS
-------------------------------

CODE:
    wav2vec2 = Wav2Vec2Extractor(
        model_name="facebook/wav2vec2-base-960h"
    )

DETAILED EXPLANATION:
--------------------

1. Wav2Vec2Extractor Class
   ------------------------
   This is our custom wrapper that encapsulates:
   
   a) Model Download & Loading:
      - Downloads from HuggingFace Hub (huggingface.co)
      - Cached in ~/.cache/huggingface/
      - Only downloads once, then uses cached version
   
   b) Architecture Setup:
      - Feature Extractor: CNN layers that process raw waveform
      - Context Network: 12 Transformer layers
      - Output: Contextualized representations
   
   c) Device Management:
      - Auto-detects CUDA availability
      - Falls back to CPU if no GPU
      - Moves model to appropriate device
   
   d) Preprocessing:
      - Normalizes audio to [-1, 1] range
      - Handles variable-length inputs
      - Creates attention masks for padding

2. Model: facebook/wav2vec2-base-960h
   ----------------------------------
   
   Architecture Details:
   - Base size: 12 layers, 768 hidden dimensions, 12 attention heads
   - 960h: Trained on 960 hours of LibriSpeech clean + other
   - Parameters: ~95 million
   - Model size: ~360 MB on disk
   
   Training Objective (Self-Supervised):
   - Contrastive task: Distinguish true quantized representation from distractors
   - Masked prediction: ~50% of time steps are masked
   - Diversity penalty: Encourages equal use of codebook entries
   
   Pre-training Process:
   1. Feature encoder (CNN) converts waveform to latent representations
   2. Quantization module discretizes to finite vocabulary
   3. Context network (Transformer) processes full sequence
   4. Model learns to predict quantized targets from masked positions

CODE:
    data2vec = Data2VecExtractor(
        model_path="facebook/data2vec-audio-base"
    )

3. Data2VecExtractor Class
   ------------------------
   Similar to Wav2Vec2Extractor but for Data2Vec model.
   
   Key Differences:
   - May require fairseq library (not just transformers)
   - Uses teacher-student architecture
   - Different preprocessing pipeline

4. Model: facebook/data2vec-audio-base
   -----------------------------------
   
   Architecture Details:
   - Similar size to Wav2Vec2: 12 layers, 768 dimensions
   - Parameters: ~95 million
   - Model size: ~360 MB
   
   Training Objective (Self-Supervised):
   - Teacher-Student framework
   - Teacher: EMA (Exponentially Moving Average) of student weights
   - Student: Main model being trained
   - Objective: Predict teacher's latent representations
   
   Key Innovation:
   - Unified architecture across modalities (vision, NLP, speech)
   - No discrete targets needed (unlike Wav2Vec2's quantization)
   - Direct regression of continuous representations

5. Error Handling
   --------------
   try:
       data2vec = Data2VecExtractor(...)
   except Exception as e:
       data2vec = None
   
   Why this matters:
   - Fairseq installation can be problematic
   - Demo should run even if Data2Vec fails
   - Fallback to random embeddings allows testing the pipeline
"""

# =============================================================================
# SECTION 4: STEP 3 - EMBEDDING EXTRACTION
# =============================================================================

"""
STEP 3: EXTRACT EMBEDDINGS
--------------------------

CODE:
    result_w2v = wav2vec2.extract_embeddings(
        audio,
        layers=[-1],
        pooling="mean"
    )
    embeddings_w2v.append(result_w2v["embeddings"]["layer_-1"].squeeze().numpy())

DETAILED EXPLANATION:
--------------------

1. extract_embeddings() Method
   ----------------------------
   
   Input:
   - audio: numpy array of shape (80,) or (80,000,) depending on duration
            Contains raw waveform amplitudes
   
   Parameters:
   - layers=[-1]:
     * List of layer indices to extract
     * -1 means the last (final) layer
     * Other options: [0] (first), [6] (middle), [0, 6, -1] (multiple)
     * Research question: Which layer has best representations?
   
   - pooling="mean":
     * Strategy to aggregate time-step embeddings
     * "mean": Average across time dimension (most common)
     * "cls": Use first position (like BERT's [CLS] token)
     * "last": Use last non-padding position
     * None: Return full sequence (batch, time, dim)

2. Internal Processing Pipeline
   ----------------------------
   
   Step A: Preprocessing
   - Convert numpy array to PyTorch tensor
   - Add batch dimension: (80000,) → (1, 80000)
   - Move to device (CPU/GPU)
   - Normalize if needed
   
   Step B: Feature Extraction (CNN)
   - Raw waveform → Convolutional layers
   - Reduces time resolution by factor of 320
   - 80,000 samples → ~250 time steps
   - Output: (1, 250, 512) - 512-dim features
   
   Step C: Contextualization (Transformer)
   - 12 layers of self-attention
   - Each layer: Multi-head attention + FFN + LayerNorm
   - Captures long-range dependencies
   - Output per layer: (1, 250, 768)
   
   Step D: Layer Selection
   - Extract specified layer(s) hidden states
   - For layers=[-1]: Get output of 12th transformer layer
   - Shape: (1, 250, 768) - (batch, time, hidden_dim)
   
   Step E: Pooling
   - "mean": Average over time dimension
   - (1, 250, 768) → (1, 768)
   - Reduces variable-length to fixed-length
   
   Step F: Output Formatting
   - Return dictionary with embeddings per layer
   - Structure: {"layer_-1": tensor([1, 768])}

3. Result Processing
   -----------------
   result_w2v["embeddings"]["layer_-1"]
   - Access the embedding for layer -1
   - Returns: torch.Tensor of shape (1, 768)
   
   .squeeze()
   - Removes dimensions of size 1
   - (1, 768) → (768,)
   - Makes it easier to work with
   
   .numpy()
   - Converts PyTorch tensor to numpy array
   - Required for scikit-learn compatibility

4. Final Embedding Matrix
   ----------------------
   After processing all 20 samples:
   - embeddings_w2v: list of 20 arrays, each (768,)
   - np.array(embeddings_w2v): converts to matrix (20, 768)
   
   Interpretation:
   - 20 rows = 20 audio samples
   - 768 columns = embedding dimensions
   - Each row is a point in 768-dimensional space
   - Similar audio should have similar (close) embeddings

5. Why the Last Layer?
   -------------------
   Research shows:
   - Early layers: Capture low-level features (pitch, timbre)
   - Middle layers: Capture phonetic/phonemic information
   - Late layers: Capture semantic/speaker information
   
   For speaker identification/emotion: Later layers are better
   For phoneme recognition: Middle layers may be better
"""

# =============================================================================
# SECTION 5: STEP 4 - COMPARISON ANALYSIS
# =============================================================================

"""
STEP 4: RUN COMPARISON ANALYSIS
-------------------------------

CODE:
    comparator = EmbeddingComparator("Wav2Vec2", "Data2Vec")
    results = comparator.run_full_comparison(
        embeddings_w2v,
        embeddings_d2v,
        labels=labels
    )

DETAILED EXPLANATION:
--------------------

1. EmbeddingComparator Initialization
   ----------------------------------
   Purpose: Framework for systematic comparison of two embedding spaces
   
   Parameters:
   - "Wav2Vec2": Name for model A (for reporting)
   - "Data2Vec": Name for model B (for reporting)

2. run_full_comparison() Method
   ----------------------------
   
   Performs FOUR categories of analysis:

   A. SIMILARITY MATRIX ANALYSIS
      --------------------------
      Goal: Do both models produce similar similarity structures?
      
      Steps:
      1. Normalize embeddings: L2 normalization
         - embeddings / ||embeddings||
      
      2. Compute pairwise cosine similarity:
         - sim[i,j] = dot(emb[i], emb[j]) / (||emb[i]|| × ||emb[j]||)
         - Result: 20×20 matrix for each model
      
      3. Compare matrices:
         - Correlation: Pearson correlation of upper triangles
         - Frobenius norm: ||sim_w2v - sim_d2v||_F
         - Relative difference: Frobenius / mean_norm
      
      Interpretation:
      - High correlation (>0.8): Models agree on what's similar
      - Low correlation (<0.5): Different similarity structures
      
      Metrics computed:
      - similarity_correlation: Pearson r between matrices
      - frobenius_difference: Absolute difference magnitude
      - relative_difference: Normalized difference
      - mean_sim_model_a/b: Average similarity score

   B. CLUSTERING QUALITY
      ------------------
      Goal: Which model produces better clusters?
      
      Steps:
      1. K-means clustering (k=5, matching n_classes)
         - Cluster embeddings from each model separately
      
      2. Silhouette Score:
         - Measures: (b - a) / max(a, b)
           where a = mean intra-cluster distance
                 b = mean nearest-cluster distance
         - Range: [-1, 1], higher is better
         - >0.5: Strong structure
         - <0: Incorrect clustering
      
      3. Adjusted Rand Index (ARI):
         - Compares clustering to ground truth labels
         - Range: [-1, 1], higher is better
         - 1.0: Perfect match
         - 0.0: Random clustering
      
      4. Compactness:
         - Average distance from cluster center
         - Lower is better (tighter clusters)
      
      Metrics computed:
      - silhouette_model_a/b: Quality of clusters
      - cluster_agreement_ari: Agreement between model clusterings
      - compactness_model_a/b: Cluster tightness

   C. DISCRIMINATIVE POWER (CLASSIFICATION)
      ------------------------------------
      Goal: Which embeddings are more discriminative?
      
      Steps:
      1. Split data: 80% train, 20% test (stratified)
      
      2. Train Logistic Regression:
         - Simple linear classifier
         - No hidden layers (tests embedding quality directly)
         - max_iter=1000 for convergence
      
      3. Evaluate:
         - Accuracy: % correct predictions
         - F1-score: Harmonic mean of precision and recall
         - Weighted average across classes
      
      Interpretation:
      - Higher accuracy = better embeddings for classification
      - Compares which model captures class-discriminative features
      
      Metrics computed:
      - accuracy_model_a/b: Classification accuracy
      - f1_model_a/b: F1-score
      - accuracy_difference: Direct comparison

   D. INFORMATION RETRIEVAL
      ---------------------
      Goal: Which model is better for retrieval tasks?
      
      Concept: Given a query audio, find similar ones
      
      Steps:
      1. For each sample, compute similarity to all others
      
      2. Recall@k:
         - Top-k similar samples
         - What fraction are same class?
         - k=1, 5, 10 (different granularity)
      
      3. Mean Reciprocal Rank (MRR):
         - Rank of first correct match
         - MRR = mean(1/rank)
         - Higher is better (perfect = 1.0)
      
      Interpretation:
      - Important for applications like: speaker verification, music retrieval
      - Tests local neighborhood structure
      
      Metrics computed:
      - recall@1/5/10: Retrieval accuracy at different k
      - mrr: Mean reciprocal rank

3. Statistical Significance
   ------------------------
   After computing metrics, runs statistical tests:
   
   - Paired t-test: Are differences significant?
   - p-value < 0.05: Statistically significant
   - Prevents over-interpreting small differences

4. Output Structure
   ----------------
   results = {
       "models": {"model_a": "Wav2Vec2", "model_b": "Data2Vec"},
       "dataset_info": {"num_samples": 20, ...},
       "similarity_comparison": {...},
       "clustering_comparison": {...},
       "classification_comparison": {...},
       "retrieval_comparison": {...}
   }
   
   Dictionary with 50+ metrics for comprehensive analysis.
"""

# =============================================================================
# SECTION 6: STEP 5 - REPORT GENERATION & VISUALIZATION
# =============================================================================

"""
STEP 5: GENERATE REPORT & VISUALIZATIONS
----------------------------------------

CODE:
    report = comparator.generate_report(results)
    visualizer.plot_similarity_matrices(...)
    visualizer.plot_dimensionality_reduction(...)
    visualizer.plot_metrics_comparison(...)

DETAILED EXPLANATION:
--------------------

1. Text Report Generation
   ----------------------
   generate_report() creates formatted output:
   
   Sections:
   - Header: Model names, dataset info
   - Similarity Analysis: Correlation coefficients
   - Clustering: Silhouette scores, ARI
   - Classification: Accuracy, F1 comparisons
   - Retrieval: Recall@k, MRR
   - Footer: Summary
   
   Example output:
   ```
   =================================================================
   AUDIO EMBEDDING COMPARISON REPORT
   =================================================================
   
   Similarity Matrix Correlation: 0.8234
   Frobenius Norm Difference: 12.4567
   
   Silhouette Score - Wav2Vec2: 0.4521
   Silhouette Score - Data2Vec: 0.5134
   
   Classification Accuracy - Wav2Vec2: 0.8500
   Classification Accuracy - Data2Vec: 0.9000
   =================================================================
   ```

2. Visualization 1: Similarity Matrices
   ------------------------------------
   
   Purpose: Visual comparison of embedding space structures
   
   Plots created:
   - Left: Wav2Vec2 similarity heatmap (20×20)
   - Middle: Data2Vec similarity heatmap (20×20)
   - Right: Difference matrix (Wav2Vec2 - Data2Vec)
   
   Interpretation:
   - Bright diagonal: Each sample is similar to itself
   - Block patterns: Samples from same class cluster together
   - Difference matrix: Shows where models disagree
   
   Technical details:
   - Colormap: 'viridis' for similarities, 'RdBu_r' for differences
   - Sorted by labels: Groups same-class samples together
   - Color bar: Indicates similarity magnitude

3. Visualization 2: t-SNE Dimensionality Reduction
   -----------------------------------------------
   
   Purpose: Visualize 768D embeddings in 2D
   
   Method: t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - Non-linear dimensionality reduction
   - Preserves local neighborhood structure
   - Better than PCA for visualization
   
   Parameters:
   - n_components=2: Project to 2D
   - perplexity=30: Balance local/global structure
   - random_state=42: Reproducible results
   
   Plots:
   - Left: Wav2Vec2 embeddings in 2D
   - Right: Data2Vec embeddings in 2D
   - Points colored by class label
   
   Interpretation:
   - Tight clusters: Model learned class-discriminative features
   - Overlapping clusters: Classes not well separated
   - Compare both plots: Which model clusters better?

4. Visualization 3: Metrics Comparison
   -----------------------------------
   
   Purpose: Bar charts comparing quantitative metrics
   
   Subplots:
   - Overall metrics (similarity, silhouette, accuracy)
   - Retrieval performance (Recall@1, @5, @10)
   - Similarity correlation
   - Cluster compactness
   
   Each subplot:
   - Side-by-side bars for Wav2Vec2 and Data2Vec
   - Error bars (if multiple runs)
   - Exact values labeled

5. Output Directory Structure
   --------------------------
   outputs/example/
   ├── similarity_matrices.png
   ├── tsne_comparison.png
   ├── metrics_comparison.png
   └── (additional files from full pipeline)

6. Research Value
   --------------
   These visualizations enable:
   
   - Qualitative assessment: "Do embeddings cluster by class?"
   - Quantitative comparison: "Which model has higher silhouette score?"
   - Pattern identification: "Are similarity structures correlated?"
   - Publication-ready figures: For thesis and papers
"""

# =============================================================================
# SUMMARY: COMPLETE RESEARCH PIPELINE
# =============================================================================

"""
SUMMARY FOR LECTURER
-------------------

This quick_start.py demonstrates a COMPLETE research pipeline for comparing
self-supervised audio representations. Here is the academic justification:

RESEARCH QUESTION:
"How do Wav2Vec2 (contrastive learning) and Data2Vec (teacher-student) 
representations differ in terms of clustering quality, discriminative power,
and information retrieval performance?"

METHODOLOGY:
1. Data Generation: Controlled synthetic dataset with known ground truth
2. Feature Extraction: Extract embeddings from final transformer layers
3. Multi-faceted Evaluation:
   - Similarity structure analysis (pairwise correlations)
   - Clustering quality (silhouette, ARI, compactness)
   - Discriminative power (linear classification)
   - Information retrieval (Recall@k, MRR)
4. Statistical Testing: Paired t-tests for significance
5. Visualization: Publication-ready figures

SCIENTIFIC CONTRIBUTION:
- Systematic comparison framework applicable to any embedding models
- Multiple evaluation perspectives (clustering, classification, retrieval)
- Statistical rigor with significance testing
- Reproducible pipeline with open-source tools

OUTPUTS FOR PUBLICATION:
- Numerical results (JSON format)
- Statistical report (text format)
- Visualization figures (PNG format)
- Complete pipeline code (Python)

This provides a minimal but complete example that can be scaled to:
- Real datasets (LibriSpeech, VoxCeleb, etc.)
- Different model variants (base, large, xls-r)
- Additional downstream tasks (ASR, emotion, speaker ID)
- Ablation studies (different layers, pooling strategies)
"""

# End of detailed explanation