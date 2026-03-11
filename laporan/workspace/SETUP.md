# Setup Guide - Audio Embedding Comparison

This guide documents how to set up the research environment for comparing Wav2Vec2 and Data2Vec audio embeddings.

## System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 - 3.11 (3.10 recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB free space (for models and data)
- **GPU**: Optional (CUDA-capable for faster processing)

## Quick Setup

### 1. Clone/Copy the Project

```bash
cd G:\skripsi\quran_matcher\laporan\workspace
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv

# Linux/macOS
python3 -m venv .venv
```

### 3. Activate Virtual Environment

```bash
# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# Linux/macOS
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

## Detailed Installation

If `pip install -r requirements.txt` fails, install packages manually in order:

### Step 1: Core Scientific Computing

```bash
pip install numpy scipy pandas
```

### Step 2: Machine Learning

```bash
pip install scikit-learn
```

### Step 3: Deep Learning (PyTorch)

**For CPU only:**
```bash
pip install torch==2.10.0 torchaudio==2.10.0
```

**For CUDA (GPU):**
```bash
pip install torch==2.10.0+cu118 torchaudio==2.10.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Check PyTorch installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Transformers & NLP

```bash
pip install transformers==5.3.0
pip install tokenizers==0.22.2
pip install regex
pip install huggingface-hub
pip install safetensors
pip install filelock
pip install pyyaml
pip install fsspec
```

### Step 5: Fairseq (for Data2Vec)

```bash
pip install fairseq==0.12.2
```

**Note:** Fairseq may require additional dependencies:
```bash
pip install hydra-core==1.0.7
pip install omegaconf==2.0.6
pip install antlr4-python3-runtime==4.8
pip install sacrebleu==2.6.0
pip install bitarray
pip install cython
pip install tabulate
pip install portalocker
pip install lxml
```

### Step 6: Audio Processing

```bash
pip install librosa==0.11.0
pip install soundfile==0.13.1
```

**Note:** SoundFile may require system libraries:
- **Windows**: Install Visual C++ Redistributable
- **Linux**: `sudo apt-get install libsndfile1`
- **macOS**: `brew install libsndfile`

### Step 7: Visualization

```bash
pip install matplotlib==3.10.8
pip install seaborn==0.13.2
```

### Step 8: Dimensionality Reduction

```bash
pip install umap-learn==0.5.11
```

**Note:** UMAP requires:
```bash
pip install numba==0.64.0
pip install llvmlite==0.46.0
pip install pynndescent==0.6.0
```

### Step 9: Utilities

```bash
pip install tqdm==4.67.3
pip install tensorboard==2.20.0
pip install jupyter
```

**Note:** TensorBoard requires:
```bash
pip install protobuf==7.34.0
pip install grpcio==1.78.0
pip install absl-py
pip install markdown
pip install werkzeug
pip install tensorboard-data-server
```

### Step 10: Network/HTTP (for HuggingFace)

```bash
pip install httpx==0.28.1
pip install httpcore==1.0.9
pip install h11
pip install anyio
pip install certifi
pip install charset-normalizer
pip install idna
pip install urllib3
pip install requests
```

## Package Version Summary

Here are the exact versions installed and verified:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.x | Programming language |
| torch | 2.10.0+cpu | Deep learning framework |
| torchaudio | 2.10.0+cpu | Audio processing for PyTorch |
| transformers | 5.3.0 | HuggingFace transformers |
| fairseq | 0.12.2 | Facebook's sequence modeling |
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.15.3 | Scientific computing |
| scikit-learn | 1.7.2 | Machine learning |
| librosa | 0.11.0 | Audio analysis |
| soundfile | 0.13.1 | Audio I/O |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| pandas | 2.3.3 | Data manipulation |
| umap-learn | 0.5.11 | Dimensionality reduction |
| numba | 0.64.0 | JIT compilation |
| llvmlite | 0.46.0 | LLVM bindings |
| tqdm | 4.67.3 | Progress bars |
| tensorboard | 2.20.0 | Visualization |
| protobuf | 7.34.0 | Protocol buffers |
| grpcio | 1.78.0 | RPC framework |
| regex | 2026.2.28 | Regular expressions |
| tokenizers | 0.22.2 | Text tokenization |
| huggingface-hub | 1.6.0 | Model hub |
| safetensors | 0.7.0 | Safe tensor serialization |
| filelock | 3.25.1 | File locking |
| pyyaml | 6.0.3 | YAML parsing |
| fsspec | 2026.2.0 | Filesystem spec |
| networkx | 3.4.2 | Graph library |
| jinja2 | 3.1.6 | Templating |
| sympy | 1.14.0 | Symbolic math |
| hydra-core | 1.0.7 | Configuration |
| omegaconf | 2.0.6 | Configuration |
| antlr4-python3-runtime | 4.8 | Parser generator |
| sacrebleu | 2.6.0 | BLEU score |
| bitarray | 3.8.0 | Bit arrays |
| cython | 3.2.4 | C extensions |
| tabulate | 0.10.0 | Table formatting |
| portalocker | 3.2.0 | File locking |
| lxml | 6.0.2 | XML processing |
| httpx | 0.28.1 | HTTP client |
| httpcore | 1.0.9 | HTTP core |
| h11 | 0.16.0 | HTTP/1.1 |
| anyio | 4.12.1 | Async I/O |
| certifi | 2026.2.25 | SSL certificates |
| charset-normalizer | 3.4.5 | Encoding detection |
| idna | 3.11 | Internationalized domains |
| urllib3 | 2.6.3 | HTTP client |
| requests | 2.32.5 | HTTP library |

## Verification

Run the verification script:

```bash
python -c "
import torch
import torchaudio
import transformers
import fairseq
import numpy
import sklearn
import scipy
import librosa
import soundfile
import matplotlib
import seaborn
import pandas
import tqdm
print('='*60)
print('All packages imported successfully!')
print('='*60)
print(f'PyTorch: {torch.__version__}')
print(f'TorchAudio: {torchaudio.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Fairseq: {fairseq.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Librosa: {librosa.__version__}')
print(f'SoundFile: {soundfile.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {seaborn.__version__}')
print(f'Pandas: {pandas.__version__}')
print('='*60)
"
```

Expected output:
```
============================================================
All packages imported successfully!
============================================================
PyTorch: 2.10.0+cpu
TorchAudio: 2.10.0+cpu
Transformers: 5.3.0
Fairseq: 0.12.2
NumPy: 2.2.6
Scikit-learn: 1.7.2
SciPy: 1.15.3
Librosa: 0.11.0
SoundFile: 0.13.1
Matplotlib: 0.13.1
Seaborn: 0.13.2
Pandas: 2.3.3
============================================================
```

## Common Issues & Solutions

### Issue 1: Fairseq installation fails

**Solution:**
```bash
# Install build tools first
pip install Cython
pip install --upgrade setuptools wheel

# Try installing fairseq again
pip install fairseq==0.12.2 --no-build-isolation
```

### Issue 2: SoundFile cannot find libsndfile

**Windows:**
- Download and install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install libsndfile1
```

**macOS:**
```bash
brew install libsndfile
```

### Issue 3: Numba/llvmlite installation fails

**Solution:**
```bash
# Install specific versions
pip install llvmlite==0.46.0
pip install numba==0.64.0
```

### Issue 4: Out of memory during model download

**Solution:**
```bash
# Set cache directory to larger drive
export HF_HOME=/path/to/large/drive
export TRANSFORMERS_CACHE=/path/to/large/drive
```

### Issue 5: CUDA out of memory

**Solution:**
Use CPU instead:
```python
# In your code
device = 'cpu'
```

Or reduce batch size:
```bash
python compare_embeddings.py --max_samples 50
```

## Environment Variables

Optional environment variables for configuration:

```bash
# HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"

# Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA settings (if using GPU)
export CUDA_VISIBLE_DEVICES=0
```

## Export/Import Environment

### Export installed packages:

```bash
pip freeze > installed_packages.txt
```

### Import on new machine:

```bash
pip install -r installed_packages.txt
```

## Testing the Setup

Run a quick test:

```bash
# Test with synthetic data
python quick_start.py
```

This will:
1. Create synthetic audio data
2. Load Wav2Vec2 and Data2Vec models
3. Extract embeddings
4. Compare and visualize results

## Directory Structure After Setup

```
workspace/
├── .venv/                      # Virtual environment
├── src/                        # Source code
│   ├── __init__.py
│   ├── wav2vec2_extractor.py
│   ├── data2vec_extractor.py
│   ├── comparison_framework.py
│   ├── visualization.py
│   └── dataset_loader.py
├── compare_embeddings.py       # Main script
├── quick_start.py              # Quick demo
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── SETUP.md                    # This file
├── .gitignore                  # Git ignore rules
└── config.yaml                 # Configuration template
```

## Next Steps

After successful setup:

1. **Quick Test:**
   ```bash
   python quick_start.py
   ```

2. **Full Comparison (Synthetic):**
   ```bash
   python compare_embeddings.py --dataset synthetic --max_samples 100
   ```

3. **LibriSpeech Dataset:**
   ```bash
   python compare_embeddings.py \
       --dataset librispeech \
       --librispeech_root /path/to/LibriSpeech \
       --max_samples 500
   ```

4. **Custom Dataset:**
   ```bash
   python compare_embeddings.py \
       --dataset custom \
       --data_dir /path/to/audio \
       --max_samples 200
   ```

## Support

If you encounter issues:

1. Check Python version: `python --version` (should be 3.8-3.11)
2. Verify virtual environment is activated
3. Check available disk space
4. Review error messages and search for solutions online
5. Ensure all dependencies are installed in correct order

## License

This setup guide is provided for research purposes. All third-party packages have their own licenses.