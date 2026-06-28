"""
Wav2Vec2 Model Wrapper
Supports both HuggingFace and Fairseq implementations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Processor
import warnings

warnings.filterwarnings('ignore')


class Wav2Vec2Extractor:
    """
    Wav2Vec2 Feature Extractor
    Extracts embeddings from different layers and supports multiple pooling strategies
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base-960h",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 sample_rate: int = 16000):
        """
        Initialize Wav2Vec2 extractor
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            sample_rate: Expected sample rate (16kHz for wav2vec2)
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.processor = None
        self.model = None
        self.hidden_size = None
        self.num_layers = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the wav2vec2 model and processor"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.hidden_size = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers
            
            print(f"Loaded Wav2Vec2: {self.model_name}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.num_layers}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio for model input
        
        Args:
            audio: Audio array (should be at 16kHz)
        
        Returns:
            Preprocessed tensor
        """
        # Ensure correct sample rate
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        # Process with wav2vec2 processor
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        return inputs.input_values.to(self.device)
    
    def extract_embeddings(self, 
                          audio: Union[np.ndarray, torch.Tensor, List],
                          layers: Optional[List[int]] = None,
                          pooling: str = "mean",
                          return_all_layers: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from specified layers
        
        Args:
            audio: Audio input (numpy array or tensor)
            layers: List of layer indices to extract (None = all layers)
            pooling: Pooling strategy ("mean", "cls", "last")
            return_all_layers: If True, return embeddings from all layers
        
        Returns:
            Dictionary with embeddings and metadata
        """
        if isinstance(audio, list):
            audio = np.array(audio)
        
        if isinstance(audio, np.ndarray):
            inputs = self.preprocess(audio)
        else:
            inputs = audio.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
        
        # Extract hidden states from all layers
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
        
        results = {
            "model": "wav2vec2",
            "pooling": pooling,
            "layers_extracted": [],
            "embeddings": {}
        }
        
        # Determine which layers to extract
        if layers is None:
            layers_to_extract = range(len(hidden_states))
        else:
            layers_to_extract = layers
        
        for layer_idx in layers_to_extract:
            if layer_idx >= len(hidden_states):
                continue
                
            layer_output = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            
            # Apply pooling
            if pooling == "mean":
                embedding = layer_output.mean(dim=1)  # (batch, hidden_dim)
            elif pooling == "cls":
                embedding = layer_output[:, 0, :]  # (batch, hidden_dim)
            elif pooling == "last":
                # Remove padding and take last non-padding token
                attention_mask = (inputs != self.processor.tokenizer.pad_token_id).long()
                seq_lengths = attention_mask.sum(dim=1) - 1
                embedding = layer_output[torch.arange(layer_output.size(0)), seq_lengths]
            else:
                embedding = layer_output  # Return full sequence
            
            results["embeddings"][f"layer_{layer_idx}"] = embedding.cpu()
            results["layers_extracted"].append(layer_idx)
        
        # Store final projection layer output
        if hasattr(outputs, 'last_hidden_state'):
            results["final_projection"] = outputs.last_hidden_state.cpu()
        
        return results
    
    def extract_multi_scale(self, 
                           audio: np.ndarray,
                           pooling: str = "mean") -> Dict[str, torch.Tensor]:
        """
        Extract embeddings at multiple scales (early, middle, late layers)
        
        Args:
            audio: Audio input
            pooling: Pooling strategy
        
        Returns:
            Multi-scale embeddings
        """
        layers = [0, self.num_layers // 2, self.num_layers]
        return self.extract_embeddings(audio, layers=layers, pooling=pooling)
    
    def compute_similarity_matrix(self, 
                                  audio_list: List[np.ndarray],
                                  layer_idx: int = -1) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix
        
        Args:
            audio_list: List of audio samples
            layer_idx: Which layer to use (-1 for last)
        
        Returns:
            Similarity matrix
        """
        embeddings = []
        
        for audio in audio_list:
            result = self.extract_embeddings(audio, layers=[layer_idx], pooling="mean")
            emb = result["embeddings"][f"layer_{layer_idx}"]
            embeddings.append(emb.squeeze().numpy())
        
        embeddings = np.stack(embeddings)
        
        # Normalize and compute cosine similarity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        return similarity_matrix
    
    def get_attention_weights(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract attention weights from transformer layers
        
        Args:
            audio: Audio input
        
        Returns:
            Attention weights if available
        """
        inputs = self.preprocess(audio)
        
        with torch.no_grad():
            outputs = self.model(inputs, output_attentions=True)
        
        if hasattr(outputs, 'attentions'):
            return torch.stack(outputs.attentions)  # (num_layers, batch, num_heads, seq_len, seq_len)
        return None
    
    def get_model_info(self) -> Dict:
        """Get model configuration info"""
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sample_rate": self.sample_rate,
            "framework": "huggingface",
            "parameters": sum(p.numel() for p in self.model.parameters())
        }