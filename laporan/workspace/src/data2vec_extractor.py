"""
Data2Vec Model Wrapper
Supports Fairseq implementation for audio
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
import os

warnings.filterwarnings('ignore')


class Data2VecExtractor:
    """
    Data2Vec Feature Extractor
    Self-supervised learning with unified architecture across modalities
    """
    
    def __init__(self, 
                 model_path: str = "facebook/data2vec-audio-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 sample_rate: int = 16000):
        """
        Initialize Data2Vec extractor
        
        Args:
            model_path: Path to fairseq checkpoint or HuggingFace model
            device: Device to run model on
            sample_rate: Expected sample rate (16kHz)
        """
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self.hidden_size = None
        self.num_layers = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the data2vec model"""
        try:
            # Try loading from fairseq first
            self._load_fairseq_model()
        except Exception as e:
            print(f"Fairseq loading failed: {e}")
            try:
                # Fallback to HuggingFace
                self._load_huggingface_model()
            except Exception as e2:
                print(f"HuggingFace loading also failed: {e2}")
                raise RuntimeError("Failed to load data2vec model")
    
    def _load_fairseq_model(self):
        """Load model using fairseq"""
        try:
            from fairseq import checkpoint_utils, tasks
            from fairseq.models.data2vec import Data2VecAudioModel
            
            # Load model checkpoint
            state = checkpoint_utils.load_checkpoint_to_cpu(self.model_path)
            
            # Build model
            args = state["args"]
            task = tasks.setup_task(args)
            model = task.build_model(args)
            model.load_state_dict(state["model"], strict=True)
            
            self.model = model.to(self.device)
            self.model.eval()
            
            self.hidden_size = args.encoder_embed_dim
            self.num_layers = args.encoder_layers
            
            print(f"Loaded Data2Vec (Fairseq): {self.model_path}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.num_layers}")
            
        except ImportError:
            raise ImportError("Fairseq not installed. Install with: pip install fairseq")
    
    def _load_huggingface_model(self):
        """Load model from HuggingFace (if available)"""
        from transformers import AutoModel, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        print(f"Loaded Data2Vec (HuggingFace): {self.model_path}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
    
    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio for model input
        
        Args:
            audio: Audio array (should be at 16kHz)
        
        Returns:
            Preprocessed tensor
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0  # Assume 16-bit PCM
        
        return torch.from_numpy(audio).unsqueeze(0).to(self.device)
    
    def extract_embeddings(self, 
                          audio: Union[np.ndarray, torch.Tensor, List],
                          layers: Optional[List[int]] = None,
                          pooling: str = "mean",
                          return_all_layers: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from specified layers
        
        Args:
            audio: Audio input
            layers: List of layer indices (None = all layers)
            pooling: Pooling strategy ("mean", "first", "last")
            return_all_layers: If True, return all layer embeddings
        
        Returns:
            Dictionary with embeddings and metadata
        """
        if isinstance(audio, list):
            audio = np.array(audio)
        
        if isinstance(audio, np.ndarray):
            inputs = self.preprocess(audio)
        else:
            inputs = audio.to(self.device)
        
        results = {
            "model": "data2vec",
            "pooling": pooling,
            "layers_extracted": [],
            "embeddings": {}
        }
        
        with torch.no_grad():
            # Data2Vec typically outputs hidden states
            if hasattr(self.model, 'forward') and not hasattr(self.model, 'extract_features'):
                # HuggingFace style
                outputs = self.model(inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            else:
                # Fairseq style
                res = self.model.extract_features(
                    inputs,
                    padding_mask=None,
                    mask=False,  # Don't apply masking for inference
                    feature_only=False
                )
                # res contains features and potentially other outputs
                if isinstance(res, tuple):
                    hidden_states = res[0]
                else:
                    hidden_states = res
        
        # Handle different output formats
        if isinstance(hidden_states, tuple):
            hidden_states_list = list(hidden_states)
        elif isinstance(hidden_states, torch.Tensor):
            hidden_states_list = [hidden_states]
        else:
            hidden_states_list = [hidden_states]
        
        # Determine which layers to extract
        if layers is None:
            layers_to_extract = range(len(hidden_states_list))
        else:
            layers_to_extract = layers
        
        for layer_idx in layers_to_extract:
            if layer_idx >= len(hidden_states_list):
                continue
                
            layer_output = hidden_states_list[layer_idx]
            if len(layer_output.shape) == 2:
                layer_output = layer_output.unsqueeze(0)  # Add batch dimension
            
            # Apply pooling
            if pooling == "mean":
                embedding = layer_output.mean(dim=1)  # (batch, hidden_dim)
            elif pooling == "first":
                embedding = layer_output[:, 0, :]  # (batch, hidden_dim)
            elif pooling == "last":
                embedding = layer_output[:, -1, :]  # (batch, hidden_dim)
            else:
                embedding = layer_output  # Return full sequence
            
            results["embeddings"][f"layer_{layer_idx}"] = embedding.cpu()
            results["layers_extracted"].append(layer_idx)
        
        return results
    
    def extract_multi_scale(self, 
                           audio: np.ndarray,
                           pooling: str = "mean") -> Dict[str, torch.Tensor]:
        """
        Extract embeddings at multiple scales
        
        Args:
            audio: Audio input
            pooling: Pooling strategy
        
        Returns:
            Multi-scale embeddings
        """
        layers = [0, self.num_layers // 2, self.num_layers - 1]
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
    
    def get_model_info(self) -> Dict:
        """Get model configuration info"""
        return {
            "model_name": self.model_path,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sample_rate": self.sample_rate,
            "framework": "fairseq",
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def compare_with_teacher(self, 
                            audio: np.ndarray,
                            student_layer: int = -1) -> Dict:
        """
        Compare student (audio encoder) with teacher representations
        Specific to data2vec architecture
        
        Args:
            audio: Audio input
            student_layer: Which student layer to compare
        
        Returns:
            Comparison metrics
        """
        # Data2Vec uses a teacher-student framework
        # This method extracts and compares both representations
        
        inputs = self.preprocess(audio)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_teacher_representation'):
                # If model has explicit teacher method
                teacher_emb = self.model.get_teacher_representation(inputs)
                student_result = self.extract_embeddings(audio, layers=[student_layer])
                student_emb = student_result["embeddings"][f"layer_{student_layer}"]
                
                # Compute similarity
                similarity = torch.cosine_similarity(
                    student_emb.flatten(), 
                    teacher_emb.flatten(), 
                    dim=0
                )
                
                return {
                    "student_teacher_similarity": similarity.item(),
                    "student_embedding": student_emb,
                    "teacher_embedding": teacher_emb
                }
            else:
                return {"error": "Teacher representation not available"}


# Alternative simplified version using HuggingFace
try:
    from transformers import Data2VecAudioModel, Data2VecAudioProcessor
    
    class Data2VecHuggingFaceExtractor(Data2VecExtractor):
        """
        Data2Vec using HuggingFace Transformers (simpler implementation)
        """
        
        def __init__(self, 
                     model_name: str = "facebook/data2vec-audio-base",
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     sample_rate: int = 16000):
            """
            Initialize Data2Vec HuggingFace extractor
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
            """Load model from HuggingFace"""
            self.processor = Data2VecAudioProcessor.from_pretrained(self.model_name)
            self.model = Data2VecAudioModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.hidden_size = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers
            
            print(f"Loaded Data2Vec (HuggingFace): {self.model_name}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.num_layers}")
        
        def preprocess(self, audio: np.ndarray) -> torch.Tensor:
            """Preprocess audio"""
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            return inputs.input_values.to(self.device)

except ImportError:
    Data2VecHuggingFaceExtractor = Data2VecExtractor