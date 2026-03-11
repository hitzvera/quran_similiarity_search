"""
Dataset Loader for Audio Comparison
Supports multiple datasets for benchmarking
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import json


class AudioDatasetLoader:
    """
    Flexible dataset loader for audio files
    Supports: LibriSpeech, custom datasets, and synthetic data
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 10.0):
        """
        Initialize dataset loader
        
        Args:
            sample_rate: Target sample rate (16kHz for wav2vec2/data2vec)
            max_duration: Maximum audio duration in seconds
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            filepath: Path to audio file
        
        Returns:
            Audio array at target sample rate
        """
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            
            # Trim or pad to max duration
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            
            return audio
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_librispeech(self, 
                         root_dir: str,
                         subset: str = "test-clean",
                         max_samples: Optional[int] = None) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load LibriSpeech dataset
        
        Args:
            root_dir: Root directory of LibriSpeech
            subset: Subset to load (test-clean, test-other, dev-clean, dev-other)
            max_samples: Maximum number of samples to load
        
        Returns:
            Tuple of (audio_list, speaker_ids, file_paths)
        """
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"LibriSpeech subset not found: {subset_dir}")
            return [], [], []
        
        audio_list = []
        speaker_ids = []
        file_paths = []
        
        # LibriSpeech structure: subset/speaker_id/chapter_id/file.flac
        speaker_dirs = sorted([d for d in os.listdir(subset_dir) 
                              if os.path.isdir(os.path.join(subset_dir, d))])
        
        print(f"Loading LibriSpeech {subset}...")
        for speaker_id in tqdm(speaker_dirs, desc="Speakers"):
            speaker_path = os.path.join(subset_dir, speaker_id)
            
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if not os.path.isdir(chapter_path):
                    continue
                
                for filename in os.listdir(chapter_path):
                    if filename.endswith('.flac'):
                        filepath = os.path.join(chapter_path, filename)
                        
                        audio = self.load_audio(filepath)
                        if audio is not None:
                            audio_list.append(audio)
                            speaker_ids.append(int(speaker_id))
                            file_paths.append(filepath)
                            
                            if max_samples and len(audio_list) >= max_samples:
                                return audio_list, speaker_ids, file_paths
        
        print(f"Loaded {len(audio_list)} samples from {len(set(speaker_ids))} speakers")
        return audio_list, speaker_ids, file_paths
    
    def load_custom_dataset(self,
                           data_dir: str,
                           label_file: Optional[str] = None,
                           max_samples: Optional[int] = None) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load custom dataset
        
        Args:
            data_dir: Directory containing audio files
            label_file: Optional JSON/CSV file with labels
            max_samples: Maximum number of samples
        
        Returns:
            Tuple of (audio_list, labels, file_paths)
        """
        audio_list = []
        labels = []
        file_paths = []
        
        # Load labels if provided
        label_dict = {}
        if label_file and os.path.exists(label_file):
            if label_file.endswith('.json'):
                with open(label_file, 'r') as f:
                    label_dict = json.load(f)
            elif label_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(label_file)
                label_dict = dict(zip(df['filename'], df['label']))
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Loading custom dataset from {data_dir}...")
        for filepath in tqdm(audio_files[:max_samples], desc="Files"):
            audio = self.load_audio(filepath)
            if audio is not None:
                audio_list.append(audio)
                file_paths.append(filepath)
                
                # Get label
                filename = os.path.basename(filepath)
                if filename in label_dict:
                    labels.append(label_dict[filename])
                else:
                    # Use directory name as label
                    labels.append(os.path.basename(os.path.dirname(filepath)))
        
        # Convert string labels to integers
        if labels and isinstance(labels[0], str):
            unique_labels = sorted(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = [label_map[l] for l in labels]
        
        print(f"Loaded {len(audio_list)} samples")
        return audio_list, labels, file_paths
    
    def load_from_list(self,
                      file_list: List[str],
                      labels: Optional[List] = None) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load audio from a list of file paths
        
        Args:
            file_list: List of audio file paths
            labels: Optional list of labels
        
        Returns:
            Tuple of (audio_list, labels, file_paths)
        """
        audio_list = []
        loaded_labels = []
        file_paths = []
        
        print(f"Loading {len(file_list)} files...")
        for i, filepath in enumerate(tqdm(file_list)):
            audio = self.load_audio(filepath)
            if audio is not None:
                audio_list.append(audio)
                file_paths.append(filepath)
                
                if labels and i < len(labels):
                    loaded_labels.append(labels[i])
                else:
                    loaded_labels.append(0)
        
        return audio_list, loaded_labels, file_paths
    
    def create_synthetic_dataset(self,
                                 n_samples: int = 100,
                                 n_classes: int = 5,
                                 duration: float = 5.0) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Create synthetic dataset with different frequency characteristics
        
        Args:
            n_samples: Number of samples per class
            n_classes: Number of classes
            duration: Duration of each sample in seconds
        
        Returns:
            Tuple of (audio_list, labels, descriptions)
        """
        audio_list = []
        labels = []
        descriptions = []
        
        print(f"Creating synthetic dataset: {n_samples*n_classes} samples, {n_classes} classes...")
        
        for class_id in range(n_classes):
            for sample_id in range(n_samples):
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                
                if class_id == 0:
                    # Low frequency sine wave
                    audio = np.sin(2 * np.pi * 200 * t)
                    desc = f"low_freq_{sample_id}"
                elif class_id == 1:
                    # High frequency sine wave
                    audio = np.sin(2 * np.pi * 2000 * t)
                    desc = f"high_freq_{sample_id}"
                elif class_id == 2:
                    # Chirp signal
                    audio = np.sin(2 * np.pi * (200 + 1800 * t / duration) * t)
                    desc = f"chirp_{sample_id}"
                elif class_id == 3:
                    # White noise
                    audio = np.random.randn(len(t))
                    desc = f"noise_{sample_id}"
                else:
                    # Mixed signal
                    audio = (np.sin(2 * np.pi * 500 * t) + 
                            0.5 * np.sin(2 * np.pi * 1500 * t) +
                            0.3 * np.random.randn(len(t)))
                    desc = f"mixed_{sample_id}"
                
                # Normalize
                audio = audio / (np.abs(audio).max() + 1e-8)
                
                audio_list.append(audio)
                labels.append(class_id)
                descriptions.append(desc)
        
        return audio_list, labels, descriptions
    
    def save_embeddings(self,
                       embeddings: np.ndarray,
                       labels: List,
                       file_paths: List[str],
                       save_path: str):
        """
        Save embeddings to disk
        
        Args:
            embeddings: Embedding array
            labels: List of labels
            file_paths: List of file paths
            save_path: Path to save (will create .npz file)
        """
        np.savez(save_path,
                embeddings=embeddings,
                labels=labels,
                file_paths=file_paths)
        print(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str) -> Tuple[np.ndarray, List, List[str]]:
        """
        Load embeddings from disk
        
        Args:
            load_path: Path to .npz file
        
        Returns:
            Tuple of (embeddings, labels, file_paths)
        """
        data = np.load(load_path, allow_pickle=True)
        embeddings = data['embeddings']
        labels = data['labels'].tolist()
        file_paths = data['file_paths'].tolist()
        
        print(f"Loaded {len(embeddings)} embeddings from {load_path}")
        return embeddings, labels, file_paths