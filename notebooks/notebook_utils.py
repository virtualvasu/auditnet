# Smart Contract Vulnerability Detector - Shared Utilities
"""
Common utilities and configurations shared across notebooks.
This module consolidates repeated code patterns to improve maintainability.
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Data manipulation
import pandas as pd
import numpy as np

# Deep learning
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class ProjectConfig:
    """Centralized project configuration"""
    
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Auto-detect project root
            current_path = Path().absolute()
            if current_path.name == 'notebooks':
                self.BASE_DIR = current_path.parent
            else:
                self.BASE_DIR = Path('/home/netweb/vasu/smart-contract-vuln-detector')
        else:
            self.BASE_DIR = Path(base_dir)
        
        # Define standard paths
        self.DATA_DIR = self.BASE_DIR / 'data/processed'
        self.RESULTS_DIR = self.BASE_DIR / 'results'
        self.MODELS_DIR = self.BASE_DIR / 'models'
        self.NOTEBOOKS_DIR = self.BASE_DIR / 'notebooks'
        
        # Model configuration
        self.MODEL_NAME = "microsoft/codebert-base"
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = 16
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.RESULTS_DIR / 'metrics',
            self.RESULTS_DIR / 'visualizations', 
            self.RESULTS_DIR / 'predictions',
            self.RESULTS_DIR / 'checkpoints',
            self.DATA_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_device(self) -> torch.device:
        """Get appropriate computing device"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def print_info(self):
        """Print configuration information"""
        print("🔧 PROJECT CONFIGURATION")
        print("=" * 40)
        print(f"Base directory: {self.BASE_DIR}")
        print(f"Data directory: {self.DATA_DIR}")
        print(f"Results directory: {self.RESULTS_DIR}")
        print(f"Models directory: {self.MODELS_DIR}")
        print(f"Model: {self.MODEL_NAME}")
        print(f"Max length: {self.MAX_LENGTH}")
        print(f"Batch size: {self.BATCH_SIZE}")
        
        device = self.get_device()
        print(f"Device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")

def setup_notebook_environment(config: Optional[ProjectConfig] = None, 
                             set_seeds: bool = True,
                             setup_plotting: bool = True) -> ProjectConfig:
    """
    Set up common notebook environment with configuration, seeds, and plotting.
    
    Args:
        config: Optional ProjectConfig instance
        set_seeds: Whether to set random seeds for reproducibility
        setup_plotting: Whether to configure matplotlib/seaborn
    
    Returns:
        ProjectConfig instance
    """
    if config is None:
        config = ProjectConfig()
    
    if set_seeds:
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    
    if setup_plotting:
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    return config

class VulnerabilityClassifier(nn.Module):
    """
    Shared vulnerability classifier architecture.
    Used across multiple notebooks (4, 5, 6).
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", 
                 num_classes: int = 2, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get the size of BERT's hidden states
        self.hidden_size = self.bert.config.hidden_size  # 768 for base models
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Store configuration
        self.num_classes = num_classes
        self.model_name = model_name
        
    def forward(self, input_ids, attention_mask, return_attention=False):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_attention:
            return logits, outputs.attentions
        return logits

class VulnerabilityDataset(Dataset):
    """
    Shared dataset class for vulnerability detection.
    Used in notebooks 3, 4, 5, 6.
    """
    
    def __init__(self, dataframe: pd.DataFrame, 
                 tokenizer, 
                 max_length: int = 512, 
                 label_type: str = 'binary'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_type = label_type
        
        if label_type == 'binary':
            self.labels = dataframe['is_vulnerable'].astype(int).values
            self.num_classes = 2
        elif label_type == 'multiclass':
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(dataframe['vulnerability_category'])
            self.num_classes = len(self.label_encoder.classes_)
        else:
            raise ValueError("label_type must be 'binary' or 'multiclass'")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        code = str(row['function_code'])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_datasets(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets"""
    train_df = pd.read_csv(config.DATA_DIR / 'train_functions.csv')
    val_df = pd.read_csv(config.DATA_DIR / 'validation_functions.csv')
    test_df = pd.read_csv(config.DATA_DIR / 'test_functions.csv')
    
    print(f"📊 DATASETS LOADED")
    print(f"Train: {len(train_df):,} samples")
    print(f"Validation: {len(val_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    return train_df, val_df, test_df

def load_model_components(config: ProjectConfig) -> Tuple[Any, Any, Dict]:
    """Load tokenizer, label encoder, and dataset configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load label encoder
    with open(config.DATA_DIR / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load dataset configuration
    with open(config.DATA_DIR / 'dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    return tokenizer, label_encoder, dataset_config

def find_latest_model(config: ProjectConfig) -> Optional[Path]:
    """Find the latest trained model file"""
    
    # Check for models in different locations
    model_locations = [
        list(config.MODELS_DIR.glob('*.pt')),
        list((config.RESULTS_DIR / 'checkpoints').glob('best_model_*.pt'))
    ]
    
    all_models = []
    for location in model_locations:
        all_models.extend(location)
    
    if not all_models:
        return None
    
    # Return the most recent model
    latest_model = sorted(all_models, key=lambda x: x.stat().st_mtime)[-1]
    return latest_model

def print_pytorch_info(config: Optional[ProjectConfig] = None):
    """Print PyTorch and CUDA information"""
    print("🚀 PYTORCH ENVIRONMENT")
    print("=" * 30)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if config:
        device = config.get_device()
        print(f"Using device: {device}")

# Quick setup function for notebooks
def quick_setup(set_seeds=True, setup_plotting=True, print_info=True) -> ProjectConfig:
    """
    One-line setup for notebooks.
    
    Usage in notebooks:
    from notebook_utils import quick_setup
    config = quick_setup()
    """
    config = setup_notebook_environment(
        set_seeds=set_seeds, 
        setup_plotting=setup_plotting
    )
    
    if print_info:
        config.print_info()
        print_pytorch_info(config)
    
    return config