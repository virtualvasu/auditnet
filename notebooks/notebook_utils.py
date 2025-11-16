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

class OptimizedLSTM(nn.Module):
    """
    Optimized LSTM architecture for vulnerability detection.
    Captures sequential patterns in Solidity code using bidirectional LSTM
    with attention mechanism and residual connections.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_size: int = 256, num_layers: int = 2, 
                 num_classes: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, use_attention: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            lstm_output_size = hidden_size * self.num_directions
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        
        # Classification layers
        final_hidden_size = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(final_hidden_size)
        
        # Residual connection for classification
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize attention weights
        if self.use_attention:
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, return_attention=False):
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding_dropout(embedded)
        
        # Pack padded sequence for efficient LSTM processing
        lengths = attention_mask.sum(dim=1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        lstm_output, unpacked_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # lstm_output: [batch_size, actual_seq_len, hidden_size * num_directions]
        
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = self.attention(lstm_output)  # [batch_size, actual_seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch_size, actual_seq_len]
            
            # Create attention mask that matches the unpacked sequence length
            batch_size, actual_seq_len = lstm_output.size(0), lstm_output.size(1)
            device = lstm_output.device
            
            # Create mask based on unpacked lengths
            unpacked_mask = torch.zeros(batch_size, actual_seq_len, device=device, dtype=torch.bool)
            for i, length in enumerate(unpacked_lengths):
                unpacked_mask[i, :length] = 1
            
            # Mask attention weights for padding tokens (use -1e4 for half precision compatibility)
            mask_value = -1e4 if attention_weights.dtype == torch.float16 else -1e9
            attention_weights = attention_weights.masked_fill(~unpacked_mask, mask_value)
            attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, actual_seq_len]
            
            # Apply attention to get context vector
            context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # [batch_size, 1, hidden_size * num_directions]
            context = context.squeeze(1)  # [batch_size, hidden_size * num_directions]
            
            final_representation = context
            
        else:
            # Use last hidden state (from both directions for bidirectional)
            if self.lstm.bidirectional:
                # Concatenate forward and backward final hidden states
                final_representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                final_representation = hidden[-1]
        
        # Classification
        output = self.dropout(final_representation)
        output = self.batch_norm(output)
        logits = self.classifier(output)
        
        if return_attention and self.use_attention:
            return logits, attention_weights
        return logits

class OptimizedCNN(nn.Module):
    """
    Optimized CNN architecture for vulnerability detection.
    Uses multiple kernel sizes to capture different structural patterns
    in Solidity code with batch normalization and global max pooling.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 num_filters: int = 128, filter_sizes: List[int] = [3, 4, 5, 6, 7],
                 num_classes: int = 2, dropout: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.use_batch_norm = use_batch_norm
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Convolutional layers for different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k//2  # Same padding
            )
            for k in filter_sizes
        ])
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters) for _ in filter_sizes
            ])
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout)
        
        # Global max pooling will be applied dynamically
        
        # Classification layers
        total_filters = len(filter_sizes) * num_filters
        
        # Multi-layer classifier with skip connections
        self.classifier = nn.ModuleDict({
            'layer1': nn.Linear(total_filters, total_filters // 2),
            'layer2': nn.Linear(total_filters // 2, total_filters // 4),
            'output': nn.Linear(total_filters // 4, num_classes),
            'skip': nn.Linear(total_filters, total_filters // 4)  # Skip connection
        })
        
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier_batch_norm1 = nn.BatchNorm1d(total_filters // 2)
        self.classifier_batch_norm2 = nn.BatchNorm1d(total_filters // 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize convolutional layers
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(conv.bias)
        
        # Initialize classifier layers
        for name, layer in self.classifier.items():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, return_attention=False):
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding_dropout(embedded)
        
        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply different convolutional filters
        conv_outputs = []
        
        for i, conv in enumerate(self.convs):
            # Convolution
            conv_out = conv(embedded)  # [batch_size, num_filters, seq_len]
            
            # Batch normalization
            if self.use_batch_norm:
                conv_out = self.batch_norms[i](conv_out)
            
            # Activation
            conv_out = self.relu(conv_out)
            conv_out = self.conv_dropout(conv_out)
            
            # Global max pooling
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]
        
        # Classification with skip connections
        x = concatenated
        
        # First layer
        x1 = self.relu(self.classifier['layer1'](x))
        x1 = self.classifier_batch_norm1(x1)
        x1 = self.classifier_dropout(x1)
        
        # Second layer
        x2 = self.relu(self.classifier['layer2'](x1))
        x2 = self.classifier_batch_norm2(x2)
        
        # Skip connection from input to second layer
        x_skip = self.classifier['skip'](x)
        x2 = x2 + x_skip  # Residual connection
        
        x2 = self.classifier_dropout(x2)
        
        # Output layer
        logits = self.classifier['output'](x2)
        
        # For CNN, we don't have meaningful attention weights like transformers or LSTM
        # But we can return the global max pooling activations as a form of "attention"
        if return_attention:
            # Return the max pooled features as pseudo-attention
            attention_like = torch.stack(conv_outputs, dim=1)  # [batch_size, num_filters, num_filter_types]
            return logits, attention_like
        
        return logits

class EnsembleModel(nn.Module):
    """
    Ensemble model combining CodeBERT, LSTM, and CNN predictions.
    Supports multiple fusion strategies: weighted voting, stacking, and attention-based fusion.
    """
    
    def __init__(self, codebert_model, lstm_model, cnn_model, 
                 num_classes: int = 2, fusion_strategy: str = 'weighted',
                 fusion_hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        
        self.codebert_model = codebert_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        
        # Freeze individual models during ensemble training (optional)
        self.freeze_individual_models()
        
        if fusion_strategy == 'weighted':
            # Simple weighted combination
            self.weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
            
        elif fusion_strategy == 'stacking':
            # Meta-learner for stacking
            self.meta_learner = nn.Sequential(
                nn.Linear(3 * num_classes, fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_size // 2, num_classes)
            )
            
        elif fusion_strategy == 'attention':
            # Attention-based fusion
            self.attention_layer = nn.Sequential(
                nn.Linear(3 * num_classes, fusion_hidden_size),
                nn.Tanh(),
                nn.Linear(fusion_hidden_size, 3),  # Attention weights for 3 models
                nn.Softmax(dim=1)
            )
            
        else:
            raise ValueError("fusion_strategy must be 'weighted', 'stacking', or 'attention'")
    
    def freeze_individual_models(self):
        """Freeze parameters of individual models"""
        for param in self.codebert_model.parameters():
            param.requires_grad = False
        for param in self.lstm_model.parameters():
            param.requires_grad = False
        for param in self.cnn_model.parameters():
            param.requires_grad = False
    
    def unfreeze_individual_models(self):
        """Unfreeze parameters of individual models for fine-tuning"""
        for param in self.codebert_model.parameters():
            param.requires_grad = True
        for param in self.lstm_model.parameters():
            param.requires_grad = True
        for param in self.cnn_model.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, return_individual_predictions=False):
        # Get predictions from individual models
        with torch.no_grad() if self.training else torch.enable_grad():
            codebert_logits = self.codebert_model(input_ids, attention_mask)
            lstm_logits = self.lstm_model(input_ids, attention_mask)
            cnn_logits = self.cnn_model(input_ids, attention_mask)
        
        # Apply softmax to get probabilities for fusion
        codebert_probs = torch.softmax(codebert_logits, dim=-1)
        lstm_probs = torch.softmax(lstm_logits, dim=-1)
        cnn_probs = torch.softmax(cnn_logits, dim=-1)
        
        if self.fusion_strategy == 'weighted':
            # Weighted combination
            weights = torch.softmax(self.weights, dim=0)  # Ensure weights sum to 1
            ensemble_probs = (weights[0] * codebert_probs + 
                            weights[1] * lstm_probs + 
                            weights[2] * cnn_probs)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)  # Convert back to logits
            
        elif self.fusion_strategy == 'stacking':
            # Concatenate predictions and pass through meta-learner
            stacked_features = torch.cat([codebert_logits, lstm_logits, cnn_logits], dim=-1)
            ensemble_logits = self.meta_learner(stacked_features)
            
        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            stacked_logits = torch.cat([codebert_logits, lstm_logits, cnn_logits], dim=-1)
            attention_weights = self.attention_layer(stacked_logits)  # [batch_size, 3]
            
            # Apply attention weights
            weighted_probs = (attention_weights[:, 0:1] * codebert_probs +
                            attention_weights[:, 1:2] * lstm_probs +
                            attention_weights[:, 2:3] * cnn_probs)
            ensemble_logits = torch.log(weighted_probs + 1e-8)
        
        if return_individual_predictions:
            return {
                'ensemble_logits': ensemble_logits,
                'codebert_logits': codebert_logits,
                'lstm_logits': lstm_logits,
                'cnn_logits': cnn_logits,
                'fusion_weights': getattr(self, 'weights', None) or 
                               getattr(self, 'attention_weights', None)
            }
        
        return ensemble_logits

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
        code = str(row['code'])
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