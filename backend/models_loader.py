"""
Model Loading Utility
Handles loading of trained models for vulnerability detection
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CodeBERTForVulnerabilityDetection(nn.Module):
    """CodeBERT model for vulnerability detection"""
    
    def __init__(self, model_name, num_classes, dropout=0.1, freeze_base=False):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LSTMClassifier(nn.Module):
    """LSTM model with attention for vulnerability detection"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=256, num_classes=1,
                 num_layers=2, dropout=0.3, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.use_attention:
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            context = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        context = self.batch_norm(context)
        logits = self.classifier(context)
        return logits


class CNNClassifier(nn.Module):
    """Multi-kernel CNN model for vulnerability detection"""
    
    def __init__(self, vocab_size, embedding_dim=256, num_filters=128,
                 filter_sizes=[3,4,5,6,7], num_classes=1, dropout=0.3, use_batch_norm=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_batch_norm = use_batch_norm
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters) for _ in filter_sizes
            ])
        
        total_filters = len(filter_sizes) * num_filters
        self.classifier = nn.ModuleDict({
            'layer1': nn.Linear(total_filters, total_filters // 2),
            'layer2': nn.Linear(total_filters // 2, total_filters // 4),
            'output': nn.Linear(total_filters // 4, num_classes),
            'skip': nn.Linear(total_filters, total_filters // 4)
        })
        
        if use_batch_norm:
            self.classifier_batch_norm1 = nn.BatchNorm1d(total_filters // 2)
            self.classifier_batch_norm2 = nn.BatchNorm1d(total_filters // 4)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        
        conved = []
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(embedded))
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            pooled = torch.max_pool1d(x, x.shape[2]).squeeze(2)
            conved.append(pooled)
        
        cat = torch.cat(conved, dim=1)
        cat = self.dropout(cat)
        
        h = self.classifier['layer1'](cat)
        if self.use_batch_norm:
            h = self.classifier_batch_norm1(h)
        h = torch.relu(h)
        h = self.dropout(h)
        
        h_skip = self.classifier['skip'](cat)
        h = h + h_skip
        
        h = self.classifier['layer2'](h)
        if self.use_batch_norm:
            h = self.classifier_batch_norm2(h)
        h = torch.relu(h)
        h = self.dropout(h)
        
        logits = self.classifier['output'](h)
        return logits


def load_model_with_checkpoint_compatibility(model, checkpoint):
    """Load model state dict with compatibility"""
    state_dict = checkpoint['model_state_dict']
    
    if any(key.startswith('bert.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key.replace('bert.', 'codebert.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    return model


def get_available_models() -> Dict:
    """Get all available trained models"""
    models_dir = Path(__file__).parent.parent / 'models'
    checkpoints_dir = Path(__file__).parent.parent / 'results' / 'checkpoints'
    
    available_models = {
        'CodeBERT (Latest)': {'type': 'codebert', 'files': []},
        'LSTM (Latest)': {'type': 'lstm', 'files': []},
        'CNN (Latest)': {'type': 'cnn', 'files': []},
        'Ensemble - Stacking (Best)': {'type': 'ensemble_stacking', 'files': []},
        'Ensemble - Attention': {'type': 'ensemble_attention', 'files': []}
    }
    
    if models_dir.exists():
        available_models['CodeBERT (Latest)']['files'].extend(list(models_dir.glob('codebert*.pt')))
        available_models['LSTM (Latest)']['files'].extend(list(models_dir.glob('lstm*.pt')))
        available_models['CNN (Latest)']['files'].extend(list(models_dir.glob('cnn*.pt')))
    
    if checkpoints_dir.exists():
        available_models['CodeBERT (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_codebert*.pt')))
        available_models['LSTM (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_lstm*.pt')))
        available_models['CNN (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_cnn*.pt')))
        available_models['Ensemble - Stacking (Best)']['files'].extend(
            list(checkpoints_dir.glob('best_ensemble_stacking*.pt'))
        )
        available_models['Ensemble - Attention']['files'].extend(
            list(checkpoints_dir.glob('best_ensemble_attention*.pt'))
        )
    
    available_models = {name: info for name, info in available_models.items() if info['files']}
    return available_models


def load_model(model_choice: str = 'auto') -> Tuple[Optional[nn.Module], Optional[AutoTokenizer], str, str]:
    """
    Load the selected model
    
    Args:
        model_choice: 'auto' for best model, or specific model name
        
    Returns:
        Tuple of (model, tokenizer, device, task_type)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    available_models = get_available_models()
    
    if not available_models:
        logger.error("No model checkpoints found!")
        return None, None, str(device), "binary"
    
    # Select model
    if model_choice == 'auto':
        if 'Ensemble - Stacking (Best)' in available_models:
            model_choice = 'Ensemble - Stacking (Best)'
        elif 'CodeBERT (Latest)' in available_models:
            model_choice = 'CodeBERT (Latest)'
        else:
            model_choice = list(available_models.keys())[0]
    
    if model_choice not in available_models:
        logger.error(f"Selected model '{model_choice}' not found!")
        return None, None, str(device), "binary"
    
    model_info = available_models[model_choice]
    model_type = model_info['type']
    model_file = max(model_info['files'], key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Loading model: {model_choice} from {model_file}")
    
    try:
        checkpoint = torch.load(model_file, map_location=device)
        
        if model_type == 'codebert':
            model_name = checkpoint.get('model_name', 'microsoft/codebert-base')
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = CodeBERTForVulnerabilityDetection(model_name, num_classes)
            model = load_model_with_checkpoint_compatibility(model, checkpoint)
            
        elif model_type == 'lstm':
            vocab_size = checkpoint.get('vocab_size', 10000)
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            model_config = checkpoint.get('model_config', {})
            embedding_dim = model_config.get('embedding_dim', 256)
            hidden_size = model_config.get('hidden_size', 256)
            num_layers = model_config.get('num_layers', 2)
            dropout = model_config.get('dropout', 0.3)
            use_attention = model_config.get('use_attention', True)
            
            tokenizer = checkpoint.get('tokenizer', None)
            model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                use_attention=use_attention
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
        elif model_type == 'cnn':
            vocab_size = checkpoint.get('vocab_size', 10000)
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            model_config = checkpoint.get('model_config', {})
            embedding_dim = model_config.get('embedding_dim', 256)
            num_filters = model_config.get('num_filters', 128)
            filter_sizes = model_config.get('filter_sizes', [3, 4, 5, 6, 7])
            dropout = model_config.get('dropout', 0.3)
            use_batch_norm = model_config.get('use_batch_norm', True)
            
            tokenizer = checkpoint.get('tokenizer', None)
            model = CNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                num_classes=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None, None, str(device), "binary"
        
        model.to(device)
        model.eval()
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer, str(device), task_type
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None, None, str(device), "binary"
