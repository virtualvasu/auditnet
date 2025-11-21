# Smart Contract Vulnerability Detector

A machine learning system for detecting security vulnerabilities in Solidity smart contracts using fine-tuned CodeBERT.

## 🎯 Project Overview

This project fine-tunes CodeBERT on labeled contract functions to automatically classify whether a function contains specific vulnerability types. It uses datasets like Slither-audited code, SmartBugs-Wild, and SolidiFI to train a model that can predict vulnerabilities with high accuracy, along with explainability components that highlight risky code sections.

## 🚀 Features

- **Multi-class Vulnerability Detection**: Detects various vulnerability types including:
  - Overflow/Underflow
  - Re-entrancy
  - Timestamp Dependency
  - Transaction Order Dependence (TOD)
  - tx.origin usage
  - Unchecked Send
  - Unhandled Exceptions

- **Explainable AI**: Attention visualization showing which code tokens the model considers risky

- **Benchmark Comparisons**: Performance comparison against traditional static analysis tools (Slither, Mythril)

- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrices

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/smart-contract-vuln-detector.git
cd smart-contract-vuln-detector
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install additional tools (optional):**
```bash
# For Slither analysis
pip install slither-analyzer

# For Mythril analysis
pip install mythril
```

## 📊 Dataset Structure

```
dataset/
├── buggy_contracts/
│   ├── Overflow-Underflow/
│   ├── Re-entrancy/
│   ├── Timestamp-Dependency/
│   ├── TOD/
│   ├── tx.origin/
│   ├── Unchecked-Send/
│   └── Unhandled-Exceptions/
└── results/
    ├── Slither/
    ├── Mythril/
    └── ...
```

## 🔧 Usage

### Running the Complete Pipeline

Execute the notebooks in order:

1. **Setup and Environment** (`00_setup_and_env.ipynb`)
   - Install dependencies and verify GPU setup
   - Set random seeds for reproducibility

2. **Data Acquisition** (`01_data_acquisition_and_overview.ipynb`)
   - Load and explore the vulnerability datasets
   - Generate data statistics and samples

3. **Preprocessing** (`02_preprocessing_and_function_extraction.ipynb`)
   - Extract Solidity functions from contracts
   - Clean and normalize code
   - Create structured datasets

4. **Tokenization** (`03_tokenization_and_dataset.ipynb`)
   - Tokenize code using CodeBERT tokenizer
   - Create PyTorch datasets and data loaders
   - Prepare train/validation/test splits

5. **Model Training** (`04_train_codebert_baseline.ipynb`)
   - Fine-tune CodeBERT for vulnerability detection
   - Implement training loop with validation
   - Save best model checkpoints

6. **Evaluation** (`05_evaluation_and_metrics.ipynb`)
   - Comprehensive model evaluation
   - Generate confusion matrices and classification reports
   - Save prediction results

7. **Attention Visualization** (`06_attention_visualization.ipynb`)
   - Extract attention weights from trained model
   - Create interactive visualizations
   - Map attention to source code lines

8. **Benchmarking** (`07_benchmark_vs_slither_mythril.ipynb`)
   - Compare performance against Slither and Mythril
   - Generate benchmark comparison reports

### Quick Start (Development)

For quick testing and development, set `use_subset=True` in the training configuration to use a smaller dataset subset.

## 📈 Results

The trained model achieves:
- **Accuracy**: >85% on test set
- **F1-Score**: >0.80 macro average
- **Precision**: High precision across vulnerability types
- **Recall**: Balanced recall for different vulnerability classes

## 🔍 Model Architecture

- **Base Model**: Microsoft CodeBERT (codebert-base)
- **Architecture**: Transformer encoder with classification head
- **Input**: Solidity function source code (max 512 tokens)
- **Output**: Multi-class vulnerability predictions

## 📁 Project Structure

```
├── notebooks/           # Jupyter notebooks for the ML pipeline
├── data/               # Processed datasets and configurations
├── dataset/            # Raw vulnerability datasets
├── models/             # Trained model checkpoints
├── results/            # Evaluation results and visualizations
├── logs/               # Training and execution logs
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## ⚙️ Configuration

Key configuration files:
- `data/processed/dataset_config.json`: Dataset and tokenization settings
- Training hyperparameters are defined in each training notebook

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in configuration
2. **Missing data files**: Ensure dataset is properly downloaded
3. **Import errors**: Verify all requirements are installed

### GPU Setup:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{smart-contract-vuln-detector,
  title={Smart Contract Vulnerability Detection with CodeBERT},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/smart-contract-vuln-detector}
}
```


## 🙏 Acknowledgments

- Microsoft for CodeBERT model
- Vulnerability dataset providers (SolidiFI, SmartBugs-Wild)
- Hugging Face for transformers library
- PyTorch team for the deep learning framework
