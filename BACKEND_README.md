# Smart Contract Vulnerability Detector - Backend API

FastAPI-based REST API for detecting vulnerabilities in Solidity smart contracts using advanced Deep Learning models.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server

```bash
# Make script executable (first time only)
chmod +x run_backend.sh

# Run the server
./run_backend.sh
```

Or directly with uvicorn:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Documentation (Swagger UI):** http://localhost:8000/docs
- **Alternative API Docs (ReDoc):** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## 📊 API Endpoints

### Phase 1 - Core Endpoints ⭐

#### 1. Health Check
```
GET /health
```
Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "operational",
  "models_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

#### 2. List Available Models
```
GET /models
```
Get information about all available trained models.

**Response:**
```json
{
  "models": [
    {
      "name": "Ensemble - Stacking (Best)",
      "type": "ensemble_stacking",
      "available": true,
      "performance": {
        "accuracy": 0.928,
        "f1_score": 0.537,
        "precision": 0.82,
        "recall": 0.72
      },
      "description": "Best ensemble model - Recommended"
    }
  ],
  "default_model": "auto",
  "total_models": 5
}
```

#### 3. List Vulnerability Types
```
GET /vulnerabilities
```
Get all supported vulnerability types with descriptions.

**Response:**
```json
{
  "total_types": 7,
  "vulnerabilities": [
    {
      "type": "reentrancy",
      "title": "Reentrancy Vulnerability",
      "description": "...",
      "severity": "HIGH"
    }
  ]
}
```

#### 4. Analyze Solidity Contract
```
POST /analyze
```
Analyze a complete Solidity contract for vulnerabilities.

**Request:**
```json
{
  "code": "pragma solidity ^0.8.0; contract Bank { ... }",
  "contract_name": "Bank",
  "model": "auto"
}
```

**Response:**
```json
{
  "status": "success",
  "contract_name": "Bank",
  "timestamp": "2026-01-29T10:30:00",
  "functions_analyzed": 3,
  "vulnerabilities_found": 2,
  "vulnerable_functions": 1,
  "functions": [
    {
      "name": "withdraw",
      "code": "function withdraw(uint amount) public { ... }",
      "vulnerable": true,
      "risk_level": "HIGH",
      "confidence": 0.94,
      "vulnerabilities": [
        {
          "type": "reentrancy",
          "severity": "HIGH",
          "description": "Potential reentrancy vulnerability",
          "recommendation": "Use nonReentrant modifier..."
        }
      ],
      "start_line": 5,
      "code_length": 150
    }
  ],
  "overall_risk": "HIGH",
  "average_confidence": 0.94,
  "model_used": "ensemble_stacking"
}
```

#### 5. Analyze Single Function
```
POST /analyze-function
```
Analyze a single Solidity function.

**Request:**
```json
{
  "code": "function withdraw(uint amount) public { ... }",
  "model": "auto"
}
```

**Response:**
```json
{
  "status": "success",
  "function_code": "...",
  "prediction": "Vulnerable",
  "confidence": 0.94,
  "vulnerability_probability": 0.94,
  "vulnerability_type": "reentrancy",
  "recommendations": "Use nonReentrant modifier...",
  "model_used": "ensemble_stacking",
  "timestamp": "2026-01-29T10:30:00"
}
```

#### 6. Get API Info
```
GET /info
```
Get general API information and configuration.

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# First, start the server in one terminal
./run_backend.sh

# In another terminal, run tests
python test_backend.py
```

The test script will verify:
- ✅ Health check
- ✅ API info endpoint
- ✅ Models listing
- ✅ Vulnerabilities listing
- ✅ Single function analysis
- ✅ Simple contract analysis
- ✅ Vulnerable contract detection

---

## 📁 Project Structure

```
backend/
├── __init__.py              # Package init
├── app.py                   # Main FastAPI application
├── models_loader.py         # Model loading utilities
├── solidity_parser.py       # Solidity code parsing
└── schemas.py               # Pydantic request/response models

run_backend.sh              # Backend startup script
test_backend.py             # Test suite
requirements.txt            # Dependencies
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Configuration
DEFAULT_MODEL=auto
DEVICE=auto  # auto, cuda, cpu

# API Settings
API_TITLE=Smart Contract Vulnerability Detector
API_DESCRIPTION=AI-powered security analysis for Solidity contracts
API_VERSION=1.0.0

# Logging
LOG_LEVEL=INFO
```

---

## 📊 Supported Vulnerability Types

| Type | Severity | Description |
|------|----------|-------------|
| **Reentrancy** | HIGH | Recursive external calls before state update |
| **Overflow/Underflow** | HIGH | Integer overflow or underflow in arithmetic |
| **Unchecked Send** | HIGH | External call return value not checked |
| **Timestamp Dependency** | MEDIUM | Reliance on manipulable block.timestamp |
| **tx.origin Abuse** | MEDIUM | Authorization using tx.origin instead of msg.sender |
| **Unhandled Exception** | MEDIUM | External call without proper error handling |
| **Transaction Order Dependence** | MEDIUM | Vulnerability to transaction ordering attacks |

---

## 🎯 Model Performance

The backend uses trained Deep Learning models:

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Ensemble - Stacking** ⭐ | 92.8% | 53.7% | 82% | 72% |
| CodeBERT | 96.2% | 61.8% | 78% | 55% |
| LSTM | 94.9% | 45.7% | 65% | 35% |
| CNN | 94.1% | 42.0% | 60% | 30% |

**Note:** Binary classification (Vulnerable vs Safe). Vulnerability type detected using pattern matching.

---

## 📝 Example Usage

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Analyze a contract
contract_code = """
pragma solidity ^0.8.0;
contract Bank {
    function withdraw(uint amount) public {
        msg.sender.send(amount);
    }
}
"""

response = requests.post(
    f"{BASE_URL}/analyze",
    json={
        "code": contract_code,
        "contract_name": "Bank"
    }
)

result = response.json()
print(f"Vulnerabilities found: {result['vulnerabilities_found']}")
print(f"Overall risk: {result['overall_risk']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "pragma solidity ^0.8.0; contract Test { }",
    "contract_name": "Test",
    "model": "auto"
  }'
```

### JavaScript/Frontend

```javascript
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: solidityCode,
    contract_name: 'MyContract',
    model: 'auto'
  })
});

const result = await response.json();
console.log(`Risk Level: ${result.overall_risk}`);
```

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Run with Docker

```bash
docker build -t vuln-detector:latest .
docker run -p 8000:8000 vuln-detector:latest
```

---

## 🐛 Troubleshooting

### Models not found
```
⚠️ Warning: No trained models found!
```
**Solution:** Train models using notebooks (04_train_codebert_baseline.ipynb, etc.)

### CUDA/GPU Issues
```python
# Force CPU mode in app.py
device = 'cpu'  # instead of auto-detection
```

### Port already in use
```bash
# Use different port
uvicorn backend.app:app --port 8001
```

### High memory usage
- Use a simpler model (CodeBERT instead of Ensemble)
- Reduce batch size
- Deploy with `--workers 1`

---

## 📚 API Documentation

Full interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🔐 Security Notes

⚠️ **Current Implementation:**
- No authentication/authorization
- CORS enabled for all origins
- Suitable for internal/development use

**For Production:**
1. Add API key authentication
2. Restrict CORS origins
3. Add rate limiting
4. Enable HTTPS
5. Add input validation/sanitization
6. Set resource limits

---

## 📈 Performance Tips

1. **Use ensemble models** for best accuracy (92.8%)
2. **Batch processing** for multiple contracts
3. **GPU deployment** for faster inference (5-10x speedup)
4. **Model caching** reduces load times
5. **Async processing** for large contracts

---

## 🤝 Contributing

To add new endpoints:

1. Define request/response schemas in `schemas.py`
2. Add endpoint function in `app.py`
3. Add tests in `test_backend.py`
4. Update this README

---

## 📄 License

MIT License - See LICENSE file

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review logs: `tail -f logs/api.log`
- Check API health: `curl http://localhost:8000/health`

---

**Built with ❤️ using FastAPI, PyTorch, and CodeBERT**
