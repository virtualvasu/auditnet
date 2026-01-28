# 🚀 Quick Start - Backend API

## 3-Step Setup

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start the Backend
```bash
chmod +x run_backend.sh
./run_backend.sh
```

### 3️⃣ Access the API
```
Open: http://localhost:8000/docs
```

---

## 🧪 Test the API

```bash
# In another terminal
python test_backend.py
```

---

## 📝 Basic Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "code": "pragma solidity ^0.8.0; contract Test { }",
        "contract_name": "Test"
    }
)

print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "pragma solidity ^0.8.0; contract Test { }",
    "contract_name": "Test"
  }'
```

### JavaScript
```javascript
const res = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    code: solidityCode,
    contract_name: "Test"
  })
});
const data = await res.json();
```

---

## 🔗 Main Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if API is running |
| `/models` | GET | List available models |
| `/vulnerabilities` | GET | List vulnerability types |
| `/analyze` | POST | Analyze Solidity contract |
| `/analyze-function` | POST | Analyze single function |
| `/info` | GET | API information |

---

## 📊 Response Example

```json
{
  "status": "success",
  "contract_name": "Bank",
  "functions_analyzed": 2,
  "vulnerabilities_found": 1,
  "overall_risk": "HIGH",
  "functions": [
    {
      "name": "withdraw",
      "vulnerable": true,
      "confidence": 0.94,
      "vulnerabilities": [
        {
          "type": "reentrancy",
          "severity": "HIGH",
          "description": "Potential reentrancy vulnerability"
        }
      ]
    }
  ]
}
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `./run_backend.sh` then use different port |
| Models not found | Train models using notebooks first |
| Import errors | `pip install -r requirements.txt` |
| CUDA errors | Models auto-fallback to CPU |

---

## 📚 Full Documentation

See **BACKEND_README.md** for complete API documentation.

---

**✨ API is production-ready! Start now:** `./run_backend.sh`
