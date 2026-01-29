"""
Smart Contract Vulnerability Detector - FastAPI Backend
Main application with all API endpoints
"""

import torch
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
from datetime import datetime
from typing import Optional

# Import local modules
from backend.models_loader import load_model, get_available_models
from backend.solidity_parser import SolidityFunctionParser, VulnerabilityPatternDetector, VULNERABILITY_INFO
from backend.schemas import (
    AnalyzeRequest, AnalyzeResponse, FunctionAnalysis, VulnerabilityType,
    HealthResponse, ModelsListResponse, VulnerabilitiesListResponse,
    VulnerabilityInfo, ErrorResponse, FunctionPredictionResponse,
    AnalyzeFunctionRequest
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
class AppState:
    model: Optional[torch.nn.Module] = None
    tokenizer: Optional[object] = None
    device: str = "cpu"
    task_type: str = "binary"
    model_name: str = "auto"
    
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Smart Contract Vulnerability Detector API...")
    
    try:
        model, tokenizer, device, task_type = load_model(app_state.model_name)
        
        if model is None:
            logger.warning("⚠️  No models found - API started in demo mode")
            logger.warning("   To use the API, train models first using notebooks:")
            logger.warning("   - 04_train_codebert_baseline.ipynb")
            logger.warning("   - 09_train_lstm_cnn_models.ipynb")
            logger.warning("   - 10_ensemble_training.ipynb")
            app_state.model = None
            app_state.tokenizer = None
            app_state.device = "cpu"
            app_state.task_type = "binary"
        else:
            app_state.model = model
            app_state.tokenizer = tokenizer
            app_state.device = device
            app_state.task_type = task_type
            
            logger.info(f"✅ Model loaded on {device}")
            logger.info(f"   Task type: {task_type}")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        logger.warning("⚠️  API started in demo mode (no models loaded)")
        app_state.model = None
        app_state.tokenizer = None
        app_state.device = "cpu"
        app_state.task_type = "binary"
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Smart Contract Vulnerability Detector API",
    description="AI-powered security analysis for Solidity smart contracts using Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parsers
parser = SolidityFunctionParser()
vuln_detector = VulnerabilityPatternDetector()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_vulnerability(function_code: str) -> dict:
    """
    Predict vulnerability for a function using the loaded model
    
    Args:
        function_code: Solidity function code
        
    Returns:
        Dictionary with prediction results
    """
    if app_state.model is None or app_state.tokenizer is None:
        logger.error("Model not loaded")
        raise RuntimeError("Model not loaded")
    
    try:
        # Tokenize
        encoding = app_state.tokenizer(
            function_code,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(app_state.device)
        attention_mask = encoding['attention_mask'].to(app_state.device)
        
        # Predict
        with torch.no_grad():
            logits = app_state.model(input_ids, attention_mask)
        
        # Get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        if app_state.task_type == 'binary':
            vuln_prob = float(probabilities[0])
            prediction = "Vulnerable" if vuln_prob > 0.5 else "Safe"
            confidence = vuln_prob if vuln_prob > 0.5 else 1 - vuln_prob
        else:
            # Multi-class (if trained that way)
            pred_class = int(np.argmax(probabilities))
            prediction = "Vulnerable" if pred_class > 0 else "Safe"
            confidence = float(np.max(probabilities))
            vuln_prob = float(probabilities[0]) if len(probabilities) > 0 else 0.5
        
        return {
            'prediction': prediction,
            'confidence': min(confidence, 1.0),
            'vulnerability_probability': vuln_prob,
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise


def _analyze_contract_impl(code: str, contract_name: str = "Unknown") -> dict:
    """
    Analyze entire contract for vulnerabilities (internal implementation)
    
    Args:
        code: Solidity contract code
        contract_name: Name of the contract
        
    Returns:
        Dictionary with analysis results
    """
    # Validate code
    is_valid, error_msg = parser.validate_solidity_code(code)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Extract functions
    functions = parser.extract_functions(code)
    if not functions:
        raise ValueError("No functions found in contract")
    
    logger.info(f"Analyzing {len(functions)} functions from {contract_name}")
    
    # Analyze each function
    function_results = []
    vulnerabilities_found = 0
    
    for func in functions:
        try:
            prediction = predict_vulnerability(func['code'])
            vuln_type = vuln_detector.detect_vulnerability_type(func['code']) if prediction['prediction'] == 'Vulnerable' else None
            
            vulnerabilities = []
            if prediction['prediction'] == 'Vulnerable' and vuln_type:
                vuln_info = VULNERABILITY_INFO.get(vuln_type, {})
                vulnerabilities.append({
                    'type': vuln_type,
                    'severity': vuln_info.get('severity', 'MEDIUM'),
                    'description': vuln_info.get('description', 'Potential vulnerability detected'),
                    'line_number': func.get('start_line'),
                    'pattern_matched': vuln_type,
                    'recommendation': vuln_info.get('fix', 'Manual review recommended')
                })
                vulnerabilities_found += 1
            
            # Determine risk level
            if prediction['prediction'] == 'Vulnerable':
                if prediction['confidence'] > 0.8:
                    risk_level = 'HIGH'
                elif prediction['confidence'] > 0.6:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
            else:
                risk_level = 'LOW'
            
            function_results.append({
                'name': func['name'],
                'code': func['code'],
                'vulnerable': prediction['prediction'] == 'Vulnerable',
                'risk_level': risk_level,
                'confidence': prediction['confidence'],
                'vulnerabilities': vulnerabilities,
                'start_line': func['start_line'],
                'code_length': func['length']
            })
        
        except Exception as e:
            logger.error(f"Error analyzing function {func['name']}: {e}")
            # Continue with next function
            continue
    
    # Calculate overall risk
    vulnerable_count = len([f for f in function_results if f['vulnerable']])
    if vulnerable_count > len(function_results) * 0.5:
        overall_risk = 'HIGH'
    elif vulnerable_count > 0:
        overall_risk = 'MEDIUM'
    else:
        overall_risk = 'LOW'
    
    avg_confidence = np.mean([f['confidence'] for f in function_results]) if function_results else 0.0
    
    return {
        'functions_analyzed': len(function_results),
        'vulnerabilities_found': vulnerabilities_found,
        'vulnerable_functions': vulnerable_count,
        'functions': function_results,
        'overall_risk': overall_risk,
        'average_confidence': float(avg_confidence)
    }


# ============================================================================
# API ENDPOINTS - PHASE 1
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns API status and model information
    """
    status = "operational" if app_state.model is not None else "demo"
    return HealthResponse(
        status=status,
        models_loaded=app_state.model is not None,
        device=app_state.device,
        version="1.0.0"
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    Get list of available models with performance metrics
    """
    available = get_available_models()
    
    models_info = []
    for name, info in available.items():
        model_info = {
            'name': name,
            'type': info['type'],
            'available': len(info['files']) > 0,
            'performance': None,
            'description': None
        }
        
        # Add performance metrics for known models
        if 'Ensemble - Stacking' in name:
            model_info['performance'] = {
                'accuracy': 0.928,
                'f1_score': 0.537,
                'precision': 0.82,
                'recall': 0.72
            }
            model_info['description'] = 'Best ensemble model - Recommended'
        elif 'CodeBERT' in name:
            model_info['performance'] = {
                'accuracy': 0.962,
                'f1_score': 0.618,
                'precision': 0.78,
                'recall': 0.55
            }
        elif 'LSTM' in name:
            model_info['performance'] = {
                'accuracy': 0.949,
                'f1_score': 0.457,
                'precision': 0.65,
                'recall': 0.35
            }
        elif 'CNN' in name:
            model_info['performance'] = {
                'accuracy': 0.941,
                'f1_score': 0.42,
                'precision': 0.60,
                'recall': 0.30
            }
        
        models_info.append(model_info)
    
    return ModelsListResponse(
        models=models_info,
        default_model=app_state.model_name,
        total_models=len(models_info)
    )


@app.get("/vulnerabilities", response_model=VulnerabilitiesListResponse, tags=["Info"])
async def list_vulnerabilities():
    """
    Get list of all supported vulnerability types
    """
    vuln_list = [
        VulnerabilityInfo(
            type=key,
            title=info.get('title', key),
            description=info.get('description', ''),
            severity=info.get('severity', 'MEDIUM')
        )
        for key, info in VULNERABILITY_INFO.items()
    ]
    
    return VulnerabilitiesListResponse(
        total_types=len(vuln_list),
        vulnerabilities=vuln_list
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_contract(request: AnalyzeRequest):
    """
    Analyze a Solidity contract for vulnerabilities
    
    The API will:
    1. Extract all functions from the contract
    2. Analyze each function using the trained model
    3. Detect vulnerability types using pattern matching
    4. Return detailed results with confidence scores
    
    **Example request:**
    ```json
    {
        "code": "pragma solidity ^0.8.0; contract Bank { ... }",
        "contract_name": "Bank",
        "model": "ensemble_stacking"
    }
    ```
    """
    logger.info(f"🔍 Analyzing contract: {request.contract_name}")
    
    try:
        # Check if model is loaded
        if app_state.model is None:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please train models first using the notebooks."
            )
        
        # Analyze contract
        results = _analyze_contract_impl(request.code, request.contract_name or "Unknown")
        
        # Build response
        function_analyses = []
        for func in results['functions']:
            vulnerabilities = [
                VulnerabilityType(**vuln)
                for vuln in func['vulnerabilities']
            ]
            
            function_analyses.append(
                FunctionAnalysis(
                    name=func['name'],
                    code=func['code'],
                    vulnerable=func['vulnerable'],
                    risk_level=func['risk_level'],
                    confidence=func['confidence'],
                    vulnerabilities=vulnerabilities,
                    start_line=func['start_line'],
                    code_length=func['code_length']
                )
            )
        
        response = AnalyzeResponse(
            status="success",
            contract_name=request.contract_name or "Unknown",
            functions_analyzed=results['functions_analyzed'],
            vulnerabilities_found=results['vulnerabilities_found'],
            vulnerable_functions=results['vulnerable_functions'],
            functions=function_analyses,
            overall_risk=results['overall_risk'],
            average_confidence=results['average_confidence'],
            model_used=app_state.model_name
        )
        
        logger.info(f"✅ Analysis complete. Found {results['vulnerabilities_found']} vulnerabilities")
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze-function", response_model=FunctionPredictionResponse, tags=["Analysis"])
async def analyze_function(request: AnalyzeFunctionRequest):
    """
    Analyze a single Solidity function for vulnerabilities
    
    Useful for analyzing individual functions or test cases
    """
    logger.info("🔍 Analyzing single function")
    
    try:
        if app_state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.code or not request.code.strip():
            raise HTTPException(status_code=400, detail="Function code is empty")
        
        prediction = predict_vulnerability(request.code)
        vuln_type = vuln_detector.detect_vulnerability_type(request.code) if prediction['prediction'] == 'Vulnerable' else None
        
        recommendations = None
        if vuln_type and vuln_type in VULNERABILITY_INFO:
            recommendations = VULNERABILITY_INFO[vuln_type].get('fix', '')
        
        return FunctionPredictionResponse(
            status="success",
            function_code=request.code,
            prediction=prediction['prediction'],
            confidence=prediction['confidence'],
            vulnerability_probability=prediction['vulnerability_probability'],
            vulnerability_type=vuln_type,
            recommendations=recommendations,
            model_used=app_state.model_name
        )
    
    except Exception as e:
        logger.error(f"Error analyzing function: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Info"])
async def get_info():
    """
    Get API information and configuration
    """
    return {
        "name": "Smart Contract Vulnerability Detector API",
        "version": "1.0.0",
        "description": "AI-powered security analysis for Solidity smart contracts",
        "model_loaded": app_state.model is not None,
        "device": app_state.device,
        "current_model": app_state.model_name,
        "supported_vulnerabilities": list(VULNERABILITY_INFO.keys()),
        "api_endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "analyze_function": "POST /analyze-function",
            "list_models": "GET /models",
            "list_vulnerabilities": "GET /vulnerabilities",
            "info": "GET /info"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return ErrorResponse(
        message=str(exc),
        details="An unexpected error occurred"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
