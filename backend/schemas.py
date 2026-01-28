"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Request Schemas
class AnalyzeRequest(BaseModel):
    """Request model for contract analysis"""
    code: str = Field(..., description="Solidity contract source code")
    contract_name: Optional[str] = Field(None, description="Name of the contract")
    model: Optional[str] = Field("auto", description="Model to use for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "pragma solidity ^0.8.0; contract Test { function withdraw() public { } }",
                "contract_name": "Test",
                "model": "ensemble_stacking"
            }
        }


class AnalyzeFunctionRequest(BaseModel):
    """Request model for single function analysis"""
    code: str = Field(..., description="Solidity function code")
    model: Optional[str] = Field("auto", description="Model to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "function withdraw(uint amount) public { msg.sender.call{value: amount}(''); }",
                "model": "ensemble_stacking"
            }
        }


class AnalyzeBatchRequest(BaseModel):
    """Request model for batch contract analysis"""
    contracts: List[Dict[str, str]] = Field(..., description="List of contracts with code and name")
    model: Optional[str] = Field("auto", description="Model to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "contracts": [
                    {"code": "pragma solidity ^0.8.0; contract A { }", "name": "ContractA"},
                    {"code": "pragma solidity ^0.8.0; contract B { }", "name": "ContractB"}
                ],
                "model": "ensemble_stacking"
            }
        }


class AnalyzeWithDetailsRequest(BaseModel):
    """Request model for detailed analysis with explanations"""
    code: str = Field(..., description="Solidity contract source code")
    contract_name: Optional[str] = Field(None, description="Name of the contract")
    model: Optional[str] = Field("auto", description="Model to use")
    include_recommendations: bool = Field(True, description="Include fix recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "pragma solidity ^0.8.0; contract Test { }",
                "contract_name": "Test",
                "model": "ensemble_stacking",
                "include_recommendations": True
            }
        }


# Response Schemas
class VulnerabilityType(BaseModel):
    """Detected vulnerability information"""
    type: str = Field(..., description="Vulnerability type")
    severity: str = Field(..., description="Severity level (HIGH, MEDIUM, LOW)")
    description: str = Field(..., description="Description of the vulnerability")
    line_number: Optional[int] = Field(None, description="Line number where vulnerability detected")
    pattern_matched: Optional[str] = Field(None, description="Pattern that triggered detection")
    recommendation: Optional[str] = Field(None, description="Recommended fix")


class FunctionAnalysis(BaseModel):
    """Analysis result for a single function"""
    name: str = Field(..., description="Function name")
    code: str = Field(..., description="Function code")
    vulnerable: bool = Field(..., description="Whether function is vulnerable")
    risk_level: str = Field(..., description="Risk level (HIGH, MEDIUM, LOW)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    vulnerabilities: List[VulnerabilityType] = Field(default_factory=list)
    start_line: int = Field(..., description="Starting line number")
    code_length: int = Field(..., description="Length of function code")


class ModelInfo(BaseModel):
    """Information about a model"""
    name: str
    type: str
    available: bool
    performance: Optional[Dict[str, float]] = None
    description: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response model for contract analysis"""
    status: str = Field(..., description="Analysis status (success, error)")
    contract_name: str = Field(..., description="Contract name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    functions_analyzed: int = Field(..., description="Number of functions analyzed")
    vulnerabilities_found: int = Field(..., description="Number of vulnerabilities found")
    vulnerable_functions: int = Field(..., description="Number of vulnerable functions")
    
    functions: List[FunctionAnalysis] = Field(..., description="Analysis of each function")
    
    overall_risk: str = Field(..., description="Overall risk level (HIGH, MEDIUM, LOW)")
    average_confidence: float = Field(..., description="Average confidence across functions")
    
    model_used: str = Field(..., description="Model used for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "contract_name": "MyContract",
                "timestamp": "2026-01-29T10:30:00",
                "functions_analyzed": 2,
                "vulnerabilities_found": 1,
                "vulnerable_functions": 1,
                "functions": [
                    {
                        "name": "withdraw",
                        "code": "function withdraw(uint amount) public { ... }",
                        "vulnerable": True,
                        "risk_level": "HIGH",
                        "confidence": 0.94,
                        "vulnerabilities": [
                            {
                                "type": "reentrancy",
                                "severity": "HIGH",
                                "description": "Potential reentrancy vulnerability"
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
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models_loaded: bool
    device: str
    version: str


class ModelsListResponse(BaseModel):
    """Response with list of available models"""
    models: List[ModelInfo]
    default_model: str
    total_models: int


class VulnerabilityInfo(BaseModel):
    """Information about a vulnerability type"""
    type: str
    title: str
    description: str
    severity: str
    examples: Optional[List[str]] = None


class VulnerabilitiesListResponse(BaseModel):
    """Response with list of all supported vulnerabilities"""
    total_types: int
    vulnerabilities: List[VulnerabilityInfo]


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FunctionPredictionResponse(BaseModel):
    """Response for single function prediction"""
    status: str
    function_code: str
    prediction: str  # "Vulnerable" or "Safe"
    confidence: float
    vulnerability_probability: float
    vulnerability_type: Optional[str] = None
    recommendations: Optional[str] = None
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
