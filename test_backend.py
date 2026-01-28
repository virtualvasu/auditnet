"""
Test script to verify backend API functionality
Run: python test_backend.py
"""

import sys
import time
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# Test Solidity contracts
SIMPLE_CONTRACT = """
pragma solidity ^0.8.0;

contract TestBank {
    mapping(address => uint) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
    }
}
"""

VULNERABLE_CONTRACT = """
pragma solidity ^0.8.0;

contract Vulnerable {
    mapping(address => uint) public balances;
    
    function withdrawVulnerable(uint amount) public {
        require(balances[msg.sender] >= amount);
        msg.sender.call{value: amount}("");
        balances[msg.sender] -= amount;
    }
    
    function txOriginVuln() public {
        require(tx.origin == msg.sender);
    }
}
"""


def wait_for_server(timeout=60):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    
    print("❌ Server did not start in time")
    return False


def test_health():
    """Test health endpoint"""
    print("\n🧪 Testing GET /health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data
        print(f"✅ PASS - Status: {data['status']}")
        print(f"   Models loaded: {data['models_loaded']}")
        print(f"   Device: {data['device']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_info():
    """Test info endpoint"""
    print("\n🧪 Testing GET /info")
    try:
        response = requests.get(f"{BASE_URL}/info", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS - API version: {data['version']}")
        print(f"   Device: {data['device']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_models():
    """Test models list endpoint"""
    print("\n🧪 Testing GET /models")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS - Found {data['total_models']} models")
        for model in data['models'][:3]:  # Show first 3
            print(f"   - {model['name']}: {model['type']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_vulnerabilities():
    """Test vulnerabilities list endpoint"""
    print("\n🧪 Testing GET /vulnerabilities")
    try:
        response = requests.get(f"{BASE_URL}/vulnerabilities", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS - Found {data['total_types']} vulnerability types")
        for vuln in data['vulnerabilities'][:3]:  # Show first 3
            print(f"   - {vuln['type']}: {vuln['severity']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_analyze_function():
    """Test analyze single function"""
    print("\n🧪 Testing POST /analyze-function")
    try:
        payload = {
            "code": "function test(uint amount) public { msg.sender.send(amount); }",
            "model": "auto"
        }
        response = requests.post(
            f"{BASE_URL}/analyze-function",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Confidence: {data['confidence']:.2f}")
        if data['vulnerability_type']:
            print(f"   Type: {data['vulnerability_type']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_analyze_contract():
    """Test analyze contract endpoint"""
    print("\n🧪 Testing POST /analyze (Simple Contract)")
    try:
        payload = {
            "code": SIMPLE_CONTRACT,
            "contract_name": "TestBank",
            "model": "auto"
        }
        response = requests.post(
            f"{BASE_URL}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS")
        print(f"   Functions analyzed: {data['functions_analyzed']}")
        print(f"   Vulnerabilities found: {data['vulnerabilities_found']}")
        print(f"   Overall risk: {data['overall_risk']}")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'N/A'}")
        return False


def test_analyze_vulnerable():
    """Test analyze vulnerable contract"""
    print("\n🧪 Testing POST /analyze (Vulnerable Contract)")
    try:
        payload = {
            "code": VULNERABLE_CONTRACT,
            "contract_name": "VulnerableContract",
            "model": "auto"
        }
        response = requests.post(
            f"{BASE_URL}/analyze",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        print(f"✅ PASS")
        print(f"   Functions analyzed: {data['functions_analyzed']}")
        print(f"   Vulnerabilities found: {data['vulnerabilities_found']}")
        print(f"   Overall risk: {data['overall_risk']}")
        if data['vulnerabilities_found'] > 0:
            print("   ⚠️  Vulnerabilities detected as expected")
        return True
    except Exception as e:
        print(f"❌ FAIL - {e}")
        print(f"   Response: {response.text if 'response' in locals() else 'N/A'}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Smart Contract Vulnerability Detector - Backend API Tests")
    print("=" * 60)
    
    # Wait for server
    print("\n⏳ Waiting for server to start...")
    if not wait_for_server():
        print("\n❌ Server is not running!")
        print("   Run: ./run_backend.sh")
        sys.exit(1)
    
    time.sleep(2)
    
    # Run tests
    tests = [
        test_health,
        test_info,
        test_models,
        test_vulnerabilities,
        test_analyze_function,
        test_analyze_contract,
        test_analyze_vulnerable,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
