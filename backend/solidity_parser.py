"""
Solidity Code Parser and Analyzer
Extracts functions and analyzes vulnerabilities in Solidity contracts
"""

import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SolidityFunctionParser:
    """Extract and parse functions from Solidity contract code"""
    
    def __init__(self):
        self.function_pattern = re.compile(
            r'function\s+(\w+)\s*\([^)]*\)\s*(?:public|private|internal|external)?\s*(?:view|pure|payable)?\s*(?:returns\s*\([^)]*\))?\s*\{',
            re.MULTILINE | re.IGNORECASE
        )
    
    def extract_functions(self, contract_code: str) -> List[Dict]:
        """
        Extract all functions from contract code
        
        Args:
            contract_code: Solidity contract source code
            
        Returns:
            List of function dictionaries with name, code, start_line, length
        """
        functions = []
        matches = list(self.function_pattern.finditer(contract_code))
        
        for i, match in enumerate(matches):
            func_name = match.group(1)
            start_pos = match.start()
            
            # Find the end of the function by counting braces
            brace_count = 0
            func_start = match.end() - 1
            func_end = None
            
            for j, char in enumerate(contract_code[func_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = func_start + j + 1
                        break
            
            if func_end is None:
                logger.warning(f"Could not find closing brace for function {func_name}")
                continue
            
            func_code = contract_code[start_pos:func_end].strip()
            func_code = self._clean_function_code(func_code)
            
            functions.append({
                'name': func_name,
                'code': func_code,
                'start_line': contract_code[:start_pos].count('\n') + 1,
                'length': len(func_code)
            })
        
        return functions
    
    def _clean_function_code(self, code: str) -> str:
        """Clean and normalize function code"""
        # Remove multiple newlines
        code = re.sub(r'\n\s*\n', '\n', code)
        # Normalize whitespace (but preserve some structure)
        code = re.sub(r'[ \t]+', ' ', code)
        # Remove single-line comments
        code = re.sub(r'//.*?\n', '\n', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code.strip()
    
    def validate_solidity_code(self, code: str) -> tuple[bool, str]:
        """
        Basic validation of Solidity code
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        if code.count('{') != code.count('}'):
            return False, "Mismatched braces"
        
        if 'function' not in code.lower():
            return False, "No function definitions found"
        
        return True, ""


class VulnerabilityPatternDetector:
    """Detect vulnerability types using pattern matching"""
    
    def __init__(self):
        self.patterns = {
            'reentrancy': [
                r'call\.value\s*\(',
                r'call\s*\{\s*value\s*:',
                r'\.send\s*\(',
                r'balances?\s*\[',
            ],
            'overflow_underflow': [
                r'\+=\s*(?!0[xX])',
                r'-=\s*(?!0[xX])',
                r'\*=\s*',
                r'balances?\[',
                r'uint(?!8)(?!16)',
            ],
            'unchecked_send': [
                r'\.send\s*\(',
                r'\.call\s*\(',
                r'transfer\s*\(',
                r'(?<!require\s*\()',
            ],
            'timestamp_dependency': [
                r'block\.timestamp',
                r'\bnow\b',
                r'block\.number',
            ],
            'tx_origin': [
                r'tx\.origin',
            ],
            'unhandled_exceptions': [
                r'\.call\s*\(',
                r'\.delegatecall\s*\(',
                r'\.staticcall\s*\(',
                r'(?<!try\s)',
            ],
            'tod': [  # Transaction Order Dependence
                r'block\.timestamp',
                r'block\.number',
                r'tx\.gasprice',
            ],
        }
    
    def detect_vulnerability_type(self, function_code: str) -> Optional[str]:
        """
        Detect vulnerability type using pattern matching
        
        Args:
            function_code: Solidity function code
            
        Returns:
            Detected vulnerability type or None
        """
        code_lower = function_code.lower()
        
        # Check for specific patterns in order of priority
        if self._has_pattern(code_lower, 'reentrancy'):
            if 'call{value:' in code_lower and 'balances[' in code_lower:
                return 'reentrancy'
        
        if self._has_pattern(code_lower, 'unchecked_send'):
            if '.send(' in code_lower and 'require(' not in code_lower:
                return 'unchecked_send'
        
        if self._has_pattern(code_lower, 'tx_origin'):
            return 'tx_origin'
        
        if self._has_pattern(code_lower, 'overflow_underflow'):
            if '+=' in code_lower or '-=' in code_lower:
                return 'overflow_underflow'
        
        if self._has_pattern(code_lower, 'timestamp_dependency'):
            if 'block.timestamp' in code_lower or 'now' in code_lower:
                return 'timestamp_dependency'
        
        if self._has_pattern(code_lower, 'unhandled_exceptions'):
            if '.call(' in code_lower and 'try' not in code_lower:
                return 'unhandled_exceptions'
        
        if self._has_pattern(code_lower, 'tod'):
            return 'tod'
        
        return None
    
    def _has_pattern(self, code: str, vuln_type: str) -> bool:
        """Check if code contains patterns for a vulnerability type"""
        if vuln_type not in self.patterns:
            return False
        
        for pattern in self.patterns[vuln_type]:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        
        return False


VULNERABILITY_INFO = {
    'reentrancy': {
        'title': 'Reentrancy Vulnerability',
        'description': 'Potential reentrancy vulnerability detected. External call before state update.',
        'severity': 'HIGH',
        'fix': 'Use nonReentrant modifier or implement Check-Effects-Interactions pattern'
    },
    'overflow_underflow': {
        'title': 'Integer Overflow/Underflow',
        'description': 'Potential integer overflow or underflow in arithmetic operations.',
        'severity': 'HIGH',
        'fix': 'Use Solidity ^0.8.0 or SafeMath library for arithmetic operations'
    },
    'unchecked_send': {
        'title': 'Unchecked External Call',
        'description': 'Return value of send/call not checked. May fail silently.',
        'severity': 'HIGH',
        'fix': 'Check return value of send/call or use transfer (which throws on failure)'
    },
    'timestamp_dependency': {
        'title': 'Timestamp Dependency',
        'description': 'Code relies on block.timestamp which can be manipulated by miners.',
        'severity': 'MEDIUM',
        'fix': 'Use block.number for time-sensitive logic or accept small time variations'
    },
    'tx_origin': {
        'title': 'tx.origin Usage',
        'description': 'Use of tx.origin for authorization can be exploited via phishing.',
        'severity': 'MEDIUM',
        'fix': 'Use msg.sender instead of tx.origin for authorization checks'
    },
    'unhandled_exceptions': {
        'title': 'Unhandled Exception',
        'description': 'External call without try-catch or proper error handling.',
        'severity': 'MEDIUM',
        'fix': 'Implement try-catch blocks or check return values for external calls'
    },
    'tod': {
        'title': 'Transaction Order Dependence',
        'description': 'Code may be vulnerable to transaction ordering attacks.',
        'severity': 'MEDIUM',
        'fix': 'Minimize dependencies on transaction order or use commit-reveal schemes'
    },
}
