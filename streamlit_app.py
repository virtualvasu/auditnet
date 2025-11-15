import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'notebooks'))

# Set page config
st.set_page_config(
    page_title="Smart Contract Vulnerability Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model architecture (same as notebook 8)
class CodeBERTForVulnerabilityDetection(nn.Module):
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

class VulnerabilitySolutions:
    """Provides solutions and recommendations for detected vulnerabilities"""
    
    def __init__(self):
        self.solutions = {
            'reentrancy': {
                'title': '🔄 Reentrancy Attack',
                'description': 'External calls before state updates can allow malicious contracts to re-enter your function.',
                'solution': '''
**Solutions:**
1. **Checks-Effects-Interactions Pattern**: Update state before external calls
2. **Reentrancy Guard**: Use OpenZeppelin's ReentrancyGuard modifier
3. **Pull Payment Pattern**: Let users withdraw funds themselves

**Fixed Code Example:**
```solidity
bool private locked;
modifier nonReentrant() {
    require(!locked, "ReentrancyGuard: reentrant call");
    locked = true;
    _;
    locked = false;
}

function withdraw(uint256 amount) public nonReentrant {
    require(balances[msg.sender] >= amount, "Insufficient balance");
    
    // Effects: Update state BEFORE external call
    balances[msg.sender] -= amount;
    
    // Interactions: External call last
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}
```''',
                'severity': 'HIGH'
            },
            'unchecked_send': {
                'title': '❌ Unchecked External Call',
                'description': 'External calls (send, call) can fail silently if return values are not checked.',
                'solution': '''
**Solutions:**
1. **Check Return Values**: Always verify success of external calls
2. **Use transfer()**: For simple Ether transfers (throws on failure)
3. **Implement Fallback**: Handle failed transfers gracefully

**Fixed Code Example:**
```solidity
// Instead of: payable(owner).send(amount);
// Use:
require(payable(owner).send(amount), "Transfer failed");

// Or better:
(bool success, ) = payable(owner).call{value: amount}("");
require(success, "Transfer failed");

// Or safest:
payable(owner).transfer(amount); // Throws on failure
```''',
                'severity': 'HIGH'
            },
            'tx_origin': {
                'title': '🎭 tx.origin Vulnerability',
                'description': 'Using tx.origin for authorization can be exploited through phishing attacks.',
                'solution': '''
**Solutions:**
1. **Use msg.sender**: Always use msg.sender instead of tx.origin
2. **Multi-signature**: Implement multi-signature authorization
3. **Access Control**: Use OpenZeppelin's AccessControl

**Fixed Code Example:**
```solidity
// Instead of: require(tx.origin == owner, "Not authorized");
// Use:
require(msg.sender == owner, "Not authorized");

// Or better with OpenZeppelin:
import "@openzeppelin/contracts/access/Ownable.sol";
contract MyContract is Ownable {
    function restrictedFunction() public onlyOwner {
        // Only owner can call this
    }
}
```''',
                'severity': 'MEDIUM'
            },
            'overflow': {
                'title': '🔢 Integer Overflow/Underflow',
                'description': 'Arithmetic operations can overflow or underflow, leading to unexpected values.',
                'solution': '''
**Solutions:**
1. **Use Solidity ^0.8.0**: Built-in overflow/underflow protection
2. **SafeMath Library**: For older Solidity versions
3. **Explicit Checks**: Manual bounds checking

**Fixed Code Example:**
```solidity
// Solidity ^0.8.0 automatically reverts on overflow
function addToBalance(uint256 amount) public {
    // This will automatically revert on overflow
    balances[msg.sender] += amount;
}

// For older versions, use SafeMath:
using SafeMath for uint256;
function addToBalance(uint256 amount) public {
    balances[msg.sender] = balances[msg.sender].add(amount);
}
```''',
                'severity': 'HIGH'
            },
            'timestamp_dependency': {
                'title': '⏰ Timestamp Dependency',
                'description': 'Relying on block.timestamp can be manipulated by miners within a small range.',
                'solution': '''
**Solutions:**
1. **Use Block Numbers**: More reliable than timestamps
2. **Oracle Services**: External time sources for critical timing
3. **Tolerance Ranges**: Accept small time variations

**Fixed Code Example:**
```solidity
// Instead of exact timestamp checks:
// require(block.timestamp > deadline, "Too early");

// Use block numbers or ranges:
require(block.number > deadlineBlock, "Too early");

// Or allow tolerance:
require(block.timestamp > deadline - 300, "Too early"); // 5 min tolerance
```''',
                'severity': 'MEDIUM'
            },
            'unhandled_exception': {
                'title': '💥 Unhandled Exception',
                'description': 'Functions may fail silently without proper error handling.',
                'solution': '''
**Solutions:**
1. **Try-Catch Blocks**: Handle external call failures
2. **Return Booleans**: Check success of operations
3. **Event Logging**: Log failures for monitoring

**Fixed Code Example:**
```solidity
function safeExternalCall(address target, bytes memory data) public returns (bool success) {
    try target.call(data) returns (bytes memory) {
        return true;
    } catch {
        emit ExternalCallFailed(target);
        return false;
    }
}
```''',
                'severity': 'MEDIUM'
            }
        }
    
    def get_vulnerability_solution(self, function_code, prediction_type):
        """Analyze function code and provide specific solution"""
        code_lower = function_code.lower()
        
        # Detect specific vulnerability patterns
        if 'call{value:' in code_lower and 'balances[' in code_lower:
            if code_lower.find('balances[') > code_lower.find('call{value:'):
                return self.solutions['reentrancy']
        
        if '.send(' in code_lower and 'require(' not in code_lower:
            return self.solutions['unchecked_send']
        
        if 'tx.origin' in code_lower:
            return self.solutions['tx_origin']
        
        if '+=' in code_lower and 'balances[' in code_lower:
            return self.solutions['overflow']
        
        if 'block.timestamp' in code_lower or 'now' in code_lower:
            return self.solutions['timestamp_dependency']
        
        if 'call(' in code_lower and 'try' not in code_lower:
            return self.solutions['unhandled_exception']
        
        # Default solution for vulnerable functions
        if prediction_type == 'Vulnerable':
            return {
                'title': '⚠️ General Vulnerability Detected',
                'description': 'This function has been flagged as potentially vulnerable by the AI model.',
                'solution': '''
**General Security Recommendations:**
1. **Follow Security Best Practices**: Use established patterns and libraries
2. **External Audits**: Have your contract audited by security experts
3. **Testing**: Implement comprehensive unit and integration tests
4. **Formal Verification**: Use tools like Mythril, Slither, or Manticore
5. **Bug Bounty**: Consider running a bug bounty program

**Common Issues to Check:**
- Reentrancy attacks
- Integer overflow/underflow
- Unchecked external calls
- Access control issues
- Front-running vulnerabilities
''',
                'severity': 'MEDIUM'
            }
        
        return None

class SolidityFunctionParser:
    """Extract functions from Solidity contract code"""
    
    def __init__(self):
        self.function_pattern = re.compile(
            r'function\s+(\w+)\s*\([^)]*\)\s*(?:public|private|internal|external)?\s*(?:view|pure|payable)?\s*(?:returns\s*\([^)]*\))?\s*\{',
            re.MULTILINE | re.IGNORECASE
        )
        
    def extract_functions(self, contract_code):
        """Extract all functions from contract code"""
        functions = []
        matches = list(self.function_pattern.finditer(contract_code))
        
        for i, match in enumerate(matches):
            func_name = match.group(1)
            start_pos = match.start()
            
            # Find the end of the function by counting braces
            brace_count = 0
            func_start = match.end() - 1
            
            for j, char in enumerate(contract_code[func_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = func_start + j + 1
                        break
            else:
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
    
    def _clean_function_code(self, code):
        """Clean and normalize function code"""
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code.strip()

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

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find model files
    models_dir = Path('models')
    checkpoints_dir = Path('results/checkpoints')
    
    model_files = []
    if models_dir.exists():
        model_files.extend(list(models_dir.glob('*.pt')))
    if checkpoints_dir.exists():
        model_files.extend(list(checkpoints_dir.glob('best_model*.pt')))
    
    if not model_files:
        st.error("❌ No model checkpoints found! Please train a model first using notebook 04")
        return None, None, None, None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    try:
        checkpoint = torch.load(latest_model, map_location=device)
        model_name = checkpoint.get('model_name', 'microsoft/codebert-base')
        num_classes = checkpoint.get('num_classes', 1)
        task_type = checkpoint.get('task', 'binary')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = CodeBERTForVulnerabilityDetection(model_name, num_classes)
        model = load_model_with_checkpoint_compatibility(model, checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, task_type
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

def predict_function_vulnerability(function_code, model, tokenizer, device, task_type, max_length=512):
    """Predict vulnerability for a single function"""
    encoding = tokenizer(
        function_code,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        
        if task_type == 'binary':
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            vulnerability_prob = probabilities[0]
            safe_prob = 1 - vulnerability_prob
            
            prediction = 'Vulnerable' if vulnerability_prob > 0.5 else 'Safe'
            confidence = max(vulnerability_prob, safe_prob)
            
        else:
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            class_names = ['Overflow-Underflow', 'Re-entrancy', 'SAFE', 'TOD', 
                          'Timestamp-Dependency', 'Unchecked-Send', 'Unhandled-Exceptions', 'tx.origin']
            prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
            vulnerability_prob = 1 - probabilities[2] if predicted_class != 2 else 0
            safe_prob = probabilities[2] if len(probabilities) > 2 else 0
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'vulnerability_probability': float(vulnerability_prob),
        'safe_probability': float(safe_prob)
    }

def main():
    st.title("🔐 Smart Contract Vulnerability Analyzer")
    st.markdown("**AI-powered security analysis using CodeBERT**")
    
    # Load model
    with st.spinner('Loading AI model...'):
        model, tokenizer, device, task_type = load_model()
    
    if model is None:
        st.stop()
    
    st.success(f"✅ Model loaded successfully on {device}")
    
    # Sidebar
    st.sidebar.title("📋 Sample Contracts")
    
    sample_vulnerable = """pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy - external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function emergencyWithdraw() public {
        require(msg.sender == owner, "Only owner");
        
        // Unchecked send - vulnerable
        payable(owner).send(address(this).balance);
    }
    
    function addToBalance(uint256 amount) public {
        // Potential overflow vulnerability
        balances[msg.sender] += amount;
    }
    
    function transferOwnership(address newOwner) public {
        // tx.origin vulnerability
        require(tx.origin == owner, "Not authorized");
        owner = newOwner;
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}"""

    sample_safe = """pragma solidity ^0.8.0;

contract SafeBank {
    mapping(address => uint256) public balances;
    address public owner;
    bool private locked;
    
    modifier nonReentrant() {
        require(!locked, "ReentrancyGuard: reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public nonReentrant {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe: state update before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    function emergencyWithdraw() public {
        require(msg.sender == owner, "Only owner");
        
        // Safe: check return value
        require(payable(owner).send(address(this).balance), "Transfer failed");
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}"""

    if st.sidebar.button("📊 Load Vulnerable Sample"):
        st.session_state.contract_code = sample_vulnerable
        st.rerun()
    
    if st.sidebar.button("✅ Load Safe Sample"):
        st.session_state.contract_code = sample_safe
        st.rerun()
    
    # Main input
    st.subheader("📝 Contract Code")
    
    contract_code = st.text_area(
        "Paste your Solidity contract code:",
        value=st.session_state.get('contract_code', ''),
        height=300,
        placeholder="pragma solidity ^0.8.0;\n\ncontract MyContract {\n    // Your contract code here...\n}"
    )
    
    contract_name = st.text_input("Contract Name (optional):", value="MyContract")
    
    if st.button("🔍 Analyze Vulnerabilities", type="primary"):
        if not contract_code.strip():
            st.error("Please enter contract code to analyze")
            return
        
        with st.spinner('Analyzing contract for vulnerabilities...'):
            # Parse functions
            parser = SolidityFunctionParser()
            functions = parser.extract_functions(contract_code)
            
            if not functions:
                st.error("No functions found in the contract. Please check your Solidity code.")
                return
            
            # Analyze each function
            results = []
            progress_bar = st.progress(0)
            solutions_provider = VulnerabilitySolutions()
            
            for i, func in enumerate(functions):
                prediction = predict_function_vulnerability(
                    func['code'], model, tokenizer, device, task_type
                )
                
                # Get vulnerability solution if vulnerable
                solution = solutions_provider.get_vulnerability_solution(
                    func['code'], prediction['prediction']
                )
                
                result = {
                    'contract_name': contract_name,
                    'function_name': func['name'],
                    'function_code': func['code'],
                    'code_length': func['length'],
                    'start_line': func['start_line'],
                    'solution': solution,
                    **prediction
                }
                results.append(result)
                progress_bar.progress((i + 1) / len(functions))
        
        # Display results
        st.success(f"✅ Analysis complete! Found {len(functions)} functions")
        
        # Summary metrics
        vulnerable_count = len([r for r in results if r['prediction'] == 'Vulnerable'])
        avg_vuln_prob = np.mean([r['vulnerability_probability'] for r in results])
        
        # Risk level
        if vulnerable_count > len(results) * 0.5:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
        elif vulnerable_count > 0 or avg_vuln_prob > 0.3:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Functions", len(results))
        
        with col2:
            st.metric("Vulnerable Functions", vulnerable_count)
        
        with col3:
            st.metric("Avg Vulnerability", f"{avg_vuln_prob:.1%}")
        
        with col4:
            st.markdown(f"**Risk Level**")
            st.markdown(f"<h3 style='color: {risk_color}'>{risk_level}</h3>", unsafe_allow_html=True)
        
        # Vulnerability Summary
        vulnerable_results = [r for r in results if r['prediction'] == 'Vulnerable']
        if vulnerable_results:
            st.divider()
            st.subheader("⚠️ Security Issues Summary")
            
            # Group vulnerabilities by type
            vuln_types = {}
            for result in vulnerable_results:
                if result.get('solution'):
                    vuln_type = result['solution']['title']
                    if vuln_type not in vuln_types:
                        vuln_types[vuln_type] = {
                            'functions': [],
                            'solution': result['solution']
                        }
                    vuln_types[vuln_type]['functions'].append(result['function_name'])
            
            for vuln_type, data in vuln_types.items():
                with st.expander(f"🚨 {vuln_type} ({len(data['functions'])} functions affected)", expanded=False):
                    st.markdown(f"**Affected Functions:** {', '.join(data['functions'])}")
                    st.markdown(f"**Severity:** {data['solution']['severity']}")
                    st.markdown(f"**Description:** {data['solution']['description']}")
                    st.markdown("**How to Fix:**")
                    st.markdown(data['solution']['solution'])
        
        st.divider()
        
        # Function analysis results
        st.subheader("📋 Function Analysis Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_filter = st.selectbox(
                "Filter by:",
                ["All Functions", "Vulnerable Only", "Safe Only"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Vulnerability Probability", "Function Name", "Confidence"]
            )
        
        # Filter and sort results
        filtered_results = results.copy()
        
        if show_filter == "Vulnerable Only":
            filtered_results = [r for r in results if r['prediction'] == 'Vulnerable']
        elif show_filter == "Safe Only":
            filtered_results = [r for r in results if r['prediction'] == 'Safe']
        
        if sort_by == "Vulnerability Probability":
            filtered_results.sort(key=lambda x: x['vulnerability_probability'], reverse=True)
        elif sort_by == "Function Name":
            filtered_results.sort(key=lambda x: x['function_name'])
        elif sort_by == "Confidence":
            filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display functions
        for i, result in enumerate(filtered_results):
            with st.container():
                # Function header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### 📋 {result['function_name']}")
                
                with col2:
                    if result['prediction'] == 'Vulnerable':
                        st.markdown("🔴 **VULNERABLE**")
                    else:
                        st.markdown("🟢 **SAFE**")
                
                with col3:
                    vuln_prob = result['vulnerability_probability']
                    st.metric("Vuln. Prob.", f"{vuln_prob:.1%}")
                
                # Progress bar for vulnerability probability
                st.progress(vuln_prob, text=f"Vulnerability: {vuln_prob:.1%}")
                
                # Details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col2:
                    st.metric("Code Length", f"{result['code_length']} chars")
                with col3:
                    st.metric("Start Line", result['start_line'])
                
                # Function code
                with st.expander("📄 View Function Code"):
                    st.code(result['function_code'], language='solidity')
                
                # Vulnerability solution (if applicable)
                if result.get('solution') and result['prediction'] == 'Vulnerable':
                    solution = result['solution']
                    with st.expander(f"🛠️ **{solution['title']} - How to Fix**", expanded=True):
                        # Severity badge
                        severity_color = {
                            'HIGH': '🔴',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }
                        st.markdown(f"**Severity:** {severity_color.get(solution['severity'], '⚪')} {solution['severity']}")
                        
                        # Description
                        st.markdown(f"**Description:** {solution['description']}")
                        
                        # Solution
                        st.markdown(solution['solution'])
                
                st.divider()
        
        # Export options
        st.subheader("📊 Export Results")
        
        # Create DataFrame for export
        export_results = []
        for result in results:
            export_result = result.copy()
            # Add solution info to export
            if result.get('solution'):
                export_result['vulnerability_type'] = result['solution']['title']
                export_result['severity'] = result['solution']['severity']
                export_result['description'] = result['solution']['description']
            else:
                export_result['vulnerability_type'] = 'None'
                export_result['severity'] = 'None'
                export_result['description'] = 'No vulnerabilities detected'
            
            # Remove the solution object from export (too complex for CSV)
            export_result.pop('solution', None)
            export_results.append(export_result)
        
        df = pd.DataFrame(export_results)
        df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"vulnerability_report_{contract_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"vulnerability_report_{contract_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()