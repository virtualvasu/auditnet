// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_48 {
    // Common state variables
    mapping(address => uint256) public balances_intou30;
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public tokens;
    mapping(string => address) public btc;
    mapping(string => address) public eth;
    mapping(address => uint256) public lockTime_intou21;
    address public owner;
    address public feeAccount;
    uint256 public totalSupply;
    bool public paused = false;

    // Events that might be referenced
    event OwnerWithdrawTradingFee(address indexed owner, uint256 amount);
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Constructor
    constructor() public {
        owner = msg.sender;
        feeAccount = msg.sender;
    }

    // Common modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    // Helper functions that might be referenced
    function availableTradingFeeOwner() public view returns (uint256) {
        return tokens[address(0)][feeAccount];
    }

    function bug_txorigin36( address owner_txorigin36) public{
    require(tx.origin == owner_txorigin36);
    }

    function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
    require(tx.origin == owner_txorigin14);
    _recipient.transfer(address(this).balance);
    }

    function balanceOf(address tokenOwner) public view returns (uint balance) {
    return balances[tokenOwner];
    }

    function div(uint a, uint b) internal pure returns (uint c) {
    require(b > 0);
    c = a / b;
    }

    function approveAndCall(address spender, uint tokens, bytes memory data) public returns (bool success) {
    allowed[msg.sender][spender] = tokens;
    emit Approval(msg.sender, spender, tokens);
    ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
    return true;
    }

    function claimReward_TOD32(uint256 submission) public {
    require (!claimed_TOD32);
    require(submission < 10);
    msg.sender.transfer(reward_TOD32);
    claimed_TOD32 = true;
    }

    function bug_unchk_send8() payable public{
    msg.sender.transfer(1 ether);}

    function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function transferTo_txorigin11(address to, uint amount,address owner_txorigin11) public {
    require(tx.origin == owner_txorigin11);
    to.call.value(amount);
    }

    function claimReward_TOD22(uint256 submission) public {
    require (!claimed_TOD22);
    require(submission < 10);
    msg.sender.transfer(reward_TOD22);
    claimed_TOD22 = true;
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
    allowed[msg.sender][_spender] = 0;
    } else {
    allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
    allowed[msg.sender][_spender] = 0;
    } else {
    allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function sendto_txorigin1(address payable receiver, uint amount,address owner_txorigin1) public {
    require (tx.origin == owner_txorigin1);
    receiver.transfer(amount);
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {
    return allowed[tokenOwner][spender];
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {
    return allowed[tokenOwner][spender];
    }

    function withdrawFunds_re_ent38 (uint256 _weiToWithdraw) public {
    require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
    require(msg.sender.send(_weiToWithdraw));
    balances_re_ent38[msg.sender] -= _weiToWithdraw;
    }

    function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
    require(tx.origin == owner_txorigin23);
    to.call.value(amount);
    }

    function acceptOwnership() public {
    require(msg.sender == newOwner);
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
    newOwner = address(0);
    }

    function increaseLockTime_intou13(uint _secondsToIncrease) public {
    lockTime_intou13[msg.sender] += _secondsToIncrease;
    }

    function play_tmstmp31(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp31 = msg.sender;}}

    function transferOwnership(address _newOwner) public onlyOwner {
    newOwner = _newOwner;
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
    allowed[msg.sender][_spender] = 0;
    } else {
    allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function transfer(address to, uint tokens) public returns (bool success);
    function bug_tmstmp20 () public payable {
    uint pastBlockTime_tmstmp20;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp20);
    pastBlockTime_tmstmp20 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function transfer(address to, uint tokens) public returns (bool success) {
    balances[msg.sender] = balances[msg.sender].sub(tokens);
    balances[to] = balances[to].add(tokens);
    emit Transfer(msg.sender, to, tokens);
    return true;
    }

    function approve(address spender, uint tokens) public returns (bool success);
    function sendto_txorigin25(address payable receiver, uint amount,address owner_txorigin25) public {
    require (tx.origin == owner_txorigin25);
    receiver.transfer(amount);
    }

    function approveAndCall(address spender, uint tokens, bytes memory data) public returns (bool success) {
    allowed[msg.sender][spender] = tokens;
    emit Approval(msg.sender, spender, tokens);
    ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
    return true;
    }

    function balanceOf(address tokenOwner) public view returns (uint balance);
    address winner_tmstmp19;
    function play_tmstmp19(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp19 = msg.sender;}}

    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
    balances[from] = balances[from].sub(tokens);
    allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
    balances[to] = balances[to].add(tokens);
    emit Transfer(from, to, tokens);
    return true;
    }

    function acceptOwnership() public {
    require(msg.sender == newOwner);
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
    newOwner = address(0);
    }

    function withdrawBalance_re_ent40() public{
    (bool success,)=msg.sender.call.value(userBalance_re_ent40[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent40[msg.sender] = 0;
    }

    function play_TOD7(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD7 = msg.sender;
    }
    }

    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
    balances[from] = balances[from].sub(tokens);
    allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
    balances[to] = balances[to].add(tokens);
    emit Transfer(from, to, tokens);
    return true;
    }

    function withdraw_ovrflow1() public {
    require(now > lockTime_intou1[msg.sender]);
    uint transferValue_intou1 = 10;
    msg.sender.transfer(transferValue_intou1);
    }

    function approveAndCall(address spender, uint tokens, bytes memory data) public returns (bool success) {
    allowed[msg.sender][spender] = tokens;
    emit Approval(msg.sender, spender, tokens);
    ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
    return true;
    }

    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {
    return allowed[tokenOwner][spender];
    }

    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
    function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
    require(tx.origin == owner_txorigin26);
    _recipient.transfer(address(this).balance);
    }

    function setReward_TOD38() public payable {
    require (!claimed_TOD38);
    require(msg.sender == owner_TOD38);
    owner_TOD38.transfer(reward_TOD38);
    reward_TOD38 = msg.value;
    }

    function play_tmstmp7(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp7 = msg.sender;}}

    function getReward_TOD25() payable public{
    winner_TOD25.transfer(msg.value);
    }

    function getReward_TOD35() payable public{
    winner_TOD35.transfer(msg.value);
    }

    function totalSupply() public view returns (uint) {
    return _totalSupply.sub(balances[address(0)]);
    }

    function claimReward_TOD12(uint256 submission) public {
    require (!claimed_TOD12);
    require(submission < 10);
    msg.sender.transfer(reward_TOD12);
    claimed_TOD12 = true;
    }

    function bug_intou7() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function play_tmstmp30(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp30 = msg.sender;}}

    function transfer(address to, uint tokens) public returns (bool success);
    function bug_unchk_send12() payable public{
    msg.sender.transfer(1 ether);}

    function claimReward_re_ent32() public {
    require(redeemableEther_re_ent32[msg.sender] > 0);
    uint transferValue_re_ent32 = redeemableEther_re_ent32[msg.sender];
    msg.sender.transfer(transferValue_re_ent32);
    redeemableEther_re_ent32[msg.sender] = 0;
    }

    function div(uint a, uint b) internal pure returns (uint c) {
    require(b > 0);
    c = a / b;
    }

    function transferOwnership(address _newOwner) public onlyOwner {
    newOwner = _newOwner;
    }

    function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
    require (tx.origin == owner_txorigin33);
    receiver.transfer(amount);
    }

    function play_tmstmp35(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp35 = msg.sender;}}

    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
    return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }

    function withdrawBal_unchk41 () public{
    uint64 Balances_unchk41 = 0;
    msg.sender.send(Balances_unchk41);}

    function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
    }

    function claimReward_re_ent11() public {
    require(redeemableEther_re_ent11[msg.sender] > 0);
    uint transferValue_re_ent11 = redeemableEther_re_ent11[msg.sender];
    msg.sender.transfer(transferValue_re_ent11);
    redeemableEther_re_ent11[msg.sender] = 0;
    }

    function setReward_TOD10() public payable {
    require (!claimed_TOD10);
    require(msg.sender == owner_TOD10);
    owner_TOD10.transfer(reward_TOD10);
    reward_TOD10 = msg.value;
    }

    function play_TOD33(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD33 = msg.sender;
    }
    }

    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
    return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }

    function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;
    }

    function sub(uint a, uint b) internal pure returns (uint c) {
    require(b <= a);
    c = a - b;
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
    allowed[msg.sender][_spender] = 0;
    } else {
    allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
    }

    function bug_tmstmp17() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining);
    function bug_unchk19() public{
    address payable addr_unchk19;
    if (!addr_unchk19.send (10 ether) || 1==1)
    {revert();}
    }

    function totalSupply() public view returns (uint);
    function bug_unchk_send21() payable public{
    msg.sender.transfer(1 ether);}

}