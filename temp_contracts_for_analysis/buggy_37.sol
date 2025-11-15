// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_37 {
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

    constructor() public {
    symbol = "AUC";
    name = "AugustCoin";
    decimals = 18;
    _totalSupply = 100000000000000000000000000;
    balances[0xe4948b8A5609c3c39E49eC1e36679a94F72D62bD] = _totalSupply;
    emit Transfer(address(0), 0xe4948b8A5609c3c39E49eC1e36679a94F72D62bD, _totalSupply);
    }

    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
    bool not_called_re_ent20 = true;
    function bug_re_ent20() public{
    require(not_called_re_ent20);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent20 = false;
    }

    function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
    require(tx.origin == owner_txorigin38);
    _recipient.transfer(address(this).balance);
    }

    function buyTicket_re_ent23() public{
    if (!(lastPlayer_re_ent23.send(jackpot_re_ent23)))
    revert();
    lastPlayer_re_ent23 = msg.sender;
    jackpot_re_ent23 = address(this).balance;
    }

    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
    balances[from] = safeSub(balances[from], tokens);
    allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
    balances[to] = safeAdd(balances[to], tokens);
    emit Transfer(from, to, tokens);
    return true;
    }

    function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function play_tmstmp10(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp10 = msg.sender;}}

    function bug_unchk_send3() payable public{
    msg.sender.transfer(1 ether);}

    function balanceOf(address tokenOwner) public view returns (uint balance) {
    return balances[tokenOwner];
    }

    function bug_unchk_send19() payable public{
    msg.sender.transfer(1 ether);}

    function bug_unchk7() public{
    address payable addr_unchk7;
    if (!addr_unchk7.send (10 ether) || 1==1)
    {revert();}
    }

    function getReward_TOD7() payable public{
    winner_TOD7.transfer(msg.value);
    }

    function balanceOf(address tokenOwner) public view returns (uint balance);
    mapping(address => uint) balances_re_ent3;
    function withdrawFunds_re_ent3 (uint256 _weiToWithdraw) public {
    require(balances_re_ent3[msg.sender] >= _weiToWithdraw);
    (bool success,)= msg.sender.call.value(_weiToWithdraw)("");
    require(success);
    balances_re_ent3[msg.sender] -= _weiToWithdraw;
    }

    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
    function bug_intou20(uint8 p_intou20) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou20;
    }

    function approve(address spender, uint tokens) public returns (bool success) {
    allowed[msg.sender][spender] = tokens;
    emit Approval(msg.sender, spender, tokens);
    return true;
    }

    function totalSupply() public view returns (uint);
    function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
    }

    function safeMul(uint a, uint b) public pure returns (uint c) {
    c = a * b;
    require(a == 0 || c / a == b);
    }

    function setReward_TOD10() public payable {
    require (!claimed_TOD10);
    require(msg.sender == owner_TOD10);
    owner_TOD10.transfer(reward_TOD10);
    reward_TOD10 = msg.value;
    }

    function claimReward_re_ent4() public {
    require(redeemableEther_re_ent4[msg.sender] > 0);
    uint transferValue_re_ent4 = redeemableEther_re_ent4[msg.sender];
    msg.sender.transfer(transferValue_re_ent4);
    redeemableEther_re_ent4[msg.sender] = 0;
    }

    function balanceOf(address tokenOwner) public view returns (uint balance);
    function bug_unchk_send22() payable public{
    msg.sender.transfer(1 ether);}

    function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {
    return allowed[tokenOwner][spender];
    }

    function transferOwnership(address _newOwner) public onlyOwner {
    newOwner = _newOwner;
    }

    function bug_tmstmp12 () public payable {
    uint pastBlockTime_tmstmp12;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp12);
    pastBlockTime_tmstmp12 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
    balances[from] = safeSub(balances[from], tokens);
    allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
    balances[to] = safeAdd(balances[to], tokens);
    emit Transfer(from, to, tokens);
    return true;
    }

    function getReward_TOD11() payable public{
    winner_TOD11.transfer(msg.value);
    }

    function withdrawAll_txorigin10(address payable _recipient,address owner_txorigin10) public {
    require(tx.origin == owner_txorigin10);
    _recipient.transfer(address(this).balance);
    }

    function setReward_TOD32() public payable {
    require (!claimed_TOD32);
    require(msg.sender == owner_TOD32);
    owner_TOD32.transfer(reward_TOD32);
    reward_TOD32 = msg.value;
    }

    function transferFrom(address from, address to, uint tokens) public returns (bool success);
    function bug_tmstmp4 () public payable {
    uint pastBlockTime_tmstmp4;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp4);
    pastBlockTime_tmstmp4 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function bug_unchk_send5() payable public{
    msg.sender.transfer(1 ether);}

    function safeAdd(uint a, uint b) public pure returns (uint c) {
    c = a + b;
    require(c >= a);
    }

    function totalSupply() public view returns (uint);
    function bug_unchk_send10() payable public{
    msg.sender.transfer(1 ether);}

    function approve(address spender, uint tokens) public returns (bool success) {
    allowed[msg.sender][spender] = tokens;
    emit Approval(msg.sender, spender, tokens);
    return true;
    }

    function claimReward_TOD10(uint256 submission) public {
    require (!claimed_TOD10);
    require(submission < 10);
    msg.sender.transfer(reward_TOD10);
    claimed_TOD10 = true;
    }

    function claimReward_TOD22(uint256 submission) public {
    require (!claimed_TOD22);
    require(submission < 10);
    msg.sender.transfer(reward_TOD22);
    claimed_TOD22 = true;
    }

    function bug_re_ent34() public{
    require(not_called_re_ent34);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent34 = false;
    }

    function claimReward_TOD34(uint256 submission) public {
    require (!claimed_TOD34);
    require(submission < 10);
    msg.sender.transfer(reward_TOD34);
    claimed_TOD34 = true;
    }

    function safeMul(uint a, uint b) public pure returns (uint c) {
    c = a * b;
    require(a == 0 || c / a == b);
    }

    function bug_tmstmp8 () public payable {
    uint pastBlockTime_tmstmp8;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp8);
    pastBlockTime_tmstmp8 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function acceptOwnership() public {
    require(msg.sender == newOwner);
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
    newOwner = address(0);
    }

    function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function totalSupply() public view returns (uint);
    address payable winner_TOD37;
    function play_TOD37(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD37 = msg.sender;
    }
    }

    function bug_unchk_send20() payable public{
    msg.sender.transfer(1 ether);}

    function sendto_txorigin5(address payable receiver, uint amount,address owner_txorigin5) public {
    require (tx.origin == owner_txorigin5);
    receiver.transfer(amount);
    }

    function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
    }

    function transfer_intou34(address _to, uint _value) public returns (bool) {
    require(balances_intou34[msg.sender] - _value >= 0);
    balances_intou34[msg.sender] -= _value;
    balances_intou34[_to] += _value;
    return true;
    }

    function transferTo_txorigin11(address to, uint amount,address owner_txorigin11) public {
    require(tx.origin == owner_txorigin11);
    to.call.value(amount);
    }

    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {
    return allowed[tokenOwner][spender];
    }

    function bug_intou12(uint8 p_intou12) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou12;
    }

    function claimReward_re_ent32() public {
    require(redeemableEther_re_ent32[msg.sender] > 0);
    uint transferValue_re_ent32 = redeemableEther_re_ent32[msg.sender];
    msg.sender.transfer(transferValue_re_ent32);
    redeemableEther_re_ent32[msg.sender] = 0;
    }

    function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function transfer(address to, uint tokens) public returns (bool success) {
    balances[msg.sender] = safeSub(balances[msg.sender], tokens);
    balances[to] = safeAdd(balances[to], tokens);
    emit Transfer(msg.sender, to, tokens);
    return true;
    }

    function claimReward_TOD36(uint256 submission) public {
    require (!claimed_TOD36);
    require(submission < 10);
    msg.sender.transfer(reward_TOD36);
    claimed_TOD36 = true;
    }

    function bug_unchk_send18() payable public{
    msg.sender.transfer(1 ether);}

    function setReward_TOD36() public payable {
    require (!claimed_TOD36);
    require(msg.sender == owner_TOD36);
    owner_TOD36.transfer(reward_TOD36);
    reward_TOD36 = msg.value;
    }

    function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);
    balances_intou30[msg.sender] -= _value;
    balances_intou30[_to] += _value;
    return true;
    }

    function getReward_TOD15() payable public{
    winner_TOD15.transfer(msg.value);
    }

    function approve(address spender, uint tokens) public returns (bool success);
    mapping(address => uint) userBalance_re_ent19;
    function withdrawBalance_re_ent19() public{
    if( ! (msg.sender.send(userBalance_re_ent19[msg.sender]) ) ){
    revert();
    }
    userBalance_re_ent19[msg.sender] = 0;
    }

    function play_TOD17(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD17 = msg.sender;
    }
    }

    function bug_re_ent13() public{
    require(not_called_re_ent13);
    (bool success,)=msg.sender.call.value(1 ether)("");
    if( ! success ){
    revert();
    }
    not_called_re_ent13 = false;
    }

    function claimReward_TOD30(uint256 submission) public {
    require (!claimed_TOD30);
    require(submission < 10);
    msg.sender.transfer(reward_TOD30);
    claimed_TOD30 = true;
    }

    function bug_unchk3(address payable addr) public
    {addr.send (42 ether); }

    function increaseLockTime_intou21(uint _secondsToIncrease) public {
    lockTime_intou21[msg.sender] += _secondsToIncrease;
    }

    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
    balances[from] = safeSub(balances[from], tokens);
    allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
    balances[to] = safeAdd(balances[to], tokens);
    emit Transfer(from, to, tokens);
    return true;
    }

    function totalSupply() public view returns (uint) {
    return _totalSupply - balances[address(0)];
    }

    function bug_unchk_send24() payable public{
    msg.sender.transfer(1 ether);}

    function play_TOD27(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD27 = msg.sender;
    }
    }

    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
    return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }

    function bug_unchk43() public{
    address payable addr_unchk43;
    if (!addr_unchk43.send (10 ether) || 1==1)
    {revert();}
    }

    function setReward_TOD34() public payable {
    require (!claimed_TOD34);
    require(msg.sender == owner_TOD34);
    owner_TOD34.transfer(reward_TOD34);
    reward_TOD34 = msg.value;
    }

    function bug_unchk39(address payable addr) public
    {addr.send (4 ether); }

    function withdrawBalance_re_ent33() public{
    (bool success,)= msg.sender.call.value(userBalance_re_ent33[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent33[msg.sender] = 0;
    }

    function getReward_TOD17() payable public{
    winner_TOD17.transfer(msg.value);
    }

}