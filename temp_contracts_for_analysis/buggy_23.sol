// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_23 {
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

    function claimReward_TOD2(uint256 submission) public {
    require (!claimed_TOD2);
    require(submission < 10);
    msg.sender.transfer(reward_TOD2);
    claimed_TOD2 = true;
    }

    function totalSupply() public view returns (uint256) {
    return _totalSupply;
    }

    function play_TOD9(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD9 = msg.sender;
    }
    }

    function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
    require (tx.origin == owner_txorigin13);
    receiver.transfer(amount);
    }

    constructor (string memory name, string memory symbol, uint8 decimals) public {
    _name = name;
    _symbol = symbol;
    _decimals = decimals;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
    _transfer(from, to, value);
    _approve(from, msg.sender, _allowed[from][msg.sender].sub(value));
    return true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
    _approve(msg.sender, spender, value);
    return true;
    }

    function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
    }

    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    }
    pragma solidity ^0.5.2;
    library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
    return 0;
    }
    uint256 c = a * b;
    require(c / a == b);
    return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0);
    uint256 c = a / b;
    return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a);
    uint256 c = a - b;
    return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a);
    return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b != 0);
    return a % b;
    }
    }

    function burn(uint256 value) public {
    _burn(msg.sender, value);
    }

    function bug_tmstmp40 () public payable {
    uint pastBlockTime_tmstmp40;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp40);
    pastBlockTime_tmstmp40 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function withdrawBalance_re_ent33() public{
    (bool success,)= msg.sender.call.value(userBalance_re_ent33[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent33[msg.sender] = 0;
    }

    function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;
    }

    function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    _approve(msg.sender, spender, _allowed[msg.sender][spender].add(addedValue));
    return true;
    }

    function increaseLockTime_intou1(uint _secondsToIncrease) public {
    lockTime_intou1[msg.sender] += _secondsToIncrease;
    }

    function bug_unchk_send21() payable public{
    msg.sender.transfer(1 ether);}

    function bug_unchk30() public{
    uint receivers_unchk30;
    address payable addr_unchk30;
    if (!addr_unchk30.send(42 ether))
    {receivers_unchk30 +=1;}
    else
    {revert();}
    }

    function burnFrom(address from, uint256 value) public {
    _burnFrom(from, value);
    }

    function _burn(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.sub(value);
    _balances[account] = _balances[account].sub(value);
    emit Transfer(account, address(0), value);
    }

    function getReward_TOD37() payable public{
    winner_TOD37.transfer(msg.value);
    }

    function transfer(address to, uint256 value) public returns (bool) {
    _transfer(msg.sender, to, value);
    return true;
    }

    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    }
    pragma solidity ^0.5.2;
    library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
    return 0;
    }
    uint256 c = a * b;
    require(c / a == b);
    return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0);
    uint256 c = a / b;
    return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a);
    uint256 c = a - b;
    return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a);
    return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b != 0);
    return a % b;
    }
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
    _approve(msg.sender, spender, _allowed[msg.sender][spender].sub(subtractedValue));
    return true;
    }

    function claimReward_TOD30(uint256 submission) public {
    require (!claimed_TOD30);
    require(submission < 10);
    msg.sender.transfer(reward_TOD30);
    claimed_TOD30 = true;
    }

    function withdraw_intou33() public {
    require(now > lockTime_intou33[msg.sender]);
    uint transferValue_intou33 = 10;
    msg.sender.transfer(transferValue_intou33);
    }

    function bug_unchk_send11() payable public{
    msg.sender.transfer(1 ether);}

    function buyTicket_re_ent37() public{
    if (!(lastPlayer_re_ent37.send(jackpot_re_ent37)))
    revert();
    lastPlayer_re_ent37 = msg.sender;
    jackpot_re_ent37 = address(this).balance;
    }

    function play_tmstmp30(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp30 = msg.sender;}}

    function getReward_TOD7() payable public{
    winner_TOD7.transfer(msg.value);
    }

    function allowance(address owner, address spender) public view returns (uint256) {
    return _allowed[owner][spender];
    }

    function _transfer(address from, address to, uint256 value) internal {
    require(to != address(0));
    _balances[from] = _balances[from].sub(value);
    _balances[to] = _balances[to].add(value);
    emit Transfer(from, to, value);
    }

    function withdrawBalance_re_ent19() public{
    if( ! (msg.sender.send(userBalance_re_ent19[msg.sender]) ) ){
    revert();
    }
    userBalance_re_ent19[msg.sender] = 0;
    }

    function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
    }

    function increaseLockTime_intou33(uint _secondsToIncrease) public {
    lockTime_intou33[msg.sender] += _secondsToIncrease;
    }

    function decimals() public view returns (uint8) {
    return _decimals;
    }

    function bug_unchk_send32() payable public{
    msg.sender.transfer(1 ether);}

    function _mint(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.add(value);
    _balances[account] = _balances[account].add(value);
    emit Transfer(address(0), account, value);
    }

    function bug_unchk_send23() payable public{
    msg.sender.transfer(1 ether);}

    function bug_unchk_send30() payable public{
    msg.sender.transfer(1 ether);}

    function play_TOD1(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD1 = msg.sender;
    }
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
    return 0;
    }
    uint256 c = a * b;
    require(c / a == b);
    return c;
    }

    function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
    require(tx.origin == owner_txorigin35);
    to.call.value(amount);
    }

    function name() public view returns (string memory) {
    return _name;
    }

    function _transfer(address from, address to, uint256 value) internal {
    require(to != address(0));
    _balances[from] = _balances[from].sub(value);
    _balances[to] = _balances[to].add(value);
    emit Transfer(from, to, value);
    }

    function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function claimReward_TOD40(uint256 submission) public {
    require (!claimed_TOD40);
    require(submission < 10);
    msg.sender.transfer(reward_TOD40);
    claimed_TOD40 = true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
    _approve(msg.sender, spender, value);
    return true;
    }

    function callme_re_ent14() public{
    require(counter_re_ent14<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent14 += 1;
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
    _approve(msg.sender, spender, _allowed[msg.sender][spender].sub(subtractedValue));
    return true;
    }

    function symbol() public view returns (string memory) {
    return _symbol;
    }

    function play_tmstmp23(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp23 = msg.sender;}}

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    _approve(msg.sender, spender, _allowed[msg.sender][spender].add(addedValue));
    return true;
    }

    function name() public view returns (string memory) {
    return _name;
    }

    function buyTicket_re_ent30() public{
    if (!(lastPlayer_re_ent30.send(jackpot_re_ent30)))
    revert();
    lastPlayer_re_ent30 = msg.sender;
    jackpot_re_ent30 = address(this).balance;
    }

    function withdraw_intou17() public {
    require(now > lockTime_intou17[msg.sender]);
    uint transferValue_intou17 = 10;
    msg.sender.transfer(transferValue_intou17);
    }

    function sendto_txorigin17(address payable receiver, uint amount,address owner_txorigin17) public {
    require (tx.origin == owner_txorigin17);
    receiver.transfer(amount);
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0);
    uint256 c = a / b;
    return c;
    }

    function getReward_TOD33() payable public{
    winner_TOD33.transfer(msg.value);
    }

    function sendto_txorigin37(address payable receiver, uint amount,address owner_txorigin37) public {
    require (tx.origin == owner_txorigin37);
    receiver.transfer(amount);
    }

    function _mint(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.add(value);
    _balances[account] = _balances[account].add(value);
    emit Transfer(address(0), account, value);
    }

    function _burn(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.sub(value);
    _balances[account] = _balances[account].sub(value);
    emit Transfer(account, address(0), value);
    }

    function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;
    }

    function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function _burn(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.sub(value);
    _balances[account] = _balances[account].sub(value);
    emit Transfer(account, address(0), value);
    }

    function totalSupply() public view returns (uint256) {
    return _totalSupply;
    }

    function callme_re_ent35() public{
    require(counter_re_ent35<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent35 += 1;
    }

    function play_TOD33(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD33 = msg.sender;
    }
    }

    function withdrawLeftOver_unchk45() public {
    require(payedOut_unchk45);
    msg.sender.send(address(this).balance);
    }

    function increaseLockTime_intou17(uint _secondsToIncrease) public {
    lockTime_intou17[msg.sender] += _secondsToIncrease;
    }

    function getReward_TOD19() payable public{
    winner_TOD19.transfer(msg.value);
    }

    function play_TOD17(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD17 = msg.sender;
    }
    }

    function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
    require(tx.origin == owner_txorigin23);
    to.call.value(amount);
    }

    function bug_unchk_send26() payable public{
    msg.sender.transfer(1 ether);}

    function _burn(address account, uint256 value) internal {
    require(account != address(0));
    _totalSupply = _totalSupply.sub(value);
    _balances[account] = _balances[account].sub(value);
    emit Transfer(account, address(0), value);
    }

    function bug_txorigin40(address owner_txorigin40) public{
    require(tx.origin == owner_txorigin40);
    }

    function withdrawFunds_re_ent17 (uint256 _weiToWithdraw) public {
    require(balances_re_ent17[msg.sender] >= _weiToWithdraw);
    (bool success,)=msg.sender.call.value(_weiToWithdraw)("");
    require(success);
    balances_re_ent17[msg.sender] -= _weiToWithdraw;
    }

    function bug_unchk39(address payable addr) public
    {addr.send (4 ether); }

    function getReward_TOD23() payable public{
    winner_TOD23.transfer(msg.value);
    }

}