// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_38 {
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

    function setReward_TOD8() public payable {
    require (!claimed_TOD8);
    require(msg.sender == owner_TOD8);
    owner_TOD8.transfer(reward_TOD8);
    reward_TOD8 = msg.value;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
    require(value <= _balances[from]);
    require(value <= _allowed[from][msg.sender]);
    require(to != address(0));
    _balances[from] = _balances[from].sub(value);
    uint256 tokensToBurn = findfourPercent(value);
    uint256 tokensToTransfer = value.sub(tokensToBurn);
    _balances[to] = _balances[to].add(tokensToTransfer);
    _totalSupply = _totalSupply.sub(tokensToBurn);
    _allowed[from][msg.sender] = _allowed[from][msg.sender].sub(value);
    emit Transfer(from, to, tokensToTransfer);
    emit Transfer(from, address(0), tokensToBurn);
    return true;
    }

    function my_func_unchk23(address payable dst) public payable{
    dst.send(msg.value);
    }

    function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
    require(tx.origin == owner_txorigin23);
    to.call.value(amount);
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].sub(subtractedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
    }

    function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
    require(tx.origin == owner_txorigin38);
    _recipient.transfer(address(this).balance);
    }

    function burnFrom(address account, uint256 amount) external {
    require(amount <= _allowed[account][msg.sender]);
    _allowed[account][msg.sender] = _allowed[account][msg.sender].sub(amount);
    _burn(account, amount);
    }

    function play_TOD11(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD11 = msg.sender;
    }
    }

    function bug_unchk_send8() payable public{
    msg.sender.transfer(1 ether);}

    function approve(address spender, uint256 value) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = value;
    emit Approval(msg.sender, spender, value);
    return true;
    }

    function sendto_txorigin17(address payable receiver, uint amount,address owner_txorigin17) public {
    require (tx.origin == owner_txorigin17);
    receiver.transfer(amount);
    }

    function getReward_TOD1() payable public{
    winner_TOD1.transfer(msg.value);
    }

    function withdraw_intou17() public {
    require(now > lockTime_intou17[msg.sender]);
    uint transferValue_intou17 = 10;
    msg.sender.transfer(transferValue_intou17);
    }

    function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
    }

    function bug_tmstmp20 () public payable {
    uint pastBlockTime_tmstmp20;
    require(msg.value == 10 ether);
    require(now != pastBlockTime_tmstmp20);
    pastBlockTime_tmstmp20 = now;
    if(now % 15 == 0) {
    msg.sender.transfer(address(this).balance);
    }
    }

    function play_tmstmp19(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp19 = msg.sender;}}

    function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
    require(value <= _balances[from]);
    require(value <= _allowed[from][msg.sender]);
    require(to != address(0));
    _balances[from] = _balances[from].sub(value);
    uint256 tokensToBurn = findfourPercent(value);
    uint256 tokensToTransfer = value.sub(tokensToBurn);
    _balances[to] = _balances[to].add(tokensToTransfer);
    _totalSupply = _totalSupply.sub(tokensToBurn);
    _allowed[from][msg.sender] = _allowed[from][msg.sender].sub(value);
    emit Transfer(from, to, tokensToTransfer);
    emit Transfer(from, address(0), tokensToBurn);
    return true;
    }

    function bug_unchk19() public{
    address payable addr_unchk19;
    if (!addr_unchk19.send (10 ether) || 1==1)
    {revert();}
    }

    function bug_unchk_send28() payable public{
    msg.sender.transfer(1 ether);}

    function name() public view returns(string memory) {
    return _name;
    }

    function play_tmstmp31(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp31 = msg.sender;}}

    function claimReward_re_ent25() public {
    require(redeemableEther_re_ent25[msg.sender] > 0);
    uint transferValue_re_ent25 = redeemableEther_re_ent25[msg.sender];
    msg.sender.transfer(transferValue_re_ent25);
    redeemableEther_re_ent25[msg.sender] = 0;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
    return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
    }

    function burn(uint256 amount) external {
    _burn(msg.sender, amount);
    }

    function allowance(address owner, address spender) public view returns (uint256) {
    return _allowed[owner][spender];
    }

    function bug_re_ent27() public{
    require(not_called_re_ent27);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent27 = false;
    }

    function bug_unchk7() public{
    address payable addr_unchk7;
    if (!addr_unchk7.send (10 ether) || 1==1)
    {revert();}
    }

    function bug_unchk_send3() payable public{
    msg.sender.transfer(1 ether);}

    function play_tmstmp38(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp38 = msg.sender;}}

    function bug_unchk_send22() payable public{
    msg.sender.transfer(1 ether);}

    function withdraw_balances_re_ent8 () public {
    (bool success,) = msg.sender.call.value(balances_re_ent8[msg.sender ])("");
    if (success)
    balances_re_ent8[msg.sender] = 0;
    }

    constructor(string memory name, string memory symbol, uint8 decimals) public {
    _name = name;
    _symbol = symbol;
    _decimals = decimals;
    }

    function sendToWinner_unchk32() public {
    require(!payedOut_unchk32);
    winner_unchk32.send(winAmount_unchk32);
    payedOut_unchk32 = true;
    }

    function increaseLockTime_intou13(uint _secondsToIncrease) public {
    lockTime_intou13[msg.sender] += _secondsToIncrease;
    }

    function getReward_TOD33() payable public{
    winner_TOD33.transfer(msg.value);
    }

    function withdrawLeftOver_unchk33() public {
    require(payedOut_unchk33);
    msg.sender.send(address(this).balance);
    }

    function claimReward_re_ent4() public {
    require(redeemableEther_re_ent4[msg.sender] > 0);
    uint transferValue_re_ent4 = redeemableEther_re_ent4[msg.sender];
    msg.sender.transfer(transferValue_re_ent4);
    redeemableEther_re_ent4[msg.sender] = 0;
    }

    function bug_txorigin8(address owner_txorigin8) public{
    require(tx.origin == owner_txorigin8);
    }

    function totalSupply() public view returns (uint256) {
    return _totalSupply;
    }

    function setReward_TOD38() public payable {
    require (!claimed_TOD38);
    require(msg.sender == owner_TOD38);
    owner_TOD38.transfer(reward_TOD38);
    reward_TOD38 = msg.value;
    }

    function claimReward_TOD8(uint256 submission) public {
    require (!claimed_TOD8);
    require(submission < 10);
    msg.sender.transfer(reward_TOD8);
    claimed_TOD8 = true;
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].add(addedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
    }

    constructor() public payable ERC20Detailed(tokenName, tokenSymbol, tokenDecimals) {
    _mint(msg.sender, _totalSupply);
    }

    function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function bug_unchk_send21() payable public{
    msg.sender.transfer(1 ether);}

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].sub(subtractedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function getReward_TOD37() payable public{
    winner_TOD37.transfer(msg.value);
    }

    function play_TOD7(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD7 = msg.sender;
    }
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].add(addedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
    require(value <= _balances[from]);
    require(value <= _allowed[from][msg.sender]);
    require(to != address(0));
    _balances[from] = _balances[from].sub(value);
    uint256 tokensToBurn = findfourPercent(value);
    uint256 tokensToTransfer = value.sub(tokensToBurn);
    _balances[to] = _balances[to].add(tokensToTransfer);
    _totalSupply = _totalSupply.sub(tokensToBurn);
    _allowed[from][msg.sender] = _allowed[from][msg.sender].sub(value);
    emit Transfer(from, to, tokensToTransfer);
    emit Transfer(from, address(0), tokensToBurn);
    return true;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
    }

    function allowance(address owner, address spender) public view returns (uint256) {
    return _allowed[owner][spender];
    }

    function balanceOf(address owner) public view returns (uint256) {
    return _balances[owner];
    }

    function bug_unchk_send13() payable public{
    msg.sender.transfer(1 ether);}

    function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
    }

    function setReward_TOD14() public payable {
    require (!claimed_TOD14);
    require(msg.sender == owner_TOD14);
    owner_TOD14.transfer(reward_TOD14);
    reward_TOD14 = msg.value;
    }

    function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].sub(subtractedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function transferTo_txorigin19(address to, uint amount,address owner_txorigin19) public {
    require(tx.origin == owner_txorigin19);
    to.call.value(amount);
    }

    function transfer_intou22(address _to, uint _value) public returns (bool) {
    require(balances_intou22[msg.sender] - _value >= 0);
    balances_intou22[msg.sender] -= _value;
    balances_intou22[_to] += _value;
    return true;
    }

    function bug_unchk27(address payable addr) public
    {addr.send (42 ether); }

    function findfourPercent(uint256 value) public view returns (uint256) {
    uint256 roundValue = value.ceil(basePercent);
    uint256 fourPercent = roundValue.mul(basePercent).div(2500);
    return fourPercent;
    }

    function approve(address spender, uint256 value) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = value;
    emit Approval(msg.sender, spender, value);
    return true;
    }

    function withdraw_intou13() public {
    require(now > lockTime_intou13[msg.sender]);
    uint transferValue_intou13 = 10;
    msg.sender.transfer(transferValue_intou13);
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
    }

    function my_func_unchk35(address payable dst) public payable{
    dst.send(msg.value);
    }

    function getReward_TOD3() payable public{
    winner_TOD3.transfer(msg.value);
    }

    function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    for (uint256 i = 0; i < receivers.length; i++) {
    transfer(receivers[i], amounts[i]);
    }
    }

    function callme_re_ent14() public{
    require(counter_re_ent14<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent14 += 1;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
    }

    function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    for (uint256 i = 0; i < receivers.length; i++) {
    transfer(receivers[i], amounts[i]);
    }
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].add(addedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;
    }

    function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
    require(tx.origin == owner_txorigin7);
    to.call.value(amount);
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
    require(spender != address(0));
    _allowed[msg.sender][spender] = (_allowed[msg.sender][spender].add(addedValue));
    emit Approval(msg.sender, spender, _allowed[msg.sender][spender]);
    return true;
    }

    function decimals() public view returns(uint8) {
    return _decimals;
    }

    function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
    require(tx.origin == owner_txorigin14);
    _recipient.transfer(address(this).balance);
    }

    function callme_re_ent42() public{
    require(counter_re_ent42<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent42 += 1;
    }

    function bug_unchk_send23() payable public{
    msg.sender.transfer(1 ether);}

    function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    for (uint256 i = 0; i < receivers.length; i++) {
    transfer(receivers[i], amounts[i]);
    }
    }

    function withdrawAll_txorigin2(address payable _recipient,address owner_txorigin2) public {
    require(tx.origin == owner_txorigin2);
    _recipient.transfer(address(this).balance);
    }

    function play_TOD3(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD3 = msg.sender;
    }
    }

    function bug_re_ent41() public{
    require(not_called_re_ent41);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent41 = false;
    }

    function bug_unchk_send26() payable public{
    msg.sender.transfer(1 ether);}

}