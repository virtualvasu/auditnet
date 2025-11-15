// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_8 {
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

    function withdrawBalance_re_ent26() public{
    (bool success,)= msg.sender.call.value(userBalance_re_ent26[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent26[msg.sender] = 0;
    }

    function play_tmstmp26(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp26 = msg.sender;}}

    function increaseLockTime_intou9(uint _secondsToIncrease) public {
    lockTime_intou9[msg.sender] += _secondsToIncrease;
    }

    function getReward_TOD35() payable public{
    winner_TOD35.transfer(msg.value);
    }

    function burnFrom(address _from, uint256 _value) public returns (bool success) {
    require(balanceOf[_from] >= _value);
    require(_value <= allowance[_from][msg.sender]);
    balanceOf[_from] -= _value;
    allowance[_from][msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(_from, _value);
    return true;
    }

    function getReward_TOD27() payable public{
    winner_TOD27.transfer(msg.value);
    }

    function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
    }

    function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);
    balances_intou14[msg.sender] -= _value;
    balances_intou14[_to] += _value;
    return true;
    }

    function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    constructor(
    uint256 initialSupply,
    string memory tokenName,
    string memory tokenSymbol
    ) public {
    totalSupply = initialSupply * 10 ** uint256(decimals);
    balanceOf[msg.sender] = totalSupply;
    name = tokenName;
    symbol = tokenSymbol;
    }

    function claimReward_re_ent25() public {
    require(redeemableEther_re_ent25[msg.sender] > 0);
    uint transferValue_re_ent25 = redeemableEther_re_ent25[msg.sender];
    msg.sender.transfer(transferValue_re_ent25);
    redeemableEther_re_ent25[msg.sender] = 0;
    }

    constructor(
    uint256 initialSupply,
    string memory tokenName,
    string memory tokenSymbol
    ) public {
    totalSupply = initialSupply * 10 ** uint256(decimals);
    balanceOf[msg.sender] = totalSupply;
    name = tokenName;
    symbol = tokenSymbol;
    }

    function withdrawBalance_re_ent12() public{
    if( ! (msg.sender.send(userBalance_re_ent12[msg.sender]) ) ){
    revert();
    }
    userBalance_re_ent12[msg.sender] = 0;
    }

    function transfer_undrflow2(address _to, uint _value) public returns (bool) {
    require(balances_intou2[msg.sender] - _value >= 0);
    balances_intou2[msg.sender] -= _value;
    balances_intou2[_to] += _value;
    return true;
    }

    function _transfer(address _from, address _to, uint _value) internal {
    require (_to != address(0x0));
    require (balanceOf[_from] >= _value);
    require (balanceOf[_to] + _value >= balanceOf[_to]);
    require(!frozenAccount[_from]);
    require(!frozenAccount[_to]);
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    }

    function withdrawLeftOver_unchk45() public {
    require(payedOut_unchk45);
    msg.sender.send(address(this).balance);
    }

    function claimReward_TOD10(uint256 submission) public {
    require (!claimed_TOD10);
    require(submission < 10);
    msg.sender.transfer(reward_TOD10);
    claimed_TOD10 = true;
    }

    function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
    require(tx.origin == owner_txorigin27);
    to.call.value(amount);
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

    function burn(uint256 _value) public returns (bool success) {
    require(balanceOf[msg.sender] >= _value);
    balanceOf[msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(msg.sender, _value);
    return true;
    }

    constructor(
    uint256 initialSupply,
    string memory tokenName,
    string memory tokenSymbol
    ) TokenERC20(initialSupply, tokenName, tokenSymbol) public {}

    function transfer_intou38(address _to, uint _value) public returns (bool) {
    require(balances_intou38[msg.sender] - _value >= 0);
    balances_intou38[msg.sender] -= _value;
    balances_intou38[_to] += _value;
    return true;
    }

    function withdraw_intou17() public {
    require(now > lockTime_intou17[msg.sender]);
    uint transferValue_intou17 = 10;
    msg.sender.transfer(transferValue_intou17);
    }

    function increaseLockTime_intou25(uint _secondsToIncrease) public {
    lockTime_intou25[msg.sender] += _secondsToIncrease;
    }

    function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function bug_unchk_send13() payable public{
    msg.sender.transfer(1 ether);}

    function bug_unchk19() public{
    address payable addr_unchk19;
    if (!addr_unchk19.send (10 ether) || 1==1)
    {revert();}
    }

    function setPrices(uint256 newSellPrice, uint256 newBuyPrice) onlyOwner public {
    sellPrice = newSellPrice;
    buyPrice = newBuyPrice;
    }

    function increaseLockTime_intou33(uint _secondsToIncrease) public {
    lockTime_intou33[msg.sender] += _secondsToIncrease;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
    _transfer(msg.sender, _to, _value);
    return true;
    }

    function play_TOD17(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD17 = msg.sender;
    }
    }

    function play_tmstmp23(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp23 = msg.sender;}}

    function claimReward_TOD4(uint256 submission) public {
    require (!claimed_TOD4);
    require(submission < 10);
    msg.sender.transfer(reward_TOD4);
    claimed_TOD4 = true;
    }

    function _transfer(address _from, address _to, uint _value) internal {
    require(_to != address(0x0));
    require(balanceOf[_from] >= _value);
    require(balanceOf[_to] + _value > balanceOf[_to]);
    uint previousBalances = balanceOf[_from] + balanceOf[_to];
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }

    function getReward_TOD23() payable public{
    winner_TOD23.transfer(msg.value);
    }

    function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
    require(tx.origin == owner_txorigin14);
    _recipient.transfer(address(this).balance);
    }

    function claimReward_TOD32(uint256 submission) public {
    require (!claimed_TOD32);
    require(submission < 10);
    msg.sender.transfer(reward_TOD32);
    claimed_TOD32 = true;
    }

    function setReward_TOD2() public payable {
    require (!claimed_TOD2);
    require(msg.sender == owner_TOD2);
    owner_TOD2.transfer(reward_TOD2);
    reward_TOD2 = msg.value;
    }

    function burn(uint256 _value) public returns (bool success) {
    require(balanceOf[msg.sender] >= _value);
    balanceOf[msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(msg.sender, _value);
    return true;
    }

    function bug_unchk_send5() payable public{
    msg.sender.transfer(1 ether);}

    function burn(uint256 _value) public returns (bool success) {
    require(balanceOf[msg.sender] >= _value);
    balanceOf[msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(msg.sender, _value);
    return true;
    }

    function bug_unchk_send23() payable public{
    msg.sender.transfer(1 ether);}

    function transfer_intou26(address _to, uint _value) public returns (bool) {
    require(balances_intou26[msg.sender] - _value >= 0);
    balances_intou26[msg.sender] -= _value;
    balances_intou26[_to] += _value;
    return true;
    }

    function sendto_txorigin9(address payable receiver, uint amount,address owner_txorigin9) public {
    require (tx.origin == owner_txorigin9);
    receiver.transfer(amount);
    }

    function approve(address _spender, uint256 _value) public
    returns (bool success) {
    allowance[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function setReward_TOD4() public payable {
    require (!claimed_TOD4);
    require(msg.sender == owner_TOD4);
    owner_TOD4.transfer(reward_TOD4);
    reward_TOD4 = msg.value;
    }

    function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
    }

    function bug_intou19() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function sendToWinner_unchk44() public {
    require(!payedOut_unchk44);
    winner_unchk44.send(winAmount_unchk44);
    payedOut_unchk44 = true;
    }

    function sell(uint256 amount) public {
    address myAddress = address(this);
    require(myAddress.balance >= amount * sellPrice);
    _transfer(msg.sender, address(this), amount);
    msg.sender.transfer(amount * sellPrice);
    }

    function _transfer(address _from, address _to, uint _value) internal {
    require(_to != address(0x0));
    require(balanceOf[_from] >= _value);
    require(balanceOf[_to] + _value > balanceOf[_to]);
    uint previousBalances = balanceOf[_from] + balanceOf[_to];
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }

    function _transfer(address _from, address _to, uint _value) internal {
    require(_to != address(0x0));
    require(balanceOf[_from] >= _value);
    require(balanceOf[_to] + _value > balanceOf[_to]);
    uint previousBalances = balanceOf[_from] + balanceOf[_to];
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
    _transfer(msg.sender, _to, _value);
    return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    require(_value <= allowance[_from][msg.sender]);
    allowance[_from][msg.sender] -= _value;
    _transfer(_from, _to, _value);
    return true;
    }

    function sendToWinner_unchk32() public {
    require(!payedOut_unchk32);
    winner_unchk32.send(winAmount_unchk32);
    payedOut_unchk32 = true;
    }

    function approve(address _spender, uint256 _value) public
    returns (bool success) {
    allowance[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    constructor(
    uint256 initialSupply,
    string memory tokenName,
    string memory tokenSymbol
    ) public {
    totalSupply = initialSupply * 10 ** uint256(decimals);
    balanceOf[msg.sender] = totalSupply;
    name = tokenName;
    symbol = tokenSymbol;
    }

    function approve(address _spender, uint256 _value) public
    returns (bool success) {
    allowance[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function buy() payable public {
    uint amount = msg.value / buyPrice;
    _transfer(address(this), msg.sender, amount);
    }

}