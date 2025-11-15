// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_32 {
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

    function balanceOf(address _owner) public view returns (uint256 value);
    bool claimed_TOD4 = false;
    address payable owner_TOD4;
    uint256 reward_TOD4;
    function setReward_TOD4() public payable {
    require (!claimed_TOD4);
    require(msg.sender == owner_TOD4);
    owner_TOD4.transfer(reward_TOD4);
    reward_TOD4 = msg.value;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function balanceOf(address _owner) public view returns (uint256 value);
    mapping(address => uint) redeemableEther_re_ent4;
    function claimReward_re_ent4() public {
    require(redeemableEther_re_ent4[msg.sender] > 0);
    uint transferValue_re_ent4 = redeemableEther_re_ent4[msg.sender];
    msg.sender.transfer(transferValue_re_ent4);
    redeemableEther_re_ent4[msg.sender] = 0;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a, "SafeMath: addition overflow");
    return c;
    }

    function play_TOD13(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD13 = msg.sender;
    }
    }

    function claimReward_TOD4(uint256 submission) public {
    require (!claimed_TOD4);
    require(submission < 10);
    msg.sender.transfer(reward_TOD4);
    claimed_TOD4 = true;
    }

    function claimReward_TOD14(uint256 submission) public {
    require (!claimed_TOD14);
    require(submission < 10);
    msg.sender.transfer(reward_TOD14);
    claimed_TOD14 = true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
    function bug_unchk_send26() payable public{
    msg.sender.transfer(1 ether);}

    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
    return allowed[_owner][_spender];
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(_from, _to, _value);
    return true;
    }

    function withdrawFunds_re_ent38 (uint256 _weiToWithdraw) public {
    require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
    require(msg.sender.send(_weiToWithdraw));
    balances_re_ent38[msg.sender] -= _weiToWithdraw;
    }

    function approve(address _spender, uint256 _value) public returns (bool success);
    mapping(address => uint) balances_intou14;
    function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);
    balances_intou14[msg.sender] -= _value;
    balances_intou14[_to] += _value;
    return true;
    }

    function claimReward_TOD32(uint256 submission) public {
    require (!claimed_TOD32);
    require(submission < 10);
    msg.sender.transfer(reward_TOD32);
    claimed_TOD32 = true;
    }

    function bug_re_ent27() public{
    require(not_called_re_ent27);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent27 = false;
    }

    function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;
    }

    function getReward_TOD35() payable public{
    winner_TOD35.transfer(msg.value);
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
    }

    function getReward_TOD31() payable public{
    winner_TOD31.transfer(msg.value);
    }

    function play_tmstmp30(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp30 = msg.sender;}}

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(_from, _to, _value);
    return true;
    }

    function bug_unchk31() public{
    address payable addr_unchk31;
    if (!addr_unchk31.send (10 ether) || 1==1)
    {revert();}
    }

    function withdraw_intou13() public {
    require(now > lockTime_intou13[msg.sender]);
    uint transferValue_intou13 = 10;
    msg.sender.transfer(transferValue_intou13);
    }

    function bug_txorigin40(address owner_txorigin40) public{
    require(tx.origin == owner_txorigin40);
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
    function bug_intou23() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(_from, _to, _value);
    return true;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a, "SafeMath: addition overflow");
    return c;
    }

    function play_tmstmp38(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp38 = msg.sender;}}

    function bug_re_ent20() public{
    require(not_called_re_ent20);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent20 = false;
    }

    function increaseLockTime_intou33(uint _secondsToIncrease) public {
    lockTime_intou33[msg.sender] += _secondsToIncrease;
    }

    constructor() public {
    address initialOwner = 0xac775cD446889ac167da466692449ece5439fc12;
    totalSupply = 180000000 * (10**uint256(decimals));
    balances[initialOwner] = totalSupply;
    emit Transfer(address(0), initialOwner, balances[initialOwner]);
    }

    function balanceOf(address _owner) public view returns (uint256 value) {
    return balances[_owner];
    }

    function UncheckedExternalCall_unchk40 () public
    { address payable addr_unchk40;
    if (! addr_unchk40.send (2 ether))
    {
    }
    else
    {
    }
    }

    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
    bool claimed_TOD30 = false;
    address payable owner_TOD30;
    uint256 reward_TOD30;
    function setReward_TOD30() public payable {
    require (!claimed_TOD30);
    require(msg.sender == owner_TOD30);
    owner_TOD30.transfer(reward_TOD30);
    reward_TOD30 = msg.value;
    }

    function bug_txorigin36( address owner_txorigin36) public{
    require(tx.origin == owner_txorigin36);
    }

    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
    return allowed[_owner][_spender];
    }

    function setReward_TOD40() public payable {
    require (!claimed_TOD40);
    require(msg.sender == owner_TOD40);
    owner_TOD40.transfer(reward_TOD40);
    reward_TOD40 = msg.value;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
    }

    function bug_re_ent13() public{
    require(not_called_re_ent13);
    (bool success,)=msg.sender.call.value(1 ether)("");
    if( ! success ){
    revert();
    }
    not_called_re_ent13 = false;
    }

    function setReward_TOD38() public payable {
    require (!claimed_TOD38);
    require(msg.sender == owner_TOD38);
    owner_TOD38.transfer(reward_TOD38);
    reward_TOD38 = msg.value;
    }

    function claimReward_TOD20(uint256 submission) public {
    require (!claimed_TOD20);
    require(submission < 10);
    msg.sender.transfer(reward_TOD20);
    claimed_TOD20 = true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success);
    address winner_tmstmp35;
    function play_tmstmp35(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp35 = msg.sender;}}

    function getReward_TOD7() payable public{
    winner_TOD7.transfer(msg.value);
    }

}