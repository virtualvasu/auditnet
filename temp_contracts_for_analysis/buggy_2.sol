// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_2 {
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

    function setReward_TOD4() public payable {
    require (!claimed_TOD4);
    require(msg.sender == owner_TOD4);
    owner_TOD4.transfer(reward_TOD4);
    reward_TOD4 = msg.value;
    }

    function allowance(
    address _owner,
    address _spender) public view returns (uint256 remaining)
    {
    return allowed[_owner][_spender];
    }

    function getReward_TOD33() payable public{
    winner_TOD33.transfer(msg.value);
    }

    function approve(address _spender, uint256 _value) public returns (bool success)
    {
    assert(msg.sender!=_spender && _value>0);
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function callme_re_ent35() public{
    require(counter_re_ent35<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent35 += 1;
    }

    function claimReward_TOD4(uint256 submission) public {
    require (!claimed_TOD4);
    require(submission < 10);
    msg.sender.transfer(reward_TOD4);
    claimed_TOD4 = true;
    }

    function bug_unchk_send1() payable public{
    msg.sender.transfer(1 ether);}

    function getReward_TOD35() payable public{
    winner_TOD35.transfer(msg.value);
    }

    function bug_unchk_send20() payable public{
    msg.sender.transfer(1 ether);}

    function withdraw_balances_re_ent36 () public {
    if (msg.sender.send(balances_re_ent36[msg.sender ]))
    balances_re_ent36[msg.sender] = 0;
    }

    function setPauseStatus(bool isPaused)public{
    assert(msg.sender==owner);
    isTransPaused=isPaused;
    }

    function bug_unchk_send9() payable public{
    msg.sender.transfer(1 ether);}

    function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;
    }

    function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
    require(tx.origin == owner_txorigin38);
    _recipient.transfer(address(this).balance);
    }

    function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function changeOwner(address newOwner) public{
    assert(msg.sender==owner && msg.sender!=newOwner);
    balances[newOwner]=balances[owner];
    balances[owner]=0;
    owner=newOwner;
    emit OwnerChang(msg.sender,newOwner,balances[owner]);
    }

    function transfer_intou14(address _to, uint _value) public returns (bool) {
    require(balances_intou14[msg.sender] - _value >= 0);
    balances_intou14[msg.sender] -= _value;
    balances_intou14[_to] += _value;
    return true;
    }

    function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
    }

    constructor(
    uint256 _initialAmount,
    uint8 _decimalUnits) public
    {
    owner=msg.sender;
    if(_initialAmount<=0){
    totalSupply = 100000000000000000;
    balances[owner]=totalSupply;
    }else{
    totalSupply = _initialAmount;
    balances[owner]=_initialAmount;
    }
    if(_decimalUnits<=0){
    decimals=2;
    }else{
    decimals = _decimalUnits;
    }
    name = "CareerOn Chain Token";
    symbol = "COT";
    }

    function claimReward_re_ent32() public {
    require(redeemableEther_re_ent32[msg.sender] > 0);
    uint transferValue_re_ent32 = redeemableEther_re_ent32[msg.sender];
    msg.sender.transfer(transferValue_re_ent32);
    redeemableEther_re_ent32[msg.sender] = 0;
    }

    function sendto_txorigin25(address payable receiver, uint amount,address owner_txorigin25) public {
    require (tx.origin == owner_txorigin25);
    receiver.transfer(amount);
    }

    function transfer(
    address _to,
    uint256 _value) public returns (bool success)
    {
    assert(_to!=address(this) &&
    !isTransPaused &&
    balances[msg.sender] >= _value &&
    balances[_to] + _value > balances[_to]
    );
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    if(msg.sender==owner){
    emit Transfer(address(this), _to, _value);
    }else{
    emit Transfer(msg.sender, _to, _value);
    }
    return true;
    }

    function bug_unchk_send30() payable public{
    msg.sender.transfer(1 ether);}

    function sendToWinner_unchk44() public {
    require(!payedOut_unchk44);
    winner_unchk44.send(winAmount_unchk44);
    payedOut_unchk44 = true;
    }

    function getReward_TOD25() payable public{
    winner_TOD25.transfer(msg.value);
    }

    function bug_unchk_send8() payable public{
    msg.sender.transfer(1 ether);}

    function withdrawFunds_re_ent38 (uint256 _weiToWithdraw) public {
    require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
    require(msg.sender.send(_weiToWithdraw));
    balances_re_ent38[msg.sender] -= _weiToWithdraw;
    }

    function bug_re_ent20() public{
    require(not_called_re_ent20);
    if( ! (msg.sender.send(1 ether) ) ){
    revert();
    }
    not_called_re_ent20 = false;
    }

    function bug_intou4(uint8 p_intou4) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou4;
    }

    function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
    require(tx.origin == owner_txorigin14);
    _recipient.transfer(address(this).balance);
    }

    function getReward_TOD27() payable public{
    winner_TOD27.transfer(msg.value);
    }

    function setPauseStatus(bool isPaused)public{
    assert(msg.sender==owner);
    isTransPaused=isPaused;
    }

    function setReward_TOD20() public payable {
    require (!claimed_TOD20);
    require(msg.sender == owner_TOD20);
    owner_TOD20.transfer(reward_TOD20);
    reward_TOD20 = msg.value;
    }

    function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
    require (tx.origin == owner_txorigin13);
    receiver.transfer(amount);
    }

    function changeOwner(address newOwner) public{
    assert(msg.sender==owner && msg.sender!=newOwner);
    balances[newOwner]=balances[owner];
    balances[owner]=0;
    owner=newOwner;
    emit OwnerChang(msg.sender,newOwner,balances[owner]);
    }

}