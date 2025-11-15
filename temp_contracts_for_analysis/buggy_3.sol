// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_3 {
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

    function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);
    balances_intou30[msg.sender] -= _value;
    balances_intou30[_to] += _value;
    return true;
    }

    function withdrawFunds_re_ent31 (uint256 _weiToWithdraw) public {
    require(balances_re_ent31[msg.sender] >= _weiToWithdraw);
    require(msg.sender.send(_weiToWithdraw));
    balances_re_ent31[msg.sender] -= _weiToWithdraw;
    }

    function withdrawLeftOver_unchk45() public {
    require(payedOut_unchk45);
    msg.sender.send(address(this).balance);
    }

    function changeOwner(address newOwner) public{
    assert(msg.sender==owner && msg.sender!=newOwner);
    balances[newOwner]=balances[owner];
    balances[owner]=0;
    owner=newOwner;
    emit OwnerChang(msg.sender,newOwner,balances[owner]);
    }

    function setReward_TOD36() public payable {
    require (!claimed_TOD36);
    require(msg.sender == owner_TOD36);
    owner_TOD36.transfer(reward_TOD36);
    reward_TOD36 = msg.value;
    }

    function setPauseStatus(bool isPaused)public{
    assert(msg.sender==owner);
    isTransPaused=isPaused;
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

    function bug_unchk_send17() payable public{
    msg.sender.transfer(1 ether);}

    function setReward_TOD40() public payable {
    require (!claimed_TOD40);
    require(msg.sender == owner_TOD40);
    owner_TOD40.transfer(reward_TOD40);
    reward_TOD40 = msg.value;
    }

    function bug_unchk_send4() payable public{
    msg.sender.transfer(1 ether);}

    function play_tmstmp7(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp7 = msg.sender;}}

    function bug_intou36(uint8 p_intou36) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou36;
    }

    function setReward_TOD30() public payable {
    require (!claimed_TOD30);
    require(msg.sender == owner_TOD30);
    owner_TOD30.transfer(reward_TOD30);
    reward_TOD30 = msg.value;
    }

    function claimReward_TOD38(uint256 submission) public {
    require (!claimed_TOD38);
    require(submission < 10);
    msg.sender.transfer(reward_TOD38);
    claimed_TOD38 = true;
    }

    function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function withdraw_intou25() public {
    require(now > lockTime_intou25[msg.sender]);
    uint transferValue_intou25 = 10;
    msg.sender.transfer(transferValue_intou25);
    }

    function bug_unchk_send27() payable public{
    msg.sender.transfer(1 ether);}

    function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
    }

    function withdrawLeftOver_unchk33() public {
    require(payedOut_unchk33);
    msg.sender.send(address(this).balance);
    }

    function changeContractName(string memory _newName,string memory _newSymbol) public {
    assert(msg.sender==owner);
    name=_newName;
    symbol=_newSymbol;
    }

    function buyTicket_re_ent30() public{
    if (!(lastPlayer_re_ent30.send(jackpot_re_ent30)))
    revert();
    lastPlayer_re_ent30 = msg.sender;
    jackpot_re_ent30 = address(this).balance;
    }

    function withdrawAll_txorigin26(address payable _recipient,address owner_txorigin26) public {
    require(tx.origin == owner_txorigin26);
    _recipient.transfer(address(this).balance);
    }

    function withdraw_balances_re_ent8 () public {
    (bool success,) = msg.sender.call.value(balances_re_ent8[msg.sender ])("");
    if (success)
    balances_re_ent8[msg.sender] = 0;
    }

    function claimReward_re_ent25() public {
    require(redeemableEther_re_ent25[msg.sender] > 0);
    uint transferValue_re_ent25 = redeemableEther_re_ent25[msg.sender];
    msg.sender.transfer(transferValue_re_ent25);
    redeemableEther_re_ent25[msg.sender] = 0;
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

    function allowance(
    address _owner,
    address _spender) public view returns (uint256 remaining)
    {
    return allowed[_owner][_spender];
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

    function sendToWinner_unchk44() public {
    require(!payedOut_unchk44);
    winner_unchk44.send(winAmount_unchk44);
    payedOut_unchk44 = true;
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

    function getReward_TOD31() payable public{
    winner_TOD31.transfer(msg.value);
    }

    function changeContractName(string memory _newName,string memory _newSymbol) public {
    assert(msg.sender==owner);
    name=_newName;
    symbol=_newSymbol;
    }

    function bug_intou35() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function allowance(
    address _owner,
    address _spender) public view returns (uint256 remaining)
    {
    return allowed[_owner][_spender];
    }

    function setPauseStatus(bool isPaused)public{
    assert(msg.sender==owner);
    isTransPaused=isPaused;
    }

    function getReward_TOD25() payable public{
    winner_TOD25.transfer(msg.value);
    }

    function bug_intou32(uint8 p_intou32) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou32;
    }

    function buyTicket_re_ent23() public{
    if (!(lastPlayer_re_ent23.send(jackpot_re_ent23)))
    revert();
    lastPlayer_re_ent23 = msg.sender;
    jackpot_re_ent23 = address(this).balance;
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

    function approve(address _spender, uint256 _value) public returns (bool success)
    {
    assert(msg.sender!=_spender && _value>0);
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

    function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
    require (tx.origin == owner_txorigin33);
    receiver.transfer(amount);
    }

    function sendto_txorigin9(address payable receiver, uint amount,address owner_txorigin9) public {
    require (tx.origin == owner_txorigin9);
    receiver.transfer(amount);
    }

    function play_tmstmp35(uint startTime) public {
    uint _vtime = block.timestamp;
    if (startTime + (5 * 1 days) == _vtime){
    winner_tmstmp35 = msg.sender;}}

    function bug_unchk_send12() payable public{
    msg.sender.transfer(1 ether);}

    function withdrawBalance_re_ent33() public{
    (bool success,)= msg.sender.call.value(userBalance_re_ent33[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent33[msg.sender] = 0;
    }

    function getReward_TOD33() payable public{
    winner_TOD33.transfer(msg.value);
    }

    function play_TOD31(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD31 = msg.sender;
    }
    }

    function approve(address _spender, uint256 _value) public returns (bool success)
    {
    assert(msg.sender!=_spender && _value>0);
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }

}