// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_15 {
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

    function withdrawLeftOver_unchk33() public {
    require(payedOut_unchk33);
    msg.sender.send(address(this).balance);
    }

    function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function approveAndCall(address _spender, uint256 _value, bytes memory _extraData)
    public
    returns (bool success) {
    tokenRecipient spender = tokenRecipient(_spender);
    if (approve(_spender, _value)) {
    spender.receiveApproval(msg.sender, _value, address(this), _extraData);
    return true;
    }
    }

    function claimReward_TOD8(uint256 submission) public {
    require (!claimed_TOD8);
    require(submission < 10);
    msg.sender.transfer(reward_TOD8);
    claimed_TOD8 = true;
    }

    function bug_txorigin36( address owner_txorigin36) public{
    require(tx.origin == owner_txorigin36);
    }

    function receiveApproval(address _from, uint256 _value, address _token, bytes calldata _extraData) external;
    }
    contract MD{
    function bug_txorigin20(address owner_txorigin20) public{
    require(tx.origin == owner_txorigin20);
    }
    string public name;
    function bug_txorigin32( address owner_txorigin32) public{
    require(tx.origin == owner_txorigin32);
    }
    string public symbol;
    function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
    require(tx.origin == owner_txorigin38);
    _recipient.transfer(address(this).balance);
    }
    uint8 public decimals = 18;
    function bug_txorigin4(address owner_txorigin4) public{
    require(tx.origin == owner_txorigin4);
    }
    uint256 public totalSupply;
    function transferTo_txorigin7(address to, uint amount,address owner_txorigin7) public {
    require(tx.origin == owner_txorigin7);
    to.call.value(amount);
    }
    mapping (address => uint256) public balanceOf;
    function transferTo_txorigin23(address to, uint amount,address owner_txorigin23) public {
    require(tx.origin == owner_txorigin23);
    to.call.value(amount);
    }
    mapping (address => mapping (address => uint256)) public allowance;
    function transferTo_txorigin27(address to, uint amount,address owner_txorigin27) public {
    require(tx.origin == owner_txorigin27);
    to.call.value(amount);
    }
    event Transfer(address indexed from, address indexed to, uint256 value);
    function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
    require(tx.origin == owner_txorigin31);
    to.call.value(amount);
    }
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    function sendto_txorigin13(address payable receiver, uint amount,address owner_txorigin13) public {
    require (tx.origin == owner_txorigin13);
    receiver.transfer(amount);
    }
    event Burn(address indexed from, uint256 value);
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
    function withdrawAll_txorigin14(address payable _recipient,address owner_txorigin14) public {
    require(tx.origin == owner_txorigin14);
    _recipient.transfer(address(this).balance);
    }
    function _transfer(address _from, address _to, uint _value) internal {
    require(_to != address(0x0));
    require(balanceOf[_from] >= _value);
    require(balanceOf[_to] + _value >= balanceOf[_to]);
    uint previousBalances = balanceOf[_from] + balanceOf[_to];
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }
    function withdrawAll_txorigin30(address payable _recipient,address owner_txorigin30) public {
    require(tx.origin == owner_txorigin30);
    _recipient.transfer(address(this).balance);
    }
    function transfer(address _to, uint256 _value) public returns (bool success) {
    _transfer(msg.sender, _to, _value);
    return true;
    }
    function bug_txorigin8(address owner_txorigin8) public{
    require(tx.origin == owner_txorigin8);
    }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    require(_value <= allowance[_from][msg.sender]);
    allowance[_from][msg.sender] -= _value;
    _transfer(_from, _to, _value);
    return true;
    }
    function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
    require(tx.origin == owner_txorigin39);
    to.call.value(amount);
    }
    function approve(address _spender, uint256 _value) public
    returns (bool success) {
    allowance[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }
    function bug_txorigin36( address owner_txorigin36) public{
    require(tx.origin == owner_txorigin36);
    }
    function approveAndCall(address _spender, uint256 _value, bytes memory _extraData)
    public
    returns (bool success) {
    tokenRecipient spender = tokenRecipient(_spender);
    if (approve(_spender, _value)) {
    spender.receiveApproval(msg.sender, _value, address(this), _extraData);
    return true;
    }
    }
    function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
    require(tx.origin == owner_txorigin35);
    to.call.value(amount);
    }
    function burn(uint256 _value) public returns (bool success) {
    require(balanceOf[msg.sender] >= _value);
    balanceOf[msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(msg.sender, _value);
    return true;
    }
    function bug_txorigin40(address owner_txorigin40) public{
    require(tx.origin == owner_txorigin40);
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
    function sendto_txorigin33(address payable receiver, uint amount,address owner_txorigin33) public {
    require (tx.origin == owner_txorigin33);
    receiver.transfer(amount);
    }
    }

    function getReward_TOD7() payable public{
    winner_TOD7.transfer(msg.value);
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

    function setReward_TOD30() public payable {
    require (!claimed_TOD30);
    require(msg.sender == owner_TOD30);
    owner_TOD30.transfer(reward_TOD30);
    reward_TOD30 = msg.value;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    require(_value <= allowance[_from][msg.sender]);
    allowance[_from][msg.sender] -= _value;
    _transfer(_from, _to, _value);
    return true;
    }

    function withdrawBalance_re_ent40() public{
    (bool success,)=msg.sender.call.value(userBalance_re_ent40[msg.sender])("");
    if( ! success ){
    revert();
    }
    userBalance_re_ent40[msg.sender] = 0;
    }

    function setReward_TOD8() public payable {
    require (!claimed_TOD8);
    require(msg.sender == owner_TOD8);
    owner_TOD8.transfer(reward_TOD8);
    reward_TOD8 = msg.value;
    }

    function callme_re_ent7() public{
    require(counter_re_ent7<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent7 += 1;
    }

    function setReward_TOD32() public payable {
    require (!claimed_TOD32);
    require(msg.sender == owner_TOD32);
    owner_TOD32.transfer(reward_TOD32);
    reward_TOD32 = msg.value;
    }

    function _transfer(address _from, address _to, uint _value) internal {
    require(_to != address(0x0));
    require(balanceOf[_from] >= _value);
    require(balanceOf[_to] + _value >= balanceOf[_to]);
    uint previousBalances = balanceOf[_from] + balanceOf[_to];
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    emit Transfer(_from, _to, _value);
    assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }

    function bug_txorigin4(address owner_txorigin4) public{
    require(tx.origin == owner_txorigin4);
    }

    function bug_unchk_send13() payable public{
    msg.sender.transfer(1 ether);}

    function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
    require(_value <= allowance[_from][msg.sender]);
    allowance[_from][msg.sender] -= _value;
    _transfer(_from, _to, _value);
    return true;
    }

    function play_tmstmp30(uint startTime) public {
    if (startTime + (5 * 1 days) == block.timestamp){
    winner_tmstmp30 = msg.sender;}}

    function bug_unchk_send23() payable public{
    msg.sender.transfer(1 ether);}

    function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
    }

    function getReward_TOD27() payable public{
    winner_TOD27.transfer(msg.value);
    }

    function increaseLockTime_intou33(uint _secondsToIncrease) public {
    lockTime_intou33[msg.sender] += _secondsToIncrease;
    }

    function bug_unchk_send7() payable public{
    msg.sender.transfer(1 ether);}

    function bug_intou8(uint8 p_intou8) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou8;
    }

    function bug_txorigin32( address owner_txorigin32) public{
    require(tx.origin == owner_txorigin32);
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

    function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
    uint64 subpot_unchk46 = 3 ether;
    winner_unchk46.send(subpot_unchk46);
    subpot_unchk46= 0;
    }

    function getReward_TOD31() payable public{
    winner_TOD31.transfer(msg.value);
    }

    function bug_intou39() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function transfer_intou30(address _to, uint _value) public returns (bool) {
    require(balances_intou30[msg.sender] - _value >= 0);
    balances_intou30[msg.sender] -= _value;
    balances_intou30[_to] += _value;
    return true;
    }

    function getReward_TOD13() payable public{
    winner_TOD13.transfer(msg.value);
    }

    function play_TOD35(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD35 = msg.sender;
    }
    }

    function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
    require(tx.origin == owner_txorigin35);
    to.call.value(amount);
    }

    function bug_intou40(uint8 p_intou40) public{
    uint8 vundflw1=0;
    vundflw1 = vundflw1 + p_intou40;
    }

    function withdrawAll_txorigin38(address payable _recipient,address owner_txorigin38) public {
    require(tx.origin == owner_txorigin38);
    _recipient.transfer(address(this).balance);
    }

    function play_TOD33(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD33 = msg.sender;
    }
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

    function transferTo_txorigin39(address to, uint amount,address owner_txorigin39) public {
    require(tx.origin == owner_txorigin39);
    to.call.value(amount);
    }

    function play_TOD13(bytes32 guess) public{
    if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
    winner_TOD13 = msg.sender;
    }
    }

    function bug_unchk39(address payable addr) public
    {addr.send (4 ether); }

    function burn(uint256 _value) public returns (bool success) {
    require(balanceOf[msg.sender] >= _value);
    balanceOf[msg.sender] -= _value;
    totalSupply -= _value;
    emit Burn(msg.sender, _value);
    return true;
    }

}