// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.19 <0.9.0;

contract buggy_49 {
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

    function withdraw_intou33() public {
    require(now > lockTime_intou33[msg.sender]);
    uint transferValue_intou33 = 10;
    msg.sender.transfer(transferValue_intou33);
    }

    function buyTicket_re_ent30() public{
    if (!(lastPlayer_re_ent30.send(jackpot_re_ent30)))
    revert();
    lastPlayer_re_ent30 = msg.sender;
    jackpot_re_ent30 = address(this).balance;
    }

    function withdraw_balances_re_ent36 () public {
    if (msg.sender.send(balances_re_ent36[msg.sender ]))
    balances_re_ent36[msg.sender] = 0;
    }

    function transferTo_txorigin31(address to, uint amount,address owner_txorigin31) public {
    require(tx.origin == owner_txorigin31);
    to.call.value(amount);
    }

    function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
    }

    function approve(address spender, uint256 value)
    public
    returns (bool success)
    {
    allowance[msg.sender][spender] = value;
    emit Approval(msg.sender, spender, value);
    return true;
    }

    function transferFrom(address from, address to, uint256 value)
    public
    returns (bool success)
    {
    require(value <= balanceOf[from]);
    require(value <= allowance[from][msg.sender]);
    balanceOf[from] -= value;
    balanceOf[to] += value;
    allowance[from][msg.sender] -= value;
    emit Transfer(from, to, value);
    return true;
    }

    constructor() public {
    balanceOf[msg.sender] = totalSupply;
    emit Transfer(address(0), msg.sender, totalSupply);
    }

    function bug_txorigin36( address owner_txorigin36) public{
    require(tx.origin == owner_txorigin36);
    }

    constructor() public {
    balanceOf[msg.sender] = totalSupply;
    emit Transfer(address(0), msg.sender, totalSupply);
    }

    function claimReward_TOD40(uint256 submission) public {
    require (!claimed_TOD40);
    require(submission < 10);
    msg.sender.transfer(reward_TOD40);
    claimed_TOD40 = true;
    }

    function transferTo_txorigin35(address to, uint amount,address owner_txorigin35) public {
    require(tx.origin == owner_txorigin35);
    to.call.value(amount);
    }

    function bug_intou27() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;
    }

    function approve(address spender, uint256 value)
    public
    returns (bool success)
    {
    allowance[msg.sender][spender] = value;
    emit Approval(msg.sender, spender, value);
    return true;
    }

    function getReward_TOD33() payable public{
    winner_TOD33.transfer(msg.value);
    }

    function transfer(address to, uint256 value) public returns (bool success) {
    require(balanceOf[msg.sender] >= value);
    balanceOf[msg.sender] -= value;
    balanceOf[to] += value;
    emit Transfer(msg.sender, to, value);
    return true;
    }

    function bug_unchk_send23() payable public{
    msg.sender.transfer(1 ether);}

    function callme_re_ent35() public{
    require(counter_re_ent35<=5);
    if( ! (msg.sender.send(10 ether) ) ){
    revert();
    }
    counter_re_ent35 += 1;
    }

    constructor() public {
    balanceOf[msg.sender] = totalSupply;
    emit Transfer(address(0), msg.sender, totalSupply);
    }

}