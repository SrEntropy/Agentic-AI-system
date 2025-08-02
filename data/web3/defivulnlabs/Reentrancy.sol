// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Reentrancy {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        require(balances[msg.sender] > 0, "No balance");
        (bool success, ) = msg.sender.call{value: balances[msg.sender]}("");
        require(success, "Withdraw failed");
        balances[msg.sender] = 0;
    }
}
