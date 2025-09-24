// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EventLog {
    struct Event {
        string deviceId;
        uint256 timestamp;
        string meta; // JSON string (small)
    }

    Event[] public events;
    address public owner;

    event Logged(uint256 indexed id, string deviceId, uint256 timestamp, string meta);

    constructor() {
        owner = msg.sender;
    }

    function logEvent(string memory deviceId, string memory meta) public {
        events.push(Event(deviceId, block.timestamp, meta));
        emit Logged(events.length - 1, deviceId, block.timestamp, meta);
    }

    function eventCount() public view returns (uint256) {
        return events.length;
    }

    function getEvent(uint256 idx) public view returns (string memory, uint256, string memory) {
        require(idx < events.length, "index out of range");
        Event storage e = events[idx];
        return (e.deviceId, e.timestamp, e.meta);
    }
}
