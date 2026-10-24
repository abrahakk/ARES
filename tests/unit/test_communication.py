from __future__ import annotations

from robots.communication.communication_manager import CommunicationManager
from robots.communication.message import Message, MessageType
from robots.communication.protocol import CommunicationProtocol, ProtocolConfig




def test_message_round_trips_through_dict():
    message = Message(
        sender_id="robot_01",
        message_type=MessageType.FRONTIER_DISCOVERED,
        payload={"x": 1.0, "y": 2.0},
        recipient_id="robot_02",
    )
    restored = Message.from_dict(message.to_dict())
    assert restored.sender_id == message.sender_id
    assert restored.message_type == MessageType.FRONTIER_DISCOVERED
    assert restored.payload == {"x": 1.0, "y": 2.0}
    assert restored.recipient_id == "robot_02"


def test_message_has_unique_ids():
    m1 = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    m2 = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    assert m1.message_id != m2.message_id


def test_message_size_bytes_is_positive():
    message = Message(sender_id="a", message_type=MessageType.EMERGENCY, payload={"reason": "test"})
    assert message.size_bytes > 0




def test_protocol_delivers_within_range_with_perfect_channel():
    protocol = CommunicationProtocol(ProtocolConfig(packet_loss_probability=0.0, max_range_meters=10.0))
    protocol.reset_step()
    message = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    assert protocol.should_deliver(message, sender_pos=(0.0, 0.0), recipient_pos=(1.0, 0.0)) is True


def test_protocol_drops_messages_out_of_range():
    protocol = CommunicationProtocol(ProtocolConfig(max_range_meters=1.0))
    protocol.reset_step()
    message = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    assert protocol.should_deliver(message, sender_pos=(0.0, 0.0), recipient_pos=(5.0, 0.0)) is False


def test_protocol_enforces_bandwidth_budget():
    protocol = CommunicationProtocol(ProtocolConfig(max_bandwidth_bytes_per_step=1, packet_loss_probability=0.0))
    protocol.reset_step()
    message = Message(sender_id="a", message_type=MessageType.HEARTBEAT, payload={"data": "x" * 100})

    assert protocol.should_deliver(message, sender_pos=(0.0, 0.0), recipient_pos=(0.0, 0.0)) is False


def test_protocol_always_drops_with_full_packet_loss():
    protocol = CommunicationProtocol(ProtocolConfig(packet_loss_probability=1.0))
    protocol.reset_step()
    message = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    assert protocol.should_deliver(message, sender_pos=(0.0, 0.0), recipient_pos=(0.0, 0.0)) is False


def test_protocol_reset_step_clears_bandwidth_usage():
    protocol = CommunicationProtocol(ProtocolConfig(max_bandwidth_bytes_per_step=10_000, packet_loss_probability=0.0))
    protocol.reset_step()
    message = Message(sender_id="a", message_type=MessageType.HEARTBEAT)
    assert protocol.should_deliver(message, (0.0, 0.0), (0.0, 0.0)) is True
    protocol.reset_step()

    assert protocol._bandwidth_used_this_step == 0



def test_communication_manager_builds_message_with_correct_sender():
    manager = CommunicationManager(robot_id="robot_01")
    message = manager.build_message(payload={"x": 1.0}, message_type=MessageType.FRONTIER_DISCOVERED)
    assert message.sender_id == "robot_01"
    assert message.payload == {"x": 1.0}


def test_communication_manager_send_calls_attached_hook():
    sent_messages = []
    manager = CommunicationManager(robot_id="robot_01")
    manager.attach_send_hook(sent_messages.append)

    message = manager.build_message(payload={})
    manager.send(message)

    assert len(sent_messages) == 1
    assert sent_messages[0] is message


def test_communication_manager_send_without_hook_does_not_raise():
    manager = CommunicationManager(robot_id="robot_01")
    message = manager.build_message(payload={})
    manager.send(message)  
