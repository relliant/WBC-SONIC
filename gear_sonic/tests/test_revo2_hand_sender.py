import ast
from pathlib import Path

import msgpack
import numpy as np
import pytest

from gear_sonic.utils.teleop.revo2_hand_sender import (
    Revo2HandSender,
    Revo2PoseProfile,
    Revo2TeleopError,
    command_packet,
    explicit_tcp_ipv4_endpoint,
    parse_feedback,
    require_fresh_trigger_source,
)


PINNED_PROFILE_PATH = (
    Path(__file__).parents[1]
    / "config"
    / "revo2"
    / "mondo_g1_6514_pico_pose_profile.v1.json"
)


class FakeSocket:
    def __init__(self, kind, module):
        self.kind = kind
        self.module = module
        self.options = {}
        self.bound = None
        self.connected = None
        self.sent = []
        self.received = []
        self.closed = False
        self.fail_send = False

    def setsockopt(self, option, value):
        self.options[option] = value

    def bind(self, endpoint):
        self.bound = endpoint

    def connect(self, endpoint):
        self.connected = endpoint

    def recv(self, flags=0):
        if not self.received:
            raise self.module.Again
        return self.received.pop(0)

    def send(self, payload, flags=0):
        if self.fail_send:
            raise self.module.Again
        self.sent.append(payload)

    def close(self, linger=0):
        self.closed = True


class FakeZMQ:
    class Again(Exception):
        pass

    XPUB = "XPUB"
    SUB = "SUB"
    LINGER = "LINGER"
    SNDHWM = "SNDHWM"
    SNDTIMEO = "SNDTIMEO"
    XPUB_NODROP = "XPUB_NODROP"
    SUBSCRIBE = "SUBSCRIBE"
    CONFLATE = "CONFLATE"
    DONTWAIT = "DONTWAIT"

    def __init__(self):
        self.sockets = {}
        self.terminated = False

    def Context(self):
        return self

    def socket(self, kind):
        socket = FakeSocket(kind, self)
        self.sockets[kind] = socket
        return socket

    def term(self):
        self.terminated = True


def profile(max_close_ratio=1.0):
    return Revo2PoseProfile(
        open_poses={"left": [0.0] * 6, "right": [0.1] * 6},
        closed_poses={"left": [1.0] * 6, "right": [0.9] * 6},
        max_close_ratio=max_close_ratio,
    )


def feedback(
    session_id="relay-session",
    *,
    seq=0,
    agent_ns=1_000_000,
    measured_ns=900_000,
    mode="stop",
    state="disarmed",
    valid=False,
    fault="",
):
    hand = {
        "state": state,
        "last_received_seq": -1,
        "last_applied_seq": -1,
        "last_applied_control_frame_index": -1,
        "received_agent_monotonic_ns": 0,
        "applied_agent_monotonic_ns": 0,
        "requested": [0.0] * 6,
        "applied": [0.0] * 6,
        "clipped_mask": [False] * 6,
        "position_raw": [0] * 6,
        "position_norm": [0.0] * 6,
        "speed_raw": [0] * 6,
        "current_raw": [0] * 6,
        "motor_state": [0] * 6,
        "measured_agent_monotonic_ns": measured_ns,
        "valid": valid,
        "fault": "",
        "reject_reason": "command_lease_expired" if not valid else "",
    }
    return {
        "schema": "revo2.feedback.v3",
        "session_id": session_id,
        "agent_boot_id": "agent-boot-1",
        "feedback_seq": seq,
        "agent_monotonic_ns": agent_ns,
        "accepted_mode": mode,
        "last_mode_seq": -1,
        "lease_valid": False,
        "lease_remaining_ms": 0,
        "mode_reject_reason": "command_lease_expired" if not valid else "",
        "global_fault": fault,
        "capabilities": {
            "direct_enable": True,
            "source_diagnostic_only": True,
            "body_control_interlock": False,
        },
        "actuated_sides": ["left", "right"],
        "hands": {"left": dict(hand), "right": dict(hand)},
    }


def decode_all(fake):
    return [msgpack.unpackb(item, raw=False) for item in fake.sockets[fake.XPUB].sent]


def make_sender(fake=None, clock=None):
    fake = fake or FakeZMQ()
    sender = Revo2HandSender(
        "tcp://192.0.2.10:5560",
        "tcp://192.0.2.20:5561",
        session_id="session-1",
        profile=profile(),
        permit_actuation=True,
        feedback_timeout_ms=500,
        zmq_module=fake,
        clock_ns=clock or (lambda: 0),
    ).start()
    return sender, fake


def update(sender, timestamp, left=0.0, right=0.0, *, received=None, enabled=True):
    return sender.update(
        source_timestamp_ns=timestamp,
        source_received_monotonic_ns=timestamp if received is None else received,
        left_trigger=left,
        right_trigger=right,
        enabled=enabled,
    )


def test_profile_maps_one_scalar_per_hand_to_proportional_whole_hand_pose():
    result = profile(max_close_ratio=0.8).targets(0.5, 0.25)
    assert np.allclose(result["left"], [0.4] * 6)
    assert np.allclose(result["right"], [0.26] * 6)
    assert profile().targets(-1.0, 2.0) == {"left": [0.0] * 6, "right": [0.9] * 6}
    with pytest.raises(Revo2TeleopError):
        profile().targets(float("nan"), 0.0)
    with pytest.raises(Revo2TeleopError, match="numeric values"):
        Revo2PoseProfile(
            open_poses={"left": [False] * 6, "right": [0.0] * 6},
            closed_poses={"left": [1.0] * 6, "right": [1.0] * 6},
        )
    with pytest.raises(Revo2TeleopError, match="max_close_ratio"):
        profile(max_close_ratio=True)


def test_mondo_g1_6514_pinned_profile_loads_and_interpolates():
    pinned = Revo2PoseProfile.from_json(PINNED_PROFILE_PATH)
    open_pose = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]
    closed_pose = [0.5, 0.9, 0.85, 0.85, 0.85, 0.85]
    midpoint = [0.3, 0.9, 0.425, 0.425, 0.425, 0.425]

    assert pinned.max_close_ratio == 1.0
    for side in ("left", "right"):
        assert np.allclose(pinned.open_poses[side], open_pose)
        assert np.allclose(pinned.closed_poses[side], closed_pose)
    assert pinned.targets(0.0, 0.0) == {"left": open_pose, "right": open_pose}
    for target in pinned.targets(0.5, 0.5).values():
        assert np.allclose(target, midpoint)
    assert pinned.targets(1.0, 1.0) == {"left": closed_pose, "right": closed_pose}


def test_revo2_requires_fresh_xrt_trigger_samples_and_explicit_ipv4_endpoints():
    require_fresh_trigger_source("xrt")
    with pytest.raises(Revo2TeleopError, match="controller-origin sample timestamps"):
        require_fresh_trigger_source("isaac-teleop")
    assert explicit_tcp_ipv4_endpoint("tcp://192.168.123.79:5560", "test") == (
        "tcp://192.168.123.79:5560"
    )
    for value in (
        "tcp://0.0.0.0:5560",
        "tcp://localhost:5560",
        "tcp://192.168.123.079:5560",
        "tcp://192.168.123.79:80",
        "ipc:///tmp/revo2",
    ):
        with pytest.raises(Revo2TeleopError):
            explicit_tcp_ipv4_endpoint(value, "test")


def test_command_packet_matches_unified_relay_exact_fields_without_controller_id():
    expected_fields = {
        "schema",
        "session_id",
        "seq",
        "control_frame_index",
        "sender_monotonic_ns",
        "lease_ms",
        "source",
        "mode",
        "actuated_sides",
        "hands",
    }
    for mode in ("disarm", "arm", "enable", "hold", "stop"):
        targets = {"left": [0.2] * 6, "right": [0.8] * 6} if mode == "enable" else None
        packet = command_packet(
            mode,
            session_id="s",
            seq=3,
            control_frame_index=4,
            sender_monotonic_ns=5,
            targets=targets,
        )
        assert set(packet) == expected_fields
        assert "controller_id" not in packet
        assert packet["schema"] == "revo2.command.v3"
        assert packet["source"] == "glove" and packet["lease_ms"] == 200
        assert packet["actuated_sides"] == ["left", "right"]
        assert all(
            packet["hands"][side]["valid"] == (mode == "enable")
            for side in ("left", "right")
        )


def test_fresh_release_sample_sends_enable_directly_without_arm_or_feedback():
    sender, fake = make_sender()
    assert update(sender, 1, 0.25, 0.75, received=0) == "await_trigger_release"
    assert decode_all(fake) == []
    assert update(sender, 2, 0.0, 0.0, received=0) == "enable"
    packets = decode_all(fake)
    assert [packet["mode"] for packet in packets] == ["enable"]
    assert packets[0]["hands"]["left"]["target"] == [0.0] * 6
    assert packets[0]["hands"]["right"]["target"] == [0.1] * 6


def test_duplicate_timestamp_does_not_send_or_extend_deadline():
    sender, fake = make_sender()
    assert update(sender, 1, received=0) == "enable"
    deadline = sender.target_deadline_ns
    assert update(sender, 1, 1.0, 1.0, received=100_000_000) is None
    assert sender.target_deadline_ns == deadline
    assert [packet["mode"] for packet in decode_all(fake)] == ["enable"]


def test_expiry_stops_once_and_new_fresh_release_sample_can_resume():
    now = {"value": 0}
    sender, fake = make_sender(clock=lambda: now["value"])
    assert update(sender, 1, received=0) == "enable"
    now["value"] = 149_999_999
    assert sender.tick() is None
    now["value"] = 150_000_000
    assert sender.tick() == "stop"
    assert sender.tick(now_ns=200_000_000) is None
    assert [packet["mode"] for packet in decode_all(fake)] == ["enable", "stop"]

    now["value"] = 200_000_001
    assert update(sender, 2, received=now["value"]) == "enable"
    assert [packet["mode"] for packet in decode_all(fake)] == ["enable", "stop", "enable"]


def test_stale_sample_stops_without_permanently_faulting_sender():
    now = {"value": 200_000_000}
    sender, fake = make_sender(clock=lambda: now["value"])
    assert update(sender, 1, received=0) == "stale"
    assert sender.latched_fault is None and decode_all(fake) == []
    now["value"] += 1
    assert update(sender, 2, received=now["value"]) == "enable"


def test_source_timestamp_rollback_stops_and_latches():
    sender, fake = make_sender()
    assert update(sender, 2, received=0) == "enable"
    with pytest.raises(Revo2TeleopError, match="moved backwards"):
        update(sender, 1, received=0)
    assert [packet["mode"] for packet in decode_all(fake)] == ["enable", "stop"]
    assert sender.latched_fault is not None


def test_queue_saturation_does_not_advance_sequence_or_retry_old_target():
    sender, fake = make_sender()
    command = fake.sockets[fake.XPUB]
    command.fail_send = True
    assert update(sender, 1, received=0) is None
    assert sender.seq == 0 and decode_all(fake) == [] and not sender.target_active
    command.fail_send = False
    assert update(sender, 1, received=0) is None
    assert update(sender, 2, received=0) == "enable"
    assert [packet["seq"] for packet in decode_all(fake)] == [0]


def test_unified_feedback_is_diagnostic_and_relay_session_need_not_match():
    sender, fake = make_sender()
    packet = feedback(session_id="relay-output-session", valid=False, state="disarmed")
    fake.sockets[fake.SUB].received.append(msgpack.packb(packet, use_bin_type=True))
    assert update(sender, 1, received=0) == "enable"
    assert sender.last_feedback.session_id == "relay-output-session"
    assert decode_all(fake)[-1]["session_id"] == "session-1"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda packet: packet.pop("capabilities"),
        lambda packet: packet["capabilities"].update({"legacy": False}),
        lambda packet: packet["capabilities"].update({"direct_enable": False}),
        lambda packet: packet["capabilities"].update({"body_control_interlock": True}),
        lambda packet: packet["capabilities"].update({"source_diagnostic_only": 1}),
    ],
)
def test_feedback_requires_exact_unified_capabilities(mutate):
    packet = feedback()
    mutate(packet)
    with pytest.raises(Revo2TeleopError, match="capabilities|schema or fields"):
        parse_feedback(packet)


def test_feedback_sequence_agent_clock_and_measurement_clock_must_advance():
    sender, fake = make_sender()
    first = feedback(seq=1, agent_ns=2_000_000, measured_ns=1_900_000)
    fake.sockets[fake.SUB].received.append(msgpack.packb(first, use_bin_type=True))
    assert update(sender, 1, received=0) == "enable"

    replay = feedback(seq=1, agent_ns=2_100_000, measured_ns=2_000_000)
    fake.sockets[fake.SUB].received.append(msgpack.packb(replay, use_bin_type=True))
    with pytest.raises(Revo2TeleopError, match="sequence did not advance"):
        update(sender, 2, received=0)
    assert decode_all(fake)[-1]["mode"] == "stop"


def test_feedback_measurement_timestamp_may_repeat_but_not_move_backwards():
    sender, fake = make_sender()
    first = feedback(seq=1, agent_ns=2_000_000, measured_ns=1_900_000)
    repeated = feedback(seq=2, agent_ns=2_100_000, measured_ns=1_900_000)
    fake.sockets[fake.SUB].received.extend(
        msgpack.packb(packet, use_bin_type=True) for packet in (first, repeated)
    )
    assert update(sender, 1, received=0) == "enable"

    rollback = feedback(seq=3, agent_ns=2_200_000, measured_ns=1_800_000)
    fake.sockets[fake.SUB].received.append(msgpack.packb(rollback, use_bin_type=True))
    with pytest.raises(Revo2TeleopError, match="measurement clock moved backwards"):
        update(sender, 2, received=0)
    assert decode_all(fake)[-1]["mode"] == "stop"


def test_feedback_fault_or_restart_stops_active_target():
    sender, fake = make_sender()
    first = feedback(seq=1, agent_ns=2_000_000, measured_ns=1_900_000)
    fake.sockets[fake.SUB].received.append(msgpack.packb(first, use_bin_type=True))
    assert update(sender, 1, received=0) == "enable"

    restarted = feedback(seq=0, agent_ns=3_000_000, measured_ns=2_900_000)
    restarted["agent_boot_id"] = "agent-boot-2"
    fake.sockets[fake.SUB].received.append(msgpack.packb(restarted, use_bin_type=True))
    with pytest.raises(Revo2TeleopError, match="restarted"):
        update(sender, 2, received=0)
    assert decode_all(fake)[-1]["mode"] == "stop"


def test_feedback_timeout_latches_after_first_direct_enable():
    now = {"value": 0}
    sender, fake = make_sender(clock=lambda: now["value"])
    assert update(sender, 1, received=0) == "enable"
    now["value"] = 501_000_000
    with pytest.raises(Revo2TeleopError, match="feedback timed out"):
        update(sender, 2, received=now["value"])
    assert decode_all(fake)[-1]["mode"] == "stop"


def test_disable_preempts_malformed_feedback_and_close_does_not_send_disarm_or_open():
    sender, fake = make_sender()
    assert update(sender, 1, received=0) == "enable"
    fake.sockets[fake.SUB].received.append(b"not-msgpack")
    assert update(sender, 1, received=0, enabled=False) == "stop"
    packet = decode_all(fake)[-1]
    assert packet["mode"] == "stop"
    assert all(not packet["hands"][side]["valid"] for side in ("left", "right"))
    count = len(decode_all(fake))
    sender.close()
    assert len(decode_all(fake)) == count
    assert all(packet["mode"] != "disarm" for packet in decode_all(fake))
    assert fake.sockets[fake.XPUB].closed and fake.sockets[fake.SUB].closed


def test_pico_manager_uses_relay_upstream_without_body_permission_or_controller_id():
    manager_path = Path(__file__).parents[1] / "scripts" / "pico_manager_thread_server.py"
    source = manager_path.read_text()
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in ("_start_revo2_hand_sender", "run_pico_manager"):
        argument_names = {
            argument.arg
            for argument in (
                functions[function_name].args.args + functions[function_name].args.kwonlyargs
            )
        }
        assert "controller_id" not in argument_names
        assert "revo2_controller_id" not in argument_names
    assert "--revo2-controller-id" not in source
    assert "SonicControlGate" not in source
    assert "poll_control_permission" not in source
    assert "body_control_permitted" not in source
    assert "source_received_monotonic_ns" in source
    assert "robot-side Revo2 glove relay" in source
