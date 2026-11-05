"""PICO trigger to BrainCo Revo2 whole-hand command adapter.

This module publishes the upstream ``revo2.command.v3`` glove contract consumed
by the robot-side source relay. It never opens a serial port and it does not
expose per-finger teleoperation: each hand has one closure scalar interpolated
between two validated six-channel poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import uuid

import msgpack
import numpy as np

COMMAND_SCHEMA = "revo2.command.v3"
FEEDBACK_SCHEMA = "revo2.feedback.v3"
COMMAND_SOURCE = "glove"
COMMAND_LEASE_MS = 200
TARGET_VALID_MS = 150
MAX_MEASUREMENT_AGE_NS = 100_000_000
HAND_CHANNELS = 6
SIDES = ("left", "right")
COMMAND_MODES = ("disarm", "arm", "enable", "hold", "stop")
UNIFIED_CAPABILITIES = {
    "direct_enable": True,
    "source_diagnostic_only": True,
    "body_control_interlock": False,
}
_FEEDBACK_FIELDS = frozenset(
    {
        "schema",
        "session_id",
        "agent_boot_id",
        "feedback_seq",
        "agent_monotonic_ns",
        "accepted_mode",
        "last_mode_seq",
        "lease_valid",
        "lease_remaining_ms",
        "mode_reject_reason",
        "global_fault",
        "capabilities",
        "actuated_sides",
        "hands",
    }
)
_FEEDBACK_HAND_FIELDS = frozenset(
    {
        "state",
        "last_received_seq",
        "last_applied_seq",
        "last_applied_control_frame_index",
        "received_agent_monotonic_ns",
        "applied_agent_monotonic_ns",
        "requested",
        "applied",
        "clipped_mask",
        "position_raw",
        "position_norm",
        "speed_raw",
        "current_raw",
        "motor_state",
        "measured_agent_monotonic_ns",
        "valid",
        "fault",
        "reject_reason",
    }
)


class Revo2TeleopError(RuntimeError):
    """Raised when the Revo2 teleoperation contract cannot fail safely."""


def require_fresh_trigger_source(input_source: str) -> None:
    """Reject sources that cannot prove when controller values were sampled."""
    if input_source != "xrt":
        raise Revo2TeleopError(
            "Revo2 actuation requires --input-source xrt: Isaac Teleop 1.4 "
            "DeviceIO does not expose controller-origin sample timestamps"
        )


def explicit_tcp_ipv4_endpoint(value: str, path: str) -> str:
    """Require the same explicit data-plane endpoint form as the robot relay."""
    if not isinstance(value, str) or not value.startswith("tcp://"):
        raise Revo2TeleopError(f"{path} must be tcp://<explicit-ipv4>:<port>")
    address, separator, port_text = value[6:].partition(":")
    parts = address.split(".")
    if (
        not separator
        or len(parts) != 4
        or any(not part.isdigit() for part in parts)
        or any(len(part) > 1 and part.startswith("0") for part in parts)
    ):
        raise Revo2TeleopError(f"{path} must use an explicit IPv4 address")
    octets = [int(part) for part in parts]
    if any(part > 255 for part in octets) or address in ("0.0.0.0", "255.255.255.255"):
        raise Revo2TeleopError(f"{path} must use a non-wildcard IPv4 address")
    if not port_text.isdigit() or (len(port_text) > 1 and port_text.startswith("0")):
        raise Revo2TeleopError(f"{path} port must be plain decimal digits")
    port = int(port_text)
    if not 1024 <= port <= 65535:
        raise Revo2TeleopError(f"{path} port must be between 1024 and 65535")
    return f"tcp://{address}:{port}"


def _six_finite_normalized(value: Any, path: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise Revo2TeleopError(f"{path} must contain six numeric values") from error
    if raw.dtype.kind not in "iuf":
        raise Revo2TeleopError(f"{path} must contain only numeric values")
    array = raw.astype(np.float64, copy=False)
    if array.shape != (HAND_CHANNELS,) or not np.isfinite(array).all():
        raise Revo2TeleopError(f"{path} must contain six finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise Revo2TeleopError(f"{path} must stay inside [0, 1]")
    return array


@dataclass(frozen=True)
class Revo2PoseProfile:
    """Validated whole-hand open and closed endpoints for both hands."""

    open_poses: Mapping[str, np.ndarray]
    closed_poses: Mapping[str, np.ndarray]
    max_close_ratio: float = 1.0
    _scaled_deltas: Mapping[str, np.ndarray] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if set(self.open_poses) != set(SIDES) or set(self.closed_poses) != set(SIDES):
            raise Revo2TeleopError("pose profile requires left and right open/closed poses")
        if (
            isinstance(self.max_close_ratio, bool)
            or not isinstance(self.max_close_ratio, Real)
            or not math.isfinite(self.max_close_ratio)
            or not 0.0 < self.max_close_ratio <= 1.0
        ):
            raise Revo2TeleopError("max_close_ratio must be finite and inside (0, 1]")
        open_poses = {
            side: _six_finite_normalized(self.open_poses[side], f"{side}.open") for side in SIDES
        }
        closed_poses = {
            side: _six_finite_normalized(self.closed_poses[side], f"{side}.closed")
            for side in SIDES
        }
        object.__setattr__(self, "open_poses", open_poses)
        object.__setattr__(self, "closed_poses", closed_poses)
        object.__setattr__(
            self,
            "_scaled_deltas",
            {
                side: self.max_close_ratio * (closed_poses[side] - open_poses[side])
                for side in SIDES
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Revo2PoseProfile":
        source = Path(path)
        try:
            value = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise Revo2TeleopError(f"cannot load Revo2 pose profile {source}: {error}") from error
        if not isinstance(value, dict):
            raise Revo2TeleopError("Revo2 pose profile must be a JSON object")
        expected = {"schema", "open", "closed", "max_close_ratio"}
        if set(value) != expected or value["schema"] != "revo2.pico_pose_profile.v1":
            raise Revo2TeleopError(
                "Revo2 pose profile must use schema revo2.pico_pose_profile.v1 and exact keys "
                + ", ".join(sorted(expected))
            )
        return cls(value["open"], value["closed"], value["max_close_ratio"])

    def targets(self, left_trigger: float, right_trigger: float) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for side, raw_trigger in (("left", left_trigger), ("right", right_trigger)):
            trigger = float(raw_trigger)
            if not math.isfinite(trigger):
                raise Revo2TeleopError(f"{side} trigger must be finite")
            closure = min(max(trigger, 0.0), 1.0)
            target = self.open_poses[side] + closure * self._scaled_deltas[side]
            result[side] = target.tolist()
        return result


def command_packet(
    mode: str,
    *,
    session_id: str,
    seq: int,
    control_frame_index: int,
    sender_monotonic_ns: int,
    targets: Mapping[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Build one exact upstream ``revo2.command.v3`` glove packet."""

    if mode not in COMMAND_MODES:
        raise Revo2TeleopError(f"invalid Revo2 command mode: {mode}")
    if not session_id:
        raise Revo2TeleopError("session_id must be nonempty")
    for name, value in (
        ("seq", seq),
        ("control_frame_index", control_frame_index),
        ("sender_monotonic_ns", sender_monotonic_ns),
    ):
        if type(value) is not int or value < 0:
            raise Revo2TeleopError(f"{name} must be a non-negative integer")

    if mode == "enable":
        if not isinstance(targets, Mapping) or set(targets) != set(SIDES):
            raise Revo2TeleopError("enable requires one target for each hand")
        hands = {
            side: {
                "target": [float(v) for v in _six_finite_normalized(targets[side], side)],
                "valid": True,
                "source_frame_index": control_frame_index,
            }
            for side in SIDES
        }
    else:
        if targets is not None:
            raise Revo2TeleopError(f"{mode} must not carry hand targets")
        hands = {
            side: {
                "target": [0.0] * HAND_CHANNELS,
                "valid": False,
                "source_frame_index": control_frame_index,
            }
            for side in SIDES
        }

    return {
        "schema": COMMAND_SCHEMA,
        "session_id": session_id,
        "seq": seq,
        "control_frame_index": control_frame_index,
        "sender_monotonic_ns": sender_monotonic_ns,
        "lease_ms": COMMAND_LEASE_MS,
        "source": COMMAND_SOURCE,
        "mode": mode,
        "actuated_sides": list(SIDES),
        "hands": hands,
    }


@dataclass(frozen=True)
class Revo2HandFeedback:
    state: str
    measured_agent_monotonic_ns: int
    valid: bool
    fault: str
    reject_reason: str


@dataclass(frozen=True)
class Revo2Feedback:
    session_id: str
    agent_boot_id: str
    feedback_seq: int
    agent_monotonic_ns: int
    accepted_mode: str
    last_mode_seq: int
    lease_valid: bool
    global_fault: str
    mode_reject_reason: str
    hands: Mapping[str, Revo2HandFeedback]


def _feedback_integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise Revo2TeleopError(f"{path} is invalid")
    return value


def _feedback_vector(value: Any, path: str, item_type: type) -> None:
    if type(value) not in (list, tuple) or len(value) != HAND_CHANNELS:
        raise Revo2TeleopError(f"{path} must contain six values")
    if any(type(item) is not item_type for item in value):
        raise Revo2TeleopError(f"{path} contains an invalid value type")


def _feedback_normalized_vector(value: Any, path: str) -> None:
    if type(value) not in (list, tuple) or len(value) != HAND_CHANNELS:
        raise Revo2TeleopError(f"{path} must contain six values")
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(item) or not 0.0 <= item <= 1.0:
            raise Revo2TeleopError(f"{path} contains an invalid normalized value")


def _parse_feedback_hand(value: Any, path: str, agent_monotonic_ns: int) -> Revo2HandFeedback:
    if type(value) is not dict or set(value) != _FEEDBACK_HAND_FIELDS:
        raise Revo2TeleopError(f"{path} fields are invalid")
    if value["state"] not in {
        "disarmed",
        "armed",
        "enabled",
        "holding",
        "stopped",
        "stale",
        "fault",
        "unavailable",
    }:
        raise Revo2TeleopError(f"{path}.state is invalid")
    for name in ("last_received_seq", "last_applied_seq", "last_applied_control_frame_index"):
        _feedback_integer(value[name], f"{path}.{name}", minimum=-1)
    for name in ("received_agent_monotonic_ns", "applied_agent_monotonic_ns"):
        _feedback_integer(value[name], f"{path}.{name}", minimum=0)
    measured_ns = _feedback_integer(
        value["measured_agent_monotonic_ns"],
        f"{path}.measured_agent_monotonic_ns",
        minimum=1,
    )
    if not 0 <= agent_monotonic_ns - measured_ns < MAX_MEASUREMENT_AGE_NS:
        raise Revo2TeleopError(f"{path} measurement is stale")
    for name in ("requested", "applied", "position_norm"):
        _feedback_normalized_vector(value[name], f"{path}.{name}")
    _feedback_vector(value["clipped_mask"], f"{path}.clipped_mask", bool)
    for name in ("position_raw", "speed_raw", "current_raw", "motor_state"):
        _feedback_vector(value[name], f"{path}.{name}", int)
    if any(item < 0 or item > 1000 for item in value["position_raw"]):
        raise Revo2TeleopError(f"{path}.position_raw is outside [0, 1000]")
    if type(value["valid"]) is not bool:
        raise Revo2TeleopError(f"{path}.valid must be boolean")
    for name in ("fault", "reject_reason"):
        if type(value[name]) is not str:
            raise Revo2TeleopError(f"{path}.{name} must be a string")
    return Revo2HandFeedback(
        state=value["state"],
        measured_agent_monotonic_ns=measured_ns,
        valid=value["valid"],
        fault=value["fault"],
        reject_reason=value["reject_reason"],
    )


def parse_feedback(value: Any) -> Revo2Feedback:
    """Strictly validate unified v3 feedback used for health diagnostics."""

    if (
        type(value) is not dict
        or set(value) != _FEEDBACK_FIELDS
        or value.get("schema") != FEEDBACK_SCHEMA
    ):
        raise Revo2TeleopError("unexpected Revo2 feedback schema or fields")
    capabilities = value["capabilities"]
    if (
        type(capabilities) is not dict
        or set(capabilities) != set(UNIFIED_CAPABILITIES)
        or any(
            type(capabilities[name]) is not bool
            or capabilities[name] is not expected_value
            for name, expected_value in UNIFIED_CAPABILITIES.items()
        )
    ):
        raise Revo2TeleopError("feedback capabilities do not identify the unified executor")
    if value["actuated_sides"] != list(SIDES):
        raise Revo2TeleopError("Revo2 feedback must report left and right hands")
    if type(value["hands"]) is not dict or set(value["hands"]) != set(SIDES):
        raise Revo2TeleopError("Revo2 feedback hand topology does not match the command")
    if value["accepted_mode"] not in COMMAND_MODES:
        raise Revo2TeleopError("Revo2 feedback contains an invalid accepted mode")
    feedback_seq = _feedback_integer(value["feedback_seq"], "feedback.feedback_seq", minimum=0)
    last_mode_seq = _feedback_integer(value["last_mode_seq"], "feedback.last_mode_seq", minimum=-1)
    agent_monotonic_ns = _feedback_integer(
        value["agent_monotonic_ns"], "feedback.agent_monotonic_ns", minimum=1
    )
    _feedback_integer(value["lease_remaining_ms"], "feedback.lease_remaining_ms", minimum=0)
    if type(value["lease_valid"]) is not bool:
        raise Revo2TeleopError("feedback lease_valid must be boolean")
    for name in ("session_id", "agent_boot_id"):
        if type(value[name]) is not str or not value[name]:
            raise Revo2TeleopError(f"feedback {name} must be a nonempty string")
    for name in ("global_fault", "mode_reject_reason"):
        if type(value[name]) is not str:
            raise Revo2TeleopError(f"feedback {name} must be a string")

    hands = {
        side: _parse_feedback_hand(
            value["hands"][side], f"feedback.hands.{side}", agent_monotonic_ns
        )
        for side in SIDES
    }
    return Revo2Feedback(
        session_id=value["session_id"],
        agent_boot_id=value["agent_boot_id"],
        feedback_seq=feedback_seq,
        agent_monotonic_ns=agent_monotonic_ns,
        accepted_mode=value["accepted_mode"],
        last_mode_seq=last_mode_seq,
        lease_valid=value["lease_valid"],
        global_fault=value["global_fault"],
        mode_reject_reason=value["mode_reject_reason"],
        hands=hands,
    )


class Revo2HandSender:
    """Non-blocking upstream glove publisher with direct-enable semantics."""

    def __init__(
        self,
        command_bind: str,
        feedback_connect: str,
        *,
        profile: Revo2PoseProfile,
        permit_actuation: bool = False,
        feedback_timeout_ms: int = 500,
        target_valid_ms: int = TARGET_VALID_MS,
        session_id: str | None = None,
        zmq_module=None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if permit_actuation is not True:
            raise Revo2TeleopError("Revo2 publishing requires explicit actuation permission")
        self.command_bind = explicit_tcp_ipv4_endpoint(command_bind, "Revo2 command bind")
        self.feedback_connect = explicit_tcp_ipv4_endpoint(
            feedback_connect, "Revo2 feedback connect"
        )
        if feedback_timeout_ms <= COMMAND_LEASE_MS:
            raise Revo2TeleopError("feedback timeout must exceed the command lease")
        if not 0 < target_valid_ms < COMMAND_LEASE_MS:
            raise Revo2TeleopError("target validity must be inside the command lease")
        self.profile = profile
        self.feedback_timeout_ns = feedback_timeout_ms * 1_000_000
        self.target_valid_ns = target_valid_ms * 1_000_000
        self.session_id = session_id or uuid.uuid4().hex
        if zmq_module is None:
            try:
                import zmq as zmq_module
            except ImportError as error:
                raise Revo2TeleopError(
                    "Revo2 teleoperation requires pyzmq (install gear_sonic[teleop])"
                ) from error
        self.zmq = zmq_module
        self.clock_ns = clock_ns
        self.context = None
        self.command = None
        self.feedback = None
        self.seq = 0
        self.frame = 0
        self.last_source_timestamp_ns: int | None = None
        self.last_feedback: Revo2Feedback | None = None
        self.last_feedback_received_ns: int | None = None
        self.last_feedback_seq: int | None = None
        self.last_feedback_agent_ns: int | None = None
        self.last_measured_ns: tuple[int, int] | None = None
        self.last_agent_boot_id: str | None = None
        self.last_requested_mode: str | None = None
        self.first_enable_ns: int | None = None
        self.target_deadline_ns: int | None = None
        self.target_active = False
        self.awaiting_trigger_release = True
        self.closed = False
        self.latched_fault: str | None = None

    def start(self) -> "Revo2HandSender":
        if self.closed:
            raise Revo2TeleopError("closed Revo2 sender cannot be restarted")
        self.context = self.zmq.Context()
        try:
            self.command = self.context.socket(self.zmq.XPUB)
            self.command.setsockopt(self.zmq.LINGER, 0)
            self.command.setsockopt(self.zmq.SNDHWM, 2)
            self.command.setsockopt(self.zmq.SNDTIMEO, 0)
            self.command.setsockopt(self.zmq.XPUB_NODROP, 1)
            self.command.bind(self.command_bind)
            self.feedback = self.context.socket(self.zmq.SUB)
            self.feedback.setsockopt(self.zmq.LINGER, 0)
            self.feedback.setsockopt(self.zmq.SUBSCRIBE, b"")
            self.feedback.setsockopt(self.zmq.CONFLATE, 1)
            self.feedback.connect(self.feedback_connect)
        except BaseException:
            self.close()
            raise
        return self

    def _publish(
        self,
        mode: str,
        *,
        targets: Mapping[str, list[float]] | None,
        now_ns: int,
    ) -> int:
        if self.command is None or self.closed:
            raise Revo2TeleopError("Revo2 sender is not open")
        packet_seq = self.seq
        packet = command_packet(
            mode,
            session_id=self.session_id,
            seq=packet_seq,
            control_frame_index=self.frame,
            sender_monotonic_ns=now_ns,
            targets=targets,
        )
        try:
            for _ in range(8):
                try:
                    self.command.recv(flags=self.zmq.DONTWAIT)
                except self.zmq.Again:
                    break
            self.command.send(
                msgpack.packb(packet, use_bin_type=True), flags=self.zmq.DONTWAIT
            )
        except self.zmq.Again:
            return -1
        except Exception as error:
            self.latched_fault = f"Revo2 command transport failed: {error}"
            raise Revo2TeleopError(self.latched_fault) from error
        self.seq += 1
        self.last_requested_mode = mode
        if mode == "enable" and self.first_enable_ns is None:
            self.first_enable_ns = now_ns
        return packet_seq

    def _cancel_active_target(self, now_ns: int) -> bool:
        should_publish = self.target_active
        try:
            if should_publish:
                self._publish("stop", targets=None, now_ns=now_ns)
        except BaseException:
            pass
        self.target_active = False
        self.target_deadline_ns = None
        return should_publish

    def _raise_fault(self, message: str, now_ns: int) -> None:
        if self.latched_fault is None:
            self.latched_fault = message
        self._cancel_active_target(now_ns)
        raise Revo2TeleopError(self.latched_fault)

    def _drain_feedback(self, now_ns: int) -> None:
        if self.feedback is None:
            return
        while True:
            try:
                payload = self.feedback.recv(flags=self.zmq.DONTWAIT)
            except self.zmq.Again:
                return
            try:
                packet = parse_feedback(msgpack.unpackb(payload, raw=False, strict_map_key=True))
            except Exception as error:
                self._raise_fault(f"invalid Revo2 feedback: {error}", now_ns)

            measured = tuple(
                packet.hands[side].measured_agent_monotonic_ns for side in SIDES
            )
            if self.last_agent_boot_id == packet.agent_boot_id:
                if self.last_feedback_seq is not None and packet.feedback_seq <= self.last_feedback_seq:
                    self._raise_fault("Revo2 feedback sequence did not advance", now_ns)
                if (
                    self.last_feedback_agent_ns is not None
                    and packet.agent_monotonic_ns <= self.last_feedback_agent_ns
                ):
                    self._raise_fault("Revo2 feedback clock did not advance", now_ns)
                if self.last_measured_ns is not None and any(
                    current < previous
                    for current, previous in zip(measured, self.last_measured_ns)
                ):
                    self._raise_fault("Revo2 hand measurement clock moved backwards", now_ns)
            elif self.last_agent_boot_id is not None:
                self._raise_fault("Revo2 agent restarted", now_ns)

            if packet.global_fault:
                self._raise_fault("Revo2 agent fault: " + packet.global_fault, now_ns)
            for side in SIDES:
                hand = packet.hands[side]
                if hand.fault:
                    self._raise_fault(f"Revo2 {side} hand fault: {hand.fault}", now_ns)
                if hand.state in ("fault", "unavailable", "stale"):
                    self._raise_fault(f"Revo2 {side} hand state is {hand.state}", now_ns)

            self.last_agent_boot_id = packet.agent_boot_id
            self.last_feedback_seq = packet.feedback_seq
            self.last_feedback_agent_ns = packet.agent_monotonic_ns
            self.last_measured_ns = measured
            self.last_feedback = packet
            self.last_feedback_received_ns = now_ns

    def _check_feedback_timeout(self, now_ns: int) -> None:
        if self.first_enable_ns is None:
            return
        reference_ns = self.last_feedback_received_ns or self.first_enable_ns
        if now_ns - reference_ns > self.feedback_timeout_ns:
            self._raise_fault("Revo2 feedback timed out", now_ns)

    def _expire_target(self, now_ns: int) -> bool:
        if (
            self.target_active
            and self.target_deadline_ns is not None
            and now_ns >= self.target_deadline_ns
        ):
            self.stop(now_ns=now_ns)
            return True
        return False

    def update(
        self,
        *,
        source_timestamp_ns: int,
        source_received_monotonic_ns: int,
        left_trigger: float,
        right_trigger: float,
        enabled: bool,
    ) -> str | None:
        """Process one PICO sample without replaying or renewing duplicate input."""

        now_ns = self.clock_ns()
        if not enabled:
            return "stop" if self.stop(now_ns=now_ns) else None
        if self.latched_fault is not None:
            raise Revo2TeleopError(self.latched_fault)

        self._drain_feedback(now_ns)
        self._check_feedback_timeout(now_ns)
        if type(source_timestamp_ns) is not int or source_timestamp_ns < 0:
            self._raise_fault("PICO source timestamp must be a non-negative integer", now_ns)
        if type(source_received_monotonic_ns) is not int or source_received_monotonic_ns < 0:
            self._raise_fault("PICO receive timestamp must be a non-negative integer", now_ns)
        for side, trigger in (("left", left_trigger), ("right", right_trigger)):
            if not math.isfinite(trigger):
                self._raise_fault(f"{side} trigger must be finite", now_ns)

        if self.last_source_timestamp_ns is not None:
            if source_timestamp_ns == self.last_source_timestamp_ns:
                self._expire_target(now_ns)
                return None
            if source_timestamp_ns < self.last_source_timestamp_ns:
                self._raise_fault("PICO source timestamp moved backwards", now_ns)
        if source_timestamp_ns == 0:
            self._expire_target(now_ns)
            return None
        self.last_source_timestamp_ns = source_timestamp_ns
        self.frame += 1

        if now_ns < source_received_monotonic_ns:
            self._raise_fault("PICO receive clock moved backwards", now_ns)
        target_deadline_ns = source_received_monotonic_ns + self.target_valid_ns
        if now_ns >= target_deadline_ns:
            self.stop(now_ns=now_ns)
            return "stale"

        if self.awaiting_trigger_release:
            if left_trigger > 0.05 or right_trigger > 0.05:
                return "await_trigger_release"
            self.awaiting_trigger_release = False

        targets = self.profile.targets(left_trigger, right_trigger)
        packet_seq = self._publish("enable", targets=targets, now_ns=now_ns)
        if packet_seq < 0:
            return None
        self.target_active = True
        self.target_deadline_ns = target_deadline_ns
        return "enable"

    def tick(self, *, now_ns: int | None = None) -> str | None:
        """Observe feedback and cancel one expired target without replaying it."""
        if self.closed:
            return None
        current_ns = self.clock_ns() if now_ns is None else now_ns
        if type(current_ns) is not int or current_ns < 0:
            raise Revo2TeleopError("Revo2 tick timestamp must be a non-negative integer")
        if self.latched_fault is not None:
            raise Revo2TeleopError(self.latched_fault)
        self._drain_feedback(current_ns)
        self._check_feedback_timeout(current_ns)
        return "stop" if self._expire_target(current_ns) else None

    def stop(self, *, now_ns: int | None = None) -> bool:
        if self.command is None or self.closed:
            return False
        self.awaiting_trigger_release = True
        if not self.target_active:
            self.target_deadline_ns = None
            return False
        current_ns = self.clock_ns() if now_ns is None else now_ns
        return self._cancel_active_target(current_ns)

    def close(self) -> None:
        if self.closed:
            return
        if self.command is not None:
            try:
                self.stop()
            except BaseException:
                pass
        self.closed = True
        for socket_name in ("feedback", "command"):
            socket = getattr(self, socket_name)
            if socket is not None:
                try:
                    socket.close(linger=0)
                finally:
                    setattr(self, socket_name, None)
        if self.context is not None:
            self.context.term()
            self.context = None
