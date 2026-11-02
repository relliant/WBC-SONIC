"""PICO trigger to BrainCo Revo2 whole-hand command adapter.

This module speaks the existing ``revo2.command.v3`` glove contract.  It never
opens a serial port and it does not expose per-finger teleoperation: each hand
has one closure scalar which is interpolated between two validated six-channel
poses.
"""

from __future__ import annotations

from dataclasses import dataclass
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
SONIC_CONTROL_STATUS_SCHEMA = "sonic.control_status.v1"
COMMAND_SOURCE = "glove"
COMMAND_LEASE_MS = 200
HAND_CHANNELS = 6
SIDES = ("left", "right")
COMMAND_MODES = ("disarm", "arm", "enable", "hold", "stop")


class Revo2TeleopError(RuntimeError):
    """Raised when the Revo2 teleoperation contract cannot fail safely."""


def require_fresh_trigger_source(input_source: str) -> None:
    """Reject sources that cannot prove when controller values were sampled."""
    if input_source != "xrt":
        raise Revo2TeleopError(
            "Revo2 actuation requires --input-source xrt: Isaac Teleop 1.4 "
            "DeviceIO does not expose controller-origin sample timestamps"
        )


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
        triggers = {"left": left_trigger, "right": right_trigger}
        result: dict[str, list[float]] = {}
        for side in SIDES:
            trigger = float(triggers[side])
            if not math.isfinite(trigger):
                raise Revo2TeleopError(f"{side} trigger must be finite")
            closure = float(np.clip(trigger, 0.0, 1.0)) * self.max_close_ratio
            target = self.open_poses[side] + closure * (
                self.closed_poses[side] - self.open_poses[side]
            )
            result[side] = [float(item) for item in np.clip(target, 0.0, 1.0)]
        return result


@dataclass(frozen=True)
class SonicControlStatus:
    sequence: int
    control_enabled: bool
    operator_stop: bool
    hand_mode: str


def parse_sonic_control_status(value: Any) -> SonicControlStatus:
    """Validate the C++ heartbeat used to interlock external hand actuation."""

    expected = {
        "schema",
        "sequence",
        "control_enabled",
        "operator_stop",
        "hand_mode",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise Revo2TeleopError("unexpected SONIC control status fields")
    if value["schema"] != SONIC_CONTROL_STATUS_SCHEMA:
        raise Revo2TeleopError("unexpected SONIC control status schema")
    if type(value["sequence"]) is not int or value["sequence"] < 0:
        raise Revo2TeleopError("SONIC control status sequence is invalid")
    if type(value["control_enabled"]) is not bool or type(value["operator_stop"]) is not bool:
        raise Revo2TeleopError("SONIC control status flags must be boolean")
    if value["hand_mode"] not in ("dex3", "external"):
        raise Revo2TeleopError("SONIC control status hand_mode is invalid")
    return SonicControlStatus(
        sequence=value["sequence"],
        control_enabled=value["control_enabled"],
        operator_stop=value["operator_stop"],
        hand_mode=value["hand_mode"],
    )


class SonicControlGate:
    """Latch-safe freshness gate for the C++ SONIC control heartbeat."""

    def __init__(self, timeout_ms: int = 150) -> None:
        if timeout_ms <= 0 or timeout_ms >= COMMAND_LEASE_MS:
            raise Revo2TeleopError("SONIC status timeout must be inside the command lease")
        self.timeout_ns = timeout_ms * 1_000_000
        self.last_sequence: int | None = None
        self.last_received_ns: int | None = None
        self.control_enabled = False
        self.latched_fault: str | None = None

    def accept(self, value: Any, *, received_ns: int) -> None:
        if self.latched_fault is not None:
            return
        if type(received_ns) is not int or received_ns < 0:
            self.latched_fault = "SONIC status receive timestamp is invalid"
            self.control_enabled = False
            return
        try:
            status = parse_sonic_control_status(value)
        except Revo2TeleopError as error:
            self.latched_fault = str(error)
            self.control_enabled = False
            return
        if self.last_sequence is not None and status.sequence <= self.last_sequence:
            self.latched_fault = "SONIC control status sequence did not advance"
            self.control_enabled = False
            return
        self.last_sequence = status.sequence
        self.last_received_ns = received_ns
        self.control_enabled = (
            status.control_enabled and not status.operator_stop and status.hand_mode == "external"
        )

    def permitted(self, *, now_ns: int) -> bool:
        if self.latched_fault is not None or self.last_received_ns is None:
            return False
        if type(now_ns) is not int or now_ns < self.last_received_ns:
            self.latched_fault = "SONIC status clock moved backwards"
            self.control_enabled = False
            return False
        if now_ns - self.last_received_ns > self.timeout_ns:
            self.control_enabled = False
        return self.control_enabled


def command_packet(
    mode: str,
    *,
    session_id: str,
    controller_id: str,
    seq: int,
    control_frame_index: int,
    sender_monotonic_ns: int,
    targets: Mapping[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Build one exact ``revo2.command.v3`` glove packet."""

    if mode not in COMMAND_MODES:
        raise Revo2TeleopError(f"invalid Revo2 command mode: {mode}")
    if not session_id or not controller_id:
        raise Revo2TeleopError("session_id and controller_id must be nonempty")
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
        "controller_id": controller_id,
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
class Revo2Feedback:
    session_id: str
    agent_boot_id: str
    feedback_seq: int
    accepted_mode: str
    last_mode_seq: int
    lease_valid: bool
    global_fault: str
    mode_reject_reason: str
    hand_valid: bool
    hand_faults: tuple[str, ...]
    hand_reject_reasons: tuple[str, ...]


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


def _parse_feedback_hand(value: Any, path: str) -> tuple[bool, str, str]:
    fields = {
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
    if type(value) is not dict or set(value) != fields:
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
    for name in (
        "last_received_seq",
        "last_applied_seq",
        "last_applied_control_frame_index",
    ):
        _feedback_integer(value[name], f"{path}.{name}", minimum=-1)
    for name in (
        "received_agent_monotonic_ns",
        "applied_agent_monotonic_ns",
        "measured_agent_monotonic_ns",
    ):
        _feedback_integer(value[name], f"{path}.{name}", minimum=0)
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
    return value["valid"], value["fault"], value["reject_reason"]


def parse_feedback(value: Any) -> Revo2Feedback:
    """Strictly validate the v3 feedback used by the sender interlock."""

    expected = {
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
        "actuated_sides",
        "hands",
    }
    if type(value) is not dict or set(value) != expected or value.get("schema") != FEEDBACK_SCHEMA:
        raise Revo2TeleopError("unexpected Revo2 feedback schema or fields")
    if value["actuated_sides"] != list(SIDES):
        raise Revo2TeleopError("Revo2 feedback must report left and right hands")
    if type(value["hands"]) is not dict or set(value["hands"]) != set(SIDES):
        raise Revo2TeleopError("Revo2 feedback hand topology does not match the command")
    if value["accepted_mode"] not in COMMAND_MODES:
        raise Revo2TeleopError("Revo2 feedback contains an invalid accepted mode")
    _feedback_integer(value["feedback_seq"], "feedback.feedback_seq", minimum=0)
    _feedback_integer(value["last_mode_seq"], "feedback.last_mode_seq", minimum=-1)
    _feedback_integer(
        value["agent_monotonic_ns"], "feedback.agent_monotonic_ns", minimum=0
    )
    _feedback_integer(
        value["lease_remaining_ms"], "feedback.lease_remaining_ms", minimum=0
    )
    if type(value["lease_valid"]) is not bool:
        raise Revo2TeleopError("feedback lease_valid must be boolean")
    for name in ("session_id", "agent_boot_id"):
        if type(value[name]) is not str or not value[name]:
            raise Revo2TeleopError(f"feedback {name} must be a nonempty string")
    for name in ("global_fault", "mode_reject_reason"):
        if type(value[name]) is not str:
            raise Revo2TeleopError(f"feedback {name} must be a string")

    hand_status = [
        _parse_feedback_hand(value["hands"][side], f"feedback.hands.{side}")
        for side in SIDES
    ]
    return Revo2Feedback(
        session_id=value["session_id"],
        agent_boot_id=value["agent_boot_id"],
        feedback_seq=value["feedback_seq"],
        accepted_mode=value["accepted_mode"],
        last_mode_seq=value["last_mode_seq"],
        lease_valid=value["lease_valid"],
        global_fault=value["global_fault"],
        mode_reject_reason=value["mode_reject_reason"],
        hand_valid=all(status[0] for status in hand_status),
        hand_faults=tuple(status[1] for status in hand_status),
        hand_reject_reasons=tuple(status[2] for status in hand_status),
    )


class Revo2HandSender:
    """Non-blocking glove publisher with feedback-gated arm/enable transitions."""

    def __init__(
        self,
        command_bind: str,
        feedback_connect: str,
        *,
        controller_id: str,
        profile: Revo2PoseProfile,
        permit_actuation: bool = False,
        feedback_timeout_ms: int = 500,
        session_id: str | None = None,
        zmq_module=None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if permit_actuation is not True:
            raise Revo2TeleopError("Revo2 publishing requires explicit actuation permission")
        if not command_bind.startswith("tcp://") or not feedback_connect.startswith("tcp://"):
            raise Revo2TeleopError("Revo2 endpoints must be explicit tcp:// endpoints")
        if not controller_id:
            raise Revo2TeleopError("Revo2 controller_id must be nonempty")
        if feedback_timeout_ms <= COMMAND_LEASE_MS:
            raise Revo2TeleopError("feedback timeout must exceed the command lease")
        self.command_bind = command_bind
        self.feedback_connect = feedback_connect
        self.controller_id = controller_id
        self.profile = profile
        self.feedback_timeout_ns = feedback_timeout_ms * 1_000_000
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
        self.last_agent_boot_id: str | None = None
        self.last_mode_seq = -1
        self.last_requested_mode: str | None = None
        self.pending_arm_seq: int | None = None
        self.arm_authorized = False
        self.first_command_ns: int | None = None
        self.enabled = False
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

    def _drain_feedback(self, now_ns: int) -> None:
        if self.feedback is None:
            return
        while True:
            try:
                payload = self.feedback.recv(flags=self.zmq.DONTWAIT)
            except self.zmq.Again:
                return
            try:
                packet = parse_feedback(
                    msgpack.unpackb(payload, raw=False, strict_map_key=True)
                )
            except Exception as error:
                self.latched_fault = f"invalid Revo2 feedback: {error}"
                raise Revo2TeleopError(self.latched_fault) from error
            if self.last_agent_boot_id == packet.agent_boot_id:
                if (
                    self.last_feedback_seq is not None
                    and packet.feedback_seq <= self.last_feedback_seq
                ):
                    self.latched_fault = "Revo2 feedback sequence did not advance"
                    raise Revo2TeleopError(self.latched_fault)
            elif self.last_agent_boot_id is not None:
                self.latched_fault = "Revo2 agent restarted"
                raise Revo2TeleopError(self.latched_fault)
            self.last_agent_boot_id = packet.agent_boot_id
            self.last_feedback_seq = packet.feedback_seq
            self.last_feedback = packet
            self.last_feedback_received_ns = now_ns

    def _publish(self, mode: str, *, targets: Mapping[str, list[float]] | None, now_ns: int) -> int:
        if self.command is None or self.closed:
            raise Revo2TeleopError("Revo2 sender is not open")
        packet_seq = self.seq
        packet = command_packet(
            mode,
            session_id=self.session_id,
            controller_id=self.controller_id,
            seq=packet_seq,
            control_frame_index=self.frame,
            sender_monotonic_ns=now_ns,
            targets=targets,
        )
        # Drain XPUB subscription notifications without ever blocking actuation.
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
        self.last_mode_seq = packet_seq
        self.last_requested_mode = mode
        if self.first_command_ns is None:
            self.first_command_ns = now_ns
        return packet_seq

    def update(
        self,
        *,
        source_timestamp_ns: int,
        left_trigger: float,
        right_trigger: float,
        enabled: bool,
    ) -> str | None:
        """Process one fresh PICO sample and advance the arm/enable state machine.

        A duplicate source timestamp sends nothing and therefore cannot renew the
        agent's lease.  A rollback latches the sender until a new process/session.
        """

        if type(source_timestamp_ns) is not int or source_timestamp_ns < 0:
            raise Revo2TeleopError("PICO source timestamp must be a non-negative integer")
        now_ns = self.clock_ns()

        # A disable/stop request is independent of source and feedback validity.
        # Malformed queued feedback must never preempt the stop publication.
        if not enabled:
            if self.enabled or self.last_requested_mode not in (None, "stop", "disarm"):
                self._publish("stop", targets=None, now_ns=now_ns)
            self.enabled = False
            self.awaiting_trigger_release = True
            self.pending_arm_seq = None
            self.arm_authorized = False
            self.last_feedback = None
            self.last_feedback_received_ns = None
            self.first_command_ns = None
            return "stop"

        self._drain_feedback(now_ns)

        # Zero means PicoReader has not produced its first real sample yet. Do not
        # use startup defaults to satisfy the trigger-release interlock or arm.
        if source_timestamp_ns == 0:
            return None
        for side, trigger in (("left", left_trigger), ("right", right_trigger)):
            if not math.isfinite(trigger):
                raise Revo2TeleopError(f"{side} trigger must be finite")

        if self.last_source_timestamp_ns is not None:
            if source_timestamp_ns == self.last_source_timestamp_ns:
                return None
            if source_timestamp_ns < self.last_source_timestamp_ns:
                self.latched_fault = "PICO source timestamp moved backwards"
                self.stop()
                raise Revo2TeleopError(self.latched_fault)
        self.last_source_timestamp_ns = source_timestamp_ns
        self.frame += 1

        if self.latched_fault:
            raise Revo2TeleopError(self.latched_fault)
        if self.awaiting_trigger_release:
            if left_trigger > 0.05 or right_trigger > 0.05:
                return "await_trigger_release"
            self.awaiting_trigger_release = False
        self.enabled = True

        feedback = self.last_feedback
        if feedback is not None and feedback.global_fault:
            self.latched_fault = "Revo2 agent fault: " + feedback.global_fault
            self.stop()
            raise Revo2TeleopError(self.latched_fault)
        if feedback is not None and feedback.mode_reject_reason:
            self.latched_fault = "Revo2 mode rejected: " + feedback.mode_reject_reason
            self.stop()
            raise Revo2TeleopError(self.latched_fault)
        if feedback is not None and not feedback.hand_valid:
            self.latched_fault = "Revo2 hand feedback is invalid"
            self.stop()
            raise Revo2TeleopError(self.latched_fault)
        if feedback is not None and any(feedback.hand_faults):
            self.latched_fault = "Revo2 hand fault: " + next(
                fault for fault in feedback.hand_faults if fault
            )
            self.stop()
            raise Revo2TeleopError(self.latched_fault)
        if feedback is not None and any(feedback.hand_reject_reasons):
            self.latched_fault = "Revo2 hand command rejected: " + next(
                reason for reason in feedback.hand_reject_reasons if reason
            )
            self.stop()
            raise Revo2TeleopError(self.latched_fault)
        feedback_reference_ns = self.last_feedback_received_ns or self.first_command_ns
        if (
            feedback_reference_ns is not None
            and now_ns - feedback_reference_ns > self.feedback_timeout_ns
        ):
            self.latched_fault = "Revo2 feedback timed out"
            self.stop()
            raise Revo2TeleopError(self.latched_fault)

        feedback_matches_session = (
            feedback is not None and feedback.session_id == self.session_id
        )

        # Every startup or stop begins a new arm cycle. Feedback saying "enable"
        # can be delayed from before a stop, so only an acknowledgement of an arm
        # sent in this cycle can authorize targets.
        if not self.arm_authorized:
            if (
                feedback_matches_session
                and feedback.accepted_mode == "arm"
                and self.pending_arm_seq is not None
            ):
                if not feedback.lease_valid:
                    arm_seq = self._publish("arm", targets=None, now_ns=now_ns)
                    if arm_seq >= 0:
                        self.pending_arm_seq = arm_seq
                    return "arm"
                if feedback.last_mode_seq < self.pending_arm_seq:
                    # Freeze the pending sequence instead of racing it forward;
                    # the next feedback sample can then acknowledge that command.
                    return "await_arm_ack"
                self.arm_authorized = True
                self.pending_arm_seq = None
            else:
                arm_seq = self._publish("arm", targets=None, now_ns=now_ns)
                if arm_seq >= 0:
                    self.pending_arm_seq = arm_seq
                return "arm"

        if not feedback_matches_session or feedback.accepted_mode in ("disarm", "stop"):
            self.arm_authorized = False
            arm_seq = self._publish("arm", targets=None, now_ns=now_ns)
            if arm_seq >= 0:
                self.pending_arm_seq = arm_seq
            return "arm"

        if feedback.accepted_mode == "arm":
            targets = self.profile.targets(left_trigger, right_trigger)
            self._publish("enable", targets=targets, now_ns=now_ns)
            return "enable"

        if feedback.accepted_mode == "enable":
            if not feedback.lease_valid:
                self.arm_authorized = False
                arm_seq = self._publish("arm", targets=None, now_ns=now_ns)
                if arm_seq >= 0:
                    self.pending_arm_seq = arm_seq
                return "arm"
            targets = self.profile.targets(left_trigger, right_trigger)
            self._publish("enable", targets=targets, now_ns=now_ns)
            return "enable"

        # Hold is never entered by this adapter. Re-arm explicitly instead of
        # silently treating a hold acknowledgement as permission to move.
        self.arm_authorized = False
        arm_seq = self._publish("arm", targets=None, now_ns=now_ns)
        if arm_seq >= 0:
            self.pending_arm_seq = arm_seq
        return "arm"

    def stop(self) -> None:
        if self.command is None or self.closed:
            return
        try:
            self._publish("stop", targets=None, now_ns=self.clock_ns())
        except BaseException:
            pass
        self.enabled = False
        self.awaiting_trigger_release = True
        self.pending_arm_seq = None
        self.arm_authorized = False
        self.last_feedback = None
        self.last_feedback_received_ns = None
        self.first_command_ns = None

    def close(self) -> None:
        if self.closed:
            return
        if self.command is not None:
            try:
                self.stop()
                self._publish("disarm", targets=None, now_ns=self.clock_ns())
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
