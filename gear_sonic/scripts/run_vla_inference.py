"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import tyro

from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)

# Hardware, camera, Dex3, and ZMQ dependencies are imported only by the default
# CLI path. The injected loop is usable without importing robot transports.
DEFAULT_ZMQ_KEYBOARD_PORT = 5580


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    from gear_sonic.utils.inference.vla_utils import prepare_observation_for_eval

    observation = prepare_observation_for_eval(robot_model, observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity

    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        from gear_sonic.utils.inference.vla_utils import concat_action

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
    monotonic=time.monotonic,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Injectable official loop
# ---------------------------------------------------------------------------


def run_inference_loop(config: InferenceConfig, runtime, *, stop_requested=lambda: False):
    """Run the official inference/publish loop against an injected runtime.

    The CLI below keeps its existing hardware path.  This entry is the same queue,
    completion-based inference cadence, latency compensation, 50 Hz row selection,
    and final-row saturation used by that path, but delegates observation, commands,
    and publication.  Injected recorded runs can therefore exercise the real timing
    loop without importing Dex3 or opening robot sockets.
    """
    monotonic = getattr(runtime, "monotonic", time.monotonic)
    sleep = getattr(runtime, "sleep", time.sleep)
    policy = runtime.policy
    motion_key = getattr(runtime, "motion_action_key", "motion_token")
    left_key = getattr(runtime, "left_action_key", "left_hand_joints")
    right_key = getattr(runtime, "right_action_key", "right_hand_joints")
    pause_loop = getattr(runtime, "initially_paused", True)
    prompt = config.prompt
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate
    frame_counter = 0
    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    busy_event = threading.Event()

    def infer(observation):
        if hasattr(runtime, "infer"):
            return runtime.infer(observation)
        action, _info = policy.get_action(observation)
        action.pop("task_progress", None)
        action.pop("action.task_progress", None)
        return {key.replace("action.", ""): value for key, value in action.items()}

    worker = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            stop_event,
            busy_event,
            lambda: runtime.prepare_observation(prompt),
            infer,
            monotonic,
        ),
        daemon=True,
    )
    worker.start()
    try:
        while not stop_requested():
            t_start = monotonic()
            command = runtime.read_command()
            if hasattr(runtime, "handle_command"):
                runtime.handle_command(command)
            if command == "p":
                pause_loop = not pause_loop
            elif isinstance(command, str) and command.startswith("prompt:"):
                changed = command[len("prompt:") :]
                if changed:
                    prompt = changed
            elif command == "i":
                pause_loop = True
                runtime.publish_initial_pose()
                cached_action_chunk = None
                action_chunk_index = 0
                frame_counter = 0
            elif command == "k":
                runtime.toggle_control()

            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = monotonic()
                if hasattr(runtime, "action_chunk_received"):
                    runtime.action_chunk_received(inference_delay, action_chunk_index)
            except queue.Empty:
                pass

            should_start = should_trigger_new_inference(
                cached_chunk_exists=cached_action_chunk is not None,
                inference_thread_running=busy_event.is_set(),
                time_since_last_inference=monotonic() - last_inference_time,
                inference_interval=inference_interval,
            )
            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if not pause_loop and cached_action_chunk:
                motion = np.asarray(get_action_field(cached_action_chunk, motion_key), dtype=np.float32)
                left = np.asarray(get_action_field(cached_action_chunk, left_key), dtype=np.float32)
                right = np.asarray(get_action_field(cached_action_chunk, right_key), dtype=np.float32)
                if motion.ndim == 3:
                    motion = motion[0]
                if left.ndim == 3:
                    left = left[0]
                if right.ndim == 3:
                    right = right[0]
                horizon = motion.shape[0] if motion.ndim == 2 else 1
                current_idx = min(action_chunk_index, horizon - 1)
                selected_motion = motion[current_idx] if motion.ndim == 2 else motion
                selected_left = left[current_idx] if left.ndim == 2 else left
                selected_right = right[current_idx] if right.ndim == 2 else right
                runtime.publish_action(
                    selected_motion,
                    np.array([frame_counter], dtype=np.int64),
                    selected_left,
                    selected_right,
                    current_idx,
                )
                frame_counter += 1
                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            elapsed = monotonic() - t_start
            remaining = 1.0 / config.action_publish_rate - elapsed
            if remaining > 0:
                sleep(remaining)
    finally:
        stop_event.set()
        worker.join()
        runtime.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )

    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _build_default_runtime(config: InferenceConfig):
    """Construct the existing hardware transports only for the production CLI."""
    from gr00t.policy.server_client import PolicyClient
    import zmq

    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    class DefaultRuntime:
        motion_action_key = "motion_token"
        left_action_key = "left_hand_joints"
        right_action_key = "right_hand_joints"
        initially_paused = True

        def __init__(self):
            self.robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
            self.policy = PolicyClient(host=config.host, port=config.port)
            self.state = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
            self.camera = ComposedCameraClientSensor(
                server_ip=config.camera_host, port=config.camera_port
            )
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
            self.keyboard = ZMQKeyboardSubscriber(
                port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
            )
            self.control_running = False
            self.control_mode = "OFF"
            self.left_closed = False
            self.right_closed = False
            self.last_token = None

        def prepare_observation(self, prompt):
            return prepare_observation_from_sensors(
                self.camera, self.state, self.robot_model, prompt, log_errors=True
            )

        def read_command(self):
            return self.keyboard.read_msg()

        def infer(self, observation):
            return run_policy_inference_and_process(self.policy, observation, self.robot_model)

        def handle_command(self, command):
            if command in ("c", "s", "f"):
                print(f"Keyboard: {command}", flush=True)
            elif command == "[":
                self.left_closed = not self.left_closed
                print(f"Initial pose left hand: {'closed' if self.left_closed else 'open'}")
            elif command == "]":
                self.right_closed = not self.right_closed
                print(f"Initial pose right hand: {'closed' if self.right_closed else 'open'}")

        def publish_action(self, motion, frame_index, left, right, _action_index):
            self.socket.send(pack_latent_action_message(motion, frame_index, left, right))
            self.last_token = np.asarray(motion, dtype=np.float32).copy()

        def _initial_hands(self):
            left = _compute_closed_hand_joints("L") if self.left_closed else np.zeros(7, np.float32)
            right = _compute_closed_hand_joints("R") if self.right_closed else np.zeros(7, np.float32)
            return left, right

        def publish_initial_pose(self):
            if self.control_running and self.control_mode == "PLANNER":
                self._send_control(True, planner=False)
            left, right = self._initial_hands()
            if config.initial_pose_blend_duration > 0 and self.last_token is not None:
                steps = max(1, round(config.action_publish_rate * config.initial_pose_blend_duration))
                start = self.last_token.copy()
                for step in range(steps):
                    began = time.monotonic()
                    alpha = (step + 1) / steps
                    token = ((1 - alpha) * start + alpha * LATENT_INITIAL_MOTION_TOKEN).astype(np.float32)
                    self.socket.send(pack_latent_action_message(
                        token, np.array([0], dtype=np.int64), left, right
                    ))
                    self.last_token = token.copy()
                    remaining = 1.0 / config.action_publish_rate - (time.monotonic() - began)
                    if remaining > 0:
                        time.sleep(remaining)
            else:
                self.socket.send(pack_latent_action_message(
                    LATENT_INITIAL_MOTION_TOKEN, np.array([0], dtype=np.int64), left, right
                ))
                time.sleep(1.0)

        def _send_control(self, start, planner=False):
            self.socket.send(build_command_message(start=start, stop=not start, planner=planner))
            time.sleep(0.01)
            self.control_running = start
            self.control_mode = "PLANNER" if start and planner else "POSE" if start else "OFF"

        def toggle_control(self):
            if self.control_running:
                self._send_control(False, planner=self.control_mode == "PLANNER")
            else:
                self._send_control(True, planner=True)

        def close(self):
            self.socket.close()
            self.context.term()
            self.state.close()
            self.keyboard.close()

    return DefaultRuntime()


def main(config: InferenceConfig, runtime=None, *, stop_requested=lambda: False):
    """Run the single official loop with default hardware or an injected runtime."""
    if runtime is None:
        runtime = _build_default_runtime(config)
    return run_inference_loop(config, runtime, stop_requested=stop_requested)

def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
