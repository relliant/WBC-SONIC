# PICO proportional whole-hand control for BrainCo Revo2

This integration keeps SONIC whole-body teleoperation on its existing ZMQ path and sends hand commands over a separate `revo2.command.v3` path to the robot-side Revo2 physical agent. It does not add ROS and does not open the USB-to-RS485 adapters from this repository.

Each controller trigger controls one complete hand:

- left trigger → left Revo2 hand;
- right trigger → right Revo2 hand;
- trigger `0` → the validated open endpoint;
- trigger `1` → the configured allowed closed endpoint;
- intermediate values → proportional interpolation between those endpoints.

The operator does not control individual fingers. The adapter generates the six protocol channels in this order: `thumb_flex`, `thumb_opposition`, `index`, `middle`, `ring`, `pinky`.

## Safety prerequisites

Do not enable hand actuation until all of the following have been checked on the target robot:

1. The physical agent and `sonic_contract.py` are the expected pinned versions and accept `revo2.command.v3` / `revo2.feedback.v3` with source `glove`.
2. The agent's expected controller ID, command endpoint, feedback endpoint, and 200 ms lease agree with the sender configuration.
3. Both USB-to-RS485 adapters enumerate under stable device paths, with the left and right hand identities verified. Do not infer side from temporary `/dev/ttyUSB*` numbering.
4. The baud rate and Revo2 slave IDs match the robot configuration (historically 460800 baud, left ID 126, right ID 127; verify rather than assuming these values).
5. A hardware owner has validated the six-number open and closed pose for each hand. Do not use all-zero/all-one endpoints unless they have been explicitly validated for that hand and firmware.
6. The SONIC process is launched with `--hand-mode external`. Running the Dex3 publisher and Revo2 actuation at the same time is unsupported.

Perform initial hardware validation with an unloaded hand, a clear workspace, reduced close endpoints, and one hand at a time.

## Pose profile

Create a local JSON file only after measuring and approving the four six-channel endpoint vectors. Wire values are normalized to `[0, 1]`; they are not radians. `max_close_ratio` scales the trigger's allowed travel within the validated open-to-closed interval.

The required schema is:

```text
{
  "schema": "revo2.pico_pose_profile.v1",
  "open": {
    "left":  [L_thumb_flex_open, L_thumb_opposition_open, L_index_open, L_middle_open, L_ring_open, L_pinky_open],
    "right": [R_thumb_flex_open, R_thumb_opposition_open, R_index_open, R_middle_open, R_ring_open, R_pinky_open]
  },
  "closed": {
    "left":  [L_thumb_flex_closed, L_thumb_opposition_closed, L_index_closed, L_middle_closed, L_ring_closed, L_pinky_closed],
    "right": [R_thumb_flex_closed, R_thumb_opposition_closed, R_index_closed, R_middle_closed, R_ring_closed, R_pinky_closed]
  },
  "max_close_ratio": VALIDATED_RATIO_IN_0_TO_1
}
```

The symbolic entries above are deliberately not a usable hardware profile. Replace every entry with a validated JSON number. The loader rejects missing/extra keys, non-finite values, vectors of the wrong length, and values outside `[0, 1]`.

For each side, the sender computes:

\[
q_{target} = q_{open} + trigger \cdot max\_close\_ratio \cdot (q_{closed} - q_{open}).
\]

## Launch

Install the teleoperation dependencies in the Python environment:

```bash
pip install -e "gear_sonic[teleop]"
```

Launch SONIC with its existing body input and disable the Dex3 backend:

```bash
(cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager --hand-mode external real)
```

Start the already-installed robot-side Revo2 physical agent using its verified pinned launcher and `glove`/native-SONIC mode. Do not replace the pinned runtime or enable a different policy-mode service for this path.

Then start the PICO manager with every safety-sensitive value explicit:

```bash
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --revo2-profile /absolute/path/to/validated-revo2-profile.json \
  --revo2-command-bind tcp://0.0.0.0:<verified-command-port> \
  --revo2-feedback-connect tcp://<robot-ip>:<verified-feedback-port> \
  --revo2-controller-id <controller-id-configured-in-agent> \
  --permit-revo2-actuation
```

`--permit-revo2-actuation` is mandatory and has no default. Supplying only part of the Revo2 configuration fails at startup. The legacy non-manager PICO path does not accept Revo2 actuation.

Revo2 actuation currently requires `--input-source xrt`. Isaac Teleop 1.4 DeviceIO exposes controller values but not controller-origin sample timestamps, so it cannot prove that a repeated trigger record is fresh; combining `--input-source isaac-teleop` with any Revo2 option therefore fails closed at startup.

The command publisher binds locally; the physical agent connects to it. The physical agent binds its feedback publisher; the PICO sender connects to that endpoint. Substitute the endpoints from the verified robot launcher rather than copying historical addresses.

## Runtime interlocks

The hand sender only renews the receiver lease when all of these conditions hold:

- the manager is in an active streaming mode, not `OFF` or `POSE_PAUSE`;
- a fresh C++ `sonic_control_status` heartbeat explicitly reports that SONIC is in `CONTROL`, uses `external` hand mode, and is not stopping;
- the independently sampled dual-trigger timestamp advanced, and the local trigger sample is no older than 150 ms;
- physical-agent feedback is fresh and matches the current session;
- an `arm` command for the current session has been acknowledged before `enable`;
- both triggers were released below `0.05` after startup or any stop.

Repeated trigger timestamps do not renew the hand lease. A trigger stream that becomes older than 150 ms, timestamp rollback, malformed/replayed agent feedback, agent restart, mode rejection, and feedback timeout latch the hand path off for the remainder of the manager process. Body tracking cannot substitute its timestamp for stale controller values. These hand faults do not stop the body loop.

A manager stop, `POSE_PAUSE`, local C++ emergency stop, stale C++ status, exception, or process shutdown sends best-effort `stop` and stops renewing the lease. Shutdown also sends best-effort `disarm`. Stop/disarm deliberately does **not** command the open pose, which avoids automatically dropping a grasped object. The physical agent's local lease timeout remains the final receiver-side loss-of-sender protection.

## Validation sequence

Before moving hardware:

```bash
python -m pytest gear_sonic/tests/test_revo2_hand_sender.py -q
```

For staged hardware acceptance:

1. Run the physical agent in identity/feedback-only checks and verify both sides.
2. Start SONIC in `external` hand mode and verify no Dex3 command topics are created or written.
3. Start the PICO manager with actuation disabled and verify body teleoperation plus the `sonic_control_status` heartbeat.
4. Enable the Revo2 sender with reduced, approved close endpoints.
5. Release both triggers, arm one hand in a clear workspace, and test small trigger travel.
6. Repeat for the other hand, then test both hands.
7. Verify PICO OFF, `POSE_PAUSE`, local C++ emergency stop, PICO disconnect, hand-agent disconnect, and network interruption all stop lease renewal and do not replay an old target after recovery.

Code and mock tests alone do not constitute hardware acceptance.
