# PICO proportional whole-hand control for BrainCo Revo2

This integration keeps SONIC whole-body teleoperation on its existing ZMQ path and publishes a separate upstream `revo2.command.v3` glove stream for the robot-side unified hand service. It does not add ROS, open the USB-to-RS485 adapters, or own the Revo2 SDK from this repository.

Each controller trigger controls one complete hand:

- left trigger → left Revo2 hand;
- right trigger → right Revo2 hand;
- trigger `0` → the validated open endpoint;
- trigger `1` → the configured allowed closed endpoint;
- intermediate values → proportional interpolation between those endpoints.

The operator does not control individual fingers. The six protocol channels are `thumb_flex`, `thumb_opposition`, `index`, `middle`, `ring`, and `pinky`.

## Architecture and ownership

```text
PICO/XRT fresh trigger sample
    -> GROOT Revo2HandSender on AGX
       XPUB bind tcp://<AGX-data-plane-IPv4>:5560
    -> G1 md_wla_hand_source.py --source glove
       SUB connect to AGX, sole XPUB bind tcp://127.0.0.1:5560
    -> persistent source-neutral unified executor
       sole SDK / USB / serial owner

unified executor feedback tcp://0.0.0.0:5561
    -> GROOT diagnostics and robot-local health probe
```

The executor service remains named `revo2-physical-agent.service`, but it no longer has glove/policy modes and does not depend on BODY heartbeat, a body run ID, `controller_id`, or an arm acknowledgement. Source selection belongs to the independent relay. Switching between `glove`, `policy`, and `manual_test` means stopping the old relay and starting one new relay; it must not restart the executor or close the serial ports.

The robot-side relay is authoritative and is supplied by the installed `md_wla_brainco` deployment. GROOT does not copy it, launch it through SSH, or manage systemd. Exactly one source relay may own the G1 loopback publisher at a time.

## Safety prerequisites

Do not enable hand actuation until all of the following have been checked on the target robot:

1. The persistent executor reports unified capabilities exactly as `{"direct_enable": true, "source_diagnostic_only": true, "body_control_interlock": false}`.
2. A robot-local `md_wla_hand_health.py` probe observes the same boot with advancing feedback, agent and per-hand measurement clocks, fresh measurements for both hands, and no fault.
3. The AGX upstream endpoint, G1 relay input endpoint, public feedback endpoint, and 200 ms lease agree.
4. Both adapters and physical hand identities have been verified using stable paths. Do not infer side from temporary `/dev/ttyUSB*` numbering.
5. A hardware owner has approved the six-number open and closed pose for each hand.
6. SONIC is launched with `--hand-mode external`, so its Dex3 publisher cannot compete with Revo2. This is backend mutual exclusion, not executor authorization.
7. No policy or manual-test source relay is already running.

Perform initial hardware validation with unloaded hands, a clear workspace, reduced close endpoints, and one hand at a time. A healthy executor does not by itself prove movement or task acceptance.

## Pose profile

The repository pins the mondo-g1-6514 profile at `gear_sonic/config/revo2/mondo_g1_6514_pico_pose_profile.v1.json`. In the documented mondo-agx checkout, its absolute path is `/mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic/config/revo2/mondo_g1_6514_pico_pose_profile.v1.json`.

Both hands currently use open `[0.1, 0.9, 0.0, 0.0, 0.0, 0.0]`, closed `[0.5, 0.9, 0.85, 0.85, 0.85, 0.85]`, and `max_close_ratio=1.0`. Wire values are normalized to `[0,1]`; they are not radians. This is an installation-specific profile, not a universal Revo2 default.

For each side, the sender computes:

\[
q_{target} = q_{open} + trigger \cdot max\_close\_ratio \cdot (q_{closed} - q_{open}).
\]

The loader rejects missing or extra keys, non-finite values, vectors of the wrong length, bool values, and values outside `[0,1]`.

## Unified command and feedback semantics

The upstream command has the exact top-level fields:

```text
schema, session_id, seq, control_frame_index, sender_monotonic_ns,
lease_ms, source, mode, actuated_sides, hands
```

`controller_id` is an illegal legacy field. `source="glove"` is a diagnostic label, not an executor permission. The relay validates the packet and then rewrites its output session, sequence, source, and timestamp before sending it to the executor; consequently executor feedback reports the relay session, not the GROOT upstream session.

A fresh trigger sample directly sends one `enable`; there is no preceding `arm` and no feedback ACK gate. Duplicate controller timestamps send nothing and cannot renew a lease. Timestamp rollback fails closed. The sender uses the locally recorded XRT receipt time for the 150 ms sample-age limit and never subtracts monotonic timestamps from different hosts. An expired target sends one `stop`, clears local target state, and is never replayed.

Wire `arm`, `hold`, `disarm`, and `stop` are cancellation modes, not permission grants. GROOT uses `stop` for cancellation. Its hand entries have `valid=false`; placeholder target numbers are not an open-hand command. Stopping therefore does not automatically release a grasped object.

Feedback is used to verify the unified capability and diagnose boot changes, non-advancing clocks, stale measurements, faults, or timeout. It does not authorize `enable`. Fresh disarmed measurements are healthy even when feedback `valid=false`.

## Launch procedure

Install the teleoperation dependencies on AGX:

```bash
pip install -e "gear_sonic[teleop]"
```

After separate site authorization, check the already-running executor locally on G1 without opening a command socket:

```bash
cd /home/unitree/lsy/md_wla_deploy
runtime/hand/venv/bin/python -I -B tools/md_wla_hand_health.py --timeout-s 30
```

Only after that probe succeeds, start this session's glove relay on G1. This command establishes an actuation data path and must not run alongside another source relay:

```bash
cd /home/unitree/lsy/md_wla_deploy
runtime/hand/venv/bin/python -I -B tools/md_wla_hand_source.py \
  --source glove \
  --input-endpoint tcp://192.168.123.79:5560
```

Start SONIC on AGX with Dex3 disabled:

```bash
(cd gear_sonic_deploy && \
  ./deploy.sh --input-type zmq_manager --output-type all --hand-mode external real)
```

Then start the PICO manager on AGX:

```bash
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --revo2-profile /mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic/config/revo2/mondo_g1_6514_pico_pose_profile.v1.json \
  --revo2-command-bind tcp://192.168.123.79:5560 \
  --revo2-feedback-connect tcp://192.168.123.164:5561 \
  --permit-revo2-actuation
```

`--permit-revo2-actuation` has no default. A partial Revo2 configuration fails at startup. Endpoints require explicit non-wildcard IPv4 addresses. The legacy non-manager PICO path does not accept hand actuation.

Revo2 requires `--input-source xrt`: Isaac Teleop 1.4 DeviceIO does not expose controller-origin sample timestamps, so it cannot prove a repeated trigger record is fresh.

## GROOT operator gates and stopping

The unified executor has no BODY interlock. GROOT nevertheless retains explicit operator-level combined behavior:

- manager `OFF` and `POSE_PAUSE` send a hand `stop` and clear the local target;
- leaving either state requires a new fresh trigger timestamp and trigger release;
- A+B+X+Y entering `OFF` requests hand stop and BODY stop through their separate paths;
- stale PICO trigger input stops only the hand path;
- BODY heartbeat loss, BODY fault, and the deployment terminal `O` key do not implicitly stop or authorize the hand;
- process shutdown best-effort stops the hand sender but does not stop the persistent executor.

The A+B+X+Y combination is an application-level pair of stop requests, not an executor BODY interlock or a physical emergency-stop confirmation. The 200 ms lease starts when the executor receives/processes a valid command; it is not an end-to-end command age and does not prove that hardware physically stopped within 200 ms.

End a session in this order: request GROOT OFF, let the manager stop/exit, stop the session-owned glove relay, and leave the persistent executor running. Relay shutdown provides another best-effort stop barrier and does not close USB.

## Offline validation

Before any hardware operation, run:

```bash
python -m pytest gear_sonic/tests/test_revo2_hand_sender.py -q
```

The fake-only tests cover the pinned profile, exact relay command fields, direct enable, duplicate and rollback handling, one-shot expiry stop, queue saturation, unified feedback capabilities, relay-session feedback, and manager decoupling from BODY permission.

Code and mock tests do not constitute deployment, action validation, physical stop validation, or hardware acceptance.
