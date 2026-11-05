# SONIC PICO 遥操作 Quick Start

## 0. 环境校验

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl        # 进入项目根目录
ip -br addr show eno1                          # AGX 数据面应为 192.168.123.79/24
ping -c 3 192.168.123.164                      # 测试到 G1 的连通性
~/TensorRT/bin/trtexec --version | head -1     # TensorRT 应为 v100700
.venv_teleop/bin/python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
ss -tlnp | grep -E "5556|5557|5560"           # AGX 上已有输出表示可能存在其他会话
```

## 1. 终端一：PICO manager

### 只控制身体（不控制 Revo2）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
# 需要 VR 可视化时增加：--vis_vr3pt --vis_smpl
```

### PICO 手柄同时控制 Revo2

Revo2 使用机器人板端常驻的 source-neutral unified executor。GROOT 不直接连接 executor 的 command socket：AGX 发布上游命令，G1 上的独立 `md_wla_hand_source.py` relay 连接 AGX，并作为唯一 publisher 绑定 G1 `tcp://127.0.0.1:5560`。不要再启动旧 glove/policy 模式 physical agent，也不要为切换 source 重启常驻 executor。

在另行获得现场操作授权后，先在 G1 本机执行只读健康检查：

```bash
cd /home/unitree/lsy/md_wla_deploy
runtime/hand/venv/bin/python -I -B tools/md_wla_hand_health.py --timeout-s 30
```

只有输出 `healthy=true` 后，才为本次 GROOT 会话启动 glove relay。下面的命令会建立动作数据通路，不是只读检查；同一时刻只能有一个 `glove`、`policy` 或 `manual_test` relay：

```bash
cd /home/unitree/lsy/md_wla_deploy
runtime/hand/venv/bin/python -I -B tools/md_wla_hand_source.py \
  --source glove \
  --input-endpoint tcp://192.168.123.79:5560
```

relay ready 后，在 AGX 启动 manager：

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl
source .venv_teleop/bin/activate

# 仓库内固定的 mondo-g1-6514 现场 profile；不要替换为全 0 / 全 1 姿态。
export REVO2_PROFILE=/mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic/config/revo2/mondo_g1_6514_pico_pose_profile.v1.json
[[ -s "$REVO2_PROFILE" ]] || { echo "Revo2 profile 不存在"; exit 1; }

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --revo2-profile "$REVO2_PROFILE" \
  --revo2-command-bind tcp://192.168.123.79:5560 \
  --revo2-feedback-connect tcp://192.168.123.164:5561 \
  --permit-revo2-actuation
```

`--revo2-command-bind` 是 AGX 上供 G1 glove relay 连接的 upstream endpoint；它不是 G1 executor 的 loopback endpoint。`--revo2-feedback-connect` 连接 unified executor 对数据面发布的反馈。当前 Revo2 路径只支持 `--input-source xrt`，并继续要求显式 profile、两个 endpoint 和 `--permit-revo2-actuation`。

命令不包含 `controller_id`。relay 会严格验证上游 packet，并重写发往 executor 的 session、sequence、source 和本机 timestamp。fresh trigger sample 无需 `arm`、BODY heartbeat 或 C++ CONTROL ACK，直接单发 `enable`；重复 timestamp 不会重发旧目标或续租。

## 2. 终端二：C++ 部署（真机）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT
source scripts/setup_env.sh

# 只使用原有 Dex3 手：
./deploy.sh --input-type zmq_manager --hand-mode dex3 real

# 使用 PICO -> Revo2 时必须禁用 Dex3：
./deploy.sh --input-type zmq_manager --output-type all --hand-mode external real

# 手动指定网卡时使用：
./deploy.sh --input-type zmq_manager --output-type all --hand-mode external eno1
```

当前默认 encoder、decoder 和 observation config 已切换为 SONIC v1.1。`external` 只负责不初始化、不发布 Dex3 命令，以保证两个手部 backend 互斥；它不再给 Revo2 授权，也不是 unified executor 的 BODY interlock。

## 3. 终端三：实时可视化（可选）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic_deploy
source ../.venv_teleop/bin/activate
python visualize_motion.py --realtime_debug_url tcp://192.168.123.79:5557
```

## 4. 头显配置

```text
PICO 连接与 AGX 相同的 Wi-Fi
XRoboToolkit -> PC Service: 192.168.102.206
确认状态显示 WORKING
Tracking: Head + Controller
Data/Control: Send
Pico Motion Tracker: Full body
```

## 5. 键位与停止边界

| 功能 | 按键 |
|---|---|
| 启动策略 / GROOT 组合停止请求 | A+B+X+Y |
| 左 Revo2 整体比例开合 | 左扳机：0=已验证张开端点，1=配置允许的闭合端点 |
| 右 Revo2 整体比例开合 | 右扳机：0=已验证张开端点，1=配置允许的闭合端点 |
| 进 / 出 POSE | A+X |
| POSE <-> PLANNER_FROZEN_UPPER | B+Y |
| 进入 VR_3PT | 左摇杆按下 |
| 停止 BODY 部署 | 部署终端按 O（仅 BODY，不会隐式停止 Revo2） |

GROOT manager 启动时处于 `OFF`。本集成明确把 `OFF` 和 `POSE_PAUSE` 作为自己的上层 hand-stop 门槛；A+B+X+Y 进入 `OFF` 时分别请求 hand stop 和 BODY stop。该组合请求不是 unified executor 内部的 BODY interlock，也不是物理停止确认。BODY heartbeat、BODY fault 和部署终端 `O` 不再自动授予或撤销手部动作。

## 6. 停止与确认

1. 使用 A+B+X+Y 让 GROOT manager 分别请求 BODY 和 hand stop。
2. manager 退出后，结束本次 glove relay；relay 会 best-effort 再发送 stop。
3. 常驻 `revo2-physical-agent.service` 保持运行，不因 source/session 切换而重启或关闭 USB。
4. 最后按需要结束 C++ 部署和可视化进程。

```bash
ss -tlnp | grep -E "5556|5557|5560"  # 在对应主机确认本会话端口已释放
```

Revo2 stop/断流会清除目标并停止续租，但不会自动发送张开姿态，以免释放正在抓取的物体。新的动作必须来自严格递增的新鲜 trigger sample；启动或 stop 后还必须先松开双扳机。200 ms 是 executor 接收有效命令后计算的本地 lease，不是端到端年龄，也不保证硬件已在 200 ms 内物理停止。

以下 BODY 电机状态检查仅供另行获授权的现场操作使用，不属于本次 GROOT/unified Revo2 启动链路：

```bash
ssh unitree@192.168.123.164 'python3 -' <<'EOF'
import time
from collections import Counter
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
received = {}
def handler(message):
    received["state"] = message
ChannelFactoryInitialize(0, "eth1")
subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(handler, 10)
deadline = time.time() + 5
while time.time() < deadline and "state" not in received:
    time.sleep(0.1)
subscriber.Close()
print(Counter(motor.mode for motor in received["state"].motor_state))
EOF
```
