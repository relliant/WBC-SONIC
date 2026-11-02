# SONIC PICO 遥操作 Quick Start

## 0. 环境校验

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl        # 进入项目根目录
ip -br addr show eno1                          # 查看有线网卡地址，应为 192.168.123.79/24
ping -c 3 192.168.123.164                      # 测试到机载 NX 的连通性
~/TensorRT/bin/trtexec --version | head -1     # 查看 TensorRT 版本，应为 v100700
.venv_teleop/bin/python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"   # 校验 torch 与 CUDA
ss -tlnp | grep -E "5556|5557|5560"                 # 检查端口占用，有输出说明已有会话在跑
```

## 1. 终端一：Pico streamer

### 只控制身体（不控制 Revo2）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl        # 进入项目根目录
source .venv_teleop/bin/activate               # 激活遥操作环境
python gear_sonic/scripts/pico_manager_thread_server.py --manager
# 需要 VR 可视化时增加：--vis_vr3pt --vis_smpl（须在 venv 内安装 pyvista，且需图形界面）
```

### Pico 手柄同时控制 Revo2

启动前必须确认机器人侧运行的是已核实的 `glove` / native-SONIC physical agent，而不是 Thor/WLA 的 `policy` 服务。下面的 controller ID 必须从**当前这次** physical-agent 启动日志中的 `revo2.physical_agent_ready.v3` 复制；agent 重启后需要重新读取。姿态文件必须包含现场验证过的左右手六维 open/closed 端点。

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl
source .venv_teleop/bin/activate

# 必须替换为现场已验证的配置；不要直接使用全 0 / 全 1 姿态。
export REVO2_PROFILE=/absolute/path/to/validated-revo2-profile.json
export REVO2_CONTROLLER_ID=pico-revo2-physical-service-YYYYMMDDTHHMMSSZ

[[ -s "$REVO2_PROFILE" ]] || { echo "Revo2 profile 不存在"; exit 1; }
[[ "$REVO2_CONTROLLER_ID" != *YYYYMMDD* ]] || { echo "必须替换 controller ID"; exit 1; }

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --revo2-profile "$REVO2_PROFILE" \
  --revo2-command-bind tcp://0.0.0.0:5560 \
  --revo2-feedback-connect tcp://192.168.123.164:5561 \
  --revo2-controller-id "$REVO2_CONTROLLER_ID" \
  --permit-revo2-actuation
```

以上命令由 AGX 在 `5560` 绑定 Revo2 命令发布端，并连接机载 NX 的 `5561` feedback 端。若实际 physical-agent launcher 使用了不同地址或端口，必须以机器人当前 launcher 为准，不能照抄历史值。当前 Revo2 路径只支持 `--input-source xrt`。

## 2. 终端二：C++ 部署（真机）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic_deploy   # 进入部署目录
export TensorRT_ROOT=$HOME/TensorRT                        # 指定 TensorRT 路径
source scripts/setup_env.sh                                # 加载环境变量

# 只使用原有 Dex3 手：
./deploy.sh --input-type zmq_manager --hand-mode dex3 real

# 使用 Pico -> Revo2 时必须禁用 Dex3，并保留 ZMQ 状态输出：
./deploy.sh --input-type zmq_manager --output-type all --hand-mode external real

# 手动指定网卡时使用：
./deploy.sh --input-type zmq_manager --output-type all --hand-mode external eno1
```

当前默认 encoder、decoder 和 observation config 已切换为 SONIC v1.1。`external` 模式不会初始化或发布 Dex3 命令；Revo2 sender 只有在 C++ SONIC 进入允许控制状态后才会发送 `enable`。

## 3. 终端三：实时可视化（可选）

```bash
cd /mnt/data/lsy/GROOT-WholeBodyControl/gear_sonic_deploy    # 进入部署目录
source ../.venv_teleop/bin/activate                          # 激活遥操作环境
python visualize_motion.py --realtime_debug_url tcp://192.168.123.79:5557    # 观察目标姿态与实测姿态
```

## 4. 头显配置

```
PICO 连接与 AGX 相同的 Wi-Fi
XRoboToolkit -> PC Service: 192.168.102.206
确认状态显示 WORKING
Tracking: Head + Controller
Data/Control: Send
Pico Motion Tracker: Full body
```

## 5. 键位

| 功能 | 按键 |
|---|---|
| 启动策略 / 急停 | A+B+X+Y |
| 左 Revo2 整体比例开合 | 左扳机：0=已验证张开端点，1=配置允许的闭合端点 |
| 右 Revo2 整体比例开合 | 右扳机：0=已验证张开端点，1=配置允许的闭合端点 |
| 进 / 出 POSE | A+X |
| POSE <-> PLANNER_FROZEN_UPPER | B+Y |
| 进入 VR_3PT | 左摇杆按下 |
| 键盘急停 | 部署终端按 O |

## 6. 停止与确认

```bash
# 部署终端按 O 停止，终端一和终端二分别 Ctrl+C
ss -tlnp | grep -E "5556|5557|5560"                 # 确认端口已释放
```

Revo2 收到停止、断流或授权丢失后会停止续租，但不会自动发送张开姿态，以免释放正在抓取的物体。physical agent 的 200 ms 本地 lease 是最终断流保护；故障后不要直接恢复旧命令，应重新启动 manager、确认双扳机已松开，再重新 arm。

```bash
ssh unitree@192.168.123.164 'python3 -' <<'EOF'    # 检查电机使能状态，期望 Counter({0: 35})
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
