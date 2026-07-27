# MTGS 两机遥控 Demo

这一版只有两个进程：

```text
你的电脑：键盘 + 显示 App
  ├─ WebSocket/TCP  → 方向、油门、刹车、模式、重置、急停
  └─ SRT/UDP        ← H.264 驾驶画面

石迪电脑：MTGS 模拟 App
  ├─ SimpleVehicleModel 和 +/-5 m 路线边界（唯一状态真源）
  ├─ CAM_L0 + CAM_F0 + CAM_R0 → 标定 150° 前向环视
  ├─ 右上角路线几何、可驾驶走廊和运动学车辆
  └─ RTX 4090 D NVENC → SRT
```

控制端丢失超过 250 ms 后，模拟端不再沿用旧指令，而是施加全制动。
`AUTO` 使用现有拟人路线跟随器；按任一驾驶键自动切到 `REMOTE`。
右上角是可信的路线/车辆模型视图，不是伪造的俯视 RGB 或深度图。

## 1. 你的电脑

需要 Python 3.10+、Tk 和 FFmpeg。首次安装：

```bash
python -m venv .venv-mtgs-driver
.venv-mtgs-driver/bin/pip install \
  -r apps/requirements-mtgs-remote-driver.txt
```

先启动本机端，把 `SHIDI_IP` 换成石迪电脑的局域网地址：

```bash
.venv-mtgs-driver/bin/python apps/mtgs_remote_driver.py \
  --server ws://SHIDI_IP:18765 \
  --video-listen \
  'srt://0.0.0.0:19001?mode=listener&latency=80&transtype=live'
```

Windows 激活命令和 Python 路径不同，但 App 参数相同。Windows 防火墙
需要允许 Python/FFmpeg 接收 UDP `19001`。

## 2. 石迪电脑

本机端开始等待后，在项目机执行：

```bash
cd /home/yawei/driving-scene-reconstruction
MTGS_REMOTE_CLIENT_HOST=YOUR_IP \
MTGS_REMOTE_CONTROL_PORT=18765 \
MTGS_REMOTE_VIDEO_PORT=19001 \
scripts/run_stage_h3_mtgs_remote.sh server
```

`YOUR_IP` 是你的电脑局域网地址。石迪电脑需要允许入站 TCP `18765`；
它会主动向你的电脑 UDP `19001` 发送 SRT 视频。默认 `8765` 在当前项目
机上已有服务占用，所以示例显式使用 `18765`。

如果不在同一可信局域网，先用 Tailscale/WireGuard 组成私网。不要把未
加密的 `ws://` 控制口直接暴露到公网。可选共享 token：

```bash
# 两边使用同一个临时值；不要提交到 Git
MTGS_REMOTE_CLIENT_HOST=YOUR_IP \
MTGS_REMOTE_CONTROL_PORT=18765 \
MTGS_REMOTE_VIDEO_PORT=19001 \
MTGS_REMOTE_TOKEN='...' scripts/run_stage_h3_mtgs_remote.sh server

.venv-mtgs-driver/bin/python apps/mtgs_remote_driver.py \
  --server ws://SHIDI_IP:18765 --token '...'
```

## 3. 控制

- `W/S` 或上下方向键：油门/制动；
- `A/D` 或左右方向键：转向；
- `P`：返回自动驾驶；
- `M`：切换人工遥控；
- `R`：重置出生点并解除急停；
- `Space`：锁存急停；
- `Esc`：退出本机端。

模拟端权威地执行运动学自行车模型、15 m/s 限速、路线终点停车和
`+/-5 m` 支撑边界。客户端只发送意图，不能直接写世界位姿。

## 4. 当前验证边界

2026-07-26 的同机双进程实测使用真实 checkpoint、SRT 和 WebSocket：
模拟端发送 120 条遥测，本机端收到 120 条并解码 97 个最新完整视频帧。
首个 GOP 建链期间丢弃旧帧是客户端的 latest-frame 策略。三相机 150°
投影覆盖为 100%；183 帧成品的三相机渲染 p50/p95 为
22.16/27.80 ms，CPU 拼接为 20.32/20.97 ms，足够
第一版约 20 Hz 的 1280x544 驾驶流。
另一轮带断言的短回归实际观察到 `AUTO` 和 `REMOTE` 两种模式及非零
应用转向，确认 W+A 接管已进入石迪端车辆循环。

这还不是两台真实电脑的 LAN 时延验收。下一步只需在两端按上述命令各
运行一次，确认防火墙、按键接管、急停、重置和 5 分钟连续连接。场景仍
只有约 84 m，动态时间固定，且没有碰撞真值；网络通路不会扩大重建覆盖。
项目机是无图形界面的 TTY，因此本次只运行了客户端的无界面接收/控制
路径；Tk 窗口本身需要在你的桌面电脑上完成第一次实机打开。

## 5. PPT 演示视频

在石迪电脑的项目目录执行：

```bash
scripts/run_stage_h3_mtgs_remote.sh ppt-demo
```

它生成一条 1280x720、20 FPS、11 秒的演示视频：上方是真实 MTGS
checkpoint 的 150° 驾驶舱，下方是控制端 App 面板。时序固定为
`AUTO → REMOTE(W/A/D) → E-STOP → RESET → AUTO`。REMOTE 指令会以
`RemoteControlPacket` 进入同一个控制权、看门狗和车辆模型，再驱动
MTGS 位姿渲染，不是后期伪造按键动画。

视频会明确标注 `SIMULATED OPERATOR`。它适合说明两端 App 的交互方式，
但不代表真人键盘、真实 WebSocket/SRT 网络或两机时延已测试。默认输出
目录是：

```text
/home/yawei/stage3_external/artifacts/mtgs_app_control_ppt_20260727_v2
```
