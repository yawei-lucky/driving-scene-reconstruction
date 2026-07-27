# MTGS 两机遥控 Demo

这一版只有两个进程：

```text
你的电脑：键盘 + 显示 App（两条连接都由这里主动发起）
  ├─ WebSocket client → 石迪 TCP 18765
  └─ SRT caller       → 石迪 UDP 19001；连接建立后接收 H.264

石迪电脑：MTGS 模拟 App（只监听，不反向访问你的电脑）
  ├─ WebSocket listener：控制输入 + 遥测返回
  ├─ SRT listener：H.264 视频发送
  ├─ SimpleVehicleModel 和 +/-5 m 路线边界（唯一状态真源）
  ├─ CAM_L0 + CAM_F0 + CAM_R0 → 标定 150° 前向环视
  ├─ 右上角路线几何、可驾驶走廊和运动学车辆
  └─ RTX 4090 D NVENC
```

控制端丢失超过 250 ms 后，模拟端不再沿用旧指令，而是施加全制动。
`AUTO` 使用现有拟人路线跟随器；按任一驾驶键自动切到 `REMOTE`。
右上角是可信的路线/车辆模型视图，不是伪造的俯视 RGB 或深度图。

这里的“监看端”和 SRT 的 `listener` 不是一回事。你的电脑虽然负责监看，
但因为网络连接由你的电脑主动发起，所以 SRT 参数必须是 `mode=caller`；
石迪电脑才使用 `mode=listener`。

## 1. 石迪电脑

先在项目机启动两个监听端点：

```bash
cd /home/yawei/driving-scene-reconstruction
MTGS_REMOTE_CONTROL_PORT=18765 \
MTGS_REMOTE_VIDEO_PORT=19001 \
scripts/run_stage_h3_mtgs_remote.sh server
```

石迪电脑的防火墙需要允许入站 TCP `18765` 和 UDP `19001`。默认 `8765`
在当前项目机上已有服务占用，所以这里显式使用 `18765`。

## 2. 你的电脑

需要 Python 3.10+、Tk 和 FFmpeg。首次安装：

```powershell
py -3 -m venv .venv-mtgs-driver
.\.venv-mtgs-driver\Scripts\python.exe -m pip install `
  -r .\apps\requirements-mtgs-remote-driver.txt
ffmpeg -protocols | Select-String srt
```

把 `SHIDI_IP` 换成石迪电脑可访问的地址：

```powershell
$ShidiIp = "师弟电脑的 IP"

.\.venv-mtgs-driver\Scripts\python.exe .\apps\mtgs_remote_driver.py `
  --server "ws://${ShidiIp}:18765" `
  --video-source "srt://${ShidiIp}:19001?mode=caller&latency=300000&pkt_size=1316&transtype=live"
```

你的电脑不需要开放入站端口，石迪电脑也不需要知道你的 IP。Windows
控制端默认真正全屏并保持 150° 画面的宽高比；`Ctrl+Enter` 切换全屏，
`Esc` 只退出全屏而不断开远程会话，`Ctrl+Q` 才退出程序。
`--no-fullscreen` 可强制窗口启动。

如果不在同一可信局域网，先用 Tailscale/WireGuard 组成私网。不要把未
加密的 `ws://` 控制口直接暴露到公网。可选共享 token：

```bash
# 石迪电脑；两边使用同一个临时值，不要提交到 Git
MTGS_REMOTE_CONTROL_PORT=18765 \
MTGS_REMOTE_VIDEO_PORT=19001 \
MTGS_REMOTE_TOKEN='...' scripts/run_stage_h3_mtgs_remote.sh server
```

```powershell
# 你的 PowerShell
$ShidiIp = "师弟电脑的 IP"
.\.venv-mtgs-driver\Scripts\python.exe .\apps\mtgs_remote_driver.py `
  --server "ws://${ShidiIp}:18765" `
  --video-source "srt://${ShidiIp}:19001?mode=caller&latency=300000&pkt_size=1316&transtype=live" `
  --token "两边相同的临时值"
```

FFmpeg 的 SRT `latency` 单位是微秒，`300000` 才是 300 ms。旧命令中的
`latency=80` 实际只有 0.08 ms，不足以覆盖远程网络的丢包重传。当前默认
视频为 8 Mbps、0.5 秒关键帧间隔。若远端链路仍不稳定，可在两边同时把
延迟提高到 600 ms，并把师弟端码率降到 6 Mbps：

```bash
# 石迪电脑
MTGS_REMOTE_SRT_LATENCY_US=600000 \
MTGS_REMOTE_VIDEO_BITRATE=6M \
scripts/run_stage_h3_mtgs_remote.sh server
```

```powershell
# 你的 PowerShell
$ShidiIp = "师弟电脑的 IP"
.\.venv-mtgs-driver\Scripts\python.exe .\apps\mtgs_remote_driver.py `
  --server "ws://${ShidiIp}:18765" `
  --video-source "srt://${ShidiIp}:19001?mode=caller&latency=600000&pkt_size=1316&transtype=live"
```

## 3. 控制

- `W/S` 或上下方向键：油门/制动；
- `A/D` 或左右方向键：转向；
- `P`：返回自动驾驶；
- `M`：切换人工遥控；
- `R`：重置出生点并解除急停；
- `Space`：锁存急停；
- `Ctrl+Enter`：切换全屏；
- `Esc`：只退出全屏，不断开控制或视频；
- `Ctrl+Q`：明确退出本机端，并触发安全急停。

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

2026-07-27 根据“你的电脑可访问石迪、石迪不能反向访问你的电脑”的实际
网络条件，默认 SRT 建链方向已翻转。独立 FFmpeg 烟测由石迪侧
`mode=listener`、控制侧 `mode=caller`，完整接收 3 秒、60/60 帧 H.264
测试流。随后用真实 checkpoint 完成同机双进程复测：远程控制 App 收到
69 个完整视频帧和 93 条遥测，观察到 `AUTO`、`REMOTE` 以及最大绝对值
1.0 的实际转向，视频与控制状态均为 connected。

同日针对远程端未全屏和频繁花屏继续修正。当前主机的
`ffmpeg -h protocol=srt` 明确显示 `latency` 单位是微秒，旧值 `80`
实际只有 0.08 ms。默认值现为 300 ms；直播码率由 12 降至 8 Mbps，
关键帧间隔由 1 秒缩短至 0.5 秒，并启用损坏包丢弃。新版真实 checkpoint
同机双进程回归运行 8 秒，客户端收到 142 个完整视频帧和 151 条遥测，
AUTO/REMOTE 接管正常，视频和控制最终状态均为 connected。全屏窗口本身
尚未在 Windows 桌面实际打开，环回测试也没有模拟公网丢包。

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
