# 可自由驾驶场景重建模拟器自动化搭建说明书

## 1. 文档目的

本文面向第一次接触场景重建的使用者。目标不是要求使用者理解
SplatAD、MTGS、坐标变换或训练细节，而是说明：准备哪些输入、自动化
系统依次做什么、每一步产生什么，以及什么情况下可以继续。

整条流水线是：

```text
定义目标
→ 准备数据
→ 校准与对齐
→ 覆盖检查
→ 训练重建
→ 驾驶画面验收
→ 生成模拟器
→ 真人试驾验收
```

阅读每个阶段时只需关注四项：

```text
输入 → 自动动作 → 输出 → 通过标准
```

任一阶段不通过都应停止，保留报告，并给出补采数据、收窄驾驶范围或
调整模型的明确建议。

## 2. 第一阶段：定义目标

**输入：** 一份场景需求，例如道路范围、目标速度、横向范围、路口
分支、相机数量和是否需要动态交通。

**自动动作：** 检查目标是否完整，并生成后续步骤共用的配置。

**输出：** `scene_spec.yaml`。

**通过标准：** 每项驾驶能力都有可测量边界。

示例：

```yaml
scene:
  name: singapore_block_demo
driving:
  route_length_m: 100
  lateral_limit_m: 5
  max_speed_mps: 15
  required_branches: [straight, right]
sensors:
  cameras: [front]
dynamic_traffic: false
```

## 3. 第二阶段：准备采集数据

**输入：** 同一区域的相机、LiDAR、车辆轨迹、时间戳和标定数据。宽范围
驾驶应优先使用多次经过、相邻车道和各个目标路口分支的数据。

**自动动作：** 校验文件完整性、数据许可、传感器数量、时间范围和
校验和，并整理为统一目录。

**输出：**

```text
raw_scene/
├── images/
├── lidar/
├── poses/
├── calibration/
└── manifest.json
```

**通过标准：** 所需传感器和路线数据齐全；只有单次单路线时，报告必须
提示可驾驶范围可能很窄。

## 4. 第三阶段：校准与统一坐标

**输入：** `raw_scene/`。

**自动动作：** 完成相机去畸变、相机/LiDAR/IMU 时间同步、多次采集轨迹
对齐，并把全部传感器姿态转换到同一个世界坐标系。

**输出：**

```text
processed_scene/
├── cameras/
├── lidar/
├── world_poses/
└── calibration_report.json
```

**通过标准：** 没有明显时间跳变、退化轨迹或超标的跨轨迹对齐误差。
对齐失败时不能通过增加训练步数解决。

## 5. 第四阶段：检查可驾驶空间覆盖

**输入：** `processed_scene/` 和 `scene_spec.yaml`。

**自动动作：** 统计每个位置的相机距离、观察方向、重复采集次数、
LiDAR 密度和遮挡情况；识别有真实数据的路线、车道和路口分支。

**输出：**

```text
coverage/
├── support_map.json
├── coverage_visualization.png
└── audit_verdict.json
```

**通过标准：** 目标路线和横向范围处于观测支持区内。输出应明确写出：

```text
路线支持：0–84 m
横向支持：±5 m
支持分支：直行、右转
不支持分支：左转
```

这里的边界将成为模拟器的运行时保护范围。未观测区域即使可以生成
合理画面，也不能标记为真实重建证据。

## 6. 第五阶段：训练场景重建

**输入：** 已对齐的数据、覆盖地图和训练配置。

**自动动作：** 划分训练/验证数据，选择重建后端，训练并保存 checkpoint。
单路线小范围可使用 SplatAD；重复采集和宽范围优先使用 MTGS 风格的
共享静态背景。动态对象作为后续独立阶段处理。

**输出：**

```text
reconstruction/
├── checkpoint/
├── config.yaml
├── metrics.json
└── preview.mp4
```

**通过标准：** checkpoint 可以重新加载，验证视角能够产生有效且数值
有限的输出，基本图像和几何指标达到预设门槛。训练通过只代表模型拟合
成功，不代表已经可以驾驶。

## 7. 第六阶段：驾驶画面验收

**输入：** checkpoint、目标驾驶范围和覆盖地图。

**自动动作：** 渲染中心线、横向偏移、转向、连续变道、路线接缝和目标
速度视频，同时记录图像有效性、渲染时间和支持余量。

**输出：**

```text
validation/
├── drive_probe.mp4
├── contact_sheet.jpg
├── metrics.json
└── verdict.json
```

**通过标准：**

- 道路、车道线、路缘和前方通道连续可辨；
- 不出现会改变驾驶判断的浮点、重影或假障碍；
- 请求位姿没有越过覆盖边界；
- 渲染速度满足目标交互帧率；
- 失败项与适用边界写入 `verdict.json`。

PSNR、SSIM 或“所有像素有限”不能单独作为驾驶通过标准。

## 8. 第七阶段：生成驾驶模拟器

**输入：** 通过验收的 checkpoint、路线、支持边界和车辆配置。

**自动动作：** 打包场景，并连接控制输入、车辆运动、世界位姿、场景
渲染、显示和证据记录。

**输出：**

```text
simulator_scene/
├── checkpoint/
├── renderer.yaml
├── route.json
├── support_map.json
├── vehicle.yaml
└── scene_manifest.json
```

运行时链路是：

```text
键盘或方向盘
→ 车辆 x/y/yaw/speed
→ 覆盖边界检查
→ 相机世界位姿
→ 重建模型渲染
→ 本机画面与驾驶日志
```

**通过标准：** 支持加速、刹车、转向和重置；离开支持区时 fail closed
并停车；每帧能够追溯到控制、车辆状态、支持余量和渲染结果。

## 9. 第八阶段：真人试驾验收

**输入：** 已打包的 `simulator_scene/`。

**自动动作：** 引导驾驶员完成加速、左右偏离、恢复、变道或转弯、
制动、边界触发和重置，并自动录制全部证据。

**输出：**

```text
human_trial/
├── video.mp4
├── controls.json
├── poses.json
├── render_metrics.json
└── final_verdict.json
```

**通过标准：** 驾驶员不需要补偿重建缺陷，目标速度和路线能够完成，
越界保护有效，画面不存在影响驾驶决策的错误。最终结论必须注明认证
范围，例如：

```text
状态：可驾驶
最大已测速度：12 m/s
最大已测横向偏移：±4 m
已测路线：当前道路及右转
传感器：单前相机
动态交通：未认证
```

## 10. 面向使用者的一键入口

最终产品应把内部步骤收敛成以下命令：

```bash
simulator prepare scene_spec.yaml
simulator audit scene_spec.yaml
simulator train scene_spec.yaml
simulator validate scene_spec.yaml
simulator package scene_spec.yaml
simulator run scene_spec.yaml
simulator report scene_spec.yaml
```

完整自动运行可进一步简化为：

```bash
simulator build scene_spec.yaml
```

失败信息应直接说明阶段、原因和下一步，例如：

```text
失败阶段：空间覆盖检查
原因：道路右侧 3–5 m 没有足够相机观测
建议：补采相邻车道，或把 lateral_limit_m 降到 2
```

上述 `simulator ...` 是目标命令设计，不是本仓库当前已有的统一 CLI。
自动化实现时，应让每个子命令可重复运行、复用已完成产物，并在输入或
配置变化时使相关缓存失效。

## 11. 本仓库当前所在位置

当前项目已经分别具备：

- PandaSet 和 TbV 的 SplatAD 数据、训练、渲染与受限驾驶证据；
- 官方 MTGS checkpoint 的加载和单前相机渲染；
- MTGS `±5 m` 离散空间探针；
- `12 m/s`、`±4 m`、72 m 连续固定时间视频；
- 基于实际路线的车辆和 fail-closed 支持适配器；
- 由模拟驾驶员实际输出油门、刹车和转向的 MTGS 车辆/渲染闭环。

当前尚未完成统一的一键 CLI。最近的实现步骤是寻找一条连续
300–500 m、具有重复采集覆盖的兼容路线，将其划分为重叠的 80–100 m
重建块，并先验证一对真实相邻块。通过后再实现 checkpoint 预加载和切换，
同时把现有实验脚本逐步包装为本说明书中的 `audit`、`train`、
`validate`、`package` 和 `run` 阶段。
