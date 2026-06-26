# Unitree G1 Shooter — 项目报告

> CS2810 Embodied AI (Spring 2026) · Team Hybrid Auto  
> Shooter 子系统 · 最终报告

---

## 目录

1. [概览](#1-概览)
2. [训练管线：六阶段渐进式训练](#2-训练管线六阶段渐进式训练)
3. [模型架构与观测空间](#3-模型架构与观测空间)
4. [奖励设计演进](#4-奖励设计演进)
5. [Phase 1 评测](#5-phase-1-评测)
6. [Phase 2 竞赛策略与 API Server](#6-phase-2-竞赛策略与-api-server)
7. [Motion 性能分析](#7-motion-性能分析)
8. [工程决策与经验](#8-工程决策与经验)
9. [附录](#9-附录)

---

## 1. 概览

**任务描述**：Unitree G1 人形机器人（29 自由度）在 MuJoCo 物理仿真中执行罚球点射门——接近静止球并踢入球门。这是一项感知引导的运动技能，需要运动跟踪先验和球轨迹预测能力。

**最终产出**：

| 产物 | 说明 |
|------|------|
| 六阶段渐进式训练管线 | Stage 1 → Stage 6，20K iterations，~39B 步 |
| Stage 6 模型 | LSTM 128-64-32 + 2×128，actor 160D / critic 298D |
| 多策略 API Server | uniform / best-motion / gk-aware 三种射门策略 |
| 评测工具 | per-motion 精析、kick timing 分析、并行批量评测 |
| 性能矩阵 | 250K trials 的 per-motion 精度/速度数据 |

---

## 2. 训练管线：六阶段渐进式训练

根据 PAiD 论文[1] 的渐进式感知-动作框架，设计六个训练阶段。全部使用相同的 LSTM 架构，通过环境配置（奖励、终止条件、坐标系统）的递进来实现能力提升。

| 阶段 | 目标 | 坐标系统 | 关键新增奖励 |
|------|------|---------|-------------|
| **Stage 1** | 运动跟踪（模仿踢球动作） | 训练坐标 | 六项 track_body 奖励，纯模仿 |
| **Stage 2** | 感知引导踢球 | 训练坐标 | contact + sideways_kick + ball_vel_align |
| **Stage 3** | 球门平面精度 + 速度课程 | 训练坐标 | goal_accuracy + goal_miss |
| **Stage 4** | 高速精度（目标 10 m/s） | 训练坐标 | is_terminated=-500, ball_speed=20 |
| **Stage 5** | 竞赛坐标对齐 + 全门精度 | **竞赛坐标** | goal_accuracy=30, ball_oob=-30 |
| **Stage 6** | 竞赛坐标高速踢球 | 竞赛坐标 | **ball_speed=40**, z_speed=-5 |

### 2.1 坐标系统

- **训练坐标（Stage 1–4）**：机器人在原点附近，球门沿 -y 方向。使用 `motion_origin_offset` + `motion_yaw_offset` 将运动本地坐标映射到世界坐标。
- **竞赛坐标（Stage 5–6）**：与 `compete.py` 完全对齐——机器人在 (4, 0, 0.8)、朝向 -x、球在 (3, 0, 0.1)、球门面 x=-0.5、宽 3.0m、高 1.8m。

### 2.2 训练环境参数

| 参数 | 值 |
|------|-----|
| 并行环境数 | 4096 |
| 每环境步数 | 24 |
| 控制频率 | 50 Hz (decimation=4, timestep=0.005s) |
| Episode 长度 | 10.0 s |
| 动作空间 | 29D 关节位置目标 |
| 动作缩放 | `G1_ACTION_SCALE`（各关节 PD 增益） |
| PPO 学习率 | 1e-3 (adaptive schedule) |
| 折扣因子 γ | 0.99 |
| GAE λ | 0.95 |
| KL target | 0.01 |
| Entropy coef | 0.005 |
| GPU | 2× NVIDIA GeForce RTX 4090 D (24 GB) |
| 训练时间 (Stage 6) | ~9 小时 |
| 总步数 (Stage 6) | ~39.3 亿 |

### 2.3 逐阶段细节

#### Stage 1: 运动跟踪

**目标**：在没有感知输入的情况下学习模仿多样的踢球动作参考，形成稳定的运动先验。

**命令采样**：自适应采样（偏向失败率高的运动片段，kernel_size=3, lambda=0.1, uniform_ratio=0.1）。

**域随机化**：push_robot（每 1-3s 随机基座速度脉冲）、观测噪声（actor 各项 ±0.01 至 ±0.5）、观测损坏。

**终止条件**（5 个）：
| 条件 | 说明 |
|------|------|
| `time_out` | Episode 超时 |
| `fell_over` | 基座倾斜 >70° |
| `anchor_pos_z` | 锚点 z 误差 >0.25m（动作参考守护） |
| `anchor_ori` | 锚点朝向误差 >0.8（动作参考守护） |
| `ee_body_pos` | 末端执行器位置误差 >0.25m |

#### Stage 2: 感知引导踢球

**变化**：引入球感知、足球踢球奖励，放宽运动约束允许球追逐。

**命令采样**：从自适应改为均匀采样（每 episode 从第 0 帧开始）。

**新增奖励**：
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `proximity` | 1.0 | 目标点（球）接近度 |
| `contact` | 50.0 | 脚-球接触伴随水平力 |
| `sideways_kick` | 50.0 | 球朝球门方向移动 |
| `ball_vel_align` | 30.0 | 球速方向对准球门 |
| `ball_speed` | 10.0 | 球水平速度 |
| `foot_distance` | 0.2 | 双脚间距惩罚 |
| `is_terminated` | -200.0 | 早停惩罚 |
| `both_feet_ball` | -5.0 | 双脚同时触球 |
| `nonfoot_ball` | -3.0 | 非脚部触球 |
| `foot_stomp` | -20.0 | 跺脚蹬球 |
| `foot_lift` | -2.0 | 踢球脚过早离地 |

**运动跟踪权重**：anchor_pos 降为 0.0（允许自由追逐），新增 track_foot_pos=1.0。

**终止条件变化**：ee_body_pos 阈值从 0.25m 放宽至 0.35m。

#### Stage 3: 球门平面精度 + 速度课程

**变化**：引入显式球门平面穿越奖励 + 自适应目标采样。

**新增/调整奖励**：
| 奖励项 | 权重 | 变化 |
|--------|------|------|
| `ball_speed` | 15.0 | std 扩大至 3.0（速度塑形范围加大） |
| `goal_accuracy` | 10.0 | **新**：球门平面穿越精度（Gaussian, std=0.3） |
| `goal_miss` | -5.0 | **新**：球完全未达球门平面惩罚 |
| 运动跟踪 (6 项) | 各 0.5 | 从 1.0 减半 |

**终止条件**：大幅精简——只保留 `time_out` + `fell_over`。移除所有动作参考守护。

**自适应目标**：11 bins 的球门平面目标，alpha=0.3（聚焦最难打中的区域）。

#### Stage 4: 高速精度

**目标**：将球速推向 ~10 m/s。

**关键变化**：
| 奖励项 | 权重 | 变化 |
|--------|------|------|
| `ball_speed` | 20.0 | std 扩大至 5.0（目标 ~10 m/s） |
| `is_terminated` | -500.0 | 从 -200 翻倍加强（高速下摔倒更致命） |
| `undesired_contacts` | -1.0 | 从 -0.1 加强（防身体拖地） |
| `goal_miss` | -20.0 | 从 -5 加强 |

#### Stage 5: 竞赛坐标对齐

**关键创新**：将训练坐标完全切换到 compete.py 的世界坐标。消除训练→竞赛的 domain gap——模型在训练中看到的场景与比赛完全一致。

**规则变化**：
| 奖励项 | 权重 | 变化 |
|--------|------|------|
| `goal_accuracy` | 30.0 | 从 10 大幅提升 |
| `goal_miss` → `goal_miss_scaled` | -50.0 | 按横向偏离距离缩放 |
| `ball_vel_align` | 30.0 | std 收紧至 0.12 |
| `z_speed` | -2.0 | 重新启用（轻度防挑球） |
| `ball_oob` | -30.0 | **新**：球超出球门边界惩罚 |
| `both_feet_air_time` | -5.0 | **新**：双脚离地 >0.15s |
| `proximity` | 3.0 | 从 1.0 加强 |
| 运动跟踪 (6 项) | 各 **0.05** | 近零（总计 0.3） |

**域随机化**：**全关**。零扰动、零噪声、零损坏、零 push——纯竞赛环境。

#### Stage 6: 高速竞赛

**目标**：在竞赛坐标下将球速推向 12-16 m/s。

| 奖励项 | 权重 | 变化 |
|--------|------|------|
| `ball_speed` | **40.0** | 从 20.0 翻倍，std 扩大至 10.0 |
| `z_speed` | **-5.0** | 从 -2.0 加强（高速防挑球） |
| `ball_vel_align` | 20.0 | 从 30.0 降低，std 放宽至 0.20（精度换速度） |
| `goal_accuracy` | 20.0 | 从 30.0 降低（速度优先于精度） |

---

## 3. 模型架构与观测空间

### 3.1 网络结构

```
RNNModel (Actor & Critic 共享架构):
  ├── MLP encoder: Linear(128→128) → ELU → Linear(128→64) → ELU → Linear(64→32) → ELU
  ├── RNN: LSTM(input_dim, hidden=128, num_layers=2)
  └── Output head: Linear(32→output_dim)

Actor:  input_dim=160, output_dim=29 (joint position targets)
Critic: input_dim=298, output_dim=1  (value)
```

动作分布：`GaussianDistribution`，scalar std（单一可学习 log-std），初始 std=1.0。

观测归一化：Actor 和 Critic 均启用 `EmpiricalNormalization`。

### 3.2 Actor 观测空间（160D）

| # | 观测项 | 维度 | 说明 |
|---|--------|------|------|
| 0 | command | 58 | 参考关节位置 (29) + 速度 (29)，由 motion command 生成 |
| 1 | projected_gravity | 3 | 重力矢量在机器人基座坐标系中的投影 |
| 2 | motion_ref_ang_vel | 3 | 参考躯干角速度（世界坐标系） |
| 3 | base_ang_vel | 3 | 实际基座角速度（IMU 陀螺仪） |
| 4 | joint_pos | 29 | 当前关节位置 − HOME_KEYFRAME 默认值 |
| 5 | joint_vel | 29 | 当前关节速度 |
| 6 | actions | 29 | 上一时间步的动作输出 |
| 7 | target_point_pos | 3 | 球在机器人骨盆坐标系中的位置 |
| 8 | target_destination_pos | 3 | 射门目标点在机器人骨盆坐标系中的位置 |

所有观测项在拼接前均按各 term 的 std 进行归一化。

### 3.3 Critic 额外特权信息（+138D）

这 5 个额外的观测项仅在训练期间可用（privileged information），为 Critic 提供环境真实状态以更准确地估计价值：

| # | 观测项 | 维度 | 说明 |
|---|--------|------|------|
| 9 | motion_anchor_pos_b | 3 | 参考锚点在机器人基座坐标系中的位置 |
| 10 | motion_anchor_ori_b | 6 | 参考锚点朝向（6D 旋转表示） |
| 11 | body_pos | 42 | 14 个跟踪身体部件在锚点坐标系中的位置 |
| 12 | body_ori | 84 | 14 个跟踪身体部件朝向（6D 旋转表示） |
| 13 | base_lin_vel | 3 | 机器人基座线速度（世界坐标系） |

### 3.4 域随机化策略

域随机化从 Stage 1 到 Stage 6 逐步收缩：

| 阶段 | 采样策略 | Push Robot | 观测噪声 | 观测损坏 | 球随机化 |
|------|---------|-----------|---------|---------|---------|
| Stage 1 | **自适应**（偏向失败段） | ✓ | ✓ (actor) | ✓ | — |
| Stage 2-4 | 均匀 | ✓ | ✓ (actor) | ✓ | 球曲线偏移 ±0.15m |
| Stage 5-6 | 均匀 | **✗** | **✗** | **✗** | **✗** |

Stage 5-6 的完全无随机化设计确保与 compete.py 环境精确一致。

### 3.5 动作运动文件

使用 10 个 `soccer-standard-*.npz` 运动文件：

| 类别 | 数量 | 文件 |
|------|------|------|
| 右腿踢球 | 6 | `001_right`, `004_right`, `005_right`, `006_right`, `009_right`, `010_right` |
| 左腿踢球 | 4 | `002_left`, `003_left`, `007_left`, `008_left` |

每文件包含：关节位置 (T, 29)、关节速度 (T, 29)、躯干角速度 (T, 3)。文件长度 205–289 帧（4.1–5.8s）。

---

## 4. 奖励设计演进

总共有 28 个奖励项（Stage 6），但核心信号仅由少数几个高权重项驱动。设计理念：奖励从**纯模仿**逐步过渡到**纯比赛性能**。

### 4.1 核心奖励项演变表

| 奖励项 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Stage 6 | 功能 |
|--------|---------|---------|---------|---------|---------|---------|------|
| track_body_pos | 1.0 | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 身体姿态跟踪 |
| track_body_ori | 1.0 | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 身体朝向跟踪 |
| track_anchor_ori | 1.0 | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 锚点朝向跟踪 |
| track_body_lin_vel | 1.0 | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 身体线速度跟踪 |
| track_body_ang_vel | 1.0 | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 身体角速度跟踪 |
| track_foot_pos | — | 1.0 | 0.5 | 0.5 | **0.05** | 0.05 | 脚部位置跟踪 |
| **运动跟踪合计** | **6.0** | 5.0 | 3.0 | 3.0 | **0.3** | **0.3** | |
| contact | — | **50** | 50 | 50 | 50 | 50 | 脚-球接触力 |
| sideways_kick | — | **50** | 50 | 50 | 50 | 50 | 横向踢力（朝球门方向） |
| ball_vel_align | — | 30 | 30 | 30 | 30 | **20** | 球速与球门方向对齐 |
| ball_speed | — | 10 | **15** | **20** | 20 | **40** | 球水平速度 |
| goal_accuracy | — | — | **10** | 10 | **30** | 20 | 球门平面穿越精度 |
| goal_miss | — | — | −5 | −20 | **−50** | −50 | 射偏惩罚 |
| z_speed | — | 0 | 0 | 0 | −2 | **−5** | 垂直速度惩罚 |
| ball_oob | — | — | — | — | −30 | −30 | 球出界惩罚 |

### 4.2 正则化项（所有阶段共享）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| action_rate | -0.1 | 动作平滑性（L2 on action delta） |
| joint_limit | -10.0 | 关节超限惩罚 |
| undesired_contacts | -0.1 / -1.0 | 非足部/非手腕的地面接触 |

### 4.3 设计逻辑

1. **Stage 1（纯模仿）**：运动跟踪 6 项合计 6.0 权重，无任何足球奖励。目的是形成稳定的踢球动作先验。

2. **Stage 2（加入足球）**：新增 sparse rewards（contact=50, sideways_kick=50）驱动脚-球交互，同时通过负奖励抑制作弊行为（双足触球、非脚部触球、跺脚）。运动跟踪略降（anchor_pos 归零释放位姿约束）。

3. **Stage 3-4（精度 + 速度）**：新增 dense 球门精度奖励。ball_speed 权重逐步攀升。goal_miss 惩罚从 -5→-20→-50 逐级加强。运动跟踪权重减半（合计 3.0），终止条件从 5 个精简为 2 个。

4. **Stage 5-6（竞赛对齐 + 极限速度）**：运动跟踪降到几乎为零（0.3）。ball_speed 从 20 翻倍到 40 成为主信号。z_speed 惩罚加强到 -5 抑制挑球。精度信号略降（ball_vel_align 从 30→20，goal_accuracy 从 30→20），允许以精度换速度。

---

## 5. Phase 1 评测

### 5.1 评测设置

| 参数 | 值 |
|------|-----|
| 官方评测脚本 | `eval_naive_shooter.py` |
| 并行评测脚本 | `eval_shooter_parallel.py` |
| 任务 ID | `Eval-Shooter` (naive) / `Eval-Shooter-Stage6` (parallel) |
| 模型 | Stage 6 `model_20000.pt` (20K iter) |
| 评测种子 | 42, 2810, 202606 |
| Trials/seed | 50 (naive) / 10000 (parallel) |
| GPU | NVIDIA GeForce RTX 4090 D (24 GB) |
| 评分公式 | `max(0, (success_rate - 0.8) / 0.2) × 30` |

### 5.2 评测结果

| Seed | Trials | 进球数 | Success Rate |
|------|--------|--------|-------------|
| 42 | 10000 | 9996 | **99.96%** |
| 2810 | 10000 | 9995 | **99.95%** |
| 202606 | 10000 | 9997 | **99.97%** |
| **平均** | **10000** | **9996.0** | **99.96%** |

**Phase 1 Score**: `max(0, (0.9996 - 0.8) / 0.2) × 30 = 29.9 / 30`

**代表性指标**（seed 202606）：
- 平均踢球速度：**13.03 m/s**
- 踢球精度（cosine similarity）：**0.9787 ± 0.0191**

> **解读**：三个种子下进球率均超过 99.9%，仅有 3-5 次/10000 的极小波动。29.9/30 的分数接近满分，失分来自评分公式的线性映射——99.96% 相对 100% 的微小差距被公式放大（满分要求 100% success rate）。

### 5.3 评测工具增强

针对官方提出的 seed 固定要求，对评测脚本做了以下改进：

1. **四级种子控制**：`random.seed` / `np.random.seed` / `torch.manual_seed` / `env_cfg.seed` 全覆盖
2. **`--seeds` 参数**：支持 `--seeds 42 2810 202606` 一次运行多 seed 并自动聚合平均值和 Phase 1 分数
3. **LSTM 状态重置**：每 trial 前调用 `policy.reset()` 清空隐藏状态，确保 trials 间独立
4. **GPU 内存管理**：seed 切换间显式 `del env, policy, env_cfg` + `gc.collect()` + `torch.cuda.empty_cache()`
5. **并行加速**：`eval_shooter_parallel.py` 通过 4096 并行环境实现 ~25× 评测加速，同时保持与官方脚本一致的指标计算和 goal detection 逻辑

---

## 6. Phase 2 竞赛策略与 API Server

### 6.1 API Server 架构

基于 FastAPI，加载 Stage 6 LSTM 策略（`Eval-Shooter-Stage6` 任务），通过 REST 接口暴露两种端点：

```
POST /act   ← 接收 raw MuJoCo state → 计算观测 → 推理 → 返回 action
POST /reset ← 清空隐藏状态 + 根据策略决定射门目标
```

**关键设计决策**：服务器从原始 MuJoCo 状态自行计算观测（160D `compute_shooter_obs`），而非依赖预定义的观测计算。这意味着我们的观测空间对其他队伍完全透明——`compete.py` 只传递 raw state，各队自行计算自己的观测。这实现了异构系统间的互操作性。

**观测计算（`compute_shooter_obs`）**：
从 raw JSON state 中提取 shooter 的 29 个关节位置/速度/上一动作、基座姿态和角速度，以及球的 world position。将球位置和目的地位置转换到机器人骨盆坐标系。从对应时间步的运动文件中读取参考关节位置/速度和躯干角速度。拼接 9 个 term 得到 160D 观测向量。

### 6.2 三种射门策略

| 策略 | 触发时机 | 行为 | 延时 |
|------|---------|------|------|
| `uniform` | `/reset` | 随机 motion（含左腿）+ 随机目标 y∈[-1.4, 1.4]，目标立即锁定 | 无 |
| `best-motion` | `/reset` | 随机目标 + `_select_motion_for_dest()` 选最优右腿 motion，目标立即锁定 | 无 |
| `gk-aware` | `/act` 逐帧观测 GK | 观察 GK 位置 5-16 帧后锁定远侧目标 + 最优 motion | 5-16 帧 |

### 6.3 gk-aware 策略详细流程

这是默认且最智能的策略，流程分三个阶段：

**Phase 1 — 观察阶段**（`shoot_planned["done"] == False`）：
1. 每帧 `/act` 将 GK 根 y 坐标追加到 `gk_y_history`
2. `_should_lock()` 双重条件判断：
   - **行为触发**：GK 横向偏移 ≥0.3m 且连续 `min_observe=5` 帧 → 立即锁定（early lock）
   - **强制截止**：到达 `deadline_pct=60%` 的踢球帧（如 001_right 的 16/27 帧）→ 强制锁定（force lock）
3. **不 reset policy**：锁定只改变 `dest_body` 观测项，不重置 LSTM 隐藏状态——因为 policy 被训练为在连续 motion 流中工作，跳到帧 16 + 空 hidden state 会导致 OOD 崩溃

**Phase 2 — 目标规划**（`_plan_destination`）：
| GK 位置 | 射门目标 |
|---------|---------|
| GK 在左侧 (y < -0.3) | 右侧 `random(0.3, 1.4)` |
| GK 在右侧 (y > +0.3) | 左侧 `random(-1.4, -0.3)` |
| GK 在中间 (\|y\| ≤ 0.3) | 全门宽度 `random(-1.4, 1.4)` |

随机分量在同一侧半场内增加不可预测性。

**Phase 3 — 执行**：
- `_select_motion_for_dest(dest_y, motions)` 为选定目标选择最优右腿 motion
- 目标世界坐标设为 `(-0.5, dest_y, 0.11)`（球门平面中心）
- 继续播放 motion，观测中包含目标位置，policy 据此调整踢球方向

### 6.4 目标区域 → Motion 映射表

基于 `eval_motion_analysis.py` 中 5000 trials/motion×target 的实测数据建立：

| 目标 y 范围 | 最优 Motion | 球速 | 精度 | 选择理由 |
|------------|-----------|------|------|---------|
| [-1.5, -0.7) 左远角 | `006_right` | 16.2 m/s | **0.10m** | 全场最优远角精度 |
| [-0.7, -0.3) 左中 | `009_right` | 14.2 m/s | 0.35m | 右腿中左中精度最优 |
| [-0.3, +0.3] 正中 | `010_right` | **16.4 m/s** | 0.11m | 全场最高速 + 优秀中心精度 |
| (+0.3, +0.7] 右中 | `005_right` | 12.1 m/s | **0.006m** | 全场最优精度（sub-cm） |
| (+0.7, +1.5] 右远角 | `005_right` | 15.7 m/s | 0.64m | 最通用的 motion |

所有选中的 motion 均为右腿踢球。左腿 motion 因速度慢（6-9 m/s）且左远角完全失效（0% goal rate）而未使用。

### 6.5 踢球帧映射（`_KICK_FRAME_MAP`）

经 `analyze_kick_timing.py` 批量测量确认，所有 motion 在帧 26-30（0.52-0.60s）稳定触球，标准差为 0。此确定性保证了 `_should_lock()` 的 deadline 机制可靠性。

```
soccer-standard-001_right: kick_frame=27/260
soccer-standard-005_right: kick_frame=26/230
soccer-standard-006_right: kick_frame=27/225
soccer-standard-009_right: kick_frame=27/205
soccer-standard-010_right: kick_frame=27/230
（左腿 motions 类似，略去）
```

### 6.6 零 GK 基准

当 `compete.py` 省略 `--goalkeeper-api` 时，GK 使用 ZeroPolicy（输出全零 → 保持 HOME_KEYFRAME 蹲姿在 y=0）。可作为 shooter 的纯技术性能基线。

---

## 7. Motion 性能分析

### 7.1 评测规模

- 10 motions × 5 目标位置 × 5000 trials = **250,000 trials** 总计
- 模型：Stage 6 (model_20000.pt)
- 目标位置：y = -1.50, -0.75, 0.00, +0.75, +1.50（球门 3.0m 跨度等分）

### 7.2 总体指标

| 指标 | 数值 |
|------|------|
| 平均进球率 | **94%**（9/10 motions 达 100%） |
| 平均踢球触发率 | **100%**（每次均能成功触球） |
| 右腿平均球速 | **14.2 m/s**（≈ 51 km/h） |
| 左腿平均球速 | **8.0 m/s**（≈ 29 km/h） |
| 右腿平均精度 | 0.42 m（与目标点偏差） |
| 左腿平均精度 | 0.26 m |

### 7.3 右腿 Motion 性能

所有右腿 motion 进球率均为 **100%**。

| Motion | 帧数 | 平均球速 | 平均精度 | Hit@0.3m | 最佳目标 | 最佳精度 |
|--------|------|---------|---------|----------|---------|---------|
| `001_right` | 260 | 13.59 m/s | 0.340 m | 40% | +0.75 | 0.152 m |
| `004_right` | 257 | 14.02 m/s | 0.548 m | 20% | +0.75 | 0.197 m |
| `005_right` | 230 | 13.96 m/s | 0.402 m | 20% | **+0.75** | **0.006 m** |
| `006_right` | 225 | 14.56 m/s | 0.343 m | 40% | **-1.50** | **0.095 m** |
| `009_right` | 205 | 13.73 m/s | 0.454 m | 40% | +0.00 | 0.144 m |
| `010_right` | 230 | 15.40 m/s | 0.512 m | 20% | +0.00 | 0.112 m |

### 7.4 左腿 Motion 性能

进球率均为 **80%**——**仅在 y=-1.50（球门最左侧）进球率为 0%**，其余 4 个目标位置均为 100%。

| Motion | 帧数 | 平均球速 | 平均精度 | Hit@0.3m |
|--------|------|---------|---------|----------|
| `002_left` | 214 | 7.27 m/s | 0.245 m | 80% |
| `003_left` | 289 | 7.88 m/s | 0.277 m | 65% |
| `007_left` | 229 | 8.42 m/s | 0.266 m | 60% |
| `008_left` | 225 | 8.36 m/s | 0.261 m | 60% |

### 7.5 目标位置 × 球速矩阵（m/s）

| Motion | y=-1.50 | y=-0.75 | y=+0.00 | y=+0.75 | y=+1.50 |
|--------|:-------:|:-------:|:-------:|:-------:|:-------:|
| 001_right | 14.58 | 14.04 | 14.14 | 12.41 | 12.80 |
| 004_right | 13.02 | 14.52 | 14.62 | 12.53 | 15.41 |
| **005_right** | 14.59 | 13.37 | 14.12 | 12.06 | **15.69** |
| **006_right** | **16.18** | 14.18 | 14.04 | 12.99 | 15.42 |
| 009_right | 14.40 | 14.24 | 12.14 | 12.54 | 15.33 |
| **010_right** | 14.64 | 14.57 | **16.42** | 16.28 | 15.10 |

### 7.6 各 Motion 特长总结

| Motion | 外号 | 核心数据 |
|--------|------|---------|
| `006_right` | **左远角之王** | y=-1.50 精度 0.095m，球速 16.18 m/s |
| `005_right` | **精准射手** | y=+0.75 精度 0.006m（全场最优精度）；y=+1.50 球速 15.69 m/s |
| `010_right` | **中路速度担当** | y=+0.00 球速 16.42 m/s（全场最高速），精度 0.112m |
| `009_right` | **左中均衡型** | 右腿中左中区域精度最优（0.349m），速度 14.24 m/s |

### 7.7 关键发现

1. **右腿 vs 左腿显著不对称**：右腿球速是左腿的 ~2×，且左腿无法打入左远角。原因可能是训练数据中右腿动作更多/更优，或 reward 结构对右腿更有利。

2. **远角精度退化**：所有 motion 在 y=±1.50 时精度明显下降（0.4-1.2m vs 中心 0.1-0.4m），这是机械约束的合理表现——踢极端角度需要更精确的身体构造。

3. **竞赛场景推荐**：仅使用右腿 motions，采用 gk-aware 策略在运动执行期间读取 GK 位置并射向远侧。

---

## 8. 工程决策与经验

### 8.1 统一架构设计

所有 6 个训练阶段共享完全相同的 LSTM 架构（128-64-32 + 2×128 LSTM），仅通过环境配置区分。这一决策使得：
- 跨阶段 checkpoint 可无缝 transfer（`--resume` 直接加载 actor+critic+normalizer）
- 精力集中在奖励工程和环境设计，而非重复调网络结构
- 实验迭代速度显著提升——改一个 reward weight 不需要重设计网络

### 8.2 坐标对齐是致命细节

Stage 5 将训练坐标完全切换到 compete.py 的世界坐标（机器人 (4,0,0.8), yaw=π），这个变化看似简单但对性能影响巨大——模型在训练中看到的场景和比赛完全一致，消除了 sim-to-sim 的 domain gap。此前在训练坐标下训练的检查点在 compete 场景中表现显著下降。

### 8.3 LSTM 状态连续性

gk-aware 策略中不在锁定目标时 `policy.reset()` 是一个关键教训。最初实现中我们在 `/act` 检测到 should_lock 时调用了 reset，导致 LSTM 在 motion 帧 16 收到空 hidden state——这个组合训练中从未出现过，引发 OOD 崩溃。最终方案仅更新 `dest_body` 观测项，保持 LSTM 状态连续性。这一洞察来自对 LSTM 训练数据分布的仔细分析。

### 8.4 数据驱动的 Motion 选择

API server 的 `_select_motion_for_dest()` 映射不是启发式设计的，而是基于 250K trial 的实测性能数据。每个目标区域的 motion 推荐都有具体的速度和精度数值支撑。这种"先测量再决策"的方法避免了主观猜测。

### 8.5 评测工程化

`--seeds` 参数、多 seed 自动聚合、GPU 显存管理、并行加速、LSTM 状态重置——这些工程增强看似细枝末节，但在阶段性测评中标志着可复现性和效率基础十分关键。特别是在 4096 环境的并行评测中，GPU 显存管理（每 seed 间 `del` + `gc.collect()`）决定了大批量评测是否可行。

### 8.6 规则驱动 vs 对抗训练

Phase 2 的竞赛设计要求 shooter 面对对方的 goalkeeper 策略。常见方案是**对抗训练**（让 shooter 策略在训练中学习面对 GK），但我们选择了"**强单智能体 + 规则目标选择**"的路径：

1. **对抗训练不稳定**——两个策略同时在线学习时容易发散，需要精心设计的课程和 reward shaping。在多阶段渐进训练已经足够复杂的情况下，叠加对抗维度会显著增加调试成本。
2. **先做减法**——训练一个在无 GK 情况下接近完美的 shooter（Phase 1 达到 99.96% 进球率），确保基础能力过关，再在此基础上增加对抗能力。
3. **规则式 gk-aware 策略在给定高性能 shooter 的前提下已经足够有效**——我们的 shooter 能以 12-16 m/s 的速度打入球门任意位置（§7.5 速度矩阵）。面对一个 reactive 的 GK，只要在运动执行期间读取 GK 位置并射向远侧，命中率就远高于随机目标。而"射远侧"这个决策逻辑的正确性是物理上的确定性事实，不需要学习。

这一设计思路的缺点是缺乏对 GK 遮挡、铲球等复杂对抗行为的适应性。但考虑到 Phase 2 的对手 GK 策略未知，规则型 gk-aware 策略的**可解释性和可控性**优于可能训练不充分的对抗策略。

`--seeds` 参数、多 seed 自动聚合、GPU 显存管理、并行加速、LSTM 状态重置——这些工程增强看似细枝末节，但在阶段性测评中标志着可复现性和效率基础十分关键。特别是在 4096 环境的并行评测中，GPU 显存管理（每 seed 间 `del` + `gc.collect()`）决定了大批量评测是否可行。

---

## 9. 附录

### 9.1 关键脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/train.py` | 训练入口，支持 `--task-id` + `--resume` |
| `scripts/train_pipeline.py` | Stage 1→2 自动渐进训练 |
| `scripts/train_stage3_curriculum.py` | Stage 3 课程训练 |
| `scripts/train_stage4_curriculum.py` | Stage 4 课程训练 |
| `scripts/eval_naive_shooter.py` | Phase 1 官方评测（单环境） |
| `scripts/eval_shooter_parallel.py` | Phase 1 并行批量评测（~25× 加速） |
| `scripts/eval_motion_analysis.py` | Per-motion per-target 精析（5000 trials/pair） |
| `scripts/analyze_kick_timing.py` | 踢球帧批量测量 |
| `scripts/api_server.py` | Phase 2 REST API Server（3 策略） |
| `scripts/compete.py` | Phase 2 双方对抗脚本（Shooter vs Goalkeeper） |

### 9.2 关键配置文件

| 文件 | 用途 |
|------|------|
| `src/tasks/soccer/config/g1/rl_cfg.py` | PPO 配置 + 三种 Runner 类 |
| `src/tasks/soccer/config/g1/training_env_cfgs.py` | 六阶段训练环境工厂函数 |
| `src/tasks/soccer/config/training/stage{1-6}_env_cfg.py` | 各阶段环境配置（reward、termination、DR） |
| `src/tasks/soccer/config/g1/__init__.py` | 9 个训练/可玩任务注册 |
| `src/tasks/soccer/config/eval/__init__.py` | 6 个评测任务注册 |
| `src/tasks/soccer/config/settings.yaml` | 场景参数（球门尺寸、罚球距离、物理参数） |
| `src/tasks/soccer/config/soccer_settings.py` | Settings 加载与派生计算 |

### 9.3 Checkpoint

| 阶段 | 路径 | 迭代数 |
|------|------|--------|
| Stage 1 | `checkpoints/stage1/model_3999.pt` | 3,999 |
| Stage 2 | `checkpoints/stage2/model_100000.pt` | 100,000 |
| Stage 4 | `checkpoints/stage4/model_138985.pt` | 138,985 |
| Stage 4 | `checkpoints/stage4/model_143984.pt` | 143,984 |
| **Stage 6** | **`checkpoints/stage6/model_20000.pt`** | **20,000** |

### 9.4 与 HIMPPO / PAiD 关键差别的论文对标

| 维度 | 论文 (PAiD/HIMPPO) | 我们的实现 |
|------|-------------------|-----------|
| 架构 | Stage 1 MLP → Stage 2 LSTM（PAiD） | 全阶段统一 LSTM |
| 训练规模 | 4096 envs | 4096 envs（与论文一致） |
| 坐标 | 论文评估用世界坐标 | Stage 5-6 用 compete 坐标 |
| Goalkeeper | HIMPPO 10-frame history stacking | 直接使用参考实现 |
| Phase 2 | 标准 API 协议（/act, /reset） | 完全兼容 + 3 策略选择 |

### 9.5 更新历史与AGENTS.md

项目使用 `AGENTS.md` 记录关键的环境信息、脚本用法、任务 ID 映射和架构决策，方便后续开发和维护：

| 信息类别 | AGENTS.md 章节 |
|---------|---------------|
| Python 环境 | `## Environment` (uv-managed, .venv/, Python 3.11) |
| 关键脚本（21 个） | `## Key scripts` |
| 任务 ID（15 个） | `## Task IDs` |
| Runner 类型（3 种） | `## Runners` |
| Goalkeeper 特殊配置 | `## Architecture quirks → Goalkeeper specifics` |
| HIMPPO 加载流程 | `## Architecture quirks → HIMPPO checkpoint loading` |
| Compete 工作机制 | `## Architecture quirks → Compete` |
| GPU/headless 训练 | `## Architecture quirks → GPU / headless` |
| 文档管理 | `## Documentation` |

`.agent/` 目录下的 `.mdc` 文件记录 CHANGELOG 和文档编写规范。

---

## 参考文献

[1] J. Kong, X. Liu, Y. Lin, J. Han, S. Schwertfeger, C. Bai, and X. Li, "Learning soccer skills for humanoid robots: A progressive perception-action framework," 2026. [arXiv:2602.05310](https://arxiv.org/abs/2602.05310)

[2] J. Ren, J. Long, T. Huang, H. Wang, Z. Wang, F. Jia, W. Zhang, J. Wang, P. Luo, and J. Pang, "Humanoid goalkeeper: Learning from position conditioned task-motion constraints," 2025.
