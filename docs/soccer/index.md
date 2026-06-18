# Soccer 文档索引

> 最后更新: 2026-06-14

## 快速入门

| 文档 | 说明 |
|------|------|
| [项目概览](overview.md) | 项目背景、功能清单、关键技术栈、架构概要 |
| [环境搭建](get_started/installation.md) | Python 虚拟环境、依赖安装 |
| [快速开始](get_started/quickstart.md) | 常用命令速查 |

## 实现指南

| 文档 | 说明 |
|------|------|
| [任务注册](implementation/task_registration.md) | 如何注册 mjlab 任务 |
| [环境配置](implementation/env_config.md) | 各类环境配置工厂说明 |
| [Shooter 实现](implementation/shooter.md) | 射手环境、观测、奖励、训练管线 |
| [Goalkeeper 实现](implementation/goalkeeper.md) | 守门员环境、观测、奖励、训练管线 |
| [训练管线](implementation/training_pipeline.md) | Stage I/II 训练流程与配置 |
| [Checkpoint 加载](implementation/checkpoint_loading.md) | 权重加载与迁移策略 |
| [对战系统](implementation/compete.md) | compete.py 与 API 服务器实现 |

## 算法参考

| 文档 | 说明 |
|------|------|
| [PPO 算法](algorithms/ppo_algorithm.md) | PPO 原理、三种架构变体（MLP/LSTM/HIMPPO）、完整超参数表 |
| [HIMPPO 架构](algorithms/himppo.md) | GoalkeeperActorCritic 网络详解、非对称 Actor-Critic 设计 |
| [奖励设计](algorithms/reward_design.md) | Shooter Stage I/II 与 Goalkeeper 的完整奖励函数拆解和权重表 |
| [域随机化](algorithms/domain_randomization.md) | 训练中的物理域随机化策略、观测噪声、自适应采样机制 |

## 架构分析

| 文档 | 说明 |
|------|------|
| [系统总览](architecture/system_overview.md) | 整体架构、模块关系 |
| [数据流](architecture/dataflow.md) | 训练与推理的数据流 |
| [观测空间](architecture/observation_spaces.md) | Shooter / Goalkeeper 的观测维度与结构 |

## API 参考

| 文档 | 说明 |
|------|------|
| [API 索引](api/index.md) | API 文档入口 |
| [任务注册](api/public/task_registry.md) | `register_mjlab_task()` API |
| [设置参考](api/public/settings.md) | SETTINGS 数据类与 settings.yaml 映射 |
| [环境配置工厂](api/public/env_cfg_factories.md) | 环境配置工厂函数 API |
| [Runner 类](api/public/runner_classes.md) | SoccerRecurrentRunner、GoalkeeperRunner |
| [共享 MDP](api/internal/mdp_shared.md) | 共享的观测、奖励、终止条件、域随机化 |
| [Shooter MDP](api/internal/mdp_shooter.md) | Shooter 专有 MDP 函数 |
| [Goalkeeper MDP](api/internal/mdp_goalkeeper.md) | Goalkeeper 专有 MDP 函数 |
| [实体配置](api/internal/entities.md) | ball、goal、ground、robot 实体 |
| [GK Actor-Critic](api/private/gk_actor_critic.md) | GoalkeeperActorCritic 网络内部实现 |

## 教程

| 文档 | 说明 |
|------|------|
| [训练指南](tutorials/training_guide.md) | 如何启动训练（Stage I/II） |
| [评测指南](tutorials/evaluation_guide.md) | 如何使用评测脚本 |
| [Phase 2 指南](tutorials/phase2_guide.md) | 锦标赛部署与 API 服务器使用 |
| [脚本使用指南](tutorials/scripts_guide.md) | 所有 scripts/ 的用途、参数与示例 |

## 专题报告

| 文档 | 说明 |
|------|------|
| [Shooter Student 蒸馏与微调方案](reports/shooter-student-distillation-plan-2026-06-14.md) | Stage II teacher 数据采集、BC 蒸馏、student PPO accuracy/speed curriculum 与配套脚本规划 |
| [Shooter Stage II 奖励失效分析](reports/shooter-stage2-reward-failure-analysis-2026-06-12.md) | 用数学形式解释夹球策略、proximity freeze、contact attribution 与修改影响范围 |
| [Shooter Stage 3 设计方案](reports/shooter-stage3-plan-2026-06-14.md) | 射门精度与速度提升方案，goal-plane accuracy reward + 速度 curriculum + 自动递进训练 |
| [Shooter Stage 6 性能报告](shooter_performance.md) | Stage 6 模型 20K 迭代评测，含 per-motion 精度/球速矩阵和 compete 推荐策略 |
| [Shooter 训练代码审计](reports/training-audit-2026-06-11.md) | Shooter 两阶段训练代码与原始 HumanoidSoccer 代码对照审计 |
| [GK Tracking 奖励调研](reports/gk-tracking-reward-survey-2026-06-11.md) | Goal-conditioned locomotion 奖励设计论文与开源实现调研 |

## 规划文档（内部参考）

| 文档 | 说明 |
|------|------|
| [输入文档](plan/00-input.md) | 分析输入 |
| [仓库扫描](plan/01-repo-scan.md) | 仓库初始扫描 |
| [文档架构](plan/02-doc-architecture.md) | 文档体系设计 |
| [任务分解](plan/03-task-breakdown.md) | 分析任务分解 |
| [委派合同](plan/04-delegation-contract.md) | SubAgent 委派规格 |
| [聚合树](plan/05-aggregation-tree.md) | 聚合策略 |
| [验证计划](plan/06-verification-plan.md) | 验证计划 |
| [状态跟踪](plan/07-status.md) | 当前进度 |
| [MDP 分析](plan/artifacts/analysis-mdp-system.md) | MDP 系统详细分析 |
| [配置分析](plan/artifacts/analysis-config-system.md) | 配置系统详细分析 |
| [脚本分析](plan/artifacts/analysis-scripts.md) | 入口脚本详细分析 |
| [RL 基础设施分析](plan/artifacts/analysis-rl-infra.md) | RL 训练基础设施分析 |
| [实体分析](plan/artifacts/analysis-entities.md) | 实体系统分析 |
| [覆盖率 R1](plan/reviews/coverage-r1.md) | 覆盖率审查 R1 |
| [交叉审查 R1](plan/reviews/cross-r1.md) | 交叉审查 R1 |
| [交叉审查 R2](plan/reviews/cross-r2.md) | 交叉审查 R2 |
| [事实核查 R1](plan/reviews/fact-check-r1.md) | 事实核查 R1 |
| [事实核查 R2](plan/reviews/fact-check-r2.md) | 事实核查 R2 |
| [修复摘要 R1](plan/reviews/fix-r1-summary.md) | 修复摘要 R1 |

---

> 返回 [根索引](../index.md)
