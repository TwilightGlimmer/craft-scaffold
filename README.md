# CraftScaffold | 从练习到独立生存

基于 DreamerV3 的 Crafter 技能学习：辅助课程设计与迁移评估。

本项目是强化学习课程小组项目，基于 **DreamerV3 + Crafter**，研究有限训练预算下的技能学习效率。
我们不从零实现 Dreamer，也不声称提出首创算法；自主工作是行为诊断、课程场景、公平对照、可恢复训练和成本审计。

**状态：正式实验进行中，尚未得出课程方法优于基线的结论。** 本仓库不包含模型权重或正式评估结果。

## 场景与强化学习方法

Crafter 是二维生存游戏，包含采集、制作和生存任务。典型技能链：
采集木头 → 工作台 → 木镐 → 采集石头 → 更高级工具。

DreamerV3 从游戏画面与公开奖励学习世界模型，在模型想象的轨迹上训练 actor 和 critic，再回到真实环境交互。
课程仅改变部分回合的初始练习条件，不改变 Dreamer 损失函数、网络、动作空间或奖励。
库存、真实位置、课程标签与制作条件仅用于环境逻辑及审计，不作为额外策略输入。

## 研究问题与实验设计

在相同的有效交互预算和辅助练习总量下，把练习集中在前期、后期撤掉帮助，能否提升正常场景中的木镐制作成功率？

| 方法 | 课程安排 | 辅助交互数 |
|---|---|---:|
| natural | 全部正常场景 | 0 |
| fixed_mix | 每四个区块的第一个为辅助场景 | 75,000 |
| fading_mix | 前 60 个区块中偶数编号区块为辅助场景，后 60 个全正常 | 75,000 |

- 每次训练 120 个区块，每区块 2,500 次有效动作，总计 300,000 次。
- 三个训练种子：11、23、41；共九次训练，串行执行。
- 辅助场景：在相邻位置设置工作台，背包增加一份木头；不赠送工具、成就、奖励或无敌状态。
- 所有组在区块边界重置，边界为截断而非死亡终局；这是所有组共同的训练时域改动。
- 最终权重全部冻结后，每次训练统一评估 100 回合正常场景；开发集和正式集随机数命名空间隔离。
- 评估使用固定随机数的 sampled eval，并非强制 argmax。
- 主要指标：木镐制作回合成功率。次要指标包括采石、成就、回报、制作机会覆盖率及成本。
- 有效交互量相同不代表实际计算量完全相同；恢复重放和不确定执行单独计账。
- 报告三个配对训练种子的结果；100 个评估回合不等于 100 次独立训练实验。保留负结果。

精确冻结协议见 [protocol-frozen.json](reports/curriculum-v1/protocol-frozen.json)。
其中历史 `next` 等描述是冻结时保留的过程字段，当前状态以本 README 为准。

## 我们实现的部分

| 文件 | 职责 |
|---|---|
| scripts/curriculum_env.py | 课程安排、辅助重置、稳定对象排序、环境状态恢复 |
| scripts/curriculum_adapter.py | Dreamer 环境接口 |
| scripts/curriculum_train.py | 同步训练、完整检查点、真实交互账本 |
| scripts/exact_replay.py | 回放缓冲与采样器恢复，UUID 序列化修正 |
| scripts/training_runtime.py | 策略状态、环境、随机数及待处理更新恢复 |
| scripts/curriculum_formal_worker.py | 九次实验串行调度、资源限制与异常恢复 |
| scripts/curriculum_evaluate.py | 固定场景评估与逐步诊断记录 |
| scripts/analyze_curriculum_evaluation.py | 评估轨迹一致性审核 |
| scripts/test_curriculum_env.py | 课程环境验收 |
| dreamerv3/ | 固定版本的上游源码及本地集成改动 |
| configs/ | 原始训练配置的归档副本 |
| reports/curriculum-v1/ | 允许发布的冻结协议与小型验收记录 |
| tools/ | 发布前仓库检查工具 |

StableCrafterEnv 对对象排序后再进行随机抽样，使恢复后的行为可复现；所有正式组及评估统一使用。
随机种子对应的轨迹可能不同于原版 Crafter，因此不能把早期开发结果作为正式对照。

## 环境与复现边界

当前发布保存的是**实际运行的服务器版本**，不是已验证的跨机器一键安装包。
Linux、Python 3.11、NVIDIA CUDA GPU；正式运行使用 JAX 0.4.38、Crafter 1.8.3、NumPy 1.26.4。
单 GPU 任务，2 个 CPU 核，显存上限 20 GiB、进程树内存上限 16 GiB。

- 项目：`/root/gpufree-data/hanzhuo/projects/crafter-worldmodel`
- 模型：`/root/gpufree-data/hanzhuo/models/crafter-worldmodel`
- 启动统一通过 `./python.sh`，使用个人 `crafter-dreamer` 环境。
- `.deps/jax0438` 是正式 JAX 覆盖层，训练入口优先加载它。
- `requirements.lock.txt` 与 `requirements-stage1.in` 是历史基础环境记录，**不能单独代表正式环境**。
- [requirements-formal.txt](requirements-formal.txt) 合并基础环境与覆盖层的发行包版本；它是环境快照，不代表在另一台机器上已验证安装成功。
- [runtime-snapshot.json](docs/runtime-snapshot.json) 记录版本来源，不含凭据。

### 已配置服务器上的操作

先运行发布检查（仅检查索引，不启动 GPU）：

```bash
./python.sh tools/check_publication.py
```

CPU 环境验收命令如下，会更新本地环境测试报告：

```bash
./python.sh scripts/test_curriculum_env.py
```

训练由管理程序统一串行运行。**已有管理程序运行时，不要再次启动。**
仅在确认没有现存管理进程和 GPU 子进程、配置与验收记录匹配的情况下：

```bash
./python.sh scripts/curriculum_formal_worker.py
```

程序在 `reports/curriculum-v1/formal-status.json` 写入状态，在 `formal-attempts.jsonl` 记录尝试。
从最近完整检查点恢复模型、优化器、回放、环境与随机数，不只恢复权重。
`physical-actions.jsonl` 保存执行前后账本；这些运行产物不进入 Git。

### 换机器运行

当前 `python.sh`、管理程序、路径保护和 release lock 含原服务器路径与 GPU 标识。
归档配置在 `configs/recovery-v1.config.yaml`，原训练入口从模型根目录的 `recovery-v1/config.yaml` 读取。
新机器需要在个人目录中安装环境、映射路径与 GPU、恢复归档配置，并以**新实验版本**重新验收、生成对应源码锁。
不得修改运行中的冻结文件或仅替换锁来跳过审核，也不得直接沿用旧验收记录证明新环境通过。
本次仓库整理没有修改训练源码或启动新的训练。

## 已完成的验证与结果边界

已保存环境安排与重置验证、回放恢复验证，以及连续 400 步与两次中断恢复的 288 个参数数组一致性检查。
另有跨区块预检和开发评估完整性记录；这些是工程验收，不是候选方法性能证据。
预检只支持所检查配置与长度，不保证所有硬件和任意训练长度均无差异。

正式三种子结果尚未汇总，现阶段不展示“胜过基线”的图表。
最终交付计划：代码与配置、三种子对照及消融、成本报告、成功/失败案例、课堂视频与 PPT。

## 发布与协作

- Git 采用白名单：仅源码、文档、配置和指定小型验收文件入库。
- 模型、回放、日志、数据集、缓存、虚拟环境、临时文件、私钥与本地配置默认忽略。
- 不使用 `git add -f` 纳入实验目录；提交前执行发布检查并审阅 `git diff --cached`。
- 发布检查不是绝对的秘密检测保证；新增配置和数据仍需人工审阅。
- 检查点使用 pickle，仅加载可信来源的检查点。
- 持续运行实验与 GitHub 发布分离；组员修改采用新分支、新实验版本，不覆盖当前冻结配置。
- 课程报告、视频、PPT 尚未完成；后续以审核后的独立目录发布。

## 来源与许可

DreamerV3 上游：[danijar/dreamerv3](https://github.com/danijar/dreamerv3)，
固定提交 `e3f02248693a79dc8b0ebd62c93683888ddaccfe`，保留 [MIT 许可证](dreamerv3/LICENSE)。
本仓库以普通文件保留实际源码，不要求递归初始化子模块。
集成改动见 [upstream-integration.patch](docs/upstream-integration.patch)。

游戏依赖：[danijar/crafter](https://github.com/danijar/crafter)，版本 1.8.3，通过依赖安装提供游戏及素材。
各第三方组件遵循其自己的许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
本组新增代码尚未指定开源许可证；公开可阅读不等于授予第三方代码之外的再分发授权。
