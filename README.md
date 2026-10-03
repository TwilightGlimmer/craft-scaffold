# CraftScaffold｜从练习到独立生存

**基于 DreamerV3 的 Crafter 技能学习：辅助课程设计与迁移评估。**

这是一个强化学习课程小组项目：利用可撤除的制作练习场景，检验有限交互预算下不同练习时间安排对正常游戏表现的影响。
**正式三种子实验仍在进行中，尚不能声称候选方法优于基线。**

## 项目做什么，RL 在哪里？

Crafter 是二维生存游戏，典型技能链是：
采集木头 → 放置工作台 → 制作木镐 → 采集石头。

DreamerV3 从图像、奖励和回合标志学习世界模型，在模型内部想象轨迹，训练 actor 和 critic，再回到游戏交互。
我们复用该算法，不修改其网络和损失；课程只改变一部分训练回合的初始条件。
辅助场景在玩家右侧放工作台、将木头库存设为 1，必要时移除该格占据对象并重新渲染画面。
不直接授予工具、成就、奖励或无敌状态。没有额外结构化特权输入；辅助条件本身会通过正常画面被看见。

## 与已有工作是什么关系？

初始状态课程、辅助重置和撤除帮助都有先例，本项目不主张算法首创，也不是首个 DreamerV3 + Crafter 课程项目。

| 相关工作 | 重合点 | 我们的区别 |
|---|---|---|
| [Reverse Curriculum](https://proceedings.mlr.press/v78/florensa17a.html)、[RFCL](https://github.com/StoneT2000/rfcl) | 从容易起点练习并扩展至目标起始分布 | 无示范轨迹，只有一个手工辅助场景，比较固定日程 |
| [SCOUT](https://arxiv.org/html/2607.26417v1) | 辅助 reset、撤除帮助、自然起点评估 | 对方按上下文与成功率调整辅助；我们不做自适应控制 |
| [DiCode](https://github.com/konstantinosmitsides/dreaming-in-code) | Craftax 中已有制作木镐辅助任务 | 对方生成任务且修改奖励、生存与终止；我们干预范围更窄 |
| [dreamer-sc](https://github.com/abhik-roy/dreamer-sc) | 同样使用 DreamerV3/Crafter、低预算技能课程 | 对方修改目标条件与奖励；我们修改初始场景。其 README 目标成绩不能当已完成结果 |

独立公开资料审查未发现与本项目完整协议几乎相同的成品，但这不是“无人做过”的证明。
审查范围及证据边界见 [相关工作审查](docs/related-work.md)。

**自主工作**：行为机会诊断、具体课程环境适配、等辅助动作数的时间安排对照、完整运行状态恢复、恢复成本审计。
这些是课程实验与工程贡献；DreamerV3、Crafter 和课程学习基本思想属于已有工作。

## 冻结实验设计

每次训练 300,000 个保留环境动作，120 个区块，每区块 2,500 动作：

| 方法 | 日程 | 辅助动作数 |
|---|---|---:|
| natural | 全部正常场景 | 0 |
| fixed_mix | 区块编号模 4 为 0 时辅助 | 75,000 |
| fading_mix | 前 60 个区块中的偶数编号辅助，后 60 个完全正常 | 75,000 |

fading 是两阶段前置并撤除，**不是平滑衰减**。辅助区块内死亡后仍按该区块规则重置。
三组均采用区块边界截断和稳定对象排序，边界截断不等同死亡终局。
训练种子 11、23、41，共九次训练；全部最终权重冻结后，各评估 100 回合自然场景。
评估使用固定 RNG 的 sampled eval，不是 argmax。主要指标为木镐制作成功率，次要指标包括采石、成就、回报、机会覆盖及成本。

结论边界：

- 相同辅助动作数不保证相同辅助重置次数、赠送次数或有效制作机会。
- 两种完整日程同时改变前期密度、后期是否撤除和经验年龄，不能单独证明“避免依赖”等机制。
- 固定 300k 步的最终性能不等于完整样本效率曲线证据。
- 保留动作预算匹配；实际物理执行、恢复重放、不确定执行及耗时分别报告。
- 100 次评估不是 100 次独立训练；报告三个配对种子及负结果。
- natural 是本协议内基线，不能直接与上游论文成绩混比。

## 快速开始（Linux / Python 3.11）

Windows 请使用 WSL2 Linux 环境；不支持原生 Windows 训练。
将仓库克隆到**自己的可写目录**，不需要原作者服务器、SSH key、固定 GPU UUID 或外部模型目录。

```bash
git clone https://github.com/TwilightGlimmer/craft-scaffold.git
cd craft-scaffold
bash setup.sh cpu
./python.sh tools/doctor.py
./python.sh tools/validate_runtime.py
```

CPU 路径用于安装、配置和环境测试，不开展正式训练。安装只写仓库内的 `.venv` 与 `.artifacts`，不修改系统 Python。
默认需要 `python3.11`，可通过 `CRAFT_BOOTSTRAP_PYTHON` 指定个人 Python 3.11。
首次安装需要网络。依赖分为 [CPU](requirements/cpu.txt) 和 [CUDA](requirements/cuda.txt)；CPU 依赖闭包已固定在 constraints.txt；CUDA 额外驱动运行库由安装器解析，运行时记录全部实际版本。

### GPU 配置与目标机器验收

在有可用 NVIDIA GPU、兼容 CUDA 12 JAX 驱动的 Linux 机器上：

```bash
bash setup.sh cuda
cp .env.example .env.local
# 编辑 .env.local，选择个人存储目录和空闲 GPU
./python.sh tools/validate_runtime.py --gpu
./python.sh scripts/curriculum_formal_worker.py
```

GPU 验收是明确的耗时操作：检查设备、400 步连续/两次恢复一致性、5,100 步课程边界预检、100 回合开发评估。
验收通过后才生成当前机器的 `local-validation.json`，正式调度器拒绝借用历史验收解锁训练。
若验收中断，换一个新的 `CRAFT_RUN` 重新验收，避免把旧产物误认为新测试结果。
不要在另一训练管理程序仍占用目标 GPU 时启动。

**可配置项**（可写进被 Git 忽略的 `.env.local`，这是可信 Bash 配置）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| CRAFT_DATA_ROOT | 仓库/.artifacts | 模型、报告、日志和缓存根目录 |
| CRAFT_RUN | portable-v2 | 实验版本名，只允许字母、数字、下划线和连字符 |
| CRAFT_GPU | 0 | 单个 GPU 编号或 UUID |
| CRAFT_THREADS | 2 | CPU 线程和亲和性上限 |
| CRAFT_PYTHON | 仓库/.venv/bin/python | 可选个人 Python 环境 |

所有命令统一经过 `./python.sh`。源码、配置、依赖或资源设置改变时使用新实验版本，不能覆盖旧源码锁。
默认单任务、20 GiB 显存和16 GiB进程树内存上限；管理程序检测停滞后从完整检查点分段恢复。
断开终端会影响前台管理程序；长训练请使用自己的 tmux 等会话管理方式。

## 输出、恢复与审计

输出在 `CRAFT_DATA_ROOT` 下按版本隔离：

```text
models/<run>/<method>-seed<seed>/    # 权重、replay、环境/随机数状态、真实交互账本
reports/<run>/                     # 本机验收、源码锁、状态、逐回合评估
logs/<run>/                        # 训练、评估和恢复日志
cache/  tmp/                       # 安装与运行缓存
```

正式矩阵入口串行训练三个方法与三个种子，按 10k 保留动作分段，保存模型、优化器、环境、policy carry、采样器与待处理更新。
管理程序只恢复完整检查点，超过资源上限或连续五次无进展会失败退出，保留诊断信息。
正式评估与开发评估使用不同随机数命名空间，不按正式测试成绩选 checkpoint。

仅加载可信来源的 pickle 检查点。跨依赖版本、跨硬件的逐位一致性没有保证。

## 代码导航

- `scripts/curriculum_env.py`：课程和环境恢复；`curriculum_adapter.py`：Dreamer 接口。
- `curriculum_train.py`、`curriculum_evaluate.py`：训练与固定评估。
- `analyze_curriculum_evaluation.py`：逐步机会/尝试/失败原因审计。
- `exact_replay.py`、`training_runtime.py`：完整恢复。
- `project_runtime.py`、`resource_guard.py`：可移植路径、源码锁与资源限制。
- `configs/train.yaml`：完整模型配置；`configs/protocol.json`：可移植版本协议模板。
- `tools/doctor.py`、`tools/validate_runtime.py`、`tests/`：环境及目标机器验收。

## 版本与验证状态

原正式实验固定在 Git 提交 `0ecf164` 的服务器版本，仍独立运行。
本版本为 **portable-v2 工程迁移**，不修改原实验文件、结果或 release lock。
`docs/legacy-v1/` 仅为历史记录，其路径已替换为占位符；其中通过状态不能证明本版本通过。
历史锁中的 hash 对应旧版原文件，不用来校验 portable-v2。
原始绝对路径仍可能存在于旧 Git 历史；它们不是凭据，本次没有重写历史。

验证范围见 [工程迁移说明](docs/portability.md)。本版本提供安装和验收入口，但不把 CPU 检查冒充新机器 GPU 端到端复现。

## 发布与许可

模型、日志、数据、环境、缓存、临时文件和本机配置不入库。
提交前运行 `./python.sh tools/check_publication.py` 并审阅 staged diff。
不使用 `git add -f` 纳入运行目录。检查器不是绝对的秘密检测保证。

上游 [DreamerV3](https://github.com/danijar/dreamerv3) 固定于
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`，保留 [MIT 许可证](dreamerv3/LICENSE) 和 [集成补丁](docs/upstream-integration.patch)。
[Crafter](https://github.com/danijar/crafter) 1.8.3 通过依赖提供游戏和素材。
见 [第三方声明](THIRD_PARTY_NOTICES.md)。本组新增代码尚未选择开源许可证，公开可读不等于授予再分发授权。
