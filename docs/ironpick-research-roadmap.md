# 600k到1m：铁镐学习与自然迁移
状态：已授权研究，先试探，通过验收后才开展大规模训练。
共同起点是自然baseline seed23、600k完整checkpoint。目标最终为1m及自然铁镐能力；石镐是前置环节，不替代最终目标。旧任务继续暂停。

## 证据审计
检查实际辅助模型首次成功经验、重复奖励、制作条件、动作概率和预测误差。
未记录的回放次数或成就信息明确未知，不能从总奖励猜成功数。
同时覆盖石镐、煤、熔炉、铁矿、铁镐，保留全22成就评价。

## 有界试探
最多两个候选，每个最多10000新增交互，总试探最多20000动作、单轮60分钟。
从同一600k完整状态派生，保留经验池和优化器，场景切换仅在明确回合边界并审计。
优先保持原生奖励，先验证连续/恢复一致性、场景及奖励规则。
不能将强制诊断动作当自主成功写入训练池。
若采用CR，必须记录优先级、采样概率和次数并验证恢复，不能将成功过采样冒称CR。
候选前后共同30辅助铁镐+30自然开发回合，报告Score、22成就及材料浪费。
辅助筛选至少24/30；若源模型已有能力，必须检验自然迁移，不能重复证明已会技能。
自然Score下降不超过源模型10%，石镐/采铁不下降超过2/30；至少3/30新增自然铁镐成功或明确前置增益才允许扩大。
再用隔离30自然场景独立确认。单源筛选不代表统计显著或跨种子稳定。
失败如需调整，先版本化协议，不静默降门槛、扩预算。

## 正式工程迭代
只扩大一个验收通过的方案，到1m；试探、重复和恢复交互另外记账，600k前成本不隐去。
每10k保存，每50k共同100自然回合及技能评测，最终隔离100自然回合。
保留600k、50k倍数、实测峰值、最新恢复、未评测和活跃依赖；CSV22和哈希齐全才清理。
单训练GPU、2核、20GiB显存、16GiB进程树内存。正式时限依据试探测速预先冻结，资源不足排队。
远程管理器负责恢复；本地30分钟定时检查，正常静默，不持续监听。
当前为选优模型工程改进，不是无偏算法比较；原3×3历史证据分开呈现，保留负结果。
只有1m checkpoint、评测CSV、来源哈希及成本审计全部完成，才算整个目标完成。

## 第一轮实际模型审计
旧同批330k在准备好条件的铁镐场景成功9/10、石镐0/10；baseline600k两者0/10。
因此后续目标仍是从600k训练到1m并改善自然铁镐能力，但优先定位前置链和迁移，不将反复练习铁镐作为默认解法。
本轮未启动新训练；回放缺少成就标签与逐样本次数，无法直接恢复准确的历史成功采样次数。

## 已启动的可行性试探：exploration-probe-v1
此试探只检验成功经验获取瓶颈，不是最终同批训练比例，也不是CR。
两组从同一600k完整恢复，各最多10000步，统一使用128步辅助回合，按两段石镐、一段铁镐安排；石镐起点均为3木3石，铁镐保持原标准准备。
两组保留原生全部22项奖励和100k旧经验。control用原采样策略；explore仅在辅助回合前32步以50%概率执行均匀随机动作，不读制作条件、不用正确动作标签、不封禁动作。
探索动作会同时写回策略的上一动作记忆，避免环境真实动作与循环模型输入不一致；评测不启用随机覆盖。
源码检查发现categorical分布未使用配置中的unimix，因此不能假设原策略有固定均匀探索下限。这里只记录实现事实，不宣称已证明根因。
先验证200步连续与100+100恢复一致性（额外400动作），通过才自动试探；CPU八个组合的首次/重复奖励和128步边界测试已通过。
源及两组各30石镐、30铁镐、30自然回合，共270回合，开发种子命名空间39721，策略39723。
原筛选和独立确认门槛不变。正式方案仍须自然混合与迁移验证，不能直接将100%辅助试探扩大到1m。
恢复预检与经验池语义审计均通过；模型、循环状态、有效经验内容、采样索引及随机流一致。尚无新的训练收益结论。

方法参考：[Curious Replay](https://proceedings.mlr.press/v202/kauvar23a/kauvar23a.pdf)用于改进已有经验的学习利用，而本试探先区分能否获得有效成功经验。
[试探协议](results/exploration-probe-protocol.json)。

新增只读验收脚本将重算22成就Score、核对三组共同场景与来源哈希、逐动作成本、探索计数及新解锁日志。
“明确前置增益”操作化为自然石镐、采铁或铁镐至少增加3/30，其他原门槛保持；独立确认仍是必需步骤。
[恢复验收证据](results/exploration-probe-preflight.json)。预检通过只证明工程恢复一致，不证明探索策略有效。

[行为探索实现与复现边界](exploration-probe-implementation.md)：公开核心介入代码，并明确当前尚非新方案的一键复现入口。

## 本轮试探中断

control最后一段超过900秒进程上限，栈停在JAX训练输出传回主机；根因尚未证实，不归因于算法。
完整状态保留到607500，实际7561新增交互中61步未保留；预检400动作另计。explore未开始，两组最终配对评测未完成。
因此没有有效效果比较，也不能宣布探索或CR无效。原截止不延长，不启动1m；先做有界、版本化的恢复停滞诊断。
[中断审计](results/exploration-probe-incomplete.json)保留全部失败成本。

恢复诊断：同一607500检查点在原GPU及另一同型号空卡各继续128步，最终607628模型哈希一致，均越过上次607561停滞位置。
该检查点并未表现为不可恢复；短测不能证明根因、彻底修复或长训练稳定。额外256动作单列，原试探仍未完成。
[恢复诊断证据](results/exploration-recovery-diagnostic.json)。未启动1m。

### Audited control continuation (600k to 610k)

The control arm completed 10,000 retained interactions and 90 evaluation episodes.
On 30 paired natural scenarios, Crafter Score changed from 11.7612 to 12.2298.
Prepared stone-pickaxe success changed from 1/30 to 0/30; iron-pickaxe remained
0/30. This does not establish targeted skill improvement.

The cost ledger records 10,061 physical training actions, including 61 replayed
after interruption, 794 retained scene-preparation actions, 400 preflight actions,
and 256 separate recovery-diagnostic actions. All 20,000 new training updates were
verified. See results/exploration-control-completed.json.

The previously unrun exploration arm has a separate, explicitly versioned
30-minute completion budget, retaining the original 10k interaction budget,
behavior settings, evaluation seeds, and screening criteria. The original
deadline was not extended. No 1m training or CR effectiveness claim follows from
the control result.

### Retained control actions identify different skill bottlenecks

The audited 10k retained interactions contain 6,765 stone-pickaxe scene actions
and 3,235 iron-pickaxe scene actions. The former include 163 stone-pickaxe
commands but only one new stone-pickaxe unlock. The latter include only one
iron-pickaxe command and no iron-pickaxe unlock. The source replay's ordered
chunk chain was checked; the last 10k non-reset transitions were joined to
the final physical-WAL record for each retained action index.

This supports deficient target-action exploration for iron pickaxe. Stone
pickaxe requires a further recipe-availability/timing diagnosis: these totals
do not tell whether its 163 commands occurred while crafting was feasible.
No claim is made that those commands were all valid opportunities, or that
CR fixes either bottleneck. Detailed counts: results/exploration-control-actions.json.

### Corrected action alignment and full environment replay

A reconstruction check caught a one-transition error in the earlier action
join: Dreamer stores the next selected action with the current observation.
The public action-count file is corrected. Target-command totals happened to
remain 163 (stone) and 1 (iron), but movement/no-op totals changed. Commands
are not necessarily executed while the player is sleeping.

Replaying all 10k retained environment actions now exactly matches every stored
image, reward, terminal flag and logged new achievement. This CPU diagnostic
cost 10,000 replayed actions plus one failed alignment-check action; neither is
new training data.

There were 493 awake stone-pickaxe opportunities with required materials and
nearby utilities, but only one target command under those conditions, yielding
one unlock. Of 163 submitted stone-pickaxe commands, 138 occurred while sleeping,
111 lacked materials and 136 lacked nearby utilities; these categories overlap.
Iron-pickaxe scenes offered 99 feasible steps, but zero feasible target commands.
The sole iron-pickaxe command lacked both materials and nearby utilities.

This supports failure to act during valid opportunities, not a broken crafting
recipe. It does not by itself identify the actor/value/world-model cause.
See results/exploration-control-recipe-replay.json.

### Completed paired exploration screen

Both arms completed 10k retained training actions, 20k updates, and the same
90 evaluation episodes. Exploration generated 11 stone-pickaxe and 3
iron-pickaxe unlocks during training, versus 1 and 0 for control. These are
behavior-policy training events, not evaluation success rates.

At evaluation without random overrides, both arms scored 0/30 on each prepared
pickaxe task. Natural Score was 12.2298 for control and 11.0612 for exploration
(source: 11.7612), with no new natural stone-pickaxe/iron/iron-pickaxe successes.
Both failed the prespecified gate. No 1m continuation is approved by this screen.

The next question is whether improved replay can turn newly encountered
successful experiences into a learned policy. The CR integration remains an
engineering prototype. Its first GPU launch failed on a missing import path
before training; a separately versioned entry fix retains the original preflight
deadline. This software failure is not a negative CR learning result.

The exploration arm also passed full 10k CPU action replay against stored
images/rewards/termination/achievements. It had 528 feasible stone-pickaxe
opportunity steps and 12 feasible target commands, with 11 new unlocks;
iron pickaxe had 83 feasible steps and 3 feasible commands, with 3 new unlocks.
This confirms increased effective exploration, while the final 0/30 evaluations
show it did not yet produce a reliable standalone policy. Repeated production
after an existing unlock is not counted as another new achievement.
CPU reconstruction cost across both arms is 20,001 actions including the failed
one-step alignment check; this is distinct from training interaction cost.
