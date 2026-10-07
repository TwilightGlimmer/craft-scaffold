# CR接入评估：先保证恢复与经验有效性

状态：代码级调研，尚未实现CR，也没有新增效果结论。

官方版本固定为 fd4342d95fca671082d9e60bd85a6730ee9e4f4d。
[CR优先级实现](https://github.com/AutonomousAgentsLab/cr-dv3/blob/fd4342d95fca671082d9e60bd85a6730ee9e4f4d/dreamerv3/embodied/replay/curious_replay.py)结合采样次数衰减项与世界模型损失项，不直接使用成功成就标签。
[官方README](https://github.com/AutonomousAgentsLab/cr-dv3/blob/fd4342d95fca671082d9e60bd85a6730ee9e4f4d/README.md)明确不支持恢复训练。
[回放基类](https://github.com/AutonomousAgentsLab/cr-dv3/blob/fd4342d95fca671082d9e60bd85a6730ee9e4f4d/dreamerv3/embodied/replay/base_prioritized_reverb.py)的load为空。不能把Reverb保存等同于自定义计数、索引和随机流完整恢复。
上述代码已通过网页读取核对；服务器直接下载超时，未保存或声称核验本地源码哈希。

## 对本项目的判断

当前ExactReplay仅支持均匀回放，但能完整保存经验与采样随机流。直接替换官方后端不满足600k完整续训要求。
若试探支持引入CR，应另存版本扩展本地回放，保存样本优先级、次数、索引、优先级树、采样随机流和待处理更新。
600k旧经验没有历史逐样本采样次数，必须明确从迁移点开始计数，报告其与从零运行的区别，不能编造旧次数。
应使用逐样本模型损失，不能以整批均值、总奖励或成就标签替代。若改变聚合和混合规则，称CR启发的变体。

## 实装前验收

- 固定输入核对官方优先级公式，检查极端损失与零次数。
- 核对重叠序列、重复抽样及淘汰样本的延迟更新；记录样本ID、概率、次数和损失。
- 连续与分段恢复得到相同采样、模型、优化器和循环状态。
- 同一起点、交互与更新预算比较均匀回放和候选；预检及恢复成本另列。
- 共同自然场景报告全部22成就，辅助成功不能替代自然迁移。

已有成功经验却利用不足时，CR值得进入下一轮试探；没有成功经验时先解决探索；模型预测正确而策略固守时，CR未必对症。
当前试探和冻结门槛不变，不因完成本调研就扩大到1m。

## 当前agent接口的进一步核查

当前loss函数在想象策略更新前保留逐样本、逐时刻的世界模型损失，形状为(batch,time)。
但train取得优化器辅助输出后重新把outs置空，只返回回放隐状态更新和平均metrics；直接从当前metrics无法还原逐样本优先级。
源码中优先级回写只是注释，不能仅打开配置开关；注释中的losses也不是当前train作用域的可用变量。
版本化改动应在聚合前保留模型损失，排除想象actor/value损失，并与应用replay context后的stepid准确对齐。需要检查重复ID和上下文裁剪，不能按原始batch位置猜测。
本次仅审计，没有改动冻结agent。[源码审计](results/cr-agent-interface-review.json)。

## 本地优先采样接口与CPU成本

当前embodied.core.replay.Replay.update已有stepid/priority回写接口，selectors.Prioritized已有序列采样树；无需先引入Reverb才能试探。
但ExactReplay尚未保存这种选择器的完整状态，agent也未输出逐样本模型损失，开配置开关仍然不够。
优先更新应过滤已淘汰ID：现有默认字典可能重新创建无效stepid项，接入时须测试这一边界。

纯CPU合成基准使用实际100000容量、33步序列、batch8和32步损失，100批次。
平均每批均匀抽样0.038ms，优先抽样0.816ms、优先级更新5.521ms；进程RSS约268MiB。
这些数字只说明已有采样树的基础开销，没有计入CR次数更新、GPU损失传回、真实回放读取或完整训练，不能直接当吞吐或收益预测。
[基准记录](results/cr-priority-cpu-benchmark.json)。测试未使用GPU，未修改或启动任何CR训练。

## 未接入训练的CPU原型

scripts/curiosity_selector.py 实现CR启发的优先选择器，六项CPU测试通过：次数衰减、模型误差响应、重叠计数、重复损失平均、淘汰ID不复活、随机流恢复一致。
参数必须显式指定，测试参数仅用于可计算断言，不是拟定的正式超参数。
本原型的次数是样本出现在抽取序列中的次数，包括上下文；序列概率用逐步优先级的均值，同批重复ID的模型损失取均值。这些是本地实现选择，不称原论文精确复现。
采样前概率、计数、树结构、随机流及模型误差一并快照；旧600k历史次数不能恢复，迁移时必须明确从零累计。
尚未接入ExactReplay或agent逐样本损失，也未用于任何训练。必须进一步完成回放与GPU完整恢复验收，不能以CPU测试通过宣称CR有效。
测试命令：python -m unittest discover -s tests -p test_curiosity_selector.py。

### Exact replay integration prototype

The experimental curiosity_replay.py now wraps the existing in-memory replay.
Five CPU integration tests verify batch/priority feedback after restore, pending
streams and FIFO eviction, explicit uniform replay migration, and rejection of
changed priority settings. It explicitly assigns the selector after construction
because an empty selector is falsey in the upstream constructor.

Migration retains old experience, sequence order, and pending streams, but starts
new replay-use counts at zero, uses the configured initial model loss, and starts
a new explicitly seeded sampling stream. It cannot reconstruct historic CR counts.
Every sampling draw, including report/eval sampling, increments counts in this
prototype. Callers must stop insertion/sampling before checkpointing or migration.

This is not yet connected to per-step world-model losses or a GPU training run.
Passing CPU tests does not establish learning improvement or end-to-end training
recovery equivalence. Only trusted local pickle checkpoints are supported.

### Experimental model-loss hook

curiosity_agent.py adds an opt-in Agent subclass returning post-replay-context
step IDs with a per-step priority. Priority is the scaled sum of dynamics,
representation, reward, continuation and reconstruction losses. Policy/value
losses are excluded and the priority is stop-gradient. This aggregation is an
explicit local CR-inspired choice, not a claim of exact original implementation
equivalence. The host selector rejects nonfinite priorities.

Four CPU tests cover loss selection, post-context shape mismatch, missing model
losses, and absence of gradients through priorities. They do not instantiate and
train a full restored Dreamer agent. The subclass still needs full checkpoint
loading, actual GPU optimizer-update and replay-feedback verification before
enabling it in any experiment. Existing entry points remain unchanged.
