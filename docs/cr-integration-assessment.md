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
