# 行为探索试探：实现与验收边界

此页描述 exploration-probe-v1 的已执行实现，不代表方法已取得收益。CR 尚未实现。
目标是区分成功经验获取不足与已有经验学习不足，然后决定是否值得从共同 baseline 600k 扩展至1m。

## 实际动作覆盖

以下代码摘自运行时 trainer。模型先正常计算动作；仅 explore 组在辅助回合前32步以50%概率覆盖为17种动作中的均匀随机动作。control 也消耗相同数量的探索随机数，但不覆盖动作。两组之后轨迹可能不同，不能声称状态始终配对。

~~~python
    def policy(*a):
        pc,act,out=agent.policy(*a,mode='train')
        # Fixed random schedule in both arms; never inspect inventory or success.
        rng=env._env.explore_rng
        coin=float(rng.random());random_action=int(rng.integers(17))
        if args.arm=='explore' and not np.any(a[1]['is_last']) and env._env.episode_steps<32 and coin<0.5:
            act=dict(act);act['action']=np.array([random_action],np.int32)
            with jax.transfer_guard('allow'):
                pc=list(pc);pc[3]={'action':[jax.numpy.array(random_action,jax.numpy.int32)]};pc=tuple(pc)
            env._env.override_count+=1
        return pc,act,out
~~~

执行动作、回放记录与循环状态中的上一动作必须一致，否则世界模型会用错误动作解释下一帧。这里的 pc[3] 依赖当前已冻结 agent 的返回结构；不能不经检查直接移植到其他 Dreamer 版本。探索随机流随完整环境状态保存，恢复预检比较了连续与分段执行。

## 场景与成本

两组每次从同一600k完整检查点恢复，包括100k经验池、优化器与运行状态。只在已结束回合处切换到试探场景。
每个辅助回合最多128步；按累计动作的128步区间安排两块石镐、一块铁镐。因此提前死亡时同一区间可包含多个回合，不承诺回合数严格2:1。
石镐初始3木3石用于提供试错余量；两组相同。评测恢复标准1木1石，不启用随机动作覆盖。
保持原生奖励，辅助起点已有前置成就不会再次奖励。这仍然是人工构造初始状态，不是专家动作监督。
这是100%辅助的短程诊断，不是最终同批训练比例；不能将其直接放大成1m方案。

## 验收

预检400物理动作另计，两组各10000新增交互；源模型及两个结果各90回合，共270回合。
自然评测使用共同环境种子，辅助成绩不得混入自然Crafter Score。
验收v2额外绑定确切源模型和610k结果目录，检查完整文件、step、模型SHA、22成就键、目标成功数、共同种子与逐动作账本。
更新数记录为累计更新数，不伪称全部发生于本轮。原始账本与恢复成本另外保留。
旧验收源码及锁保留，v2只修改结果核验，不改变训练、评测或成功门槛。
通过开发筛选仍须隔离场景确认；失败保留，不根据结果降低门槛。

## 复现范围

公开仓库现有可执行入口仍覆盖旧版三组课程协议。本文代码片段用于审阅实际介入位置，并非已移植完成的新方案一键入口。
新试探依赖运行时v5场景、完整600k状态和审计器；验证选定方案后须一并移植并执行恢复与评测一致性测试。
[协议](results/exploration-probe-protocol.json)、[恢复证据](results/exploration-probe-preflight.json)、[验收修订](results/exploration-probe-audit-v2.json)与[研究路线](ironpick-research-roadmap.md)应联合阅读。

## 已移植的场景模块

scripts/skill_starts.py 提供 prepare_native(env, skill, rotation)，覆盖八种技能与自然出生，使用公开的 StableCrafterEnv。
该模块保留原生全部成就奖励，前置准备动作仅用于构造合法初始历史，不进入智能体回放。
此模块不加载模型、不启动训练，不代表完整新训练入口已移植；旧版课程入口保持原样。
CPU检查：python -m unittest discover -s tests -p test_skill_starts.py。
与冻结运行时的72种起点及后续随机动作进行等价检查，结果见 [场景等价证据](results/skill-starts-equivalence.json)。

## 已移植的探索随机流组件

scripts/bounded_exploration.py 将行为随机探索独立成不读取奖励或任务标签的组件，保存并恢复随机流、覆盖次数和协议参数。
四项CPU测试覆盖与运行时原始逻辑一致、连续/恢复一致、终局及32步后不覆盖、恢复时拒绝配置变更。
调用方仍必须把返回动作同步到环境、回放和模型上一动作记忆；组件本身不替代这项集成验收。现有运行时未改用本模块。
测试：python -m unittest discover -s tests -p test_bounded_exploration.py。
