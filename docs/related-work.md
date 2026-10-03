# 独立相关工作审查（2026-10-03）

输入仅为公开仓库 `0ecf164`；审查者未接收先前聊天或访问项目服务器。
阅读 README、冻结协议、课程、训练、评估、恢复与验收实现，再独立查找论文和开源代码。

结论：未找到整套协议几乎完全相同的公开项目；核心思想已有先例，不支持算法首创声明。
没有做全网代码相似度检测、复现外部实验或证明文献空白。

| 工作 | 已核实内容 | 区别与证据限制 |
|---|---|---|
| [SCOUT](https://arxiv.org/html/2607.26417v1) | 多上下文辅助 reset，按成功率撤除/恢复辅助，自然起点评估 | 方法思想很近；非固定单场景日程。本次未找到可核实官方开源实现 |
| [RFCL](https://github.com/StoneT2000/rfcl) / [论文](https://arxiv.org/abs/2405.03379) | 从示范后段状态开始，反向和前向课程 | 我们没有示范状态或自适应起点控制 |
| [Reverse Curriculum](https://proceedings.mlr.press/v78/florensa17a.html) | 自动生成与筛选易到难初始状态 | 不能将辅助起点概念当本组发明 |
| [DiCode](https://github.com/konstantinosmitsides/dreaming-in-code) | Craftax 环境代码课程，有木镐练习任务，且改奖励/终止/生存条件 | 同类应用已存在；我们的窄 reset 干预与同量动作日程对照不同 |
| [dreamer-sc](https://github.com/abhik-roy/dreamer-sc) | 同栈低预算项目，目标条件网络、成就依赖和奖励课程 | 不同干预；README 目标成绩不是完成结果，README 与当前代码部分不一致 |
| [Craftax 学生项目](https://github.com/perfectteatimer/model-based-rl-for-craftex/blob/main/scripts/v5_curriculum.sh) | v5 curriculum 实际调整想象 horizon 和 replay | 不是辅助初始状态，不应仅凭名称判断完全重合 |

[DiCode 的实际木镐任务](https://github.com/konstantinosmitsides/dreaming-in-code/blob/de6f4a024fa5e7f0ff1ecb8811aeedbd09e3c612/src/minicraftax/tasks/seed_tasks/crafting.py)
和 [任务基类](https://github.com/konstantinosmitsides/dreaming-in-code/blob/de6f4a024fa5e7f0ff1ecb8811aeedbd09e3c612/src/minicraftax/tasks/base_task.py)
证明辅助制作任务已经实现；不只是一段未来计划。
[dreamer-sc 课程源码](https://github.com/abhik-roy/dreamer-sc/blob/13a20a86ea95e08d029bb9f7483e3df332cac516/semantic/curriculum.py)
与其 README 需分开读，不推定正式受控实验已完成。

自主工作应表述为：具体任务适配、机会与行为诊断、有限预算下的时间安排对照、状态恢复和成本审计。
不能表述为首次课程学习、无需游戏先验、已解决长程探索或已证明样本效率提高。
同辅助动作数并不等于同辅助重置次数；日程对照不单独识别防止依赖或世界模型改进的因果机制。
