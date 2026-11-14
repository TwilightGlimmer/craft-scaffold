# Crafter Scale-v3 项目资料索引

本文件是事实、数据与素材目录，不是课程报告正文。组员可据此分别按课程模板撰写个人报告；请结合自己的真实分工组织叙述，不要把此索引原样作为报告提交。

## 项目信息

- 环境：Crafter 自然生存环境；独立评测使用原生奖励与自然策略交互。
- 学习系统：DreamerV3 世界模型、Actor、Critic 和序列回放训练流程。
- 研究问题：在已测 600k baseline 的能力基础上，研究 Curious Replay（CR）思路与不同成功轨迹 bank 对高阶工具技能的影响。
- 展示训练范围：600k 已测 baseline 锚点 + 1M 动作续训，终点至 1.6M。
- 共同续训初始化：610k 完整 checkpoint。600k 是评测锚点，610k 是实际共同初始化权重，二者不可混写。
- Branch/Source 均采用 CR 类回放；比较变量是成功轨迹 bank 的来源/组织，不是 CR 对均匀回放的消融。
- 没有将训练 reward 当作自然评测 Score；每个表列检查点的综合分来自 100 局自然场景评测。

## 数据与图表

- `assets/scores-full.csv`：600k–1.6M 已验证节点的 100 局综合 Score、关键成就率、检查点和评测文件核验字段。
- `assets/score-curve-600k-1600k.svg`：曲线矢量源；节点来自 CSV，不对缺失 checkpoint 插值。
- `../../assets/scale-v3-course-score-curve.png`：GitHub 展示用 PNG；共享服务器包中的同图副本为 `assets/score-curve.png`。
- `assets/crafting-actions-audit.csv`：清醒时制作行为审计；睡眠期间动作排除。“配方未满足”以材料/必要设施不齐定义。
- `../../assets/scale-v3-course-stone-pickaxe-frame.png`：GitHub 上 Branch 石镐解锁关键帧；共享服务器包中的副本位于 `assets/gameplay-stone-pickaxe-frame.png`。

## 结果速查

| 节点 | 路线 | 100 局平均自然 Score | 石镐率 | 采集铁矿率 | 铁镐率 |
|---|---|---:|---:|---:|---:|
| 600k | baseline | 11.87 | 0% | 0% | 0% |
| 1.0M | Branch | 19.78 | 40% | 0% | 0% |
| 1.0M | Source | 17.99 | 28% | 4% | 0% |
| 1.6M | Branch | 23.31 | 52% | 7% | 0% |
| 1.6M | Source | 19.40 | 47% | 2% | 0% |
| 1.56M 已测峰值 | Branch | 26.12 | 59% | 11% | 0% |
| 1.37M 已测峰值 | Source | 20.22 | 48% | 5% | 0% |

成就率为 100 局中至少完成一次对应成就的 episode 比例。峰值检查点与 1.6M 终点分开列出；分数随 checkpoint 不保证单调。每路线单种子结果是工程比较信号，不构成多种子显著性或单因素因果证明。

## 演示素材

`videos/` 中为 baseline、Branch、Source 各一段完整 MP4。其 checkpoint、episode、seed、episode 成就数、100 局均分、逐步动作重放校验状态和 SHA256 见 `videos/video-manifest.json`。三条视频对应完整自然 episode；单局成就数不等于 checkpoint 的 100 局成功率。完整视频还保存在共享服务器：

共享项目根目录下的 `outputs/reports/scale-v3-course-report-demo/group-report-package/videos/`。

## 实验结论的边界

- 可支持：600k baseline 至 1.6M 两路线均出现综合分与石镐率提升；Branch 的已测峰值高于 Source。
- 仍未支持：铁镐学习成功（所列节点均为 0%）；CR 单独导致提升；多种子统计显著。
- 行为审计中配方未满足尝试比例下降，但成功前重复制作比例没有同步下降。
- CR 调节经验抽样分布，不是成就奖励或目标成就权重。实现细节以项目代码和协议记录为准。

## 证据定位

`EVIDENCE.md` 给出结果来源、评测验证边界及附件哈希；`videos/video-manifest.json` 给出三段 MP4 的逐条来源。项目脚本与训练配置在仓库 `scripts/`、`configs/` 及 `docs/results/scale-v3-600k-to-1500k/`。
