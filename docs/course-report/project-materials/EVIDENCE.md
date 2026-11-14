# 数据与演示材料核验记录

本包区分 100 局 checkpoint 评测与单局 MP4 演示。所有分数、视频信息和哈希应以此处列出的逐点 CSV 与 manifest 为准；这份文件仅描述来源和限制，不构成报告正文。

## 曲线和表格数据

- 曲线范围：600k 已测 baseline 锚点至 1.6M 报告终点；Branch/Source 实际从共同的 610k 完整 checkpoint 开始续训。
- 逐点数据：`assets/scores-full.csv`。
- 曲线矢量源：`assets/score-curve-600k-1600k.svg`。
- PNG 曲线：`assets/score-curve.png`。
- 清醒制作审计：`assets/crafting-actions-audit.csv`。
- 评测口径：每个显示节点为 100 局自然场景评测，namespace/policy seed 与 SHA256 字段见 CSV；训练 reward 不等于自然 Score。

## 三段完整 MP4

| 文件 | checkpoint | Episode | 本局成就数 | 100 局平均 Score | SHA256 |
|---|---:|---:|---:|---:|---|
| `videos/baseline-full-episode.mp4` | 600k baseline | 71 | 13/22 | 11.8689691 | `bb5968803ad8e433d070ba70ec6dec15db75715536e612cbe80ff394726aa8d4` |
| `videos/branch-full-episode.mp4` | Branch 1.56M | 37 | 17/22 | 26.1207934 | `492b2326f263c97a416c173c0aaf488ed4c0644049bc870c549349c5cff75ccd` |
| `videos/source-full-episode.mp4` | Source 1.37M | 14 | 15/22 | 20.2234471 | `feececa4b6f2eadc89b7b5a2810eabe940fb384609755e74303cc22fdfa6e0dd` |

选择规则：每条路线先选 600k–1.6M 范围内经 100 局验证的最高平均自然分 checkpoint，再从它的 100 个自然 episode 里选解锁成就类型最多的完整一局；并列时取步数更长的一局。重放逐步核对角色位置、背包、episode 结束状态和最终成就。单局成就数不代表 checkpoint 平均水平。

完整的 checkpoint SHA256、评测文件 SHA256、seed、回放校验说明、帧率和时长见 `videos/video-manifest.json`。MP4 二进制保存在本地交付目录及共享服务器的项目资料包；仓库发布检查不接受视频二进制，因此 GitHub 仓库只保留视频索引与哈希，不通过重命名或拆分绕过检查。

附件 SHA256：

- `assets/scores-full.csv`: `c90fe4405439cc09d316194ac87d790226413ca7a56e2a06d7f540aa8dca057a`
- `assets/score-curve-600k-1600k.svg`: `1b2c405e96c6b9b9e18b23f4f654f195276b384da1877948d5e454f843a9dddc`
- `assets/score-curve.png`: `e6168690b5e9346302ad3691c09a1ebf655f9a1700e6e8d6bc95fd6823b3429f`
- `assets/crafting-actions-audit.csv`: `05c9bb16c1ca8ec3b27191b0fbb34ac2ae526d10733ed235bc65c671ca159825`
- `../../assets/scale-v3-course-stone-pickaxe-frame.png`: `e8310ab907dd7e71203a1ddbfa3244580066a87dc2a7f5ccd18005d93c3e3cfd`
- `videos/video-manifest.json`: `1293bbe0657c0a6ba24dae2fe7b4a441e02f97df6c864a9b6f38228c8587a081`

## 结论限制

- 每条路线一个训练种子，结果不得表述为多种子统计结论。
- Branch/Source 均采用 CR 类回放；本配对比较主要改变成功轨迹 bank，不是 CR 对均匀回放的消融。
- Branch/Source 的 bank 内容有差异，不能把结果归因于单一变量。
- 1.6M endpoint 与 1.56M/1.37M 已测峰值是不同节点，不能混用。
- 铁镐率仍为 0%；视频中的一局成功不能代替自然评测率。
