# 数据与演示材料核验记录

本包区分 100 局 checkpoint 评测与单局 MP4 演示。所有分数、视频信息和哈希应以此处列出的逐点 CSV 与下方核验记录为准；这份文件仅描述来源和限制，不构成报告正文。

## 曲线和表格数据

- 曲线范围：600k 已测 baseline 锚点至 1.6M 报告终点；Branch/Source 实际从共同的 610k 完整 checkpoint 开始续训。
- 逐点数据：`assets/scores-full.csv`。
- 曲线矢量源：`assets/score-curve-600k-1600k.svg`。
- PNG 曲线：`assets/score-curve.png`。
- 清醒制作审计：`assets/crafting-actions-audit.csv`。
- 评测口径：每个显示节点为 100 局自然场景评测，namespace/policy seed 与 SHA256 字段见 CSV；训练 reward 不等于自然 Score。

## 视频来源与完整性校验

### Baseline — checkpoint 600,000

- checkpoint SHA256：bd026df853ee976d780bec13824dc4a21aeaed587673aa3170735d9a128b4443
- 100 局自然评测均分：11.86896906
- 演示局：Episode 71；环境 seed 412844997；408 步；解锁 13/22：collect_drink, collect_sapling, collect_stone, collect_wood, defeat_zombie, eat_cow, make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone, place_table, wake_up
- 重放校验：exact logged action count, positions, inventories, terminal state and final achievements matched during deterministic replay
- 视频文件：baseline-full-episode.mp4；6032148 bytes；8 fps；440 帧；时长 55.0 秒
- 视频 SHA256：bb5968803ad8e433d070ba70ec6dec15db75715536e612cbe80ff394726aa8d4
- episode/actions/summary 评测记录 SHA256：7ec20f29eeb9003723e2ae8c6740d6014c9770b0d7454e9e47aa152b6256ec5a / 868259ebd2ee16b70b3c7d9710d32e2af633980ce0998aa077cea832ff59d00b / 4da7737a5394324357fce289f3d18a739e61e2ba1e8606a3bc5943a7845cb6f5

### Branch — checkpoint 1,560,000

- checkpoint SHA256：feac446c4fa92ba00ef38fd064951cd102b6cc64b779b1cc275980e2a0032943
- 100 局自然评测均分：26.12079344
- 演示局：Episode 37；环境 seed 4176110122；225 步；解锁 17/22：collect_coal, collect_drink, collect_iron, collect_sapling, collect_stone, collect_wood, defeat_zombie, eat_cow, make_stone_pickaxe, make_stone_sword, make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone, place_table, wake_up
- 重放校验：exact logged action count, positions, inventories, terminal state and final achievements matched during deterministic replay
- 视频文件：branch-full-episode.mp4；3769455 bytes；8 fps；257 帧；时长 32.12 秒
- 视频 SHA256：492b2326f263c97a416c173c0aaf488ed4c0644049bc870c549349c5cff75ccd
- episode/actions/summary 评测记录 SHA256：d23af879beff40e04062efa2f8a1b206e14112ed56a6ef13d759a0de78217ce8 / 235ade4759d93aa105b69afd35b8352682f015ced2a08b3f58bde1ecac8676d4 / 806d6d955e4b6ee1531a7a30edec9887e49d71327218ec48ccb15dda96cef2c9

### Source — checkpoint 1,370,000

- checkpoint SHA256：4728b2d35d91e09b16fb7cf3a938ddfa2c6d1df6d5d93537ee9b89ba3e716cf6
- 100 局自然评测均分：20.22344710
- 演示局：Episode 14；环境 seed 3853937150；237 步；解锁 15/22：collect_coal, collect_drink, collect_iron, collect_sapling, collect_stone, collect_wood, defeat_zombie, make_stone_pickaxe, make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone, place_table, wake_up
- 重放校验：exact logged action count, positions, inventories, terminal state and final achievements matched during deterministic replay
- 视频文件：source-full-episode.mp4；3964440 bytes；8 fps；269 帧；时长 33.62 秒
- 视频 SHA256：feececa4b6f2eadc89b7b5a2810eabe940fb384609755e74303cc22fdfa6e0dd
- episode/actions/summary 评测记录 SHA256：e547ad179a421561b1124c1e40f95de13697b56edb7d90077baaa7884a7058a3 / b6dacc1356eae20c631416a551dcf9fb1ee1090c589a83a85dac38cebc5e769c / 02f1d4a9d15e3bee61ebffb58d17971f0a121904ee92393b981085c89e2b161c


## 结论限制

- 每条路线一个训练种子，结果不得表述为多种子统计结论。
- Branch/Source 均采用 CR 类回放；本配对比较主要改变成功轨迹 bank，不是 CR 对均匀回放的消融。
- Branch/Source 的 bank 内容有差异，不能把结果归因于单一变量。
- 1.6M endpoint 与 1.56M/1.37M 已测峰值是不同节点，不能混用。
- 铁镐率仍为 0%；视频中的一局成功不能代替自然评测率。
