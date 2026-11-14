# Scale-v3 项目资料与评测素材

此目录提供供组员分别撰写课程报告使用的项目事实、结果数据、图表源文件、行为审计和演示索引。它不是小组报告正文，也不是任何成员可直接提交的课程报告。每位成员应另用课程提供的 LaTeX 模板，结合自己的实际工作写作。

## 素材地图

- `PROJECT-MATERIALS.md`：项目术语、协议边界、关键结果与文件路径的事实索引。
- `assets/scores-full.csv`：两路线 600k–1.6M 的逐检查点自然评测数据；每路线 101 个已验证节点。
- `assets/score-curve-600k-1600k.svg`：可编辑曲线图源；与 CSV 对齐，不插值缺失节点。
- `assets/score-curve.png`：同曲线 PNG。
- `assets/crafting-actions-audit.csv`：清醒时制作行为审计，含睡眠动作排除和配方未满足定义。
- `assets/gameplay-stone-pickaxe-frame.png`：Branch 一局解锁石镐的关键帧。
- `videos/video-manifest.json`：baseline、Branch、Source 三段完整 MP4 的 checkpoint、episode、seed、单局成就数、重放验证和 SHA256。
- `EVIDENCE.md`：数据边界、解释限制、素材哈希和 MP4 获取位置。

## 可核对结果摘要

| 检查点 | 路线 | 100 局平均自然 Score | 石镐率 | 采集铁矿率 | 铁镐率 |
|---|---|---:|---:|---:|---:|
| 600k | baseline 锚点 | 11.87 | 0% | 0% | 0% |
| 1.0M | Branch | 19.78 | 40% | 0% | 0% |
| 1.0M | Source | 17.99 | 28% | 4% | 0% |
| 1.6M | Branch | 23.31 | 52% | 7% | 0% |
| 1.6M | Source | 19.40 | 47% | 2% | 0% |
| 1.56M 峰值 | Branch | 26.12 | 59% | 11% | 0% |
| 1.37M 峰值 | Source | 20.22 | 48% | 5% | 0% |

以上是工程评测摘要，不是报告正文。自然 Score 与训练 reward 不同；checkpoint 峰值与 1.6M 端点分开标注。Branch/Source 都采用 CR 类回放，bank 来源不同，因此不是 CR 与均匀回放对照。两臂各单训练种子，结论不能泛化成多种子因果证据。

## MP4 获取

发布检查不接受视频二进制，仓库仅保留 MP4 的 manifest 与 SHA256。可播放文件保存在共享服务器目录：

`outputs/reports/scale-v3-course-report-demo/group-report-package/videos/`

本地交付包也附有三段 MP4。请先按 `videos/video-manifest.json` 核对文件哈希，再用于课堂报告或 PPT。

## 报告写作使用

报告应采用课程模板要求的章节。组员可从资料索引挑选与本人职责相关的项目介绍、实验方法、数据、图片和视频；组员分工、本人贡献、个人反思以及课堂观摩内容必须分别按真实情况撰写。本仓库不提供可直接提交的共用报告正文、个人报告正文或 PDF。
