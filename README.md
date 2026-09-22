# 中资投资级美元债市场风险计量与压力测试预研工具

> Chinese Investment-Grade USD Bond Market Risk Measurement & Stress-Testing Prototype

风控部实习预研项目：搭建轻量化风险计量与压力测试原型，验证不同计量方法在新兴市场（中资投资级美元债）债券中的适用性，为风控模型优化与日常监控提供研究参考。

> **本项目为实验性预研（pre-research），产出为研究结论与原型工具，不用于生产决策。**

## 数据来源（均为公开免费渠道）

| 数据 | 来源 | 频率 |
|---|---|---|
| 美债收益率曲线（1M–30Y） | U.S. Treasury (treasury.gov) | 日度 |
| ChinaAMC 亚洲美元投资级债 ETF（**9141.HK**，组合代理） | Yahoo Finance | 日度 |
| 人民币兑美元即期汇率（CNY/USD） | FRED `DEXCHUS` | 日度 |
| 新兴市场投资级公司债利差（OAS） | FRED `BAMLEMIBHGCRPIOAS` 族 | 日度 |
| 美联储政策利率 / 美国 CPI | FRED `FEDFUNDS` / `CPIAUCSL` | 月 / 日 |
| 全球金融风险事件时间线 | IMF / 美联储 / 公开财经媒体 | — |

## 四阶段路线

| 阶段 | 内容 | 计划周期 |
|---|---|---|
| 一 | 数据体系搭建与风险因子拆解（利率 / 信用利差 / 汇率） | Week 1 |
| 二 | 风险计量模型开发与回测验证（参数法 & 历史模拟法 VaR） | Week 2 |
| 三 | 压力测试框架构建与测算分析（历史 / 假设情景） | Week 3 |
| 四 | 工具整合与终期成果输出（原型 + 预研报告 + PPT） | Week 4 |

## 文档

- [第 1 周计划 · 第一阶段（Markdown）](下周计划_第一阶段_数据体系搭建与风险因子拆解.md)
- [第 1 周计划 · 第一阶段（PDF）](下周计划_第一阶段_数据体系搭建与风险因子拆解.pdf)
- [第 2 周计划 · 第二阶段（Markdown）](下周计划_第二阶段_风险计量模型开发与回测验证.md)
- [第 3 周计划 · 第三阶段（Markdown）](下周计划_第三阶段_压力测试框架构建与测算分析.md)
- [第 4 周计划 · 第四阶段（Markdown）](下周计划_第四阶段_工具整合与终期成果输出.md)
- **[VaR 计量口径与回测协议（阶段二 9/14 定稿，9/15–9/17 逐日回填）](docs/phase2_spec.md)**
- **[模型验证分析报告（阶段二正式交付，9/18 收口）](docs/phase2_model_validation.md)**
- **[压力情景库（第三阶段交付物 1：冲击参数与情景说明）](docs/phase3_scenario_library.md)**
- **[压力测试分析报告（第三阶段交付物 2：损失 / 回撤 / 因子贡献度 / 尾部风险点）](docs/phase3_stress_testing.md)**
- **[《数据说明文档》（第一阶段正式交付，9/11 收口）](docs/phase1_data_description.md)**
- **[第 1 周周报（周会要点 / 遗留问题 / 阶段二衔接）](docs/week1_report.md)**
- **[第 2 周周报（阶段二要点 / 推荐口径 / 阶段三参数交接清单）](docs/week2_report.md)**
- **[第 2 周实习周报（正式版，含图表与代码片段）](docs/week2_report_formal.md)**
- **[第 3 周周报（阶段三要点 / 遗留问题 / 阶段四交接清单）](docs/week3_report.md)**
- **[第 3 周实习周报（正式版，含图表与代码片段）](docs/week3_report_formal.md)**
- **[第 3 周补充修订周报（正式版：主情景库 1 日持有期口径修订 + 28 处缺陷纠正）](docs/week3_revision_formal.md)**
- [数据源清单与下载说明](docs/data_sources.md)
- [数据清洗说明（对齐 + 缺失值，含完整率报告解读）](docs/data_cleaning.md)
- [异常值识别与事件校验说明（异常判定表解读）](docs/outlier_check.md)
- [三大核心风险因子构建与校验说明（含利差剥离与久期估计）](docs/factors.md)
- [报价陈旧出路对照（周度 / 官方 NAV / 真实久期修正 → 采用 NAV，阶段二输入口径）](docs/staleness_remedy.md)
- [工具整合前置盘点：路径常量收敛清单（阶段四 Day 1 施工依据）](docs/tool_path_convergence.md)
- **[工具使用说明（环境搭建 / 一键与分阶段运行 / 脚本职责 / 输出物清单 / 图-脚本-数据对应 / 硬断言语义 / 已知边界）](docs/tool_usage.md)**

*每周计划与交付物随进度在本仓库更新。*

## 目录规划

```
raw_data/    原始数据（CSV）
clean_data/  清洗后数据（CSV；含异常候选/判定表）
events/      风险事件参考时间线（CSV，供异常校验与压力情景复用）
factors/     构建的风险因子表
results/     风险计量结果（VaR/回测/分段，阶段二起）
code/        数据获取 / 清洗 / 建模代码
figures/     可视化图表
docs/        数据说明文档与研究文档
```

## 结果目录（`results/`，阶段二）

阶段二共 17 个结果文件，按用途分四类：

| 文件 | 内容 | 产出日 |
|---|---|---|
| `baseline_var.csv` | 无条件正态 VaR 基线（复权主口径 1273 日：日 σ 0.266% / 年化 4.2233%；忽略均值口径 VaR95 0.4376% / VaR99 0.6189%，另列含均值口径 0.4341% / 0.6154%） | 9/14 |
| `baseline_var_tr.csv` | 双口径基线 σ / 1 日 VaR（复权主口径 + 人民币次口径 + 除权审计底） | 9/14 |
| `factor_exposure.csv`、`factor_corr.csv` | 组合因子暴露 δ 与因子相关阵 | 9/14 |
| `regimes.csv` | 滚动 60 日年化波动率 + 平稳/高波动分段标记（**同期事后标记**） | 9/14 |
| `doubtful_days.csv` | 16 条「存疑」异常日清单（11 ETF / 3 dOAS / 2 FX；只标注不改数） | 9/14 |
| `var_parametric.csv` | 参数法 VaR 序列（1023 行 × 双口径，5 个模型变体） | 9/15 |
| `var_attribution.csv` | VaR 方差归因（利率 Δ5Y/Δ10Y、利差代理、汇率） | 9/15 |
| `garch_refit_trace_{main,rmb}.csv` | GARCH(1,1) 逐日重估轨迹（含边界解标记） | 9/15 |
| `var_historical.csv` | 历史模拟 VaR 序列（1023 行，HS250/500/750 + FHS-E/FHS-G） | 9/16 |
| `backtest_results.csv` | **回测检验主表**（262 行 × 33 列：失败率 + Kupiec + Christoffersen + ES） | 9/17 |
| `backtest_exceptions.csv` | 违规判定表（832 行，含存疑日 / 事件邻近 / 3141 同向性标注） | 9/17 |
| `model_scorecard.csv` | 逐模型 × 口径 × 置信度评分卡（门槛一/二判定 + 校准偏差 + 资本占用 + 覆盖落差） | 9/18 |
| `baseline_decision.csv` | **基准口径定案表**（主/次口径 × 95%/99% 基准、备选、淘汰及理由） | 9/18 |
| `delta_transmission_events.csv` | δ 线性传导逐事件窗偏差（1500 行：85 事件 × 窗口 × δ 来源 × 口径） | 9/18 |
| `delta_transmission_summary.csv` | δ 传导偏差汇总（24 行：按窗口 / 口径 / δ 来源 / M2-vs-M3 聚合） | 9/18 |

口径与检验方法见 [VaR 计量口径与回测协议](docs/phase2_spec.md)；
结果解读、模型适用性结论、**基准口径定案**与 **δ 传导精度验证**见 [模型验证分析报告](docs/phase2_model_validation.md) §7.2 / §7.3。

## 结果目录（`results/`，阶段三）

阶段三共 9 个结果文件，前三个为测算主表，后六个为稳健性分析。
**全部口径为 1 日持有期**（按老师指示与阶段二 VaR 对齐后重做并重跑）：

| 文件 | 内容 | 产出日 |
|---|---|---|
| `stress_scenarios.csv` | **压力情景库**（112 行 × 15 列：15 情景 × 因子 × 幅度测度；含 `measure` 标记与数据可得性标记） | 9/21，9/24 补 X2，9/25 1 日重做 |
| `stress_impact.csv` | **压力测试测算主表**（30 行 × 38 列：情景 × 双口径的损失读数 `benchmark_loss_pct` 与伴随列 `loss_realized_pct` / 回撤 / 缓冲 / 1 日对标 / 尾部标记；损失类列**正值 = 损失**。信用区间两列 `credit_range_lo_pct` / `hi_pct` 与同期限点估计 `worst_day_modeled_pct` 均为 **1 日量**） | 9/22–9/23，9/24 口径修订，9/25 1 日重做，9/21 信用区间对齐 1 日 |
| `stress_factor_contrib.csv` | 因子贡献度分解（150 行 × 10 列：Δ5Y / Δ10Y / ΔOAS / 汇率 / **未解释残差**单列；`share_pct` 为同行内绝对贡献归一） | 9/23 |
| `stress_coverage.csv` | 覆盖度检验：情景损失在历史分布中的分位数与是否被实测最差单日超越（30 行 × 15 列） | 9/24 |
| `stress_worst_windows.csv` | 样本内各期限（1/3/5/6 日）× 双口径的实测最差窗口榜（每组合前 10，80 行 × 8 列；1 日为主口径，3/5/6 日仅作附录对照的行内标注） | 9/24 |
| `stress_window_sens.csv` | 历史情景窗口起止平移 ±1 / ±2 日的损失与因子幅度（50 行 × 12 列：`loss_pct` 窗内累计、`mdd_pct` 窗内最大回撤、`worst_day_pct` 窗内最差单日） | 9/24，9/25 1 日重做 |
| `stress_shift_coverage.csv` | 平移感知的**库级**覆盖度（含 √t 参考值三列；1 日两行为主口径，3/5/6 日六行为附录对照，8 行 × 13 列） | 9/24，9/25 1 日重做 |
| `stress_buffer_sens.csv` | 五档缓冲（0 / 0.1997 / 0.3365 / 0.8390 / 1.0000 pp）下的尾部判定（150 行 × 9 列） | 9/24，9/25 1 日重做 |
| `stress_anchor_sens.csv` | 锚点窗长改取 1 / 3 / 5 / 6 日时假设情景梯度的平移幅度（12 行 × 10 列，1 日为主口径） | 9/24，9/25 1 日重做 |

情景构造依据与数据边界见 [压力情景库](docs/phase3_scenario_library.md)；
三路径映射方法、缓冲量级来源、风险承受能力评估、**五项稳健性检验**与**方法局限**见
[压力测试分析报告](docs/phase3_stress_testing.md)。

9/24 的覆盖度检验查出第一版情景库漏掉了样本内 6 日期限上最差的一段冲击
（2022-03-08 ~ 03-15，俄乌叠加加息周期开启），据此补入补充情景 X2。按当时口径
（库内最深损失取 `results/stress_coverage.csv` 的 `loss_cum_pct` 列，为**窗内累计**；缺口值取重做前的 `git show 8e537c7:results/stress_coverage.csv` 的 `exceeded_by_pct` 列），
6 日期限主口径的库内最深损失由 H1 的 1.5368% 提高到 X2 的 3.0008%，覆盖缺口
由 1.5261pp 收窄至 0.0621pp；次口径同向，由 1.1118pp 收窄至 0.1327pp。
5 日期限主口径仍差 **0.5330pp**（库内 H3 2.2896% 对实测最差 2.8226%），**未修补**；
5 日次口径恰好持平（H3 2.1359% 对实测最差 2.1359%，缺口 0.0000pp），
6 日次口径**仍差 0.1327pp**（X2 2.2096% 对实测最差 2.3423%），并非无缺口。
**这些读数属多期口径**：它们拿 3 日锚点的情景损失去比 5 / 6 日的实测最差窗口，
两侧期限本就不匹配。1 日重做后情景侧已无多日损失量，该比较不再成立，
既不能说缺口已修补、也不能说缺口扩大，现移入《压力测试分析报告》与正式周报的附录，并标注为不同量纲、不构成覆盖度结论。
1 日期限的库级覆盖另有一组读数（主口径库内最深 2.5363%、次口径 3.1708%，
均深于实测最差单日 1.3449% / 1.4739%），见 `stress_shift_coverage.csv` 中
`is_main_horizon == True` 的两行。

**1 日持有期重做**（按老师指示「主情景库保持 1 日持有期，与 VaR 口径对齐」）改了四处：
假设情景锚点由全样本 3 日累计极值改为**全样本单日极值**并落库为 `peak_1d` 行；
历史与补充情景的损失读数由窗内最大回撤改为**窗内最差单日**（落库为 `worst_day` 行），
窗内累计与最大回撤保留为伴随列；判定阈值与偏差缓冲同步降到 1 日口径
（1 日 99% 期望损失主 0.6659% / 次 0.7024%，缓冲 0.3365pp）；
`stress_impact.csv` 的 `mdd_pct` 列符号由负值改为正值，使全表损失类列一律同向。
3 日口径的 `peak_3d` 行与多期覆盖度读数保留为附录对照、不参与判定。
重做的全部后果见 [压力测试分析报告](docs/phase3_stress_testing.md) §五。

## 图表目录（`figures/`，阶段二）

| 文件 | 内容 | 产出日 |
|---|---|---|
| `var_series_compare.png`、`var_garch_sigma.png`、`var_hs_vs_parametric.png` | 参数法 VaR 序列 / GARCH σ 轨迹 / 参数法与历史模拟对比 | 9/15 |
| `var_hs_quantile_path.png`、`var_fhs_compare.png` | 历史分位数路径 / Filtered-HS 对比 | 9/16 |
| `backtest_exceptions.png`、`backtest_coverage.png` | 违规时间线 + 事件标注 / 失败率点估计 + Kupiec 接受区间 | 9/17 |
| `baseline_model_tradeoff.png` | 95% / 99% 分面：失败率（纵）对资本占用指数（横），含名义线与审慎带 | 9/18 |
| `delta_transmission.png` | 预测 vs 实际 / 偏差随冲击幅度的变化 / 偏差分布（主口径 · 3 日窗 · 事件前 δ） | 9/18 |

## 图表目录（`figures/`，阶段三）

| 文件 | 内容 | 产出日 |
|---|---|---|
| `stress_scenario_library.png` | 假设情景三方向 × 三级梯度（锚点 = 全样本最差单日，1 日持有期；虚线为轻度档锚点 = 已实现的最差单日） | 9/21，9/25 1 日重做 |
| `stress_mapping.png` | 三条传导路径：利率沿用 δ / 信用给 95% 区间 / 汇率口径搬移 | 9/22 |
| `stress_loss_vs_var.png` | 情景 **1 日**损失 vs **1 日** 99% ES（全库同期限，菱形为阶段二登记的判定线 0.6659% / 0.7024%） | 9/23，9/25 1 日重做 |
| `stress_factor_contrib.png` | 因子贡献度分解（残差以纹理单列，不与因子混计） | 9/23 |
| `stress_coverage.png` | 情景损失在历史 **1 日**损失分布中的位置（含 99% / 99.9% 分位与实测最差单日线） | 9/24，9/25 1 日重做 |

**一键复现**（推荐入口，19 步按依赖顺序自动串起，默认跳过取数、不联网）：

```bash
python code/run_all.py            # 全量；跑完自动对账 results/ 与 figures/
python code/run_all.py --check    # 只对账不重跑
python code/run_all.py --only phase3
python code/run_all.py --list     # 列执行清单
```

在干净 venv 里清空 `results/*.csv` 与 `figures/*.png` 后重跑，48 个产物与清空前
**逐字节相同**。注意两点：**不能用 `--only phase2,phase3` 重建全部产物**
（`baseline_var_tr.csv` 是阶段一产物，`var_backtest.py` 依赖它）；
验证可复现性必须**先清空再重建**，在旧产物上校验发现不了空转的脚本。

分阶段手动执行（须按序，`var_historical.py` 依赖 `var_parametric.csv` 的 σ 列）：

```bash
python code/build_factors_nav.py && python code/build_tr_factors.py && python code/prep_phase2.py
python code/var_parametric.py && python code/var_historical.py && python code/var_backtest.py
python code/recommend_baseline.py && python code/verify_delta_transmission.py
python code/stress_scenarios.py && python code/stress_impact.py && python code/stress_robustness.py
```

阶段三三步须按序（`stress_impact.py` 消费 `stress_scenarios.csv`，
`stress_robustness.py` 消费 `stress_impact.csv`）；
三者只读因子表与阶段二结果 CSV，不重算 VaR，可独立重跑。

环境搭建、脚本职责表、图 ← 脚本 ← 数据三列对应、硬断言语义与已知边界见
[工具使用说明](docs/tool_usage.md)。

脚本内置多处**硬断言**，任一不满足即报错终止：

1. `stress_scenarios.py`——主/次口径收益差须逐值等于汇率项（log 收益可加性，`max|rmb − tr − fx| < 1e-9`）；
2. `stress_impact.py`——HS250 @ common773 的四个基准配置须与阶段二报告 §7.2 登记值一致；
3. `stress_impact.py`——**情景库中的每个情景都须进入测算**（此前情景清单被硬编码，
   新增的 X2 曾因此静默漏算，直到覆盖度检验才暴露）；
4. `stress_robustness.py`——1 日重做新增的一组口径一致性断言：主期限须从
   `stress_impact.csv` 的 `horizon_days` 读出且恒为 1 日；窗口敏感性表的最差单日须与
   `benchmark_loss_pct` 逐值对账、`mdd_pct` 须与伴随列逐值对账；缓冲网格中标着
   「本报告取值」的那一档须逐条复现基准表的损失值与尾部判定，且该标签指向的档位值
   须与常量 `BUFFER` 一致。
