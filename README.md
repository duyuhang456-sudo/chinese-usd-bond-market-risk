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
- **[VaR 计量口径与回测协议（阶段二 9/14 定稿，9/15–9/17 逐日回填）](docs/phase2_spec.md)**
- **[模型验证分析报告（阶段二正式交付，9/18 收口）](docs/phase2_model_validation.md)**
- **[压力情景库（第三阶段交付物 1：冲击参数与情景说明）](docs/phase3_scenario_library.md)**
- **[压力测试分析报告（第三阶段交付物 2：损失 / 回撤 / 因子贡献度 / 尾部风险点）](docs/phase3_stress_testing.md)**
- **[《数据说明文档》（第一阶段正式交付，9/11 收口）](docs/phase1_data_description.md)**
- **[第 1 周周报（周会要点 / 遗留问题 / 阶段二衔接）](docs/week1_report.md)**
- **[第 2 周周报（阶段二要点 / 推荐口径 / 阶段三参数交接清单）](docs/week2_report.md)**
- **[第 2 周实习周报（正式版，含图表与代码片段）](docs/week2_report_formal.md)**
- [数据源清单与下载说明](docs/data_sources.md)
- [数据清洗说明（对齐 + 缺失值，含完整率报告解读）](docs/data_cleaning.md)
- [异常值识别与事件校验说明（异常判定表解读）](docs/outlier_check.md)
- [三大核心风险因子构建与校验说明（含利差剥离与久期估计）](docs/factors.md)
- [报价陈旧出路对照（周度 / 官方 NAV / 真实久期修正 → 采用 NAV，阶段二输入口径）](docs/staleness_remedy.md)

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

阶段三共 3 个结果文件：

| 文件 | 内容 | 产出日 |
|---|---|---|
| `stress_scenarios.csv` | **压力情景库**（73 行 × 15 列：14 情景 × 因子 × 三种幅度测度；含数据可得性标记） | 9/21 |
| `stress_impact.csv` | **压力测试测算主表**（28 行 × 30 列：情景 × 双口径的损失 / 回撤 / 缓冲 / 同期限与 1 日对标 / 尾部标记） | 9/22–9/23 |
| `stress_factor_contrib.csv` | 因子贡献度分解（140 行 × 10 列：Δ5Y / Δ10Y / ΔOAS / 汇率 / **未解释残差**单列） | 9/23 |

情景构造依据与数据边界见 [压力情景库](docs/phase3_scenario_library.md)；
三路径映射方法、缓冲量级来源、风险承受能力评估与**方法局限**见 [压力测试分析报告](docs/phase3_stress_testing.md)。

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
| `stress_scenario_library.png` | 情景库三方向幅度（历史实际 vs 假设三梯度，含史上最差单日参照线） | 9/21 |
| `stress_mapping.png` | 三条传导路径：利率沿用 δ / 信用给 95% 区间 / 汇率口径搬移 | 9/22 |
| `stress_loss_vs_var.png` | 情景损失 vs **同期限**历史 99% ES（菱形为各自窗口长度的经验分位） | 9/23 |
| `stress_factor_contrib.png` | 因子贡献度分解（残差以纹理单列，不与因子混计） | 9/23 |

**一键复现**（须按序，`var_historical.py` 依赖 `var_parametric.csv` 的 σ 列；
后两个脚本只消费 CSV、不重算 VaR，可独立重跑）：

```bash
python code/build_factors_nav.py && python code/build_tr_factors.py && python code/prep_phase2.py
python code/var_parametric.py && python code/var_historical.py && python code/var_backtest.py
python code/recommend_baseline.py && python code/verify_delta_transmission.py
python code/stress_scenarios.py && python code/stress_impact.py
```

阶段三两步须按序（`stress_impact.py` 消费 `stress_scenarios.csv`）；
两者只读因子表与阶段二结果 CSV，不重算 VaR，可独立重跑。
`stress_scenarios.py` 另含对账断言：主/次口径收益差须逐值等于汇率项（log 收益可加性），
差值非零即报错终止。
