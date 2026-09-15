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
- **[VaR 计量口径与回测协议（阶段二 9/14 定稿，9/15–9/17 逐日回填）](docs/phase2_spec.md)**
- **[模型验证分析报告（阶段二正式交付，9/18 收口）](docs/phase2_model_validation.md)**
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

阶段二共 13 个结果文件，按用途分三类：

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

口径与检验方法见 [VaR 计量口径与回测协议](docs/phase2_spec.md)；
结果解读与模型适用性结论见 [模型验证分析报告](docs/phase2_model_validation.md)。

**一键复现**（须按序，`var_historical.py` 依赖 `var_parametric.csv` 的 σ 列）：

```bash
python code/build_factors_nav.py && python code/build_tr_factors.py && python code/prep_phase2.py
python code/var_parametric.py && python code/var_historical.py && python code/var_backtest.py
```
