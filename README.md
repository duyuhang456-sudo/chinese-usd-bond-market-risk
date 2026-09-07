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
- **[《数据说明文档》（第一阶段正式交付，9/11 收口）](docs/phase1_data_description.md)**
- **[第 1 周周报（周会要点 / 遗留问题 / 阶段二衔接）](docs/week1_report.md)**
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
code/        数据获取 / 清洗 / 建模代码
figures/     可视化图表
docs/        数据说明文档与研究文档
```
