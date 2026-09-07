# 数据源清单与下载说明

> 对应需求文档第 3 章。所有数据均来自公开免费渠道。
> 原始 CSV 存放于 `raw_data/` 并随仓库同步（弃用的银行股文件除外）。

## 一、数据源清单（已核对）

| # | 数据类别 | 内容 | 来源 | 代码/标识 | 频率 | 本次取得范围 |
|---|---|---|---|---|---|---|
| 1 | 无风险利率 | 美债收益率曲线 1M–30Y | [treasury.gov](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2026) | — | 日 | 2021-08 ~ 2026-09（1274 行） |
| 2 | 标的行情 | ChinaAMC 亚洲美元投资级债 ETF（**正式标的**） | Yahoo Finance | **9141.HK** | 日 | 2021-08 ~ 2026-09（1250 行）✓ |
| 3 | 汇率 | CNY/USD 日度中间价 | [FRED](https://fred.stlouisfed.org/series/DEXCHUS) | DEXCHUS | 日 | 2021-08 ~ 2026-08（1270 行） |
| 4 | 信用利差 | 新兴市场投资级公司债 OAS | [FRED](https://fred.stlouisfed.org/series/BAMLEMIBHGCRPIOAS) | BAMLEMIBHGCRPIOAS | 日 | 2023-09 ~ 2026-09（787 行）⚠️ |
| 5 | 宏观基准 | 联邦基金有效利率 | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS) | FEDFUNDS | 月 | 2021-08 ~ 2026-08（61 行） |
| 6 | 宏观基准 | 美国 CPI 指数 | [FRED](https://fred.stlouisfed.org/series/CPIAUCSL) | CPIAUCSL | 月 | 2021-08 ~ 2026-07（59 行） |
| 7 | 事件参考 | 金融风险事件时间线 | IMF/美联储/财经媒体 | — | — | 待建（用于异常值校验与压力情景） |

### ⚠️ 数据可得性说明
1. **FRED 利差系列（BAMLEMIBHGCRPIOAS）自 2023-09 才有数据**（约 3 年，非 5 年）。需求文档中该代码为扫描件 OCR，经核对以本表代码为准；历史偏短属源数据限制，信用利差因子及其外部校验需接受该窗口。
2. **CPI 最新值有发布滞后**（本次到 2026-07），属正常。
3. **需求文档 3.2 的“iShares 中国投资级美元债 ETF（MCHB）”在真实市场不存在**。经 Yahoo 官方接口核实，`MCHB` 实为 **Mechanics Bancorp（美国银行股，EQUITY/Nasdaq）**，Yahoo 搜索亦无任何 iShares 中国美元债基金对应此代码。该数据已弃用（`raw_data/mchb_MechanicsBancorp_WRONG_弃用.csv`）。**2026-09-07 已确定改用 9141.HK 为正式标的（见三）。**

## 二、下载方式

```bash
# 首次：创建环境
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 下载全部（FRED + 国债曲线 + 正式标的 9141.HK）
./.venv/bin/python code/download_all.py

# 或按源单独下载
./.venv/bin/python code/download_fred.py
./.venv/bin/python code/download_treasury.py
./.venv/bin/python code/download_benchmark.py
```

脚本说明：
- `code/download_fred.py`：FRED 直下 CSV（无需 key）。
- `code/download_treasury.py`：treasury.gov 按年抓取（2021~当年）后拼接；site 改版后需带 `type/_format` 参数。
- `code/download_benchmark.py`：yfinance 取正式标的 **9141.HK**，带指数退避重试，保存为 `raw_data/benchmark_9141HK.csv`。

## 三、标的代码核实记录（MCHB 疑点 → 正式标的 9141.HK）

- **问题**：文档 3.2 指定 `MCHB = iShares 中国投资级美元债 ETF`，但 Yahoo 官方接口核实结果为 **Mechanics Bancorp 银行股**（2026-09-07，代理修复后直连确认；`instrumentType=EQUITY`）。此前手动导出拿到的 1280 行实为**银行股股价**（5 年累计约 −48%、年化波动约 57%），与债券基金特征严重不符，**已弃用**。
- **候选真实代理（均已下载，特征正常）**：

| 候选 | 柜台 | Yahoo 区间 | 行数 | 5年累计 | 年化波动 | 文件 |
|---|---|---|---|---|---|---|
| ChinaAMC Asia USD IG Bond ETF | **9141.HK（USD）** | 2021-08~2026-09 | 1250 | +5.0% | 3.6% | `raw_data/proxy_9141HK_AsiaUSDIG.csv` |
| 同上 | 3141.HK（HKD） | 2021-08~2026-09 | 1253 | +5.9% | 7.5% | `raw_data/proxy_3141HK_AsiaUSDIG.csv` |

- **决定（2026-09-07）**：正式标的采用 **9141.HK（USD 柜台）**。下载脚本：`code/download_benchmark.py`（ticker 可改，保存为 `raw_data/benchmark_9141HK.csv`）。HKD 柜台 3141.HK 保留备用（`raw_data/alt_3141HK_AsiaUSDIG_HKD.csv`）。

> Yahoo 下载数据仅供个人研究使用，故不入库、不对外分发。
