# 数据源清单与下载说明

> 对应需求文档第 3 章。所有数据均来自公开免费渠道。
> 原始 CSV 存放于 `raw_data/`（本地，不入库），用脚本可随时重新下载。

## 一、数据源清单（已核对）

| # | 数据类别 | 内容 | 来源 | 代码/标识 | 频率 | 本次取得范围 |
|---|---|---|---|---|---|---|
| 1 | 无风险利率 | 美债收益率曲线 1M–30Y | [treasury.gov](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2026) | — | 日 | 2021-08 ~ 2026-09（1274 行） |
| 2 | 标的行情 | iShares 中国投资级美元债 ETF | Yahoo Finance | MCHB | 日 | 2021-08 ~ 2026-09（1280 行）⚠️ |
| 3 | 汇率 | CNY/USD 日度中间价 | [FRED](https://fred.stlouisfed.org/series/DEXCHUS) | DEXCHUS | 日 | 2021-08 ~ 2026-08（1270 行） |
| 4 | 信用利差 | 新兴市场投资级公司债 OAS | [FRED](https://fred.stlouisfed.org/series/BAMLEMIBHGCRPIOAS) | BAMLEMIBHGCRPIOAS | 日 | 2023-09 ~ 2026-09（787 行）⚠️ |
| 5 | 宏观基准 | 联邦基金有效利率 | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS) | FEDFUNDS | 月 | 2021-08 ~ 2026-08（61 行） |
| 6 | 宏观基准 | 美国 CPI 指数 | [FRED](https://fred.stlouisfed.org/series/CPIAUCSL) | CPIAUCSL | 月 | 2021-08 ~ 2026-07（59 行） |
| 7 | 事件参考 | 金融风险事件时间线 | IMF/美联储/财经媒体 | — | — | 待建（用于异常值校验与压力情景） |

### ⚠️ 数据可得性说明
1. **FRED 利差系列（BAMLEMIBHGCRPIOAS）自 2023-09 才有数据**（约 3 年，非 5 年）。需求文档中该代码为扫描件 OCR，经核对以本表代码为准；历史偏短属源数据限制，信用利差因子及其外部校验需接受该窗口。
2. **CPI 最新值有发布滞后**（本次到 2026-07），属正常。
3. **MCHB 已通过手动导出获取**（2026-09-07），但存在两点保留：**(a)** 导出文件为 xlsx 且**缺 Volume 列**（已转存为规范 CSV）；**(b) 行情特征与“投资级美元债 ETF”不符**——5 年累计总回报约 −48%、年化波动约 57%、历史区间回撤一度达 −90%，更像高波动权益类产品。**疑为代码与描述不匹配，需与导师确认标的后再用于因子拆解。**

## 二、下载方式

```bash
# 首次：创建环境
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 下载全部（FRED + 国债曲线 + MCHB）
./.venv/bin/python code/download_all.py

# 或按源单独下载
./.venv/bin/python code/download_fred.py
./.venv/bin/python code/download_treasury.py
./.venv/bin/python code/download_mchb.py
```

脚本说明：
- `code/download_fred.py`：FRED 直下 CSV（无需 key）。
- `code/download_treasury.py`：treasury.gov 按年抓取（2021~当年）后拼接；site 改版后需带 `type/_format` 参数。
- `code/download_mchb.py`：yfinance 取 MCHB，带指数退避重试。

## 三、MCHB 获取记录（Yahoo 脚本限流，走手动导出）

- **2026-09-07**：Yahoo 脚本访问被限流（429/反爬），改为浏览器手动导出。导出文件实为 **xlsx 且缺 Volume 列**，已用 pandas 转存为规范 CSV 至 `raw_data/mchb.csv`（Date/Open/High/Low/Close/Adj Close，1280 行）。
- 若后续需补 **Volume**：待 Yahoo 解除限流后用 `code/download_mchb.py` 重试，或手动在 history 页确认导出是否含成交量列。

> 若浏览器也提示该代码不存在，说明 MCHB 或非真实在交易标的，需与导师确认替代标的（记录在案）。
> Yahoo 下载数据仅供个人研究使用，故不入库、不对外分发。
