"""FRED 数据下载（CSV 直下，无需 API key）

消费：无（需联网）
产出：raw_data/ 下的 DEXCHUS、BAMLEMIBHGCRPIOAS、FEDFUNDS、CPIAUCSL
口径：覆盖需求文档第 3 章——3.3 CNY/USD 日度中间价（DEXCHUS）、3.4 新兴市场投资级(High
      Grade) 公司债 OAS（BAMLEMIBHGCRPIOAS）、3.5 美联储政策利率 / 美国 CPI
      （FEDFUNDS / CPIAUCSL）。
边界：需求文档 3.4 节给出的代码是扫描件 OCR 的结果，经核对为 BAMLEMIBHGCRPIOAS（OAS
      利差）。该系列自 2023-09 起才有数据（约 3 年），属数据可得性限制，在数据说明文档中标注。
用法：./.venv/bin/python code/download_fred.py
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from common import START, save_csv

# code -> 中文说明
FRED_SERIES: dict[str, str] = {
    "DEXCHUS": "人民币兑美元即期汇率 CNY/USD（日度）",
    "BAMLEMIBHGCRPIOAS": "新兴市场投资级公司债利差 OAS（日度）",
    "FEDFUNDS": "联邦基金有效利率（月度）",
    "CPIAUCSL": "美国 CPI 指数（月度）",
}


def download_one(code: str) -> pd.DataFrame:
    """下载单个 FRED 序列的完整历史。

    参数：
        code: FRED 系列代码，如 "DEXCHUS"。

    返回：
        date + value 两列的表，date 已转为 datetime，value 为空的交易日已剔除。
        不做时间窗口截取，截取由 download_fred() 负责。
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def download_fred(series: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """下载全部 FRED 序列，按近 5 年窗口截取后逐个归档到 raw_data/。

    参数：
        series: {代码: 中文说明} 的字典。不传时用模块级的 FRED_SERIES。

    返回：
        {代码: 截取后的数据表} 字典。每个表为 date + 以代码命名的值列，
        已写入 raw_data/fred_<代码>.csv。
    """
    series = series or FRED_SERIES
    out: dict[str, pd.DataFrame] = {}
    for code, desc in series.items():
        print(f"[FRED] {code}  {desc}")
        df = download_one(code)
        df = df[df["date"] >= START].copy()          # 近5年窗口
        df = df.rename(columns={"value": code})       # 值列以代码命名
        save_csv(df, f"fred_{code}.csv")
        out[code] = df
    return out


if __name__ == "__main__":
    download_fred()
