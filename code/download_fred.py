"""FRED 数据下载：直接下 CSV，不用 API key（阶段一 第 1 天 · 9/7）

消费：无（要联网）
产出：raw_data/ 下的 DEXCHUS、BAMLEMIBHGCRPIOAS、FEDFUNDS、CPIAUCSL 四个 csv
口径：覆盖需求文档第 3 章的四项——3.3 人民币兑美元日度中间价（DEXCHUS）、3.4 新兴市场
      投资级(High Grade) 公司债 OAS 利差（BAMLEMIBHGCRPIOAS）、3.5 美联储政策利率和美国
      CPI（FEDFUNDS / CPIAUCSL）。
边界：需求文档 3.4 节给的代码是从扫描件 OCR 出来的，核对后确认是 BAMLEMIBHGCRPIOAS
      （OAS 利差）。这个系列 2023-09 才有数据，一共才三年左右，属数据可得性的限制，
      数据说明文档里标注了。
用法：./.venv/bin/python code/download_fred.py
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from common import START, save_csv

# 系列代码 -> 中文说明
FRED_SERIES: dict[str, str] = {
    "DEXCHUS": "人民币兑美元即期汇率 CNY/USD（日度）",
    "BAMLEMIBHGCRPIOAS": "新兴市场投资级公司债利差 OAS（日度）",
    "FEDFUNDS": "联邦基金有效利率（月度）",
    "CPIAUCSL": "美国 CPI 指数（月度）",
}


def download_one(code: str) -> pd.DataFrame:
    """下载单个 FRED 序列的全历史。code 是系列代码，比如 "DEXCHUS"。

    返回 date + value 两列，date 已转成 datetime，value 为空的交易日已经删掉。
    这里不截时间窗，截窗是 download_fred() 的事。
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
    """下载全部 FRED 序列，截到近 5 年窗口，逐个归档到 raw_data/。

    series 是 {代码: 中文说明} 的字典，不传就用模块级的 FRED_SERIES。
    返回 {代码: 截取后的表} 的字典，每张表是 date 加一列以代码命名的值，也已写进
    raw_data/fred_<代码>.csv。
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
