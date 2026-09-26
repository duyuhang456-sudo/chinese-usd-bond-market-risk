"""正式标的下载：ChinaAMC 亚洲美元投资级债 ETF（9141.HK，美元柜台）（阶段一 第 1 天 · 9/7）

消费：无（要联网，走 Yahoo Finance）
产出：raw_data/benchmark_9141HK.csv（9141.HK 日度 OHLC / Adj Close / Volume）
口径：对应需求文档 3.2「标的行情数据」，拿它当中资/亚洲投资级美元债组合的代理标的。文档
      原本指定 MCHB（iShares 中国投资级美元债 ETF），但经 Yahoo 核实，这个代码实际是
      Mechanics Bancorp 银行股，市场上没有对应的 iShares 基金；经确认（2026-09-07），
      正式标的改用 9141.HK。要换成 HKD 柜台（3141.HK）或别的 ticker，改 TICKER 和 OUT
      两个常量就行。
用法：./.venv/bin/python code/download_benchmark.py
"""
from __future__ import annotations

import time

import pandas as pd

from common import START, save_csv

TICKER = "9141.HK"
OUT = "benchmark_9141HK.csv"
DESC = "ChinaAMC Asia USD Investment Grade Bond ETF (USD)"
MAX_ATTEMPT = 5


def _fetch() -> pd.DataFrame:
    """向 Yahoo 拉一次 9141.HK 的日线，不重试。

    返回的表列为 Date / Open / High / Low / Close / Adj Close / Volume，Date 已经去掉
    时区。取不到数据就抛 RuntimeError。
    """
    import yfinance as yf
    hist = yf.Ticker(TICKER).history(start=START, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"{TICKER}: 未取到任何数据")
    df = hist.reset_index()
    df.columns = [c.title() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def download_benchmark(attempts: int = MAX_ATTEMPT) -> pd.DataFrame:
    """下载正式标的 9141.HK，归档到 raw_data/benchmark_9141HK.csv，返回这张日线表。

    attempts 是最多试几次；每失败一次就等 20×第几次 秒再重试（线性退避）。全都试完
    还是不行就抛 RuntimeError，把最后一次的错误带上。返回的表和 _fetch() 的列一样。
    """
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            df = _fetch()
            save_csv(df, OUT, date_col="Date")
            return df
        except Exception as e:  # noqa: BLE001 - 网络类错误统一重试
            last_err = e
            wait = 20 * i
            print(f"[BENCHMARK] {TICKER} 第 {i}/{attempts} 次失败：{e!r}，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"{TICKER} 下载失败：{last_err!r}")


if __name__ == "__main__":
    download_benchmark()
