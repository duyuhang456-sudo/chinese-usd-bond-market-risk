"""正式标的下载 —— ChinaAMC 亚洲美元投资级债 ETF（9141.HK，美元柜台）

消费：无（需联网，Yahoo Finance）
产出：raw_data/ 下的 9141.HK 日度 OHLC / Adj Close / Volume
口径：对应需求文档 3.2「标的行情数据」，作为中资/亚洲投资级美元债组合的代理标的。文档原
      指定 MCHB（iShares 中国投资级美元债 ETF），经 Yahoo 核实该代码实为 Mechanics Bancorp
      银行股，市场中不存在对应 iShares 基金；经确认（2026-09-07），正式标的改用 9141.HK。
      若需 HKD 柜台（3141.HK）或换其它 ticker，改 TICKER 与 OUT 即可。
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
    """向 Yahoo 拉取一次 9141.HK 的日线，不做重试。

    返回：
        日线表，列为 Date / Open / High / Low / Close / Adj Close / Volume，
        Date 已去掉时区。取不到数据时抛 RuntimeError。
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
    """下载正式标的 9141.HK 并归档到 raw_data/benchmark_9141HK.csv。

    参数：
        attempts: 最大尝试次数。每次失败后等待 20×第几次 秒再重试（线性退避）。

    返回：
        下载到的日线表，与 _fetch() 的列一致。

    异常：
        全部尝试失败时抛 RuntimeError，并带上最后一次的错误。
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
