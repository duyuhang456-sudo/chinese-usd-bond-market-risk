"""
MCHB（iShares 中国投资级美元债 ETF）行情下载 —— Yahoo Finance

覆盖第 3.2 节数据源：日度收盘价、涨跌幅、成交量等。
Yahoo 对脚本访问较敏感（429 / 反爬页），本脚本带指数退避重试；
仍失败时会给出手动下载指引，原始 CSV 放入 raw_data/mchb.csv 即可。

备用思路（记录在案，便于后续切换）：
  - 换网络出口（Yahoo 常按出口 IP 限流）；
  - 浏览器手动在 finance.yahoo.com/quote/MCHB 历史页导出 CSV。
"""
from __future__ import annotations

import time

import pandas as pd

from common import START, save_csv

TICKER = "MCHB"
MAX_ATTEMPT = 5


def _fetch() -> pd.DataFrame:
    import yfinance as yf
    ticker = yf.Ticker(TICKER)
    hist = ticker.history(start=START, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"{TICKER}: 未取到任何数据")
    df = hist.reset_index().rename(columns=str.title)
    # yfinance 的 Date 列带时区，去掉
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[[c for c in keep if c in df.columns]]


def download_mchb(attempts: int = MAX_ATTEMPT) -> pd.DataFrame:
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            df = _fetch()
            save_csv(df, "mchb.csv", date_col="Date")
            return df
        except Exception as e:  # noqa: BLE001 - 网络类错误统一重试
            last_err = e
            wait = 20 * i
            print(f"[MCHB] 第 {i}/{attempts} 次失败：{e!r}，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(
        f"{TICKER} 下载失败（{last_err!r}）。"
        "请在浏览器打开 finance.yahoo.com/quote/MCHB/history 手动导出 CSV，"
        "另存为 raw_data/mchb.csv 后继续。"
    )


if __name__ == "__main__":
    download_mchb()
