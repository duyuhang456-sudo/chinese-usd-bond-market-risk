"""美国国债收益率曲线下载（treasury.gov，日度）

消费：无（需联网）
产出：raw_data/ 下的美债收益率曲线日度表（1 Mo–30 Yr 各期限到期收益率）
口径：覆盖需求文档第 3.1 节数据源。URL 模板（site 改版后需带 query 参数）：
      /daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve
      &field_tdr_date_value={year}&_format=csv
      按年份（2021..今年）抓取后拼接、去重，并按近 5 年窗口截取。
用法：./.venv/bin/python code/download_treasury.py
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

from common import START, save_csv

BASE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&_format=csv"
)

YEARS = list(range(2021, dt.date.today().year + 1))   # 2021..当前年
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


def download_one_year(year: int) -> pd.DataFrame:
    """下载某一整年的国债收益率曲线。

    参数：
        year: 年份，如 2022。

    返回：
        该年的原始 CSV 内容，列为各期限（1 Mo … 30 Yr）与 Date，未做清洗与截取。
    """
    url = BASE.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download_treasury() -> pd.DataFrame:
    """下载 2021 年至今的全部国债收益率曲线，拼接去重后归档。

    返回：
        拼接、按日期升序去重（保留最后一条）、截到 START 之后的长表，
        已写入 raw_data/treasury_yield_curve.csv。Date 列已改名为 date。
    """
    frames = []
    for year in YEARS:
        print(f"[TREASURY] downloading {year} ...")
        frames.append(download_one_year(year))

    all_df = pd.concat(frames, ignore_index=True)
    all_df.columns = [c.strip() for c in all_df.columns]
    all_df = all_df.rename(columns={"Date": "date"})
    all_df["date"] = pd.to_datetime(all_df["date"], format="%m/%d/%Y")
    all_df = (
        all_df[all_df["date"] >= pd.Timestamp(START)]       # 近5年窗口
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    save_csv(all_df, "treasury_yield_curve.csv")
    return all_df


if __name__ == "__main__":
    download_treasury()
