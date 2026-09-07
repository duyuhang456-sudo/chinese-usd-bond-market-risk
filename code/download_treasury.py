"""
美国国债收益率曲线下载（treasury.gov，日度）

覆盖第 3.1 节数据源：1 月 - 30 年各期限到期收益率。
URL 模板（site 改版后需带 query 参数）：
  /daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve
   &field_tdr_date_value={year}&_format=csv
按年份(2021..今年)抓取后拼接、去重，并按近 5 年窗口截取。
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
    url = BASE.format(year=year)
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def download_treasury() -> pd.DataFrame:
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
