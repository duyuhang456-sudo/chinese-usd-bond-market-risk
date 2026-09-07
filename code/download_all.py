"""
一键下载全部数据源（第 1 阶段 · 第 1 天）
用法：
    cd 仓库根目录
    ./.venv/bin/python code/download_all.py
"""
from __future__ import annotations

import pandas as pd

import download_fred
import download_mchb
import download_treasury


def main() -> None:
    print("=" * 60)
    print("1/3  下载 FRED 系列（汇率/利差/政策利率/CPI）")
    print("=" * 60)
    fred = download_fred.download_fred()

    print("\n" + "=" * 60)
    print("2/3  下载美国国债收益率曲线（treasury.gov）")
    print("=" * 60)
    tsy = download_treasury.download_treasury()

    print("\n" + "=" * 60)
    print("3/3  下载 MCHB ETF 行情（Yahoo Finance）")
    print("=" * 60)
    mchb: pd.DataFrame | None = None
    try:
        mchb = download_mchb.download_mchb()
    except RuntimeError as e:
        print(f"[MCHB] 跳过：{e}")

    # 汇总
    print("\n" + "=" * 60)
    print("下载汇总")
    print("=" * 60)
    for code, df in fred.items():
        print(f"  fred_{code}.csv      {len(df):>5} 行  "
              f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    if tsy is not None and len(tsy):
        print(f"  treasury_yield_curve.csv  {len(tsy):>5} 行  "
              f"{tsy['date'].min().date()} ~ {tsy['date'].max().date()}")
    if mchb is not None and len(mchb):
        print(f"  mchb.csv            {len(mchb):>5} 行  "
              f"{mchb['Date'].min().date()} ~ {mchb['Date'].max().date()}")
    print("\n完成。原始 CSV 见 raw_data/")


if __name__ == "__main__":
    main()
