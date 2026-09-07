"""
一键下载全部数据源（第 1 阶段 · 第 1 天）
用法：
    cd 仓库根目录
    ./.venv/bin/python code/download_all.py

数据源（对应 docs/data_sources.md）：
  1) FRED       汇率 / 新兴市场IG利差 / 政策利率 / CPI
  2) Treasury   美债收益率曲线（无风险利率基准）
  3) Benchmark  正式标的 9141.HK（中资/亚洲美元投资级债代理）
"""
from __future__ import annotations

import pandas as pd

import download_benchmark
import download_fred
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
    print("3/3  下载正式标的 9141.HK（ChinaAMC 亚洲美元投资级债 ETF）")
    print("=" * 60)
    bench: pd.DataFrame | None = None
    try:
        bench = download_benchmark.download_benchmark()
    except RuntimeError as e:
        print(f"[BENCHMARK] 跳过：{e}")

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
    if bench is not None and len(bench):
        print(f"  {download_benchmark.OUT}  {len(bench):>5} 行  "
              f"{bench['Date'].min().date()} ~ {bench['Date'].max().date()}")
    print("\n完成。原始 CSV 见 raw_data/")


if __name__ == "__main__":
    main()
