"""一键下载全部数据源（阶段一 第 1 天 · 9/7）

消费：无（要联网）
产出：raw_data/ 下的 FRED 各序列、美债收益率曲线、9141.HK 行情
口径：三个来源各管一块——FRED 出汇率 / 新兴市场 IG 利差 / 政策利率 / CPI；Treasury 出
      美债收益率曲线，当无风险利率基准用；Benchmark 出正式标的 9141.HK，拿来当中资/亚洲
      美元投资级债的代理。三个子脚本都只负责「下载并原样归档」，日期只做 ISO 化和窗口
      截取，不清洗。
边界：9141.HK 的日度 NAV 不在这里，那条由 download_nav.py 单独跑。
用法：在仓库根目录执行 ./.venv/bin/python code/download_all.py
"""
from __future__ import annotations

import pandas as pd

import download_benchmark
import download_fred
import download_treasury


def main() -> None:
    """挨个下载三类数据源，最后打一份汇总，是阶段一的取数入口。

    从 FRED / treasury.gov / Yahoo 三个外部接口拿数（都要联网），产出 raw_data/ 下的
    fred_*.csv、treasury_yield_curve.csv 和 benchmark_9141HK.csv。
    标的下载失败只打印一句跳过、不中断，另外两类照样落地；9141.HK 的日度 NAV 不在
    这里，归 download_nav.py 管。
    """
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

    # 下载完，打一份汇总
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
