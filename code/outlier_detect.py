"""异常值识别：先挑出「疑似」，不下结论（阶段一 Day3 · 9/9 上午 · 数据清洗②）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、benchmark_9141HK_clean.csv、
      fred_DEXCHUS_clean.csv、fred_BAMLEMIBHGCRPIOAS_clean.csv
产出：clean_data/outlier_candidates.csv
口径：对每条「日变化」序列同时跑四个 3σ 信号，中一个就记成候选：global z（拿全样本算）、
      roll60 z（最近 60 个交易日）、roll20 z（最近 20 个交易日，专抓平静期里的突刺）、
      pre60 z（拿前 60 天当基线，标准差只用到 t−1；冲击刚起来那几天波动率还没抬升，用它才
      抓得住事件第一天）。四个信号之外还有一道绝对下限护栏，免得波动极低的时候被微噪声刷屏。
      单位统一：美债 5Y/10Y 和 EM IG OAS 取日变化 ×100 换成 bp（OAS 只有 2023-09 起）；
      9141.HK 和 DEXCHUS 取对数收益 %。港股休市那几天是填充出来的，日收益 ≈0%，不会被
      记成异常。
边界：这一步只挑候选，不判真假。哪条候选真算异常，交给 adjudicate_outliers.py 逐条裁定。
用法：./.venv/bin/python code/outlier_detect.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, ensure_dirs

ensure_dirs()


def detect(series: pd.Series, name: str, kind: str,
           floors: dict[str, float]) -> pd.DataFrame:
    """把一条对齐好的日度序列变成日变化，再用四个 3σ 信号挑疑似异常。

    series 是已经对齐到主日历的日度序列（索引是日期）；name 是序列名（d5y_bp / d10y_bp /
    ETF_ret_pct / FX_ret_pct / dOAS_bp），同时也是查下限护栏的键；kind 决定日变化怎么算，
    "logret" 取对数收益 %，"diffbp" 取一阶差分 ×100 换成 bp，其余的取一阶差分、原单位；
    floors 是 {序列名: 绝对下限}，日变化绝对值不到下限的，哪怕 z>3 也不算候选。波动极低的
    时候标准差很小，一点点动静就能顶过 3 倍，这道下限就是拦这个的。

    返回候选表，列有 series / date(YYYY-MM-DD) / value / 四个 z 值 / hit（中了哪几个信号，
    用 + 连起来）。没有候选、或者序列还不到 40 个观测，返回空表。
    """
    s = pd.to_numeric(series).dropna().sort_index()
    if kind == "logret":                     # 收益类：对数收益 %
        chg = np.log(s).diff() * 100.0
    elif kind == "diffbp":                   # 收益率 / 利差：日变化换算成 bp
        chg = s.diff() * 100.0
    else:                                    # 普通日变化，原单位
        chg = s.diff()
    chg = chg.dropna()
    if len(chg) < 40:
        return pd.DataFrame()

    mu, sd = chg.mean(), chg.std()
    z_global = (chg - mu) / sd

    z60 = (chg - chg.rolling(60, min_periods=20).mean()) / chg.rolling(60, min_periods=20).std()
    z20 = (chg - chg.rolling(20, min_periods=10).mean()) / chg.rolling(20, min_periods=10).std()
    # 拿前 60 天当基线，shift(1) 把当日摘出去
    mbase = chg.rolling(60, min_periods=20).mean().shift(1)
    sbase = chg.rolling(60, min_periods=20).std().shift(1)
    zpre = (chg - mbase) / sbase

    floor = floors.get(name, 0.0)
    hit = (z_global.abs() > 3) | (z60.abs() > 3) | (z20.abs() > 3) | (zpre.abs() > 3)
    hit = hit & (chg.abs() >= floor)         # 最后过一道下限护栏：动静太小的一律不算
    cand = pd.DataFrame({
        "series": name, "value": chg[hit], "global_z": z_global[hit],
        "roll60_z": z60[hit], "roll20_z": z20[hit], "pre60_z": zpre[hit],
    })
    cand["hit"] = cand.apply(
        lambda r: "+".join(k for k, z in
                           [("global", abs(r.global_z)), ("r60", abs(r.roll60_z)),
                            ("r20", abs(r.roll20_z)), ("pre60", abs(r.pre60_z))]
                           if z > 3), axis=1)
    cand = cand.reset_index().rename(columns={"index": "date"})
    cand["date"] = cand["date"].dt.strftime("%Y-%m-%d")
    return cand


def main() -> None:
    """对五条日变化序列各跑一遍四个 3σ 信号，列出候选异常。

    消费 clean_data/ 下的 treasury、9141.HK、DEXCHUS、OAS 四张清洗表；产出
    clean_data/outlier_candidates.csv，带四个 z 值和命中信号列。这一步只挑候选、不下判定，
    哪条候选真算异常由 adjudicate_outliers.py 逐条裁定。
    """
    C = lambda n: pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = C("treasury_yield_curve_clean.csv")
    etf = C("benchmark_9141HK_clean.csv")
    fx = C("fred_DEXCHUS_clean.csv")
    oas = C("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    floors = {   # 绝对下限护栏：波动小到这份上的算微噪声，z>3 也不记候选
        "d5y_bp": 8.0, "d10y_bp": 8.0,        # 美债 5Y/10Y：日变化满 8bp 才够看
        "ETF_ret_pct": 0.5,                    # 9141.HK 日收益：满 0.5% 才够看
        "FX_ret_pct": 0.45,                    # DEXCHUS 汇率日收益：满 0.45% 才够看
        "dOAS_bp": 5.0,                        # OAS 利差：日变化满 5bp 才够看
    }

    jobs = [
        ("d5y_bp", tsy["5 Yr"], "diffbp"),
        ("d10y_bp", tsy["10 Yr"], "diffbp"),
        ("ETF_ret_pct", etf["Adj Close"], "logret"),
        ("FX_ret_pct", fx["DEXCHUS"], "logret"),
        ("dOAS_bp", oas["BAMLEMIBHGCRPIOAS"], "diffbp"),
    ]

    out: list[pd.DataFrame] = []
    print(f"{'序列':<12}{'样本':>5}  候选数  (global/r60/r20/pre60 命中计数)")
    stats: dict[str, dict] = {}
    for name, col, kind in jobs:
        df = detect(col, name, kind, floors)
        out.append(df)
        # 顺手打点背景统计，用来判断上面那组阈值定得合不合理
        s = pd.to_numeric(col).dropna()
        chg = np.log(s).diff() * 100 if kind == "logret" else s.diff() * 100
        chg = chg.dropna()
        cnt = {k: int((df[k].abs() > 3).sum()) for k in
               ["global_z", "roll60_z", "roll20_z", "pre60_z"]} if len(df) else {}
        stats[name] = {"n": int(len(chg)), "cand": int(len(df)), "cnt": cnt,
                       "std": float(chg.std()), "max": float(chg.abs().max())}
        print(f"{name:<12}{len(chg):>5}  {len(df):>4}     {cnt}   "
              f"(σ={chg.std():.2f}, max|Δ|={chg.abs().max():.1f})")

    allc = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["series", "date", "value", "global_z", "roll60_z", "roll20_z", "pre60_z", "hit"])
    allc = allc.sort_values(["series", "date"])
    allc.to_csv(CLEAN / "outlier_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已写] clean_data/outlier_candidates.csv  {len(allc)} 条候选（含多信号命中列）")


if __name__ == "__main__":
    main()
