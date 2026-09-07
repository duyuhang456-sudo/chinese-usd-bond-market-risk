"""
异常值识别（第 1 阶段 · 9/9 数据清洗② 上午部分）

思路（对应计划：滚动/分段 3σ）：
  对每条「日变化」序列做多信号 3σ 识别，任一命中即记为疑似异常候选：
    - global z      ：相对全样本均值/标准差
    - roll60 z      ：相对最近 60 个交易日（trailing）均值/标准差（regime-relative）
    - roll20 z      ：相对最近 20 个交易日（trailing）——捕捉相对近期平静期的突刺
    - pre60 z       ：相对「前 60 日」基线（std 只用到 t-1，不含当日）——
                      冲击初期的波动率还没有抬升，用它能抓住事件首日
  另有绝对下限护栏，避免收益率类序列在波动极低期被微噪声过度触发。

  序列口径（单位统一）：
    美债 5Y/10Y   ：日变化 ×100 = bp
    9141.HK       ：Adj Close 对数收益 %（对齐到美股交易日，港股休市填充日≈0% 不算异常）
    DEXCHUS       ：对数收益 %
    EM IG OAS     ：日变化 ×100 = bp（仅 2023-09 起）

用法： ./.venv/bin/python code/outlier_detect.py
输出： clean_data/outlier_candidates.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import REPO

CLEAN = REPO / "clean_data"


def detect(series: pd.Series, name: str, kind: str,
           floors: dict[str, float]) -> pd.DataFrame:
    """对一条已对齐的日度序列做日变化 + 多信号 3σ 识别。"""
    s = pd.to_numeric(series).dropna().sort_index()
    if kind == "logret":                     # 收益类：对数收益 %
        chg = np.log(s).diff() * 100.0
    elif kind == "diffbp":                   # 收益率/利差：日变化转 bp
        chg = s.diff() * 100.0
    else:                                    # 简单日变化（原单位）
        chg = s.diff()
    chg = chg.dropna()
    if len(chg) < 40:
        return pd.DataFrame()

    mu, sd = chg.mean(), chg.std()
    z_global = (chg - mu) / sd

    z60 = (chg - chg.rolling(60, min_periods=20).mean()) / chg.rolling(60, min_periods=20).std()
    z20 = (chg - chg.rolling(20, min_periods=10).mean()) / chg.rolling(20, min_periods=10).std()
    # 前 60 日基线（不含当日）
    mbase = chg.rolling(60, min_periods=20).mean().shift(1)
    sbase = chg.rolling(60, min_periods=20).std().shift(1)
    zpre = (chg - mbase) / sbase

    floor = floors.get(name, 0.0)
    hit = (z_global.abs() > 3) | (z60.abs() > 3) | (z20.abs() > 3) | (zpre.abs() > 3)
    hit = hit & (chg.abs() >= floor)         # 下限护栏
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
    C = lambda n: pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = C("treasury_yield_curve_clean.csv")
    etf = C("benchmark_9141HK_clean.csv")
    fx = C("fred_DEXCHUS_clean.csv")
    oas = C("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    floors = {   # 绝对下限护栏：低于该量级即便 z>3 也过滤（微噪声）
        "d5y_bp": 8.0, "d10y_bp": 8.0,        # 8bp 起
        "ETF_ret_pct": 0.5,                    # 0.5% 起
        "FX_ret_pct": 0.45,                    # 0.45% 起
        "dOAS_bp": 5.0,                        # 5bp 起
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
        # 背景统计（帮助判断阈值是否合理）
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
