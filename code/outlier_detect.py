"""异常值识别（阶段一 9/9 数据清洗② 上午部分）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、benchmark_9141HK_clean.csv、
      fred_DEXCHUS_clean.csv、fred_BAMLEMIBHGCRPIOAS_clean.csv
产出：clean_data/outlier_candidates.csv
口径：对每条「日变化」序列做多信号 3σ 识别，任一命中即记为候选：global z（全样本）、
      roll60 z（最近 60 交易日）、roll20 z（最近 20 交易日，抓相对平静期的突刺）、pre60 z
      （前 60 日基线，std 只用到 t−1，冲击初期波动率尚未抬升，用它能抓住事件首日）。另有
      绝对下限护栏，避免低波动期被微噪声过度触发。单位统一：美债 5Y/10Y 与 EM IG OAS 取
      日变化 ×100 = bp（OAS 仅 2023-09 起）；9141.HK 与 DEXCHUS 取对数收益 %，港股休市的
      填充日收益 ≈0%，不会被记成异常。
用法：./.venv/bin/python code/outlier_detect.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, ensure_dirs

ensure_dirs()


def detect(series: pd.Series, name: str, kind: str,
           floors: dict[str, float]) -> pd.DataFrame:
    """对一条已对齐的日度序列做日变化，再用四个 3σ 信号识别疑似异常。

    参数：
        series: 已对齐到主日历的日度序列，索引为日期。
        name: 序列名（d5y_bp / d10y_bp / ETF_ret_pct / FX_ret_pct / dOAS_bp），
            同时用作下限护栏的查表键。
        kind: 日变化的算法。"logret" 取对数收益 %，"diffbp" 取一阶差分 ×100（bp），
            其余取一阶差分原单位。
        floors: {序列名: 绝对下限}。日变化绝对值低于下限的即使 z>3 也不算候选，
            用于滤掉波动极低期的微噪声（此时标准差很小，微小变动也会 z>3）。

    返回：
        候选异常表，列为 series / date(YYYY-MM-DD) / value / 四个 z 值 /
        hit（命中的信号名，用 + 连接）。无候选或序列短于 40 个观测时返回空表。
    """
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
    """对五条日变化序列跑多信号 3σ 识别，输出候选异常清单。

    脚本契约：
        消费：clean_data/ 下的 treasury、9141.HK、DEXCHUS、OAS 四张清洗表。
        产出：clean_data/outlier_candidates.csv（含四个 z 值与命中信号列）。
        边界：本步只「识别候选」，不下判定；哪条候选真算异常由
            adjudicate_outliers.py 逐条裁定。

    返回：
        None。
    """
    C = lambda n: pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = C("treasury_yield_curve_clean.csv")
    etf = C("benchmark_9141HK_clean.csv")
    fx = C("fred_DEXCHUS_clean.csv")
    oas = C("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    floors = {   # 绝对下限护栏：低于该量级的波动属微噪声，即便 z>3 也不记候选
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
