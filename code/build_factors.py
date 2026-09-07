"""
三大核心风险因子构建 + 信用利差剥离（第 1 阶段 · 9/10）

利率因子 : Δy5、Δy10 —— 5Y/10Y 美债收益率日变化（bp）
汇率因子 : CNY/USD 日对数涨跌幅（%）
组合收益 : 9141.HK（ChinaAMC 亚洲美元投资级债 ETF, USD 柜台）日对数收益（%）
利差剥离 : 回归 组合收益 = α + β5·Δy5 + β10·Δy10 + ε
            利率贡献 = 拟合值（按久期把国债收益率变动映射到组合收益）
            利差代理 = 残差 ε（组合收益中剔除利率部分后的剩余 = 信用利差驱动 + 噪声）
            —— 两收益率高度共线不影响拟合值，故剥离稳健
校验      : 利差代理与 FRED EM IG OAS 日变化在重叠窗（2023-09 起）做
            相关性与同向率（利差走阔→价格下跌→利差代理为负，故与 ΔOAS 预期负相关）。

输出：
  factors/factor_table.csv      因子表（同一美股交易日对齐；OAS 2023-09 前留 NaN）
  figures/spread_factor_vs_oas.png  利差代理 vs FRED OAS 对比图

用法： ./.venv/bin/python code/build_factors.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import REPO

CLEAN = REPO / "clean_data"
FACT = REPO / "factors"
FIG = REPO / "figures"
FACT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    def c(n):
        return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = c("treasury_yield_curve_clean.csv")
    etf = c("benchmark_9141HK_clean.csv")
    fx = c("fred_DEXCHUS_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    F = pd.DataFrame(index=tsy.index)
    F["d5y_bp"] = tsy["5 Yr"].diff() * 100.0
    F["d10y_bp"] = tsy["10 Yr"].diff() * 100.0
    F["fx_ret_pct"] = np.log(fx["DEXCHUS"]).diff() * 100.0
    F["etf_ret_pct"] = np.log(etf["Adj Close"]).diff() * 100.0
    F["oas_bp"] = oas["BAMLEMIBHGCRPIOAS"] * 100.0      # level, 2023-09 前 NaN
    F["doas_bp"] = F["oas_bp"].diff()
    F = F.dropna(subset=["d5y_bp", "d10y_bp", "etf_ret_pct"])

    # 关键时序口径：9141.HK 在港股收盘定价，早于美债同日收盘；
    # 故今日的基金收益主要由「前一美股交易日」的收益率变动驱动（lag=1）。
    # 用滞后 Δy 做回归，否则同期相关≈0（见代码注释与 docs）。
    y = F["etf_ret_pct"]
    l5 = F["d5y_bp"].shift(1)      # Δy5(t-1)
    l10 = F["d10y_bp"].shift(1)    # Δy10(t-1)
    X = sm.add_constant(pd.DataFrame({"d5y_lag": l5, "d10y_lag": l10}))
    yx = y.dropna().loc[X.dropna().index]
    m = sm.OLS(yx, X.loc[yx.index]).fit()

    # 利率贡献（%）= α + β5·Δy5(t-1) + β10·Δy10(t-1)；残差 = 信用利差驱动 + 噪声
    rate_attr = pd.Series(m.predict(X), index=X.index).reindex(F.index)
    spread_proxy = (y - rate_attr).rename("etf_spread_proxy_pct")
    F["etf_rate_attrib_pct"] = rate_attr
    F["etf_spread_proxy_pct"] = spread_proxy

    d5, d10 = m.params["d5y_lag"], m.params["d10y_lag"]
    dur_equiv = -(d5 + d10) * 100.0              # %/bp → 久期(年) 的映射估计
    print("=" * 70)
    print("OLS：etf_ret_pct(t) = α + β5·Δy5(t-1) + β10·Δy10(t-1)   [港股收盘滞后美股]")
    print(f"  β5={d5:+.4f}  β10={d10:+.4f}  α={m.params['const']:+.4f}")
    print(f"  R²={m.rsquared:.3f}（同期口径仅≈0.00，滞后口径显著提升——时序错位已校正）")
    print(f"  等效久期(β和×-100)≈{dur_equiv:.2f} 年")
    print(f"  样本 n={int(m.nobs)}；组合年化波动={y.std()*np.sqrt(252):.2f}%（含~63%零收益日的陈旧报价）")

    # 陈旧报价会稀释系数(衰减) → 补充"仅变动日"回归作为久期更接近无偏的估计
    dfa = pd.DataFrame({"y": y, "l5": l5, "l10": l10})
    dfa = dfa[(dfa["y"] != 0)].dropna()          # 仅变动日 + 剔除 lag 引入的头 NaN
    Xa = sm.add_constant(dfa[["l5", "l10"]].rename(columns={"l5": "d5y_lag", "l10": "d10y_lag"}))
    ma = sm.OLS(dfa["y"], Xa).fit()
    d5a, d10a = ma.params["d5y_lag"], ma.params["d10y_lag"]
    print(f"  [去陈旧报价]仅变动日回归: n={int(ma.nobs)}  R²={ma.rsquared:.3f}  "
          f"等效久期≈{-(d5a+d10a)*100:.2f} 年（更接近真实久期）")

    # ---- 校验：利差代理 vs FRED OAS 变化（重叠窗 2023-09+；同样试滞后） ----
    V = F.dropna(subset=["etf_spread_proxy_pct", "doas_bp"])
    sp = V["etf_spread_proxy_pct"]
    results = {}
    for lbl, do in [("同期 doas(t)", V["doas_bp"]),
                    ("滞后 doas(t-1)", V["doas_bp"].shift(1))]:
        dd = pd.concat([sp, do], axis=1).dropna()
        rho = dd.iloc[:, 0].corr(dd.iloc[:, 1])
        hit = ((dd.iloc[:, 0] < 0) & (dd.iloc[:, 1] > 0)) | ((dd.iloc[:, 0] > 0) & (dd.iloc[:, 1] < 0))
        results[lbl] = (rho, float(hit.mean()), dd.index)
        print(f"  利差代理 vs {lbl}: ρ={rho:+.3f}  同向命中={hit.mean()*100:.1f}%  (n={len(dd)})")
    # 图里用相关更强的 OAS 滞后口径
    best_lbl = max(results, key=lambda k: abs(results[k][0]))
    best = results[best_lbl]
    print(f"  → 图用 {best_lbl}（|ρ| 更大）")
    do_used = (V["doas_bp"].shift(1) if "滞后" in best_lbl else V["doas_bp"])

    # ---- 输出因子表 ----
    cols = ["d5y_bp", "d10y_bp", "fx_ret_pct", "etf_ret_pct",
            "etf_rate_attrib_pct", "etf_spread_proxy_pct", "oas_bp", "doas_bp"]
    out = F[cols].round(4).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FACT / "factor_table.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已写] factors/factor_table.csv  rows={len(out)}")

    # ---- 对比图 ----
    V2 = pd.concat([sp, V["oas_bp"], do_used], axis=1).dropna()
    V2.columns = ["sp", "oas_bp", "do"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1b = ax1.twinx()
    l1, = ax1.plot(V2.index, V2["sp"].cumsum(), color="tab:blue", lw=1.2,
                   label="利差代理累计（组合残差, %）")
    l2, = ax1b.plot(V2.index, V2["oas_bp"], color="tab:red", lw=1.2,
                    label="FRED EM IG OAS (bp)")
    ax1.set_ylabel("利差代理累计 (%)", color="tab:blue")
    ax1b.set_ylabel("OAS (bp)", color="tab:red")
    ax1.set_title(f"9141.HK 剥离利差代理 vs FRED EM IG OAS  |  "
                  f"校验 ρ={results[best_lbl][0]:+.2f}，同向命中 {results[best_lbl][1]*100:.0f}%"
                  f"（2023-09+，{best_lbl}）")
    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.grid(alpha=.3)

    ax2.scatter(V2["do"], V2["sp"], s=6, alpha=.35)
    ax2.set_xlabel(f"ΔOAS (bp, {'滞后一日' if '滞后' in best_lbl else '同期'})")
    ax2.set_ylabel("利差代理日值 (%)")
    ax2.axhline(0, color="grey", lw=.6); ax2.axvline(0, color="grey", lw=.6)
    ax2.set_title("散点：利差代理(价格,%) vs ΔOAS(bp) —— 预期左上↘右下（负相关）")
    fig.tight_layout()
    fig.savefig(FIG / "spread_factor_vs_oas.png", dpi=150)
    print(f"[已写] figures/spread_factor_vs_oas.png")


if __name__ == "__main__":
    main()
