"""三大风险因子正式表（NAV 口径 · 阶段二及后续任务的默认输入）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、fred_DEXCHUS_clean.csv、
      fred_BAMLEMIBHGCRPIOAS_clean.csv、nav_9141HK_clean.csv、benchmark_9141HK_clean.csv
产出：factors/factor_table_nav.csv、figures/spread_factor_vs_oas_nav.png
口径：9141.HK 市价报价陈旧（~63% 零收益日），日度 VaR 会被系统性低估（~24%，对照实验见
      docs/staleness_remedy.md），故组合收益改用官方 NAV 日度（零收益仅 ~6%）。与市价口径
      factor_table.csv 有三处差异：收益序列换成每单位资产净值(USD) 的对数收益；时序对齐用
      lag=0 同日剥离（NAV 与美股同日估值，市价口径才用 lag=1），NAV 同日口径利率剥离
      R²=0.668、真实久期≈3.7y，市价 lag1 仅 0.23/1.74y；利差代理对同日 ΔOAS。列结构与
      factor_table.csv 一致：date、d5y_bp、d10y_bp、fx_ret_pct、etf_ret_pct、
      etf_rate_attrib_pct、etf_spread_proxy_pct、oas_bp、doas_bp。
边界：OAS 是 EM 级非紧基准，ρ 与命中率只作参考，且该列 2023-09 之前为空。
用法：./.venv/bin/python code/build_factors_nav.py
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

from common import CLEAN, FACT, FIG, ensure_dirs

ensure_dirs()


def main() -> None:
    """用官方 NAV 日度收益重建因子表，作阶段二及后续任务的默认输入。

    脚本契约：
        消费：clean_data/treasury_yield_curve_clean.csv、
            clean_data/nav_9141HK_clean.csv（取 nav_usd 列）、
            clean_data/fred_DEXCHUS_clean.csv、
            clean_data/fred_BAMLEMIBHGCRPIOAS_clean.csv。
        产出：factors/factor_table_nav.csv（列与市价口径 factor_table.csv 一致：
            date、d5y_bp、d10y_bp、fx_ret_pct、etf_ret_pct、etf_rate_attrib_pct、
            etf_spread_proxy_pct、oas_bp、doas_bp）、
            figures/spread_factor_vs_oas_nav.png。
        断言/边界：无 assert，输入表缺失即抛 FileNotFoundError。ρ 与同向命中率
            同时按同日 ΔOAS(t) 与滞后 ΔOAS(t-1) 打印，作图取同日口径；OAS 自
            2023-09 起才有值，该段之前的校验样本为空。因子表以美债日历为索引，
            fx_ret_pct 在汇率缺失的尾段留 NaN。

    返回：
        None。

    备注：
        NAV 与美股同日估值，故此处用 lag=0；市价口径的 lag=1 不能用在这里。
        除权 NAV 会把分红跳空计成波动，正式 VaR 输入要用 build_tr_factors.py
        的复权总收益表，本表保留作对照与审计底表。
    """
    def c(n):
        return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = c("treasury_yield_curve_clean.csv")
    nav = c("nav_9141HK_clean.csv")
    fx = c("fred_DEXCHUS_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    F = pd.DataFrame(index=tsy.index)
    F["d5y_bp"] = tsy["5 Yr"].diff() * 100.0
    F["d10y_bp"] = tsy["10 Yr"].diff() * 100.0
    F["fx_ret_pct"] = np.log(fx["DEXCHUS"]).diff() * 100.0
    F["etf_ret_pct"] = np.log(nav["nav_usd"]).diff() * 100.0     # NAV 日收益 %
    F["oas_bp"] = oas["BAMLEMIBHGCRPIOAS"] * 100.0
    F["doas_bp"] = F["oas_bp"].diff()
    F = F.dropna(subset=["d5y_bp", "d10y_bp", "etf_ret_pct"])

    # NAV 与美股同日 → lag=0 同日剥离（勿用市价的 lag=1）
    y = F["etf_ret_pct"]
    X = sm.add_constant(pd.DataFrame({"d5y_same": F["d5y_bp"],
                                      "d10y_same": F["d10y_bp"]}))
    yx = y.dropna().loc[X.dropna().index]
    m = sm.OLS(yx, X.loc[yx.index]).fit()

    rate_attr = pd.Series(m.predict(X), index=X.index).reindex(F.index)
    F["etf_rate_attrib_pct"] = rate_attr
    F["etf_spread_proxy_pct"] = (y - rate_attr)

    d5, d10 = m.params["d5y_same"], m.params["d10y_same"]
    dur = -(d5 + d10) * 100.0
    zshare = float((y == 0).mean() * 100.0)
    print("=" * 72)
    print("OLS（NAV 同日）：etf_ret(t) = α + β5·Δy5(t) + β10·Δy10(t) + ε")
    print(f"  β5={d5:+.4f}  β10={d10:+.4f}  α={m.params['const']:+.4f}")
    print(f"  R²={m.rsquared:.3f}   等效久期(β和×-100)≈{dur:.2f} 年   "
          f"n={int(m.nobs)}   年化波动={y.std()*np.sqrt(252):.2f}%   零收益占比={zshare:.1f}%")

    # 校验：利差代理 vs 同日 ΔOAS（重叠窗 2023-09 起，只作参考）
    V = F.dropna(subset=["etf_spread_proxy_pct", "doas_bp"])
    sp = V["etf_spread_proxy_pct"]
    for lbl, do in [("同日 ΔOAS(t)", V["doas_bp"]),
                    ("滞后 ΔOAS(t-1)", V["doas_bp"].shift(1))]:
        dd = pd.concat([sp, do], axis=1).dropna()
        rho = dd.iloc[:, 0].corr(dd.iloc[:, 1])
        hit = ((dd.iloc[:, 0] < 0) & (dd.iloc[:, 1] > 0)) | \
              ((dd.iloc[:, 0] > 0) & (dd.iloc[:, 1] < 0))
        print(f"  利差代理 vs {lbl}: ρ={rho:+.3f}  同向命中={hit.mean()*100:.1f}%  (n={len(dd)})")
    do_used = V["doas_bp"]   # NAV 同日 → 用同日 ΔOAS 作图

    # 输出（列结构与市价 factor_table.csv 一致）
    cols = ["d5y_bp", "d10y_bp", "fx_ret_pct", "etf_ret_pct",
            "etf_rate_attrib_pct", "etf_spread_proxy_pct", "oas_bp", "doas_bp"]
    out = F[cols].round(4).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FACT / "factor_table_nav.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已写] factors/factor_table_nav.csv  rows={len(out)}")

    # 图：NAV 剥离利差代理 vs FRED OAS
    V2 = pd.concat([sp, V["oas_bp"], do_used], axis=1).dropna()
    V2.columns = ["sp", "oas_bp", "do"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1b = ax1.twinx()
    l1, = ax1.plot(V2.index, V2["sp"].cumsum(), color="tab:green", lw=1.2,
                   label="利差代理累计（NAV残差, %）")
    l2, = ax1b.plot(V2.index, V2["oas_bp"], color="tab:red", lw=1.2,
                    label="FRED EM IG OAS (bp)")
    ax1.set_ylabel("利差代理累计 (%)", color="tab:green")
    ax1b.set_ylabel("OAS (bp)", color="tab:red")
    ax1.set_title("官方NAV(9141.HK) 剥离利差代理 vs FRED EM IG OAS  |  同日口径 R²=%.2f，"
                  "真实久期≈%.1fy" % (m.rsquared, dur))
    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.grid(alpha=.3)
    ax2.scatter(V2["do"], V2["sp"], s=6, alpha=.35)
    ax2.set_xlabel("ΔOAS (bp, 同日)")
    ax2.set_ylabel("利差代理日值 (%)")
    ax2.axhline(0, color="grey", lw=.6); ax2.axvline(0, color="grey", lw=.6)
    ax2.set_title("散点：利差代理(价格,%) vs ΔOAS(bp)——NAV 为同日口径")
    fig.tight_layout()
    fig.savefig(FIG / "spread_factor_vs_oas_nav.png", dpi=150)
    print("[已写] figures/spread_factor_vs_oas_nav.png")


if __name__ == "__main__":
    main()
