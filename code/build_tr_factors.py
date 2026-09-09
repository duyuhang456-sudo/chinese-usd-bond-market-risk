"""
复权（总收益）+ 人民币双口径 正式因子表（阶段二 VaR 输入 · 导师反馈落地）

背景：9141.HK 为**派息型** ETF（官方口径：每季度考虑派息；样本窗 2021-08~2026-09
内共 20 次除息，ex-date 全在 1/4/7/10 月上旬，HKD 0.11→0.13/单位，两柜台同额、以
HKD 派发）。直接用**除权 NAV** 算日收益会把「分红除息跳空」计成市场波动 → 系统性
高估 σ，进而抬高 VaR 基准（导师反馈 ①）。本脚本在除权因子表基础上加回每期派息：

  主口径 USD 总收益  etf_ret_tr_pct(t) = log[ (nav_{t-1}·e^{r_t} + D_t) / nav_{t-1} ]
                                        （D_t>0 仅在除息日；D_t = div_usd_9141）
  次口径人民币总收益 etf_ret_rmb_pct(t) = etf_ret_tr_pct + fx_ret_pct   （对数相加，
                                        ≈ USD 资产按 CNY/USD 折算给人民币投资者）
  因子剥离（lag=0 同日）在复权收益上重做 → etf_rate_attrib_tr / etf_spread_proxy_tr

输出为阶段二 VaR 的**正式输入** `factors/factor_table_nav_tr.csv`（除权口径
factor_table_nav.csv 保留作对照/审计底表）。

用法： ./.venv/bin/python code/build_tr_factors.py
输出： factors/factor_table_nav_tr.csv、results/baseline_var_tr.csv、figures/tr_return_series.png
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

RAW = REPO / "raw_data"
CLEAN = REPO / "clean_data"
FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
for d in (FACT, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

Z95, Z99 = 1.6449, 2.3263


def normal_var(sigma_day: float) -> tuple[float, float]:
    return Z95 * sigma_day, Z99 * sigma_day


def main() -> None:
    # --- 输入 ---------------------------------------------------------------
    F = pd.read_csv(FACT / "factor_table_nav.csv", parse_dates=["date"]).set_index("date")
    nav = pd.read_csv(CLEAN / "nav_9141HK_clean.csv", parse_dates=["date"]).set_index("date")["nav_usd"]
    div = pd.read_csv(RAW / "dividends_9141HK.csv", parse_dates=["ex_date"])
    # 只保留落在主日历样本窗内的派息（窗外派息跳空不在本因子表内，勿映射到窗口首日）
    div = div[(div["ex_date"] >= F.index[0]) & (div["ex_date"] <= F.index[-1])].copy()

    # --- 除息日映射到主日历（主日历=美债交易日；除息跳空落在 ≥ ex_date 的首个交易日）
    idx = F.index
    pos = idx.searchsorted(div["ex_date"].values, side="left")   # np.datetime64
    valid = pos < len(idx)
    ex_at = pd.Series(idx[pos[valid]], index=div.index[valid], name="map_date")
    map_df = div.loc[valid].copy()
    map_df["map_date"] = ex_at.values

    nav_prev = nav.shift(1).reindex(idx)             # 除息前一交易日净值(USD/单位)
    yield_pct = pd.Series(0.0, index=idx)
    is_ex = pd.Series(False, index=idx)
    rows = []
    for _, r in map_df.iterrows():
        md = r["map_date"]
        np_prev = nav_prev.at[md]
        if not np.isnan(np_prev) and np_prev > 0:
            dy = r["div_usd_9141"] / np_prev            # 除息收益率（简单）
            yield_pct.at[md] = dy * 100.0
            is_ex.at[md] = True
            rows.append((md, r["ex_date"], r["div_usd_9141"], np_prev, dy * 100.0))

    # --- 复权（主口径 USD 总收益） -------------------------------------------
    r_simple_ex = np.expm1(F["etf_ret_pct"] / 100.0)          # 除权 NAV 日简单收益
    r_tr_simple = r_simple_ex + yield_pct / 100.0             # 除息日加回派息
    F["etf_ret_pct"] = F["etf_ret_pct"]                        # 除权（参考列，保留）
    F["etf_ret_tr_pct"] = np.log1p(r_tr_simple) * 100.0        # USD 复权总收益 %
    F["etf_ret_rmb_pct"] = F["etf_ret_tr_pct"] + F["fx_ret_pct"]  # 人民币总收益 %（对数相加）
    F["is_exdate"] = is_ex
    F["div_yield_pct"] = yield_pct

    # --- 因子剥离重做（TR 口径，lag=0 同日） ----------------------------------
    y = F["etf_ret_tr_pct"]
    X = sm.add_constant(pd.DataFrame({"d5y": F["d5y_bp"], "d10y": F["d10y_bp"]}))
    yx = y.dropna().loc[X.dropna().index]
    m = sm.OLS(yx, X.loc[yx.index]).fit()
    rate_attr = pd.Series(m.predict(X), index=X.index).reindex(F.index)
    F["etf_rate_attrib_tr_pct"] = rate_attr
    F["etf_spread_proxy_tr_pct"] = y - rate_attr
    d5, d10 = m.params["d5y"], m.params["d10y"]
    dur = -(d5 + d10) * 100.0

    # --- 除息日核对（校验：映射日期当天除权收益 ≈ −派息 + 市场，加回后恢复正常）----
    print("=" * 78)
    print("除息日复权核对（20 期样本窗内；ex-date 映射到主日历）")
    print(f"  {'ex-date':10s} {'map_date':10s} {'div$':>6s} {'NAV_prev':>9s} "
          f"{'dy%':>7s} {'除权ret%':>9s} {'复权ret%':>9s}")
    for (md, ed, d_, np_, dyp) in rows:
        er = float(F.at[md, "etf_ret_pct"])
        tr = float(F.at[md, "etf_ret_tr_pct"])
        print(f"  {str(ed.date()):10s} {str(md.date()):10s} {d_:6.4f} {np_:9.4f} {dyp:7.3f} "
              f"{er:9.3f} {tr:9.3f}")

    # --- 统计与基线（三口径对照）----------------------------------------------
    print("=" * 78)
    stats_rows = []
    for name, s in [("除权 USD(参考)", F["etf_ret_pct"]),
                    ("复权 USD(主口径)", F["etf_ret_tr_pct"]),
                    ("人民币总收益(次口径)", F["etf_ret_rmb_pct"])]:
        s = s.dropna()
        sd = s.std()
        sa = sd * np.sqrt(252.0)
        v95, v99 = normal_var(sd)
        stats_rows.append({
            "口径": name, "n": int(s.size), "零收益%": float((s == 0).mean() * 100.0),
            "日均%": float(s.mean()), "σ_day%": sd, "σ_ann%": sa,
            "VaR95_μ0%": v95, "VaR99_μ0%": v99,
        })
        print(f"  {name:<12s} n={s.size:5d} 零%={float((s==0).mean()*100):5.1f} "
              f"σ_ann={sa:5.2f}%  1日95%VaR={v95:.4f}%  99%VaR={v99:.4f}%")
    print(f"  TR 复权剥离：β5={d5:+.4f} β10={d10:+.4f}  R²={m.rsquared:.3f}  "
          f"等效久期≈{dur:.2f}y  n={int(m.nobs)}")

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(RES / "baseline_var_tr.csv", index=False, encoding="utf-8-sig")
    print(f"\n[已写] results/baseline_var_tr.csv\n[已写] 上图核对")

    # --- 输出正式表 ----------------------------------------------------------
    cols = ["d5y_bp", "d10y_bp", "fx_ret_pct", "oas_bp", "doas_bp",
            "etf_ret_pct", "etf_ret_tr_pct", "etf_ret_rmb_pct",
            "etf_rate_attrib_tr_pct", "etf_spread_proxy_tr_pct",
            "is_exdate", "div_yield_pct"]
    out = F[cols].round(4).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FACT / "factor_table_nav_tr.csv", index=False, encoding="utf-8-sig")
    print(f"[已写] factors/factor_table_nav_tr.csv  rows={len(out)}  "
          f"除息日命中={int(is_ex.sum())}/20")

    # --- 图：三口径累计收益对照 -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(F.index, F["etf_ret_pct"].cumsum(), lw=1.0, color="grey",
            label="除权 NAV（含除息跳空，参考）")
    ax.plot(F.index, F["etf_ret_tr_pct"].cumsum(), lw=1.2, color="tab:blue",
            label="复权 USD 总收益（主口径）")
    ax.plot(F.index, F["etf_ret_rmb_pct"].cumsum(), lw=1.0, color="tab:red",
            label="人民币总收益（次口径，缺 CNY 尾段）")
    ax.scatter(F.index[is_ex], F.loc[is_ex, "etf_ret_tr_pct"].cumsum(),
               marker="v", s=26, color="tab:blue", zorder=5,
               label=f"除息日（{int(is_ex.sum())} 期，已加回）")
    ax.set_title("9141.HK 组合收益三口径累计（除权 vs 复权 USD 主口径 vs 人民币次口径）")
    ax.set_ylabel("累计对数收益 (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "tr_return_series.png", dpi=150)
    print("[已写] figures/tr_return_series.png")


if __name__ == "__main__":
    main()
