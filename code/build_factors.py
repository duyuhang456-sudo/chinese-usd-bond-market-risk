"""
三大核心风险因子构建 + 信用利差剥离（第 1 阶段 · 9/10，含 9141 vs 3141 对照）

利率因子 : Δy5、Δy10 —— 5Y/10Y 美债收益率日变化（bp）
汇率因子 : CNY/USD 日对数涨跌幅（%）
组合收益 : 中国华夏亚洲美元投资级债 ETF，两个柜台分别试跑：
             9141.HK（USD 柜台，正式标的）——报价陈旧（~63% 零收益日）
             3141.HK（HKD 柜台，对照）——报价较活，但收益掺入 HKD/USD 联系汇率噪声
利差剥离 : 回归 组合收益 = α + β5·Δy5(t-1) + β10·Δy10(t-1) + ε（港股先收盘 → 滞后 1）
            利率贡献 = 拟合值；利差代理 = 残差 ε
校验      : 利差代理与 FRED EM IG OAS 日变化（重叠窗 2023-09 起，滞后口径）的相关/同向率。
对比      : 同一套方法跑两个柜台 → factors/benchmark_comparison.csv + 终端对照表，
            由“回归解释力/久期/利差校验/报价活性”综合决定阶段二用哪个。

输出：
  factors/factor_table.csv                 9141.HK 因子表（正式标的，口径不变）
  factors/factor_table_3141HK.csv          3141.HK 因子表（对照）
  factors/benchmark_comparison.csv         两柜台指标对照表
  figures/spread_factor_vs_oas.png         9141.HK 利差代理 vs FRED OAS 图
  figures/spread_factor_vs_oas_3141HK.png  3141.HK 同款图

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


def staleness_metrics(adj: pd.Series, flag: pd.Series) -> dict:
    """在“真实报价日”（flag==0，即该柜台确实有行情的日子）内度量报价陈旧度：
    相邻两个真实报价日收盘价相同的占比、最长连续不变、平均每几天才更新一次。
    （flag>0 的填充日不是真实报价，先剔除，避免把“港股休市被填充”误算成陈旧。）"""
    real = flag.to_numpy() == 0
    p = adj.to_numpy()[real]
    chg = np.diff(p) != 0.0                      # 相邻真实报价日之间价格是否变化
    if len(chg) == 0:
        return {"n_real": 0, "zero_pct": float("nan"), "max_flat": 0, "avg_update_d": float("nan")}
    zero_pct = float((~chg).mean() * 100.0)
    longest = cur = 0
    for c in ~chg:
        cur = cur + 1 if c else 0
        longest = max(longest, cur)
    n_chg = max(int(chg.sum()), 1)
    return {
        "n_real": int(len(chg) + 1),
        "zero_pct": zero_pct,
        "max_flat": int(longest + 1),            # 连续不变的天数（含两端的两个报价日）
        "avg_update_d": round((len(chg) + 1) / n_chg, 2),
    }


def run(tag: str, etf_clean: str, out_table: str, out_fig: str,
        tsy: pd.DataFrame, fx: pd.DataFrame, oas: pd.DataFrame) -> dict:
    """对单个柜台跑完整流程：因子表 + 利差剥离回归 + 校验 + 图，返回对比指标。"""
    etf = pd.read_csv(CLEAN / etf_clean, parse_dates=["date"]).set_index("date")
    adj, flag = etf["Adj Close"], etf.get("flag", pd.Series(0, index=etf.index))

    F = pd.DataFrame(index=tsy.index)
    F["d5y_bp"] = tsy["5 Yr"].diff() * 100.0
    F["d10y_bp"] = tsy["10 Yr"].diff() * 100.0
    F["fx_ret_pct"] = np.log(fx["DEXCHUS"]).diff() * 100.0
    F["etf_ret_pct"] = np.log(adj).diff() * 100.0
    F["oas_bp"] = oas["BAMLEMIBHGCRPIOAS"] * 100.0
    F["doas_bp"] = F["oas_bp"].diff()
    F = F.dropna(subset=["d5y_bp", "d10y_bp", "etf_ret_pct"])

    # 港股先收盘 → 今日收益由前一美股交易日 Δy 驱动；用滞后回归（同期口径≈0）
    y = F["etf_ret_pct"]
    l5 = F["d5y_bp"].shift(1)
    l10 = F["d10y_bp"].shift(1)
    X = sm.add_constant(pd.DataFrame({"d5y_lag": l5, "d10y_lag": l10}))
    yx = y.dropna().loc[X.dropna().index]
    m = sm.OLS(yx, X.loc[yx.index]).fit()

    rate_attr = pd.Series(m.predict(X), index=X.index).reindex(F.index)
    F["etf_rate_attrib_pct"] = rate_attr
    F["etf_spread_proxy_pct"] = (y - rate_attr)

    d5, d10 = m.params["d5y_lag"], m.params["d10y_lag"]
    dur_full = -(d5 + d10) * 100.0

    # 仅变动日回归（去陈旧 → 久期更接近无偏）
    dfa = pd.DataFrame({"y": y, "l5": l5, "l10": l10})
    dfa = dfa[(dfa["y"] != 0)].dropna()
    Xa = sm.add_constant(dfa[["l5", "l10"]].rename(columns={"l5": "d5y_lag", "l10": "d10y_lag"}))
    ma = sm.OLS(dfa["y"], Xa).fit()
    dur_active = -((ma.params["d5y_lag"] + ma.params["d10y_lag"]) * 100.0)

    print("=" * 70)
    print(f"[{tag}] OLS：etf_ret(t) = α + β5·Δy5(t-1) + β10·Δy10(t-1)")
    print(f"  全样本  n={int(m.nobs)}  R²={m.rsquared:.3f}  等效久期≈{dur_full:.2f} 年"
          f"  年化波动={y.std()*np.sqrt(252):.2f}%")
    print(f"  仅变动日  n={int(ma.nobs)}  R²={ma.rsquared:.3f}  等效久期≈{dur_active:.2f} 年")

    # 校验：利差代理 vs FRED OAS（滞后 ΔOAS(t-1) 口径，见 docs）
    V = F.dropna(subset=["etf_spread_proxy_pct", "doas_bp"])
    sp, do = V["etf_spread_proxy_pct"], V["doas_bp"].shift(1)
    dd = pd.concat([sp, do], axis=1).dropna()
    rho = dd.iloc[:, 0].corr(dd.iloc[:, 1])
    hit = ((dd.iloc[:, 0] < 0) & (dd.iloc[:, 1] > 0)) | ((dd.iloc[:, 0] > 0) & (dd.iloc[:, 1] < 0))
    hit_rate = float(hit.mean() * 100.0)
    print(f"  利差代理 vs 滞后ΔOAS(t-1):  ρ={rho:+.3f}  同向命中={hit_rate:.1f}%  (n={len(dd)})")

    # 陈旧度（真实报价日口径）
    sm_ = staleness_metrics(adj, flag)
    print(f"  报价陈旧: 真实报价日 {sm_['n_real']} 天 / 相邻报价不变 {sm_['zero_pct']:.1f}% / "
          f"最长连续不变 {sm_['max_flat']} 天 / 平均 {sm_['avg_update_d']} 天才更新一次")

    # 输出
    cols = ["d5y_bp", "d10y_bp", "fx_ret_pct", "etf_ret_pct",
            "etf_rate_attrib_pct", "etf_spread_proxy_pct", "oas_bp", "doas_bp"]
    out = F[cols].round(4).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FACT / out_table, index=False, encoding="utf-8-sig")
    print(f"  [已写] factors/{out_table}  rows={len(out)}")

    # 图：利差代理累计 vs OAS 水平（双轴）+ ΔOAS 散点
    V2 = pd.concat([sp, V["oas_bp"], do], axis=1).dropna()
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
    ax1.set_title(f"{tag} 剥离利差代理 vs FRED EM IG OAS  |  校验 ρ={rho:+.2f}，"
                  f"同向命中 {hit_rate:.0f}%（2023-09+，滞后 ΔOAS(t-1)）")
    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.grid(alpha=.3)
    ax2.scatter(V2["do"], V2["sp"], s=6, alpha=.35)
    ax2.set_xlabel("ΔOAS (bp, 滞后一日)")
    ax2.set_ylabel("利差代理日值 (%)")
    ax2.axhline(0, color="grey", lw=.6); ax2.axvline(0, color="grey", lw=.6)
    ax2.set_title("散点：利差代理(价格,%) vs ΔOAS(bp) —— 预期负相关")
    fig.tight_layout()
    fig.savefig(FIG / out_fig, dpi=150)
    print(f"  [已写] figures/{out_fig}")

    return {"tag": tag,
            "zero_pct": sm_["zero_pct"], "max_flat": sm_["max_flat"],
            "avg_update_d": sm_["avg_update_d"],
            "vol_ann_pct": round(float(y.std() * np.sqrt(252)), 2),
            "r2_full": round(float(m.rsquared), 3), "dur_full": round(dur_full, 2),
            "n_active": int(ma.nobs), "r2_active": round(float(ma.rsquared), 3),
            "dur_active": round(dur_active, 2),
            "rho_lag": round(float(rho), 3), "hit_lag_pct": round(hit_rate, 1)}


def main() -> None:
    def c(n):
        return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = c("treasury_yield_curve_clean.csv")
    fx = c("fred_DEXCHUS_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    # 9141.HK（正式标的）→ 保留既有文件名口径；3141.HK（对照）→ 带柜台后缀
    res = [
        run("9141.HK", "benchmark_9141HK_clean.csv",
            "factor_table.csv", "spread_factor_vs_oas.png", tsy, fx, oas),
        run("3141.HK", "benchmark_3141HK_clean.csv",
            "factor_table_3141HK.csv", "spread_factor_vs_oas_3141HK.png", tsy, fx, oas),
    ]

    # ---- 对照表 ----
    rows = [
        ("报价日相邻不变占比 %", "zero_pct", "%", "越小越活（报价陈旧度核心）"),
        ("最长连续不变 天", "max_flat", "天", "越大越陈旧"),
        ("平均更新间隔 交易日/次", "avg_update_d", "天", "越小越活"),
        ("组合年化波动(全样本) %", "vol_ann_pct", "%", "含零收益日，越陈旧越被压低"),
        ("利差剥离回归 R²(全样本)", "r2_full", "", "越大=利率解释力越强"),
        ("等效久期(全样本) 年", "dur_full", "年", "被陈旧稀释，应明显低于仅变动日"),
        ("仅变动日回归 R²", "r2_active", "", "去陈旧后的解释力"),
        ("仅变动日样本 n", "n_active", "", ""),
        ("等效久期(仅变动日) 年", "dur_active", "年", "更接近真实久期"),
        ("利差代理 vs 滞后ΔOAS 的 ρ", "rho_lag", "", "应<0，|ρ| 越大越接近纯利差"),
        ("利差代理 vs 滞后ΔOAS 命中 %", "hit_lag_pct", "%", ">50%=符号正确，越大越好"),
    ]
    cmp = pd.DataFrame({r["tag"]: r for r in res}).T   # index=tag, columns=metric键
    print("\n" + "=" * 78)
    print("9141.HK(USD柜台,正式) vs 3141.HK(HKD柜台,对照) —— 同法对照")
    print("=" * 78)
    head = f"  {'指标':<28}{'9141.HK':>11}{'3141.HK':>11}   注"
    print(head)
    out_rows = []
    for label, col, unit, note in rows:
        v0, v1 = cmp.loc["9141.HK", col], cmp.loc["3141.HK", col]
        u = f"{unit}" if unit else ""
        print(f"  {label:<28}{str(v0)+u:>11}{str(v1)+u:>11}   {note}")
        out_rows.append({"metric": label, "unit": unit,
                         "9141.HK": v0, "3141.HK": v1})
    pd.DataFrame(out_rows).to_csv(FACT / "benchmark_comparison.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[已写] factors/benchmark_comparison.csv")


if __name__ == "__main__":
    main()
