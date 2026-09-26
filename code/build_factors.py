"""三大核心风险因子构建 + 信用利差剥离（阶段一 9/10，含 9141.HK vs 3141.HK 柜台对照）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、fred_DEXCHUS_clean.csv、
      fred_BAMLEMIBHGCRPIOAS_clean.csv（原始单位 %，代码内 ×100 转 bp）、
      benchmark_9141HK_clean.csv、benchmark_3141HK_clean.csv
产出：factors/factor_table.csv（9141.HK 正式标的）、factors/factor_table_3141HK.csv（对照）、
      factors/benchmark_comparison.csv；figures/spread_factor_vs_oas.png 及 _3141HK 版
口径：利率因子 Δy5/Δy10 是美债 5Y/10Y 收益率的日变化（bp），汇率因子是 CNY/USD 的对数
      涨跌幅（%）。利差剥离走 OLS：etf_ret(t) = α + β5·Δy5(t−1) + β10·Δy10(t−1) + ε，
      利率贡献取拟合值，利差代理取残差 ε。港股比美股先收盘，所以因子要滞后一天；同期口径
      的 R² 接近 0，不能用。等效久期 = −(β5+β10)×100。OAS 是 EM 级非紧基准，ρ 和同向
      命中率只当参考。
边界：OAS 列 2023-09 之前是空的，ρ 和命中率只覆盖那之后，两个柜台的 n 也可能不一样。
      本脚本只做对照，不拿它改写正式因子表。真实报价日不到两天时，zero_pct 和 avg_update_d
      返回 nan。
用法：./.venv/bin/python code/build_factors.py
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


def staleness_metrics(adj: pd.Series, flag: pd.Series) -> dict:
    """在真实报价日上量一量柜台报价有多陈旧。

    adj 是复权收盘价，索引是交易日；flag 是同一索引上的填充标记，0 表示当天真有行情，
    >0 是休市填充日。返回 dict 四个键：n_real 是参与统计的真实报价日天数，zero_pct 是
    相邻两个真实报价日收盘价没变的比例（%），max_flat 是最长连续不变的天数（含两端的
    报价日），avg_update_d 是平均隔几个交易日才更新一次。真实报价日不到两天，zero_pct
    和 avg_update_d 给 nan，n_real 给 0、max_flat 给 0。

    填充日要先剔掉，不然「港股休市被填充」会被当成报价陈旧。价格变没变是拿 != 0.0
    严格比的，不留容差。
    """
    real = flag.to_numpy() == 0
    p = adj.to_numpy()[real]
    chg = np.diff(p) != 0.0                      # 相邻两个真实报价日之间，价格到底动没动
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
        "max_flat": int(longest + 1),            # 最长连续多少天不动，两端的报价日也算进去
        "avg_update_d": round((len(chg) + 1) / n_chg, 2),
    }


def run(tag: str, etf_clean: str, out_table: str, out_fig: str,
        tsy: pd.DataFrame, fx: pd.DataFrame, oas: pd.DataFrame) -> dict:
    """对单个柜台跑一整套：建因子表、剥利差、做校验、出图。

    tag 是柜台名（比如 "9141.HK"），终端打印和对照表列名都用它。etf_clean 是 clean_data/
    下的 ETF 清洗文件名，要有 date 和 Adj Close，flag 列可选，没有就全按真实报价日算。
    out_table 是因子表文件名（写到 factors/ 下），out_fig 是图文件名（写到 figures/ 下）。
    tsy 是美债收益率表，含 "5 Yr"、"10 Yr" 列，索引是交易日；fx 是汇率表，含 DEXCHUS 列；
    oas 是 FRED EM IG OAS 表，含 BAMLEMIBHGCRPIOAS 列（原始单位是 %，代码里 ×100 转 bp）。

    返回 dict，装对照表要用的 12 项：tag、zero_pct、max_flat、avg_update_d、vol_ann_pct、
    r2_full、dur_full、n_active、r2_active、dur_active、rho_lag、hit_lag_pct。其中
    rho_lag 是利差代理和滞后 ΔOAS(t-1) 的相关系数，hit_lag_pct 是两者符号相反的占比。

    港股比美股先收盘，所以利率因子取滞后一天的 Δy(t-1)，同期口径解释力接近 0、不能用。
    等效久期由 −(β5+β10)×100 得到；全样本那个久期被零收益日稀释过，只看变动日回归出来的
    久期更接近真实水平。OAS 是 EM 级非紧基准，ρ 和命中率只当参考。
    """
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

    # 港股比美股先收盘，今天的收益是被前一个美股交易日的 Δy 推着走的，所以要滞后一天；同期口径解释力接近 0
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

    # 只在有变动的日子上再回归一次：把零收益日剔掉，久期更接近真实值
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

    # 校验：利差代理对 FRED OAS。两地收盘有时差，所以对的是滞后一天的 ΔOAS(t−1)
    V = F.dropna(subset=["etf_spread_proxy_pct", "doas_bp"])
    sp, do = V["etf_spread_proxy_pct"], V["doas_bp"].shift(1)
    dd = pd.concat([sp, do], axis=1).dropna()
    rho = dd.iloc[:, 0].corr(dd.iloc[:, 1])
    hit = ((dd.iloc[:, 0] < 0) & (dd.iloc[:, 1] > 0)) | ((dd.iloc[:, 0] > 0) & (dd.iloc[:, 1] < 0))
    hit_rate = float(hit.mean() * 100.0)
    print(f"  利差代理 vs 滞后ΔOAS(t-1):  ρ={rho:+.3f}  同向命中={hit_rate:.1f}%  (n={len(dd)})")

    # 陈旧度，只在真实报价日上算
    sm_ = staleness_metrics(adj, flag)
    print(f"  报价陈旧: 真实报价日 {sm_['n_real']} 天 / 相邻报价不变 {sm_['zero_pct']:.1f}% / "
          f"最长连续不变 {sm_['max_flat']} 天 / 平均 {sm_['avg_update_d']} 天才更新一次")

    # 写因子表，只留下游要用的列
    cols = ["d5y_bp", "d10y_bp", "fx_ret_pct", "etf_ret_pct",
            "etf_rate_attrib_pct", "etf_spread_proxy_pct", "oas_bp", "doas_bp"]
    out = F[cols].round(4).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(FACT / out_table, index=False, encoding="utf-8-sig")
    print(f"  [已写] factors/{out_table}  rows={len(out)}")

    # 图：上格是利差代理累计对 OAS 水平（两个纵轴），下格是 ΔOAS 散点
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
    """9141.HK 和 3141.HK 两个柜台跑同一套方法，把对照结论打出来也写出去。

    读 clean_data 下的 treasury_yield_curve_clean.csv、fred_DEXCHUS_clean.csv、
    fred_BAMLEMIBHGCRPIOAS_clean.csv、benchmark_9141HK_clean.csv、benchmark_3141HK_clean.csv。

    写 factors/factor_table.csv、factors/factor_table_3141HK.csv、
    factors/benchmark_comparison.csv（列是 metric/unit/9141.HK/3141.HK）、
    figures/spread_factor_vs_oas.png、figures/spread_factor_vs_oas_3141HK.png，
    外加终端上的对照表。

    没有 assert，但任一输入表缺了，read_csv 直接抛 FileNotFoundError。OAS 列在 2023-09
    之前是空的，rho_lag 和 hit_lag_pct 只覆盖那之后，两个柜台的 n 也可能不一样。本脚本只做
    对照，不拿它改写正式因子表。
    """
    def c(n):
        return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")

    tsy = c("treasury_yield_curve_clean.csv")
    fx = c("fred_DEXCHUS_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")

    # 9141.HK 是正式标的，文件名照旧；3141.HK 只做对照，后面挂个柜台后缀
    res = [
        run("9141.HK", "benchmark_9141HK_clean.csv",
            "factor_table.csv", "spread_factor_vs_oas.png", tsy, fx, oas),
        run("3141.HK", "benchmark_3141HK_clean.csv",
            "factor_table_3141HK.csv", "spread_factor_vs_oas_3141HK.png", tsy, fx, oas),
    ]

    # ---- 两柜台的对照表：左边指标名，右边两个柜台的数 ----
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
    cmp = pd.DataFrame({r["tag"]: r for r in res}).T   # 转过来：行是柜台，列是指标
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
