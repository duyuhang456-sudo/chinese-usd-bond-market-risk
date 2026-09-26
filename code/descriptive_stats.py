"""9/11 描述性统计 + 质量复核（第一阶段收口 · 默认 NAV 口径）

消费：factors/factor_table_nav.csv（正式因子表，见 docs/factors.md 第六节）
产出：factors/descriptive_stats_nav.csv、figures/descriptive_hist_qq_nav.png、
      figures/portfolio_return_decomp_nav.png；质量复核结论打印至控制台后汇入《数据说明文档》
口径：分布特征含均值 / 标准差 / 偏度 / 峰度 / 正态性；组合累计收益按「总收益 = 利率贡献 +
      利差代理」拆解。图只画核心序列，画太多反而稀释信息。
用法：./.venv/bin/python code/descriptive_stats.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import CLEAN, FACT, FIG, ensure_dirs

ensure_dirs()

# (列名, 中文名, 是否收益类（收益类才年化 σ）, 量纲, 窗口说明)
SERIES = [
    ("etf_ret_pct",          "组合收益 (NAV, %)",    True,  "%/日", "2021-08 起"),
    ("etf_spread_proxy_pct", "信用利差代理 (残差, %)",
     True, "%/日", "2021-08 起"),
    ("fx_ret_pct",           "汇率因子 CNY/USD (%)", True,  "%/日", "2021-08 起"),
    ("etf_rate_attrib_pct",  "利率贡献 (NAV, %)",    True,  "%/日", "2021-08 起"),
    ("d5y_bp",               "利率因子 Δ5Y (bp)",    False, "bp/日", "2021-08 起"),
    ("d10y_bp",              "利率因子 Δ10Y (bp)",   False, "bp/日", "2021-08 起"),
    ("doas_bp",              "外部利差 ΔOAS (bp)",   False, "bp/日", "2023-09 起"),
]
# 直方图/QQ 图展示的核心序列（图太多则信息稀释）
FIG_Q = [("etf_ret_pct", "组合收益(NAV)"), ("etf_spread_proxy_pct", "利差代理"),
         ("fx_ret_pct", "汇率"), ("d10y_bp", "Δ10Y")]


def main() -> None:
    """对 NAV 口径因子表出分布特征表、直方图/QQ 图与收益拆解图，并打印质量复核。

    脚本契约：
        消费：factors/factor_table_nav.csv、clean_data/master_calendar.csv。
        产出：factors/descriptive_stats_nav.csv（每行一个因子，列为窗口、n、
            日均、日标准差、年化σ、偏度、超额峰度、JB-p、min/max、>0 与 =0 占比）、
            figures/descriptive_hist_qq_nav.png、
            figures/portfolio_return_decomp_nav.png；质量复核只打印到控制台。
        断言/边界：无 assert。模块常量 SERIES 决定统计哪些列、量纲与窗口标签，
            其中窗口标签（如 doas_bp 标为 2023-09 起）是手写字符串，不随数据实际
            起点变化；非空检查只覆盖 6 个因子列并要求全表非空，oas_bp/doas_bp 允许
            前段为 NaN；与主日历比对日期差集，有差集时打印 "!!" 并列出日期，但不中断。

    返回：
        None。
    """
    F = pd.read_csv(FACT / "factor_table_nav.csv", parse_dates=["date"]).set_index("date")
    F = F[F.index.notna()]
    print("=" * 78)
    print(f"[输入] factors/factor_table_nav.csv  rows={len(F)}  "
          f"span={F.index.min().date()} ~ {F.index.max().date()}")

    # ---- 1) 分布特征表 ----
    rows = []
    for col, zh, ann, unit, win in SERIES:
        x = F[col].dropna()
        s = float(x.std())
        rows.append({
            "因子": zh, "窗口": win, "n": int(len(x)),
            "均值(日)": float(x.mean()),
            "标准差(日)": s,
            "年化σ(%)": round(s * np.sqrt(252), 4) if ann else np.nan,
            "偏度": float(x.skew()), "峰度(超额)": float(x.kurt()),
            "JB-p": float(stats.jarque_bera(x).pvalue),
            "min": float(x.min()), "max": float(x.max()),
            ">0占比%": round(float((x > 0).mean() * 100), 1),
            "=0占比%": round(float((x == 0).mean() * 100), 1),
        })
    D = pd.DataFrame(rows).set_index("因子")
    D.to_csv(FACT / "descriptive_stats_nav.csv", encoding="utf-8-sig")
    print("\n[已写] factors/descriptive_stats_nav.csv")
    print(D.round(4).to_string())

    # ---- 2) 金融规律核对 ----
    print("\n" + "=" * 78 + "\n金融规律核对（收益类应呈尖峰厚尾、偏度多为负、均值≈0）")
    for col, zh, ann, _, _ in SERIES:
        x = F[col].dropna()
        k = float(x.kurt()); sk = float(x.skew())
        if ann:
            shape = "尖峰厚尾" if k > 0 else "≈正态/薄尾"
            skew = "负偏(左尾)" if sk < 0 else ("正偏(右尾)" if sk > 0 else "近似对称")
            print(f"  {zh:<22} 偏度={sk:+.2f} 超额峰度={k:+.2f}   {shape} / {skew}")

    # ---- 3) 图 1：直方图 + QQ 图 ----
    nq = len(FIG_Q)
    fig, axes = plt.subplots(nq, 2, figsize=(12, 2.6 * nq))
    for i, (col, zh) in enumerate(FIG_Q):
        x = F[col].dropna().values
        ax = axes[i, 0]
        ax.hist(x, bins=60, density=True, alpha=.6, color="#4C72B0",
                label=f"n={len(x)}")
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, x.mean(), x.std()), "r-", lw=1.2, label="正态拟合")
        ax.set_title(f"{zh} 直方图（偏度 {pd.Series(x).skew():+.2f} / 超额峰度 {pd.Series(x).kurt():+.2f}）")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        ax2 = axes[i, 1]
        stats.probplot(x, dist="norm", plot=ax2)
        ax2.get_lines()[1].set_color("red"); ax2.set_title(f"{zh} QQ 图")
        ax2.grid(alpha=.25)
    fig.suptitle("因子分布形态核验（NAV 口径，日度）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(FIG / "descriptive_hist_qq_nav.png", dpi=150)
    print("\n[已写] figures/descriptive_hist_qq_nav.png")

    # ---- 4) 图 2：组合累计收益拆解 ----
    cum = F["etf_ret_pct"].cumsum()
    cr = F["etf_rate_attrib_pct"].cumsum()
    cs = F["etf_spread_proxy_pct"].cumsum()
    fig2, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(F.index, cum, color="k", lw=1.4, label="NAV 累计收益（总）")
    ax.plot(F.index, cr, color="tab:blue", lw=1.1, label="其中：利率贡献（累计）")
    ax.plot(F.index, cs, color="tab:orange", lw=1.1, label="其中：信用利差代理（累计）")
    ax.axhline(0, color="grey", lw=.6)
    ax.set_title("组合累计收益拆解（NAV 口径）：总收益 ≈ 利率贡献 + 信用利差代理")
    ax.legend(); ax.grid(alpha=.3)
    ax.set_ylabel("累计 (%)")
    fig2.tight_layout()
    fig2.savefig(FIG / "portfolio_return_decomp_nav.png", dpi=150)
    print("[已写] figures/portfolio_return_decomp_nav.png")

    # ---- 5) 质量复核（供《数据说明文档》证据链，控制台即可） ----
    print("\n" + "=" * 78 + "\n质量复核")
    for c in ["d5y_bp", "d10y_bp", "fx_ret_pct", "etf_ret_pct",
              "etf_rate_attrib_pct", "etf_spread_proxy_pct"]:
        nn = int(F[c].notna().sum())
        print(f"    因子表 {c:<24} {nn}/{len(F)} 非空   {'OK' if nn == len(F) else '!!'}")
    oas_start = F["oas_bp"].dropna().index.min().date()
    print(f"    因子表 oas_bp/doas_bp      自 {oas_start} 起非空（2023-09 前 NaN 为设计）  OK")

    cal = pd.read_csv(CLEAN / "master_calendar.csv", parse_dates=["date"])
    extra = set(F.index) - set(cal["date"])
    print(f"    主日历天数={len(cal)}  因子表={len(F)}  "
          f"{'日期与主日历完全一致  OK' if not extra else ('!! 差集=' + str(extra))}")


if __name__ == "__main__":
    main()
