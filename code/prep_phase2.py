"""阶段二 Day1 建模准备：基线 VaR / 因子暴露 / 协方差校验 / 平稳-高波动分段 / 存疑日清单

消费：factors/factor_table_nav_tr.csv（正式输入，复权 USD 总收益主口径；除权表
      factor_table_nav.csv 只留作对照与审计底表）、events/risk_events_timeline.csv、
      clean_data/outlier_judgment.csv
产出：results/baseline_var.csv（无条件正态基线）、factor_exposure.csv（因子暴露 δ）、
      factor_corr.csv（因子相关矩阵）、regimes.csv（逐日滚动 60 日年化实现波动 +
      平稳/高波动标注）、doubtful_days.csv（阶段一 16 条存疑异常日）；
      figures/phase2_rollvol_regimes.png
口径：主口径 = USD 本位复权收益 etf_ret_tr_pct（δ_fx=0，VaR 上报口径）；人民币次口径见
      results/baseline_var_tr.csv 对照表（build_tr_factors.py 产出）。VaR 只在无条件正态
      下算，不含 GARCH 与历史模拟。波动分段有两个触发：慢性态为 60 日滚动年化 σ ≥ 全样本
      2/3 分位；危机窗为事件日前后各 7 个自然日内 |日收益| 触及 1.2%（4.51 × 主口径日
      σ 0.266）则整窗记高波动，由实现收益触发，不人工挑日子。前 59 个交易日凑不满 60 日窗，
      用 expanding σ 顶替。存疑日只按 verdict 含「存疑」二字筛，不改判、不合并。
用法：./.venv/bin/python code/prep_phase2.py
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

from common import CLEAN, FACT, RES, FIG, EVENTS_CSV, ensure_dirs

ensure_dirs()

Q_HIGH = 2 / 3          # 高波动 = 滚动年化波动 ≥ 全样本 2/3 分位
RV_WIN = 60             # 滚动实现波动窗口（交易日）
Z = {"95%": 1.6449, "99%": 2.3263}


def main() -> None:
    """为阶段二备齐基线 VaR、因子暴露、协方差校验、波动分段与存疑日清单。

    脚本契约：
        消费：factors/factor_table_nav_tr.csv（复权 USD 总收益主口径表，用 date、
              etf_ret_tr_pct、d5y_bp、d10y_bp、etf_spread_proxy_tr_pct 五列）、
              events/risk_events_timeline.csv（date、event_cn）、
              clean_data/outlier_judgment.csv（verdict 含「存疑」的行）。
        产出：results/baseline_var.csv（无条件正态基线，含观测数、日σ、年化σ，
              95%/99% 两行各给忽略均值与含均值的 VaR）、
              results/factor_exposure.csv（四个因子各一行：暴露δ 与含义）、
              results/factor_corr.csv（Δ5Y / Δ10Y / 利差代理的相关系数矩阵）、
              results/regimes.csv（date、rv60_ann_%、chronic_high、crisis_window、regime）、
              results/doubtful_days.csv（阶段一判为存疑的异常日）、
              figures/phase2_rollvol_regimes.png（60 日滚动年化波动 + 高波动段阴影）。
        断言/边界：无硬断言，但有几处口径边界。VaR 只在无条件正态下算，不含 GARCH 或历史模拟。
              波动分段取两种触发：慢性态为 60 日滚动年化 σ 不低于全样本 2/3 分位；危机窗为事件日
              前后各 7 个自然日内 |日收益| 触及 1.2%（约 4.5 倍主口径日σ）则整窗记高波动，
              不靠人工挑日子。前 59 个交易日凑不满 60 日窗口，用 expanding σ 顶替。
              存疑日清单只按 verdict 里是否含「存疑」二字筛，不改判、不合并。
    """
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    r = F["etf_ret_tr_pct"].dropna()          # 主口径：复权 USD 总收益（正式 VaR 标的序列）
    print(f"[输入] factor_table_nav_tr rows={len(F)}  复权 USD 收益 n={len(r)}")

    # ---- 1) 基线：无条件正态 VaR（1 日，USD 本位） ----
    mu, sd = float(r.mean()), float(r.std())
    rows = []
    for cl, z in Z.items():
        rows.append({"置信度": cl, "z": z,
                     "VaR_忽略均值(%)": round(z * sd, 4),
                     "VaR_含均值(%)": round(-(mu - z * sd), 4)})
    bl = pd.DataFrame(rows)
    bl.insert(0, "观测数", len(r))
    bl.insert(1, "日σ(%)", round(sd, 4))
    bl.insert(2, "年化σ(%)", round(sd * np.sqrt(252), 4))
    bl.to_csv(RES / "baseline_var.csv", index=False, encoding="utf-8-sig")
    print("\n[1] 基线 VaR（无条件正态）→ results/baseline_var.csv")
    print(bl.to_string(index=False))

    # ---- 2) 因子暴露 δ 与协方差校验 ----
    # 主口径：复权收益 ≡ β5·Δ5Y + β10·Δ10Y + 1·利差代理（同日 TR 剥离，见 build_tr_factors.py）
    y = F["etf_ret_tr_pct"]
    X = sm.add_constant(pd.DataFrame({
        "d5y_bp": F["d5y_bp"], "d10y_bp": F["d10y_bp"]}))
    dfa = pd.concat([y, X], axis=1).dropna()
    m = sm.OLS(dfa["etf_ret_tr_pct"], dfa[["const", "d5y_bp", "d10y_bp"]]).fit()
    b5, b10 = float(m.params["d5y_bp"]), float(m.params["d10y_bp"])
    dur = -(b5 + b10) * 100

    fac = pd.DataFrame({
        "Δ5Y_bp": F["d5y_bp"],
        "Δ10Y_bp": F["d10y_bp"],
        "利差代理_%": F["etf_spread_proxy_tr_pct"],
    }).dropna()
    Sigma = fac.cov()                                   # 因子协方差矩阵
    dvec = pd.Series({"Δ5Y_bp": b5, "Δ10Y_bp": b10, "利差代理_%": 1.0})   # δ
    sig_p = float(np.sqrt(dvec.values @ Sigma.values @ dvec.values))       # δ'Σδ
    expo = pd.DataFrame({
        "因子": ["利率 Δ5Y", "利率 Δ10Y", "信用利差代理", "汇率 CNY/USD"],
        "暴露δ": [b5, b10, 1.0, 0.0],
        "含义": ["% 收益 / bp", "% 收益 / bp", "% 收益 / % 残差",
                 "主口径 USD 本位→0；人民币次口径=+1（见 baseline_var_tr）"],
    })
    expo.to_csv(RES / "factor_exposure.csv", index=False, encoding="utf-8-sig")
    corr = fac.corr().round(3)
    corr.to_csv(RES / "factor_corr.csv", encoding="utf-8-sig")
    print("\n[2] 因子暴露 → results/factor_exposure.csv")
    print(expo.to_string(index=False))
    print(f"    同日回归：β5={b5:+.4f} β10={b10:+.4f}（等效久期≈{dur:.2f}y）")
    print(f"    因子协方差校验：√(δ'Σδ)={sig_p:.4f} %/日  vs  经验 σ={sd:.4f} %/日"
          f"（差={sig_p-sd:+.5f} 个百分点——几乎一致，拆解自洽）")
    print("    因子相关系数矩阵（Δ5Y/Δ10Y 高共线，协方差法已吸收）：")
    print(corr.to_string())

    # ---- 3) 平稳/高波动分段 ----
    # 3a) 慢性高波动态：滚动 60 日年化实现波动 ≥ 全样本 2/3 分位（集中 2022-23 结构性高波动）
    rv = r.rolling(RV_WIN).std() * np.sqrt(252)
    th = float(rv.quantile(Q_HIGH))
    head = r.index[:RV_WIN - 1]
    exp = r.expanding().std().iloc[:RV_WIN - 1] * np.sqrt(252)   # 开头不足 60 日用 expanding
    rv.loc[head] = exp.values
    chronic = (rv >= th)

    # 3b) 急性危机窗：事件时间线前后各 7 个自然日（约 ±5 交易日），窗内 |NAV 日收益| 达
    #     1.2%（4.51 × 日 σ）则整窗记危机。由实现收益触发，不靠人工挑日子，可复现；
    #     捕捉 2025-04 关税这类短促冲击。
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    CRISIS_BAR = 1.2                       # %（≈4.5 × 主口径日σ 0.266，复权后略升）
    crisis = pd.Series(False, index=r.index)
    ev_hit: list[str] = []
    rv_abs = r.abs()
    for _, e in ev.iterrows():
        win = r.index[(r.index >= e["date"] - pd.Timedelta(days=7)) &
                      (r.index <= e["date"] + pd.Timedelta(days=7))]
        if len(win) >= 1 and rv_abs.loc[win].max() >= CRISIS_BAR:
            crisis.loc[win] = True
            ev_hit.append(f"{e['date'].date()} {e['event_cn'][:18]} |NAV|max={rv_abs.loc[win].max():.2f}%")
    regime = pd.DataFrame({"date": r.index, "rv60_ann_%": rv.values,
                           "chronic_high": chronic.values, "crisis_window": crisis.values})
    regime["regime"] = np.where((regime["chronic_high"] | regime["crisis_window"]),
                                "高波动", "平稳")
    regime.to_csv(RES / "regimes.csv", index=False, encoding="utf-8-sig")
    cnt = regime["regime"].value_counts()
    print("\n[3] 平稳/高波动分段 → results/regimes.csv")
    print(f"    慢性高波动：60日年化σ≥{th:.2f}%（上{(1-Q_HIGH)*100:.0f}%分位）"
          f"→ {int(chronic.sum())} 天；叠加危机窗 → 高波动合计 {cnt.get('高波动',0)} 天"
          f"（{cnt.get('高波动',0)/len(regime)*100:.0f}%），平稳 {cnt.get('平稳',0)} 天")
    hi = regime[regime["regime"] == "高波动"]
    yrs = {str(y): int((hi['date'].dt.year == y).sum()) for y in sorted(hi['date'].dt.year.unique())}
    print(f"    高波动分布(年)：{yrs}")
    print(f"    命中的危机窗（事件 ±内 |NAV|≥{CRISIS_BAR}%）：")
    for s in ev_hit:
        print(f"      · {s}")

    # ---- 4) 16 条「存疑」异常日清单（供 9/17 复核） ----
    j = pd.read_csv(CLEAN / "outlier_judgment.csv", encoding="utf-8-sig")
    d = j[j["verdict"].astype(str).str.contains("存疑", na=False)].copy()
    d.to_csv(RES / "doubtful_days.csv", index=False, encoding="utf-8-sig")
    print(f"\n[4] 存疑日 → results/doubtful_days.csv  共 {len(d)} 条"
          f"（{d['series'].value_counts().to_dict()}）")

    # ---- 图：滚动实现波动 + 高波动阴影 ----
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(regime["date"], regime["rv60_ann_%"], lw=0.9, color="#4C72B0",
            label=f"{RV_WIN}日滚动年化波动")
    ax.axhline(th, color="grey", ls="--", lw=0.8, label=f"慢性高波动阈值 {th:.2f}%")
    ax.fill_between(regime["date"], 0, regime["rv60_ann_%"].max(),
                    where=(regime["regime"] == "高波动"), color="tomato",
                    alpha=.16, label="高波动（慢性态 ∪ 危机窗）")
    ax.set_ylim(0, None); ax.legend(); ax.grid(alpha=.3)
    ax.set_title(f"阶段二分段：60日滚动年化实现波动 + 高波动段（NAV 口径）—— 占 "
                 f"{cnt.get('高波动',0)/len(regime)*100:.0f}% 样本")
    ax.set_ylabel("年化波动 (%)")
    fig.tight_layout()
    fig.savefig(FIG / "phase2_rollvol_regimes.png", dpi=150)
    print("[图] figures/phase2_rollvol_regimes.png")


if __name__ == "__main__":
    main()
