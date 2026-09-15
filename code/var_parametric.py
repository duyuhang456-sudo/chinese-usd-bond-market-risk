"""
阶段二 Day2（9/15）参数法 VaR —— 无条件正态 / GARCH(1,1) 条件 / 因子协方差 δ-normal（双口径）

输入：factors/factor_table_nav_tr.csv（复权主口径 etf_ret_tr_pct + 人民币次口径 etf_ret_rmb_pct，
      源见 build_tr_factors.py）；δ 由复权主口径全样本同日回归估计（与 prep_phase2 一致）。
口径：1 日持有期；95% / 99%；**VaR = z_c · σ（μ=0，spec §1）**；逐日样本外前推。

四个参数化变体（均逐日滚出样本外预测，供 9/17 回测直接消费）：
  M1   无条件正态    ：σ 取**滚动 250 交易日**窗（[t−250, t−1]），与回测协议 §7.1 同窗
  M1g  GARCH(1,1)    ：**扩展窗逐日重估**（arch 7.2.0，正态为主 + Student-t 对照），
                       σ_t = 对 t 的一步向前条件波动预测（仅用 ≤ t−1 信息）
  M1e  EWMA(λ=0.94)  ：spec §8.3 的回退对照——实测 α+β≈0.996（近 IGARCH），故并行报告
  M1δ  因子 δ-normal ：滚动 250 窗估三因子（Δ5Y/Δ10Y/利差代理）协方差 Σ_w，σ_p=√(δ′Σ_w δ)

人民币次口径（对照）：同法在 etf_ret_rmb_pct 上重做；δ_rmb = δ + 汇率暴露 1。
CNY 尾 5 交易日源未发布 → 该段 NaN 记档（不伪造），故次口径样本略短。

产出：results/var_parametric.csv（逐日：收益 / 各模型 σ_t / 各模型 VaR95·99，主口径 + 次口径分列）
      figures/var_garch_sigma.png   波动率估计对比（GARCH σ_t vs EWMA vs 无条件，双口径分面）
      figures/var_series_compare.png 收益 + 三模型 VaR（口径 × 置信度 小多图）

用法： ./.venv/bin/python code/var_parametric.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import REPO
from var_common import (WIN, Z, LAM, C_M1, C_GARCH, C_ALT, C_GREY, INK2,
                        sigma_uncond, sigma_ewma, var_cols)

FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SCALE = 10.0                    # GARCH 输入缩放（% → 约 1 量级，改善优化器收敛，输出再除回）
C_EWMA = C_ALT                  # 本脚本第三序列槽位 = EWMA（见 var_common 的槽位约定）


# ---------------------------------------------------------------- 估计器
# sigma_uncond（M1）/ sigma_ewma（M1e）/ 违规判定已抽到 var_common.py，供 9/16、9/17 共用。
def sigma_garch(r: pd.Series, dist: str = "normal") -> tuple[pd.Series, dict, pd.DataFrame]:
    """M1g：扩展窗逐日重估 GARCH(1,1)，σ_t = 一步向前条件波动（信息 ≤ t−1）。

    同时记录每次重估的 α/β：近 IGARCH 数据下 MLE 会落到 **α→0、β→1 的边界**，
    此时 σ_t 退化为常数（不再是条件波动）——必须逐日留痕并如实披露，不能默认它是「GARCH 条件化」。
    """
    sig = pd.Series(np.nan, index=r.index)
    tr = pd.DataFrame(np.nan, index=r.index, columns=["alpha", "beta", "persist"])
    for t in range(WIN, len(r)):
        y = r.iloc[:t].values * SCALE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, vol="GARCH", p=1, q=1, dist=dist,
                             mean="Constant").fit(disp="off", show_warning=False)
        f = res.forecast(horizon=1, reindex=False)
        sig.iloc[t] = float(np.sqrt(f.variance.iloc[-1, 0])) / SCALE
        p = res.params
        tr.loc[r.index[t]] = [p["alpha[1]"], p["beta[1]"], p["alpha[1]"] + p["beta[1]"]]
    # 全样本拟合（仅用于报告系数/似然/平稳性，不参与逐日预测）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = arch_model(r.values * SCALE, vol="GARCH", p=1, q=1, dist=dist,
                          mean="Constant").fit(disp="off", show_warning=False)
    p = full.params
    diag = {"dist": dist, "mu": p.get("mu", np.nan) / SCALE,
            "omega": p["omega"] / SCALE ** 2, "alpha": p["alpha[1]"], "beta": p["beta[1]"],
            "persist": p["alpha[1]"] + p["beta[1]"], "loglik": full.loglikelihood,
            "nu": p.get("nu", np.nan), "n": int(full.nobs)}
    return sig, diag, tr


def sigma_dnorm(fac: pd.DataFrame, delta: pd.Series, win: int = WIN) -> pd.Series:
    """M1δ：滚动窗因子协方差 Σ_w → σ_p = √(δ′Σ_w δ)。"""
    sig = pd.Series(np.nan, index=fac.index)
    X = fac.values
    d = delta.values
    for t in range(win, len(fac)):
        W = X[t - win:t]
        if not np.isfinite(W).all():
            continue
        sig.iloc[t] = float(np.sqrt(d @ np.cov(W, rowvar=False) @ d))
    return sig


def var_attribution(fac: pd.DataFrame, delta: pd.Series, win: int = WIN) -> pd.Series:
    """因子方差贡献份额（%）：对每窗算 δ_i·(Σ_w δ)_i / δ′Σ_w δ 后取均值。

    这是 δ-normal 相对「直接法」的**增量价值**所在——恒等的 σ 之下给出「利率 vs 残差」的拆解。
    """
    X, d = fac.values, delta.values
    share = np.full((len(fac), len(d)), np.nan)
    for t in range(win, len(fac)):
        W = X[t - win:t]
        if not np.isfinite(W).all():
            continue
        Sd = np.cov(W, rowvar=False) @ d
        tot = float(d @ Sd)
        if tot > 0:
            share[t] = d * Sd / tot * 100.0
    return pd.DataFrame(share, index=fac.index, columns=fac.columns).mean()


# ---------------------------------------------------------------- 主流程
def main() -> None:
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    r_main = F["etf_ret_tr_pct"].astype(float)
    r_rmb = F["etf_ret_rmb_pct"].astype(float)

    # 次口径缺失必须是「连续尾段」（CNY 源滞后），否则轮动下标会错位——显式护栏
    nan_pos = np.flatnonzero(r_rmb.isna().values)
    assert nan_pos.size == 0 or nan_pos[-1] == len(r_rmb) - 1, "人民币次口径 NaN 非尾部连续，需检查"
    n_tail = int(nan_pos.size)
    r_rmb_v = r_rmb.dropna()
    print(f"[输入] factor_table_nav_tr rows={len(F)}  主口径 n={r_main.notna().sum()}  "
          f"次口径 n={len(r_rmb_v)}（CNY 尾 {n_tail} 日记 NaN，不伪造）")

    # ---- δ：复权主口径全样本同日回归 -------------------------------------
    X = sm.add_constant(pd.DataFrame({"d5y": F["d5y_bp"], "d10y": F["d10y_bp"]}))
    dfa = pd.concat([r_main.rename("y"), X], axis=1).dropna()
    m = sm.OLS(dfa["y"], dfa[["const", "d5y", "d10y"]]).fit()
    b5, b10 = float(m.params["d5y"]), float(m.params["d10y"])
    delta = pd.Series({"d5y_bp": b5, "d10y_bp": b10, "spread": 1.0})
    delta_rmb = pd.Series({"d5y_bp": b5, "d10y_bp": b10, "spread": 1.0, "fx": 1.0})
    print(f"[δ] 复权主口径同日回归 β5={b5:+.4f} β10={b10:+.4f} "
          f"（等效久期≈{-(b5+b10)*100:.2f}y，R²={m.rsquared:.3f}）")

    # ---- 各模型 σ_t -------------------------------------------------------
    out = pd.DataFrame(index=F.index)
    out["ret_tr_pct"] = r_main

    s_unc = sigma_uncond(r_main)
    s_ewm = sigma_ewma(r_main)
    s_g, diag_n, tr_n = sigma_garch(r_main, "normal")
    s_gt, diag_t, _ = sigma_garch(r_main, "t")
    fac_m = pd.DataFrame({"d5y_bp": F["d5y_bp"], "d10y_bp": F["d10y_bp"],
                          "spread": F["etf_spread_proxy_tr_pct"]})
    s_dn = sigma_dnorm(fac_m, delta)

    for name, s in [("sig_uncond", s_unc), ("sig_garch", s_g), ("sig_garch_t", s_gt),
                    ("sig_ewma", s_ewm), ("sig_dnorm", s_dn)]:
        out[name] = s
    for tag, s in [("m1", s_unc), ("garch", s_g), ("garch_t", s_gt),
                   ("ewma", s_ewm), ("dnorm", s_dn)]:
        for k, v in var_cols(s, tag).items():
            out[k] = v

    # ---- 人民币次口径（对照） --------------------------------------------
    out["ret_rmb_pct"] = r_rmb
    s_unc_r = sigma_uncond(r_rmb_v).reindex(F.index)
    s_g_r, diag_r, tr_r = sigma_garch(r_rmb_v, "normal")
    s_g_r = s_g_r.reindex(F.index)
    fac_r = pd.DataFrame({"d5y_bp": F["d5y_bp"], "d10y_bp": F["d10y_bp"],
                          "spread": F["etf_spread_proxy_tr_pct"], "fx": F["fx_ret_pct"]})
    s_dn_r = sigma_dnorm(fac_r, delta_rmb)
    s_ewm_r = sigma_ewma(r_rmb_v).reindex(F.index)
    for name, s in [("rmb_sig_uncond", s_unc_r), ("rmb_sig_garch", s_g_r),
                    ("rmb_sig_ewma", s_ewm_r), ("rmb_sig_dnorm", s_dn_r)]:
        out[name] = s
    for tag, s in [("rmb_m1", s_unc_r), ("rmb_garch", s_g_r),
                   ("rmb_ewma", s_ewm_r), ("rmb_dnorm", s_dn_r)]:
        for k, v in var_cols(s, tag).items():
            out[k] = v

    # ---- 样本外切片（第 251 个收益日起）----------------------------------
    oos = out.iloc[WIN:].copy()
    oos.index.name = "date"
    oos.round(6).to_csv(RES / "var_parametric.csv", encoding="utf-8-sig")
    print(f"\n[1] 逐日样本外预测 → results/var_parametric.csv  rows={len(oos)}"
          f"（{oos.index[0].date()} ~ {oos.index[-1].date()}）")

    # ---- GARCH 诊断 -------------------------------------------------------
    print("\n[2] GARCH(1,1) 全样本拟合诊断（逐日预测为扩展窗重估，非本组系数）：")
    for lbl, d in [("主口径 · 正态", diag_n), ("主口径 · Student-t", diag_t), ("次口径 · 正态", diag_r)]:
        extra = f" ν={d['nu']:.2f}" if np.isfinite(d["nu"]) else ""
        print(f"    {lbl:<18s} μ={d['mu']:+.4f}%/日  ω={d['omega']:.5f}  α={d['alpha']:.4f} "
              f"β={d['beta']:.4f}  α+β={d['persist']:.4f}  logL={d['loglik']:.1f}{extra}")
    print(f"    说明：α+β≈1（近 IGARCH，冲击近乎永久）——按 spec §8.3 已并行报告 EWMA(λ={LAM}) 回退对照；")
    print(f"          Student-t 的 ν 偏大（≈{diag_t['nu']:.0f}），厚尾程度弱于阶段一超额峰度暗示，两分布差异有限。")

    # 边界解留痕：MLE 落到 α→0、β→1 时 σ_t 是常数，等于把「GARCH 条件化」偷换成无条件波动
    for lbl, tr in [("主口径", tr_n), ("次口径", tr_r)]:
        bad = tr[tr["alpha"] < 1e-4]
        if len(bad):
            print(f"    [边界解] {lbl}：{len(bad)}/{len(tr)} 次逐日重估落到 α→0、β→1（σ_t 退化为常数，"
                  f"非条件波动）——区间 {bad.index[0].date()} ~ {bad.index[-1].date()}；"
                  f"其余 {len(tr)-len(bad)} 次为正常内点解。这批日子 GARCH 与无条件口径等价，"
                  f"对比时不可当作「GARCH 条件化」的独立证据。")
        else:
            print(f"    [边界解] {lbl}：无（全部逐日重估均为内点解 α>0）。")
    tr_n.to_csv(RES / "garch_refit_trace_main.csv", encoding="utf-8-sig")
    tr_r.to_csv(RES / "garch_refit_trace_rmb.csv", encoding="utf-8-sig")
    bad_r = tr_r.index[tr_r["alpha"] < 1e-4]

    # ---- 模型对比表 -------------------------------------------------------
    print("\n[3] 样本外平均 VaR（μ=0，%/日）与平均 σ：")
    rows = []
    for lbl, col95, col99, sigcol in [
            ("M1 无条件(滚动250)", "m1_var_95", "m1_var_99", "sig_uncond"),
            ("M1g GARCH(1,1)", "garch_var_95", "garch_var_99", "sig_garch"),
            ("M1g GARCH-t", "garch_t_var_95", "garch_t_var_99", "sig_garch_t"),
            ("M1e EWMA(0.94)", "ewma_var_95", "ewma_var_99", "sig_ewma"),
            ("M1δ δ-normal", "dnorm_var_95", "dnorm_var_99", "sig_dnorm")]:
        rows.append({"模型": lbl, "平均σ(%)": oos[sigcol].mean(),
                     "VaR95(%)": oos[col95].mean(), "VaR99(%)": oos[col99].mean(),
                     "σ_max(%)": oos[sigcol].max()})
    cmp_main = pd.DataFrame(rows)
    print(cmp_main.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rows_r = []
    for lbl, col95, col99, sigcol in [
            ("M1 无条件(滚动250)", "rmb_m1_var_95", "rmb_m1_var_99", "rmb_sig_uncond"),
            ("M1g GARCH(1,1)", "rmb_garch_var_95", "rmb_garch_var_99", "rmb_sig_garch"),
            ("M1e EWMA(0.94)", "rmb_ewma_var_95", "rmb_ewma_var_99", "rmb_sig_ewma"),
            ("M1δ δ-normal", "rmb_dnorm_var_95", "rmb_dnorm_var_99", "rmb_sig_dnorm")]:
        rows_r.append({"模型": lbl, "平均σ(%)": oos[sigcol].mean(),
                       "VaR95(%)": oos[col95].mean(), "VaR99(%)": oos[col99].mean()})
    cmp_rmb = pd.DataFrame(rows_r)
    print("\n    人民币次口径（对照）：")
    print(cmp_rmb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ratio = oos["rmb_garch_var_99"].mean() / oos["garch_var_99"].mean()
    print(f"    次口径 / 主口径（GARCH 99%）平均比 = {ratio:.3f}"
          f"（超出 1 的部分 ≈ 汇率折算风险贡献）")
    print(f"    次口径有效样本 {int(oos['ret_rmb_pct'].notna().sum())}"
          f"/{len(oos)} 日（CNY 尾 {n_tail} 日为 NaN，逐日对照从该段起留空）")

    # ---- δ-normal 的等价性与归因（重要披露）------------------------------
    dmax = float((oos["sig_dnorm"] - oos["sig_uncond"]).abs().max())
    dmax_r = float((oos["rmb_sig_dnorm"] - oos["rmb_sig_uncond"]).abs().max())
    print(f"\n[4] M1δ 与 M1 的 σ 最大差异：主口径 {dmax:.2e}、次口径 {dmax_r:.2e} 个百分点"
          f"（后者非零段仅来自 fx 尾部 NaN 窗）→ **两者数值上恒等**：")
    print("    第三因子「利差代理」被定义为同一次 OLS 的残差，故任意窗口内 δ′Σ_w δ ≡ var_w(r) 是代数恒等式。")
    print("    因此 M1δ **不构成独立模型证据**；它的价值在归因与压力映射（下表），见 spec §4。")
    NAME = {"d5y_bp": "Δ5Y", "d10y_bp": "Δ10Y", "spread": "利差代理（残差）", "fx": "汇率 CNY/USD"}
    attr_m = var_attribution(fac_m, delta)
    attr_r = var_attribution(fac_r, delta_rmb)
    # 注意：两个 Series 因子集不同（次口径多一个 fx），DataFrame 会按索引并集**字典序**排行 → 必须显式 reindex
    attr = (pd.DataFrame({"主口径（复权 USD）": attr_m, "人民币次口径": attr_r})
            .reindex(["d5y_bp", "d10y_bp", "spread", "fx"]).rename(index=NAME))
    attr.to_csv(RES / "var_attribution.csv", encoding="utf-8-sig")
    print("\n    因子方差贡献份额（窗口均值，%）：")
    print(attr.round(2).to_string())
    c_m = attr["主口径（复权 USD）"]
    c_r = attr["人民币次口径"]
    print(f"    主口径：利率两因子合计 {c_m['Δ5Y'] + c_m['Δ10Y']:.1f}%、残差 {c_m['利差代理（残差）']:.1f}%"
          f"（残差份额 ≈ 1−R²=0.243，与利率剥离 R²=0.757 呼应）")
    print(f"    次口径：利率 {c_r['Δ5Y'] + c_r['Δ10Y']:.1f}%、残差 {c_r['利差代理（残差）']:.1f}%、"
          f"汇率 {c_r['汇率 CNY/USD']:.1f}% → 产出 results/var_attribution.csv")

    # ---- 图 1：波动率估计对比（双口径分面）-------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)
    for ax, lbl, cols in [(axes[0], "主口径（复权 USD）",
                           [("sig_uncond", "无条件（滚动 250 日）", C_M1),
                            ("sig_garch", "GARCH(1,1)", C_GARCH),
                            ("sig_ewma", "EWMA(0.94)", C_EWMA)]),
                          # 两面板都只画「相互独立」的三个估计量：δ-normal ≡ M1 会精确压在同一条线上，
                          # 画出来等于藏一条线；该等价性改由 [4] 段文字与 results/var_attribution.csv 披露
                          (axes[1], "人民币次口径",
                           [("rmb_sig_uncond", "无条件（滚动 250 日）", C_M1),
                            ("rmb_sig_garch", "GARCH(1,1)", C_GARCH),
                            ("rmb_sig_ewma", "EWMA(0.94)", C_EWMA)])]:
        labs = []
        for col, name, c in cols:
            s = oos[col]
            # σ 以 bp 计（1% = 100bp）：直接标 % 会把 26.7bp 读成 27%/日，量级差 100 倍
            ax.plot(s.index, s * 100, lw=1.2, color=c, label=name)
            if s.notna().any():
                labs.append([float(s.dropna().iloc[-1]) * 100, "  " + name.split("（")[0], c])
        # 末端三条线收敛到同一水平 → 直标必须做最小间距撑开，否则文字互相压叠
        labs.sort()
        for i in range(1, len(labs)):
            labs[i][0] = max(labs[i][0], labs[i - 1][0] + 2.8)
        for y, txt, c in labs:
            ax.text(oos.index[-1], y, txt, va="center", ha="left", fontsize=7.5, color=c)
        ax.set_title(lbl + " · 条件波动率 σ_t（日，bp）", fontsize=10, loc="left")
        ax.set_ylabel("条件日波动 σ_t (bp)")
        ax.margins(x=0.14)
        ax.grid(alpha=.25, lw=.6)
    # 次口径早期 α→0/β→1 的边界解区段：GARCH 线在此处是常数，必须标出来，否则读图会误以为它真在条件化
    if len(bad_r):
        axes[1].axvspan(bad_r[0], bad_r[-1], color=INK2, alpha=.08, lw=0)
        axes[1].text(bad_r[0], 0.045, " 阴影段：MLE 落 α→0/β→1 边界，GARCH\n σ_t 退化为常数（非条件波动）",
                     transform=axes[1].get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    # 图例放左下：2022 年的波动尖峰集中在左上，图例放上面会盖住数据
    axes[0].legend(loc="lower left", fontsize=8, framealpha=.9)
    fig.suptitle("参数法波动率估计对比：GARCH(1,1) vs EWMA vs 无条件（样本外逐日）", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "var_garch_sigma.png", dpi=150)
    print("\n[图] figures/var_garch_sigma.png")

    # ---- 图 2：收益 + 三模型 VaR（口径 × 置信度，小多图）-----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.4), sharex=True)
    panels = [("主口径（复权 USD）", "ret_tr_pct", "m1", "garch", "ewma"),
              ("人民币次口径", "ret_rmb_pct", "rmb_m1", "rmb_garch", "rmb_ewma")]
    for irow, (lbl, rcol, t1, t2, t3) in enumerate(panels):
        for icol, c in enumerate(["95", "99"]):
            ax = axes[irow, icol]
            ax.plot(oos.index, oos[rcol], lw=.75, color=C_GREY, label="日收益")
            ax.plot(oos.index, -oos[f"{t1}_var_{c}"], lw=1.2, color=C_M1, label="M1 无条件")
            ax.plot(oos.index, -oos[f"{t2}_var_{c}"], lw=1.2, color=C_GARCH, label="M1g GARCH")
            ax.plot(oos.index, -oos[f"{t3}_var_{c}"], lw=1.2, color=C_EWMA, label="M1e EWMA")
            ax.set_title(f"{lbl} · {c}% VaR（负值 = 损失阈值）", fontsize=9.5, loc="left")
            ax.grid(alpha=.25, lw=.6)
            ax.margins(x=0.01)
            if irow == 1:
                ax.set_xlabel("")
            if icol == 0:
                ax.set_ylabel("日收益 / VaR (%)")
    # 图例提到画布顶部：四联图里任何面板内嵌图例都会压住收益尖峰
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.938))
    fig.suptitle("三参数模型的逐日样本外 VaR 与实现收益（复权口径，2022-08 ~ 2026-09）", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(FIG / "var_series_compare.png", dpi=150)
    print("[图] figures/var_series_compare.png")


if __name__ == "__main__":
    main()
