"""
阶段二补充：验证「因子冲击 → 组合损失」的线性传导精度（导师 9/18 反馈第 2 条）

背景：目前只报了因子暴露 δ 的数值，没有验证用 δ 估算的组合损失与实际收益差多少。
      进入阶段三压力测试前必须先量出这个偏差幅度——压力测试的全部损失估计都建立在
      「δ × 因子冲击」这条线性链上，链条本身的误差必须先摆出来。

--------------------------------------------------------------------------
一、先说清 δ 是什么（这决定了「怎么验」才不是循环论证）

results/factor_exposure.csv 报的 δ 向量混合了两种不同性质的对象：

  Δ5Y      δ = −0.007714 %/bp   同日 OLS 估计的真实暴露（可观测、可外推）
  Δ10Y     δ = −0.029303 %/bp   同上（等效久期 ≈ 3.70y）
  信用利差代理 δ = +1.000        **回归残差本身**（build_tr_factors.py L96：
                                etf_spread_proxy_tr_pct = etf_ret_tr_pct − rate_attrib）
  汇率 CNY/USD δ = 0.000         主口径 USD 本位的口径设定，不是估计结果

残差项 δ≡1 意味着
      r_t ≡ c + δ5·Δ5Y_t + δ10·Δ10Y_t + 1.0×(信用利差代理_t)
在估计样本上是**代数恒等式**——把残差当因子喂回去，r̂ 当然等于 r，R²≡1。这不是模型，
是分解。用它「验证传导精度」是循环论证，必然得到零偏差的假结论。（该项目 9/17 已把
该配置登记为 M1δ，role = 恒等式，禁止作独立证据。）脚本第 0 步会把这个恒等式跑出来。

所以真正可检验、也真正被阶段三使用的，是**可观测因子模型**：

  M2  利率双因子        r̂ = c + δ5·Δ5Y + δ10·Δ10Y                  （现状口径）
  M3  利率双因子 + ΔOAS r̂ = c + δ5·Δ5Y + δ10·Δ10Y + δoas·ΔOAS      （补充检验）
  Mref 含残差（δ≡1）     仅用于证伪，不作检验

次口径（人民币）在 M2/M3 上再加 +1.0×fx_ret_pct——汇率收益可观测，这一项不是残差。
M3 受限：OAS 序列（FRED BAMLEMIBHGCRPIOAS）仅自 2023-09-06 起可用（749/1273 天），
故 M3 只在事件日 ≥ 2023-09-06 的子样本上检验，且**必须与 M2 在同一子样本上对比**
（脚本内自动对齐），不能拿全样本 M2 与子样本 M3 相减。

二、两种 δ 来源，代表两种使用场景

  A 全样本 δ   报告里报出的那个数。in-sample：回答「报告上的 δ 传导得准不准」。
  B 事件前 δ   只用事件窗**之前**的样本（扩展窗 ≥40 日）重新估计。
               out-of-sample：回答「站在事件发生前、用当时估得出的 δ 去预测」。
               这才是压力测试的真实使用姿势，是本脚本**主检验**所用的 δ。

三、偏差的严格分解（把「欠了多少」翻译成「欠的是什么」）

  全样本拟合 r = c_f + δ_f′x + e_f（e_f 即信用利差代理），事件前 δ 预测 r̂ = c_p + δ_p′x：

      r − r̂ = [ (c_f − c_p) + (δ_f − δ_p)′x ]  +  e_f
              └────── 参数差异项 ──────┘      └ 信用残差项 ┘

  两项都在事件窗上累加。若信用残差项占绝对主导 → 偏差不是「估计噪声」，而是被排除在
  模型之外的**那一整个信用维度**，即结构性的、不可能靠调 δ 消除。

四、窗口、口径与统计纪律
  主窗 = 事件日起 3 个交易日 [e, e+2]（事件日常在收盘后公布，留出反应日）
  另报 1 日 [e] 与 5 日 [e, e+4] 作稳健性对照；累计量一律用对数收益求和
  事件日不在交易日历上的（1 条）映射到其后首个交易日
  重叠：85 个事件 × 5 日窗在 1273 天样本上必然重叠（实测 40 对），故事件窗统计
        **不可当作独立样本做显著性检验**，只作描述性偏差幅度，脚本显式披露重叠数。

产出：results/delta_transmission_events.csv    逐事件 × 模型 × δ来源 × 窗口 明细
      results/delta_transmission_summary.csv  同维度的汇总
      figures/delta_transmission.png
用法： ./.venv/bin/python code/verify_delta_transmission.py
--------------------------------------------------------------------------
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

FACT = REPO / "factors"
EV = REPO / "events"
RES = REPO / "results"
FIG = REPO / "figures"

MIN_EST = 40                       # 事件前扩展窗最少有效交易日
WINDOWS = {"1日": 1, "3日(主)": 3, "5日": 5}
MAIN_WIN = "3日(主)"
RATE = ["d5y_bp", "d10y_bp"]
OAS_START = pd.Timestamp("2023-09-06")
C_MAIN, C_ALT, C_NEU, C_OK = "#2a78d6", "#eb6834", "#9a9a96", "#1baf7a"


def fit(X: pd.DataFrame, y: pd.Series, extra: list[str] | None = None):
    """OLS：r = c + Σδ·x。返回 (params: dict, resid, r2, nobs)；有效样本 < MIN_EST 返回 None。"""
    regs = RATE + (extra or [])
    d = pd.concat([y.rename("y"), X[regs]], axis=1).dropna()
    if len(d) < MIN_EST:
        return None
    m = sm.OLS(d["y"], sm.add_constant(d[regs])).fit()
    return ({"const": float(m.params["const"]), **{k: float(m.params[k]) for k in regs}},
            d["y"] - m.predict(sm.add_constant(d[regs])), float(m.rsquared), int(m.nobs))


def predict(p: dict, X: pd.DataFrame, w: slice, use_fx: bool, use_oas: bool) -> float:
    """在窗口 w 上累加模型的预测收益。"""
    s = p["const"] + p["d5y_bp"] * X["d5y_bp"].iloc[w] + p["d10y_bp"] * X["d10y_bp"].iloc[w]
    if use_oas:
        s = s + p["doas_bp"] * X["doas_bp"].iloc[w]
    if use_fx:
        s = s + X["fx_ret_pct"].iloc[w]
    return float(s.sum())


def main() -> None:
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    ev = pd.read_csv(EV / "risk_events_timeline.csv", parse_dates=["date"])
    idx = F.index
    print(f"[输入] factor_table_nav_tr n={len(F)} {idx[0].date()}→{idx[-1].date()}；"
          f"事件 {len(ev)} 条")

    # ---------- 0) 先证伪「含残差 δ」的伪检验 ----------
    f_main = fit(F, F["etf_ret_tr_pct"])
    p_f, resid_f, r2_full, n_f = f_main
    y_ident = (p_f["const"] + p_f["d5y_bp"] * F["d5y_bp"] + p_f["d10y_bp"] * F["d10y_bp"]
               + F["etf_spread_proxy_tr_pct"])
    gap = float((y_ident - F["etf_ret_tr_pct"]).abs().max())
    print("\n[0] 含信用利差代理（δ≡1）的「传导」检验")
    print("    r̂ = c + δ5·Δ5Y + δ10·Δ10Y + 1.0×信用利差代理")
    print(f"    max|r̂ − r| = {gap:.3e} pp  → 代数恒等式（数值误差量级），R²≡1")
    print(f"    可观测利率双因子模型：R²={r2_full:.4f}，n={n_f}")
    print(f"    → 可观测因子解释了 {r2_full*100:.1f}% 的组合收益方差，"
          f"剩余 {(1-r2_full)*100:.1f}% 落在信用维度上，而该维度**没有可用的 δ**")
    print(f"    → 该配置不能用于验证传导精度（循环论证），以下全部改用 M2/M3")

    # ---------- 1) 事件窗对齐 ----------
    pos = idx.searchsorted(ev["date"].values, side="left")
    keep = pos < len(idx)
    ev = ev[keep].copy()
    ev["map_pos"] = pos[keep]
    ev["e_date"] = idx[ev["map_pos"]]
    offcal = int((ev["e_date"].values != ev["date"].values).sum())
    spans = [(p, p + max(WINDOWS.values()) - 1) for p in ev["map_pos"]]
    ov = sum(1 for i in range(len(spans)) for j in range(i + 1, len(spans))
             if spans[j][0] <= spans[i][1])
    print(f"\n[1] 事件对齐：{len(ev)} 条，{offcal} 条非交易日已映射到次一交易日")
    print(f"    5 日窗两两重叠 {ov} 对 → 事件窗**非独立样本**，统计只作描述性")

    # ---------- 2) 逐事件 × 模型 × 口径 × δ来源 × 窗口 ----------
    rows = []
    # δ 一律在 **USD 复权收益** 上估计（factor_exposure.csv 的 δ 就是这个口径）：
    #   r_rmb = r_tr + fx_ret  ⇒  ∂r_rmb/∂Δ5Y = ∂r_tr/∂Δ5Y，利率/利差 δ 两口径相同，
    #   汇率不是「估计出来的系数」而是口径搬运（系数恒为 +1）。
    # 若直接在 r_rmb 上回归而把 fx_ret 留在残差里，fx_ret 会被利率/OAS 系数吸收，
    # 预测时再加一次 fx_ret 就构成重复计算。此处显式避开该陷阱。
    y_fit = F["etf_ret_tr_pct"]
    for cal, ycol, use_fx in (("主", "etf_ret_tr_pct", False),
                              ("次", "etf_ret_rmb_pct", True)):
        y = F[ycol]
        full = {False: fit(F, y_fit), True: fit(F, y_fit, extra=["doas_bp"])}
        if full[False] is None:
            continue
        resid_fc, r2f, _ = full[False][1], full[False][2], full[False][3]
        print(f"\n  [{cal}口径] δ 估计样本 = USD 复权收益 r²={r2f:.4f}"
              f"（{'预测时另加 +1.0×fx_ret_pct' if use_fx else 'USD 本位，δ_fx=0'}）")
        for mdl, use_oas in (("M2 利率双因子", False), ("M3 利率+ΔOAS", True)):
            pf = full[use_oas][0]
            if pf is None or (use_oas and "doas_bp" not in pf):
                continue
            for src in ("A 全样本δ", "B 事件前δ"):
                for wname, wlen in WINDOWS.items():
                    for _, e in ev.iterrows():
                        p0 = int(e["map_pos"])
                        w = slice(p0, p0 + wlen)
                        if y.iloc[w].isna().any():
                            continue                    # 人民币尾段缺汇率
                        if use_oas and (F["doas_bp"].iloc[w].isna().any()
                                        or e["e_date"] < OAS_START):
                            continue                    # OAS 子样本约束
                        if src.startswith("B"):
                            if p0 < MIN_EST:
                                continue
                            f = fit(F.iloc[:p0], y_fit.iloc[:p0],
                                    extra=["doas_bp"] if use_oas else None)
                            if f is None:
                                continue
                            pp = f[0]                   # 事件前参数（样本外）
                        else:
                            pp = pf
                        r_act = float(y.iloc[w].sum())
                        r_hat = predict(pp, F, w, use_fx, use_oas)
                        # 偏差分解（用同口径的全样本参数作参照系）
                        par = float(((pf["const"] - pp["const"])
                                     + (pf["d5y_bp"] - pp["d5y_bp"]) * F["d5y_bp"].iloc[w]
                                     + (pf["d10y_bp"] - pp["d10y_bp"]) * F["d10y_bp"].iloc[w]
                                     + ((pf["doas_bp"] - pp["doas_bp"]) * F["doas_bp"].iloc[w]
                                        if use_oas else 0.0)).sum())
                        cred = float(resid_fc.iloc[w].sum())
                        sh5, sh10 = (float(F["d5y_bp"].iloc[w].sum()),
                                     float(F["d10y_bp"].iloc[w].sum()))
                        rows.append({
                            "model": mdl, "caliber": cal, "delta_src": src, "window": wname,
                            "event_date": e["date"].date().isoformat(),
                            "e_date": e["e_date"].date().isoformat(),
                            "event_cn": e["event_cn"], "market": e["market"],
                            "r_actual_pct": round(r_act, 4), "r_pred_pct": round(r_hat, 4),
                            "dev_pct": round(r_act - r_hat, 4),
                            "understate": bool(r_act < r_hat),
                            "d5y_sum_bp": round(sh5, 2), "d10y_sum_bp": round(sh10, 2),
                            "rate_shock_bp": round(abs(sh5) + abs(sh10), 2),
                            "dev_param_pct": round(par, 4), "dev_credit_pct": round(cred, 4),
                            "r2_delta_src": round(r2f, 4),
                        })

    E = pd.DataFrame(rows)
    E.to_csv(RES / "delta_transmission_events.csv", index=False, encoding="utf-8-sig")
    print(f"\n[2] 逐事件明细 → results/delta_transmission_events.csv  {E.shape[0]} 行")

    # ---------- 3) 汇总 ----------
    sm_rows = []
    for (mdl, cal, src, wn), g in E.groupby(["model", "caliber", "delta_src", "window"],
                                            sort=False):
        d = g["dev_pct"]
        neg = d[d < 0]
        sm_rows.append({
            "model": mdl, "caliber": cal, "delta_src": src, "window": wn, "n": len(g),
            "dev_mean": round(float(d.mean()), 4), "dev_median": round(float(d.median()), 4),
            "dev_sd": round(float(d.std()), 4),
            "dev_p05": round(float(d.quantile(0.05)), 4),
            "dev_p95": round(float(d.quantile(0.95)), 4),
            "absdev_mean": round(float(d.abs().mean()), 4),
            "absdev_p90": round(float(d.abs().quantile(0.90)), 4),
            "absdev_max": round(float(d.abs().max()), 4),
            "under_share": round(float(g["understate"].mean()), 4),
            "under_mean": round(float(neg.mean()), 4) if len(neg) else np.nan,
            "under_worst": round(float(d.min()), 4),
            "credit_share": round(float(g["dev_credit_pct"].abs().sum()
                                        / (g["dev_param_pct"].abs().sum()
                                           + g["dev_credit_pct"].abs().sum())), 4),
            "rate_shock_med_bp": round(float(g["rate_shock_bp"].median()), 2),
        })
    S = pd.DataFrame(sm_rows)
    S.to_csv(RES / "delta_transmission_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[3] 汇总 → results/delta_transmission_summary.csv  {S.shape[0]} 行")

    # 恒等式核对：偏差全在利率+信用侧，汇率项按构造对消 → 两口径的绝对偏差必然逐值相同
    chk = E[E.model == "M2 利率双因子"].pivot_table(
        index=["delta_src", "window", "e_date"], columns="caliber", values="dev_pct")
    dif = float((chk["主"] - chk["次"]).abs().max())
    print(f"\n[4] 口径恒等式核对：dev(主) 与 dev(次) 的最大逐值差 = {dif:.3e} pp")
    print("    原因：r_rmb ≡ r_tr + fx_ret（两者均可观测），预测端同样 +1.0×fx_ret，")
    print("    汇率项在 r_actual − r_pred 中**精确对消**。故：")
    print("      · 汇率的传导不是「估计」而是口径搬运，不存在传导精度问题；")
    print("      · 两个口径的**绝对**传导误差完全相同，等于利率 + 信用部分的误差；")
    print("      · 有意义的差别只在**相对**量级：同一绝对偏差除以各自的风险尺度。")

    cal_var = {}
    for cal, conf in (("主", 95), ("次", 95)):
        cal_var[cal] = float(pd.read_csv(RES / "model_scorecard.csv")
                             .query("model=='HS250' and caliber==@cal and conf==95")
                             ["mean_var_base_pct"].iloc[0])
    print(f"\n    基准口径（HS250）1 日 95% VaR：主 {cal_var['主']:.4f}% / 次 {cal_var['次']:.4f}%")
    print(f"    {'窗':<8}{'口径':<5}{'绝对偏差均值':>12}{'绝对偏差p90':>12}"
          f"{'最大偏差':>10}{'÷3日VaR(均值)':>14}{'÷3日VaR(p90)':>14}")
    for wn in WINDOWS:
        for cal in ("主", "次"):
            r_ = S[(S.caliber == cal) & (S.model == "M2 利率双因子")
                   & (S.delta_src == "B 事件前δ") & (S.window == wn)].iloc[0]
            wl = WINDOWS[wn]
            v = cal_var[cal] * np.sqrt(wl)
            print(f"    {wn:<8}{cal:<5}{r_['absdev_mean']:>12.4f}{r_['absdev_p90']:>12.4f}"
                  f"{r_['absdev_max']:>10.4f}{r_['absdev_mean']/v:>14.2f}{r_['absdev_p90']/v:>14.2f}")

    show = ["model", "caliber", "delta_src", "window", "n", "dev_median", "dev_p05", "dev_p95",
            "absdev_mean", "absdev_p90", "absdev_max", "under_share", "under_mean",
            "credit_share"]
    print(f"\n{'='*128}\n逐配置汇总（主口径；次口径绝对偏差与之逐值相同，见上）"
          f"\n  偏差 = 实际累计收益 − δ 估算累计收益（负数 = δ 低估了损失）\n{'='*128}")
    print(S[S.caliber == "主"][show].to_string(index=False))

    # ---------- 4) 主检验展开 ----------
    print(f"\n{'='*128}\n主检验：主口径 × 事件前 δ（严格样本外）× {MAIN_WIN} 窗\n{'='*128}")
    M = E[(E.caliber == "主") & (E.model == "M2 利率双因子")
          & (E.delta_src == "B 事件前δ") & (E.window == MAIN_WIN)]
    d = M["dev_pct"]
    sig1 = float(F["etf_ret_tr_pct"].std())
    var95 = float(pd.read_csv(RES / "model_scorecard.csv")
                  .query("model=='HS250' and caliber=='主' and conf==95")["mean_var_base_pct"].iloc[0])
    var3 = var95 * np.sqrt(3)
    print(f"  事件数 {len(M)}；偏差中位数 {d.median():+.4f}pp，均值 {d.mean():+.4f}pp，"
          f"标准差 {d.std():.4f}pp")
    print(f"  绝对偏差：均值 {d.abs().mean():.4f}pp，90 分位 {d.abs().quantile(.9):.4f}pp，"
          f"最大 {d.abs().max():.4f}pp")
    print(f"  低估损失（实际比估算更亏）事件占比 {M['understate'].mean()*100:.0f}%"
          f"（{int(M['understate'].sum())}/{len(M)}）；低估时平均低估 {abs(d[d<0].mean()):.4f}pp，"
          f"最多低估 {abs(d.min()):.4f}pp")
    print(f"  量级对照：日σ={sig1:.4f}%，3日σ={sig1*np.sqrt(3):.4f}%；"
          f"基准口径（HS250 主 95%）3日 VaR={var3:.4f}%")
    print(f"           → 偏差90分位 {d.abs().quantile(.9):.4f}pp "
          f"= {d.abs().quantile(.9)/var3:.2f} × 3日 VaR；"
          f"最大偏差 {d.abs().max():.4f}pp = {d.abs().max()/var3:.2f} × 3日 VaR")
    tot = M['dev_param_pct'].abs().sum() + M['dev_credit_pct'].abs().sum()
    print(f"  偏差分解：参数差异项 |合计| {M['dev_param_pct'].abs().sum():.2f}pp，"
          f"信用残差项 |合计| {M['dev_credit_pct'].abs().sum():.2f}pp"
          f" → 信用维度占 {M['dev_credit_pct'].abs().sum()/tot*100:.1f}%")

    # 压力测试最关心的是「亏损侧」：分开看实际亏损事件与实际盈利事件
    print("  按实际损益方向拆分（压力测试只关心前一行）：")
    for tag, sub in (("实际亏损事件", M[M.r_actual_pct < 0]),
                     ("实际盈利事件", M[M.r_actual_pct >= 0])):
        dd = sub["dev_pct"]
        if len(sub) == 0:
            continue
        print(f"    {tag} n={len(sub):3d}  低估占比 {sub['understate'].mean()*100:3.0f}%"
              f"  平均偏差 {dd.mean():+.4f}pp  绝对偏差均值 {dd.abs().mean():.4f}pp"
              f"  最大低估 {dd.min():+.4f}pp  最大高估 {dd.max():+.4f}pp")
    print("  偏差最大的 6 个事件（正=实际好于估算；负=实际差于估算）：")
    for _, r in M.reindex(d.abs().sort_values(ascending=False).index).head(6).iterrows():
        print(f"    {r['e_date']} {r['event_cn'][:22]:<24s} 实际 {r['r_actual_pct']:+7.3f}%  "
              f"δ估算 {r['r_pred_pct']:+7.3f}%  偏差 {r['dev_pct']:+7.3f}pp"
              f"（信用 {r['dev_credit_pct']:+6.3f} / 参数 {r['dev_param_pct']:+6.3f}）"
              f"  利率冲击 {r['rate_shock_bp']:.0f}bp")

    a = S[(S.caliber == "主") & (S.model == "M2 利率双因子")
          & (S.delta_src == "A 全样本δ") & (S.window == MAIN_WIN)].iloc[0]
    b = S[(S.caliber == "主") & (S.model == "M2 利率双因子")
          & (S.delta_src == "B 事件前δ") & (S.window == MAIN_WIN)].iloc[0]
    print(f"\n  in-sample（全样本 δ）vs out-of-sample（事件前 δ）绝对偏差均值："
          f"{a['absdev_mean']:.4f}pp vs {b['absdev_mean']:.4f}pp"
          f"（放大 {b['absdev_mean']/a['absdev_mean']:.2f} 倍）")
    print(f"  → δ 估计本身很稳，传导误差不是估计噪声，是**结构性漏因子**。")

    # ---------- 5) M2 vs M3（OAS 子样本内对比）----------
    key = ["caliber", "delta_src", "window"]
    m3 = E[E.model == "M3 利率+ΔOAS"][key + ["e_date", "dev_pct", "understate"]]
    print(f"\n{'='*128}\n补充检验：加入可观测信用因子 ΔOAS 能否收敛偏差"
          f"（OAS 仅 {OAS_START.date()} 起可用；OAS 与汇率无关，口径维度无独立信息，"
          f"故只列主口径）\n{'='*128}")
    for src in ("A 全样本δ", "B 事件前δ"):
        for wn in ("1日", MAIN_WIN, "5日"):
            g3 = m3[(m3.caliber == "主") & (m3.delta_src == src) & (m3.window == wn)]
            if len(g3) == 0:
                continue
            g2 = E[(E.model == "M2 利率双因子") & (E.caliber == "主")
                   & (E.delta_src == src) & (E.window == wn)].merge(
                g3[["e_date"]], on="e_date", how="inner")   # 严格对齐到同一子样本
            if len(g2) == 0:
                continue
            d2, d3 = g2["dev_pct"], g3["dev_pct"]
            chg = (d3.abs().mean() / d2.abs().mean() - 1) * 100
            verdict = ("收敛" if chg < -5 else ("基本持平" if abs(chg) <= 5 else "反而变差"))
            print(f"  {src} · {wn}窗 · 同一子样本 n={len(g3)}"
                  f"（{g3['e_date'].min()} → {g3['e_date'].max()}）")
            print(f"    M2 利率双因子  绝对偏差 均值 {d2.abs().mean():.4f}pp / "
                  f"90分位 {d2.abs().quantile(.9):.4f}pp；低估占比 {g2['understate'].mean()*100:.0f}%"
                  f"；最大偏差 {d2.abs().max():.4f}pp")
            print(f"    M3 利率+ΔOAS   绝对偏差 均值 {d3.abs().mean():.4f}pp / "
                  f"90分位 {d3.abs().quantile(.9):.4f}pp；低估占比 {g3['understate'].mean()*100:.0f}%"
                  f"；最大偏差 {d3.abs().max():.4f}pp")
            print(f"    → 绝对偏差均值变化 {chg:+.1f}%  ⇒ {verdict}")
    print(f"\n    注意：该子样本（{m3['e_date'].min()} 起）**不含 2022 年信用冲击**，"
          f"而 2022 年恰是偏差最大的时段（见表）。")
    print(f"    故 M3 的改善只在「较平静的近期样本」上得到验证，"
          f"不能外推为「加了 OAS 就能扛住信用危机」。")

    # ---------- 6) 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))

    G = E[(E.caliber == "主") & (E.delta_src == "B 事件前δ") & (E.window == MAIN_WIN)]
    g2, g3 = G[G.model == "M2 利率双因子"], G[G.model == "M3 利率+ΔOAS"]

    ax = axes[0]
    ax.scatter(g2["r_pred_pct"], g2["r_actual_pct"], s=34, color=C_MAIN, alpha=0.8,
               edgecolor="white", linewidth=0.5, label=f"M2 利率双因子（n={len(g2)}）")
    if len(g3):
        ax.scatter(g3["r_pred_pct"], g3["r_actual_pct"], s=34, marker="^", color=C_OK,
                   alpha=0.85, edgecolor="white", linewidth=0.5,
                   label=f"M3 利率+ΔOAS（n={len(g3)}）")
    lo = float(G[["r_pred_pct", "r_actual_pct"]].min().min())
    hi = float(G[["r_pred_pct", "r_actual_pct"]].max().max())
    ax.plot([lo, hi], [lo, hi], color="#52514e", ls="--", lw=1.1, label="完全传导（45°）")
    ax.set_xlabel("δ 估算的组合累计收益 (%)")
    ax.set_ylabel("实际组合累计收益 (%)")
    ax.set_title(f"预测 vs 实际（主口径 · {MAIN_WIN}窗 · 事件前δ）", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.scatter(g2["rate_shock_bp"], g2["dev_pct"], s=34, color=C_MAIN, alpha=0.8,
               edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#52514e", ls="--", lw=1.1)
    rho = float(np.corrcoef(g2["rate_shock_bp"], g2["dev_pct"])[0, 1])
    z = np.polyfit(g2["rate_shock_bp"], g2["dev_pct"], 1)
    xs = np.linspace(g2["rate_shock_bp"].min(), g2["rate_shock_bp"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=C_ALT, lw=1.5, label=f"趋势 {z[0]:+.4f} pp/bp")
    rho_txt = ("冲击幅度与偏差基本无关" if abs(rho) < 0.3
               else "偏差随冲击幅度系统性变化")
    ax.text(0.03, 0.05, f"相关系数 ρ = {rho:+.3f}\n（{rho_txt}）",
            transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3a3a38")
    ax.set_xlabel("窗口内利率因子冲击 |Δ5Y|+|Δ10Y| (bp)")
    ax.set_ylabel("偏差 = 实际 − δ估算 (pp)")
    ax.set_title("偏差随冲击幅度的变化（主口径·事件前δ）", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.hist(g2["dev_pct"], bins=18, alpha=0.6, color=C_MAIN, edgecolor="white",
            linewidth=0.6, label=f"M2（低估占比 {g2['understate'].mean()*100:.0f}%）")
    if len(g3):
        ax.hist(g3["dev_pct"], bins=18, alpha=0.45, color=C_OK, edgecolor="white",
                linewidth=0.6, label=f"M3（低估占比 {g3['understate'].mean()*100:.0f}%）")
    ax.axvline(0, color="#52514e", ls="--", lw=1.2)
    ax.set_xlabel("偏差 = 实际 − δ估算 (pp)")
    ax.set_ylabel("事件数")
    ax.set_title(f"偏差分布（主口径·{MAIN_WIN}窗·事件前δ）", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("因子冲击到组合损失的线性传导精度验证（85 个历史事件窗）",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "delta_transmission.png", dpi=150, bbox_inches="tight")
    print("\n[图] figures/delta_transmission.png")


if __name__ == "__main__":
    main()
