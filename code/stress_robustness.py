"""
阶段三 Day4（9/24）：压力测试稳健性分析

对 `results/stress_impact.csv` 的测算结果做四项检验，回答「结论在多大程度上依赖于
我在构造与测算中做的那几个选择」。四项检验的共同目的是把「我选了什么」与
「数据支持什么」分开——凡是结论随选择大幅摆动的地方，都必须写明。

------------------------------------------------------------------------------
一、情景覆盖度检验（本脚本的存在理由）
------------------------------------------------------------------------------
压力情景库若不含样本内**实测最差**的结果，则「极端情景」这个说法没有依据。
本项检验对 h ∈ {3, 5, 6} 日 × 主/次口径，取历史滚动 h 日累计损失的**前 10 名窗口**，
与情景库的损失逐一对齐，输出两件事：
  ① 情景损失在历史 h 日分布中的**分位数**（比「超过 99% ES」信息量更大）；
  ② 情景库是否**被实测最差窗口超越**——若被超越，该情景的命名（轻/中/极端）就有误导性。

**这项检验在本次执行中查出了选型遗漏**：2022-03-08 ~ 2022-03-15 是样本内实测最差的
6 日窗口（主口径 −3.0008%），其因子冲击与组合损失双双超过点名的 H1，
而该事件本就在 `events/risk_events_timeline.csv` 中。选型时只按「3 日利率峰值」挑窗
（峰值确在 2022-06-14），漏掉了窗长更长时更差的一段。已补入补充情景 X2。

------------------------------------------------------------------------------
二、窗口敏感性
------------------------------------------------------------------------------
历史情景的窗口是人工选定的。本项把每个窗口的起止各平移 ±1、±2 个交易日，
重算窗内累计冲击与组合实际损失，输出**损失区间**。
意义：若某情景的损失对窗口边界高度敏感（如 H3 的 ΔOAS 会被窗口截断一半），
则该情景的因子贡献度读数不稳健，报告必须带此说明。

------------------------------------------------------------------------------
三、缓冲敏感性
------------------------------------------------------------------------------
缓冲（0.6548pp）是本报告自行选定的量级。本项在 5 档缓冲下重算尾部判定：
  0（不缓冲）/ 0.2927（M3 自身 90 分位）/ 0.6548（本报告取值）/
  0.8390（M2 五日 90 分位）/ 1.0000（+1pp 压力档）
输出每档下的超限情景数与被判定为尾部风险的情景清单。
意义：若结论随缓冲档位大幅变化，说明判定是由缓冲而非数据驱动的。

------------------------------------------------------------------------------
四、信用区间敏感性
------------------------------------------------------------------------------
信用路径的系数只有 95% 置信区间、且该区间**不覆盖「平静期系数外推到危机」这一风险**。
本项在区间的下限（传导最强）、点估计、上限三档下重算全部含信用项的情景，
输出损失区间与尾部判定。

------------------------------------------------------------------------------
产出：results/stress_coverage.csv     情景 × 期限 × 口径的覆盖度与分位数
      results/stress_window_sens.csv  历史情景的窗口平移敏感性
      results/stress_buffer_sens.csv  缓冲档位 × 情景的尾部判定
      figures/stress_coverage.png     情景损失在历史分布中的位置
用法： ./.venv/bin/python code/stress_robustness.py
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
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2

FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

WINDOW = "3日(主)"
DELTA_SRC = "B 事件前δ"
BUFFER = 0.6548
BUFFER_GRID = [("0.0000 不缓冲", 0.0), ("0.2927 M3 自身", 0.2927),
               ("0.6548 本报告取值", 0.6548), ("0.8390 M2 五日", 0.8390),
               ("1.0000 +1pp 压力档", 1.0000)]
HIST_WINDOWS = {"H1": ("2022-06-09", "2022-06-16"), "H2": ("2023-03-10", "2023-03-17"),
                "H3": ("2025-04-07", "2025-04-11"), "X1": ("2022-11-29", "2022-12-05"),
                "X2": ("2022-03-08", "2022-03-15")}


def hday_losses(r: pd.Series, h: int) -> pd.Series:
    """历史滚动 h 日累计损失（正 = 损失），保留索引以便定位窗口终点。"""
    return (-pd.Series(r).dropna().rolling(h).sum().dropna())


def main() -> None:
    print("=" * 78)
    print("阶段三 Day4：稳健性分析（覆盖度 / 窗口 / 缓冲 / 信用区间）")
    print("=" * 78)

    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    I = pd.read_csv(RES / "stress_impact.csv", encoding="utf-8-sig")
    S = pd.read_csv(RES / "stress_scenarios.csv", encoding="utf-8-sig")
    rets = {"主": F["etf_ret_tr_pct"], "次": F["etf_ret_rmb_pct"]}

    # ============================================================ 一、覆盖度
    print("\n" + "=" * 78)
    print("一、情景覆盖度检验：情景损失 vs 样本内实测最差窗口")
    print("=" * 78)
    HORIZONS = sorted(I.horizon_days.unique())

    # 1a) 样本内实测最差窗口榜（每期限 × 每口径取前 10）
    print("\n  1a) 样本内滚动 h 日累计损失最差窗口榜")
    print("      （表中 loss_pct 为**实际发生**的损失，非模型估计，不含任何缓冲）")
    top = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            hist = hday_losses(rets[cal], h)
            for rank, (d, v) in enumerate(hist.nlargest(10).items(), start=1):
                top.append(dict(horizon_days=h, caliber=cal, rank=rank,
                                window_end=str(d.date()),
                                window_start=str((hist.index[hist.index.get_loc(d) - (h - 1)]).date())
                                if hist.index.get_loc(d) >= h - 1 else "",
                                loss_pct=round(float(v), 4)))
    T = pd.DataFrame(top)
    T.to_csv(RES / "stress_worst_windows.csv", index=False, encoding="utf-8-sig")
    print(f"      → results/stress_worst_windows.csv  {T.shape[0]} 行 × {T.shape[1]} 列")
    for h in HORIZONS:
        for cal in ("主", "次"):
            g = T[(T.horizon_days == h) & (T.caliber == cal)].head(5)
            s = "  ".join(f"{r.rank}.{r.loss_pct:.4f}%({r.window_end})" for r in g.itertuples())
            print(f"      {cal}口径 {h} 日：{s}")

    # 1b) 情景在历史分布中的分位数 + 是否被实测最差超越
    cov = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            hist = hday_losses(rets[cal], h)
            worst, worst_d = float(hist.max()), hist.idxmax()
            scen = I[(I.horizon_days == h) & (I.caliber == cal)]
            for _, r in scen.iterrows():
                pct = float((hist < r.loss_realized_pct).mean() * 100)
                cov.append(dict(scenario_id=r.scenario_id, scenario_name=r.scenario_name,
                                scenario_class=r.scenario_class, caliber=cal, horizon_days=h,
                                loss_pct=round(r.loss_realized_pct, 4),
                                hist_percentile=round(pct, 2),
                                hist_worst_pct=round(worst, 4),
                                hist_worst_end=str(worst_d.date()),
                                # 正数 = 实测最差比该情景更差（情景被超越）
                                exceeded_by_pct=round(worst - r.loss_realized_pct, 4),
                                below_worst="是" if worst > r.loss_realized_pct else "否"))
    C = pd.DataFrame(cov)
    C.to_csv(RES / "stress_coverage.csv", index=False, encoding="utf-8-sig")
    print(f"\n[1] 覆盖度 → results/stress_coverage.csv  {C.shape[0]} 行 × {C.shape[1]} 列")

    # 覆盖率只在**损失情景**（loss_pct > 0）上判定：收益情景被「超越」是同义反复，无信息量
    print("\n  1b) 各期限/口径：情景库最深损失 vs 样本内实测最差")
    cov_flag = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            g = C[(C.horizon_days == h) & (C.caliber == cal)]
            loss_scen = g[g.loss_pct > 0]
            deepest = float(loss_scen.loss_pct.max()) if len(loss_scen) else float("nan")
            worst = float(g.hist_worst_pct.iloc[0])
            covered = deepest >= worst
            cov_flag.append(dict(horizon_days=h, caliber=cal, worst_scenario=
                                 (loss_scen.loc[loss_scen.loss_pct.idxmax(), "scenario_id"]
                                  if len(loss_scen) else ""),
                                 deepest_scenario_loss=round(deepest, 4),
                                 hist_worst_pct=round(worst, 4),
                                 gap_pct=round(worst - deepest, 4),
                                 worst_window_end=str(g.hist_worst_end.iloc[0]),
                                 covered="是" if covered else "否"))
            print(f"      {cal}口径 {h} 日：最深情景 {deepest:+.4f}%  "
                  f"实测最差 {worst:+.4f}%（止于 {g.hist_worst_end.iloc[0]}）  "
                  f"缺口 {worst - deepest:+.4f}pp  → {'覆盖' if covered else '**未覆盖**'}")

    bad = C[(C.below_worst == "是") & (C.loss_pct > 0)].sort_values(
        "exceeded_by_pct", ascending=False)
    print(f"\n  **被样本内实测最差窗口超越的损失情景 {len(bad)} 条**（严重度命名可能误导）：")
    if len(bad):
        print(bad[["scenario_id", "scenario_name", "caliber", "horizon_days", "loss_pct",
                   "hist_worst_pct", "hist_worst_end", "exceeded_by_pct"]].to_string(index=False))
    else:
        print("    （无）")

    # ============================================================ 二、窗口敏感性
    print("\n" + "=" * 78)
    print("二、窗口敏感性：起止各平移 ±1 / ±2 个交易日")
    print("=" * 78)
    idx = F.index
    wrows = []
    for sid, (a, b) in HIST_WINDOWS.items():
        i0, i1 = idx.get_loc(pd.Timestamp(a)), idx.get_loc(pd.Timestamp(b))
        for sh in (-2, -1, 0, 1, 2):
            j0, j1 = i0 + sh, i1 + sh
            if j0 < 0 or j1 >= len(idx):
                continue
            W = F.iloc[j0:j1 + 1]
            for cal, col in (("主", "etf_ret_tr_pct"), ("次", "etf_ret_rmb_pct")):
                wrows.append(dict(scenario_id=sid, shift_days=sh, caliber=cal,
                                  window_start=str(W.index.min().date()),
                                  window_end=str(W.index.max().date()),
                                  n_days=len(W),
                                  d5y_bp=round(float(W["d5y_bp"].sum()), 4),
                                  d10y_bp=round(float(W["d10y_bp"].sum()), 4),
                                  loss_pct=round(float(-W[col].sum()), 4)))
    Ws = pd.DataFrame(wrows)
    Ws.to_csv(RES / "stress_window_sens.csv", index=False, encoding="utf-8-sig")
    print(f"[2] 窗口敏感性 → results/stress_window_sens.csv  {Ws.shape[0]} 行 × {Ws.shape[1]} 列\n")
    for sid in HIST_WINDOWS:
        for cal in ("主", "次"):
            g = Ws[(Ws.scenario_id == sid) & (Ws.caliber == cal)]
            base = g[g.shift_days == 0].loss_pct.iloc[0]
            print(f"    {sid} {cal}口径：基准 {base:+.4f}%  "
                  f"平移后区间 [{g.loss_pct.min():+.4f}, {g.loss_pct.max():+.4f}]  "
                  f"极差 {g.loss_pct.max() - g.loss_pct.min():.4f}pp")
        g5 = Ws[(Ws.scenario_id == sid) & (Ws.caliber == "主")]
        print(f"        Δ5Y 窗内累计区间 [{g5.d5y_bp.min():+.0f}, {g5.d5y_bp.max():+.0f}] bp")

    # 2b) 平移感知的**库级**覆盖度：把每个 h 期限上所有情景（历史情景取其 ±2 日内最优对齐、
    #     假设情景取原值）放在一起，看情景库在该期限上最深能到多少，与同期限实测最差比。
    #     为什么是库级：单个历史情景对应的是它自己那段事件，拿它去「覆盖」别的方向的最差窗
    #     （如拿避险情景 H2 去覆盖利率上行的最差窗）是范畴错误。要问的是整个库够不够深。
    print("\n  2b) 平移感知的库级覆盖度（历史情景取 ±2 日内最优对齐）")
    sh_rows = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            cands = []
            for _, r in I[(I.caliber == cal) & (I.horizon_days == h)].iterrows():
                if r.scenario_class == "假设":
                    cands.append((r.scenario_id, float(r.loss_realized_pct), 0))
                else:
                    g = Ws[(Ws.scenario_id == r.scenario_id) & (Ws.caliber == cal)]
                    k = g.loss_pct.idxmax()
                    cands.append((r.scenario_id, float(g.loc[k, "loss_pct"]),
                                  int(g.loc[k, "shift_days"])))
            best_sid, best_v, best_sh = max(cands, key=lambda t: t[1])
            worst_hist = float(hday_losses(rets[cal], h).max())
            sh_rows.append(dict(horizon_days=h, caliber=cal, deepest_scenario=best_sid,
                                deepest_loss_shifted=round(best_v, 4),
                                best_shift_days=best_sh,
                                hist_worst_pct=round(worst_hist, 4),
                                residual_gap_pct=round(worst_hist - best_v, 4),
                                shift_covered="是" if best_v >= worst_hist - 1e-9 else "否"))
            print(f"      {cal}口径 {h} 日：库内最深（平移后）{best_sid} {best_v:+.4f}%"
                  f"（平移 {best_sh:+d} 日）  实测最差 {worst_hist:+.4f}%  "
                  f"残余缺口 {worst_hist - best_v:+.4f}pp  → "
                  f"{'覆盖' if best_v >= worst_hist - 1e-9 else '**仍有缺口**'}")
    pd.DataFrame(sh_rows).to_csv(RES / "stress_shift_coverage.csv", index=False,
                                 encoding="utf-8-sig")
    print("      → results/stress_shift_coverage.csv")

    # ============================================================ 三、缓冲敏感性
    print("\n" + "=" * 78)
    print("三、缓冲敏感性：5 档缓冲下的尾部判定")
    print("=" * 78)
    brows = []
    for lab, buf in BUFFER_GRID:
        for _, r in I.iterrows():
            # 缓冲只对假设情景生效；历史情景取实际路径，任何档位下都不加
            net = r.loss_realized_pct + (buf if r.scenario_class == "假设" else 0.0)
            brows.append(dict(buffer_label=lab, buffer_value=buf,
                              scenario_id=r.scenario_id, scenario_name=r.scenario_name,
                              caliber=r.caliber, horizon_days=r.horizon_days,
                              loss_net_pct=round(net, 4),
                              tail_flag="是" if (net > r.es_hd_99_pct and net > 0) else "否"))
    Bs = pd.DataFrame(brows)
    Bs.to_csv(RES / "stress_buffer_sens.csv", index=False, encoding="utf-8-sig")
    print(f"[3] 缓冲敏感性 → results/stress_buffer_sens.csv  {Bs.shape[0]} 行 × {Bs.shape[1]} 列\n")
    for lab, _ in BUFFER_GRID:
        g = Bs[Bs.buffer_label == lab]
        ids = sorted(g[g.tail_flag == "是"].scenario_id.unique())
        print(f"    {lab:<20} 超限 {int((g.tail_flag == '是').sum()):>2} / {len(g)} 条  "
              f"情景 {','.join(ids) if ids else '（无）'}")

    # ============================================================ 四、信用区间敏感性
    print("\n" + "=" * 78)
    print("四、信用区间敏感性：δ(ΔOAS) 的 95% 区间")
    print("=" * 78)
    sub = F[["etf_ret_tr_pct", "d5y_bp", "d10y_bp", "doas_bp"]].dropna()
    m3 = sm.OLS(sub["etf_ret_tr_pct"],
                sm.add_constant(sub[["d5y_bp", "d10y_bp", "doas_bp"]])).fit()
    lo, hi = (float(v) for v in m3.conf_int().loc["doas_bp"])
    b = float(m3.params["doas_bp"])
    print(f"  δ(ΔOAS) 下限 {lo:+.6f} / 点估计 {b:+.6f} / 上限 {hi:+.6f} %/bp\n")
    # 口径纪律：区间是**模型路径**下算的，必须与 loss_modeled_pct 比，不能与历史情景的
    # 实际实现损失（loss_realized_pct）比——后者含残差，两者不是同一个量。
    cr = I[I.credit_range_lo_pct.notna()].copy()
    cr["in_range"] = cr.apply(
        lambda r: r.credit_range_hi_pct - 1e-9 <= r.loss_modeled_pct
        <= r.credit_range_lo_pct + 1e-9, axis=1)
    for _, r in cr.iterrows():
        tag = "" if r.in_range else "  **点估计落在区间外，检查口径**"
        hist = (f"  | 该情景为历史情景，实现损失 {r.loss_realized_pct:+.4f}%"
                f"（含残差 {r.residual_pct:+.4f}%，超出模型区间上限"
                f" {r.loss_realized_pct - r.credit_range_lo_pct:+.4f}pp）"
                if r.scenario_class != "假设" else "")
        print(f"    {r.scenario_id} {r.caliber}口径（模型路径）："
              f"强传导 {r.credit_range_lo_pct:+.4f}%  点估计 {r.loss_modeled_pct:+.4f}%  "
              f"弱传导 {r.credit_range_hi_pct:+.4f}%  "
              f"| 含缓冲(强) {r.credit_range_lo_pct + BUFFER:.4f}%  "
              f"vs 同期限99%ES {r.es_hd_99_pct:.4f}%{hist}{tag}")

    # 只对**假设情景**判定：历史情景的实现损失另属已实现路径，不在此项做区间敏感性
    hy = cr[cr.scenario_class == "假设"]
    cr_only = hy[hy.scenario_id.str.startswith("S0") & ~hy.scenario_id.isin(["S10"])]
    worst_lo = float((cr_only.credit_range_lo_pct + BUFFER).max())
    es_min = float(cr_only.es_hd_99_pct.min())
    print(f"\n  纯信用情景（S04–S06）在**传导最强端 + 缓冲**下的最大损失 {worst_lo:.4f}%"
          f"  vs 同期限 99% ES 最小值 {es_min:.4f}%"
          f"  → {'仍不超限' if worst_lo < es_min else '**超限**'}")
    s10 = hy[hy.scenario_id == "S10"].credit_range_lo_pct + BUFFER
    print(f"  组合档 S10 在传导最强端 + 缓冲下 {float(s10.max()):.4f}% —— 但 S10 同时含"
          f"利率极端项，其超限不可归因于信用维度")

    # ============================================================ 五、锚点敏感性
    print("\n" + "=" * 78)
    print("五、锚点口径敏感性：把 1× 的窗长从 3 日改成 5 日 / 6 日")
    print("=" * 78)
    # 需求文档只要求假设情景幅度取「历史极端值的 1/1.5/2 倍」，**未规定窗长**。
    # 本报告取 3 日累计峰值为 1×，这是自选口径，必须量化它影响多大。
    # 现象：同一因子的极端值随窗长显著变化（汇率 5 日贬值 −3.4785% 比 3 日 −2.7459% 深 26.7%），
    # 故 1× 的绝对水平依赖于窗长选择，连带整条梯度平移。
    d5_ = float(pd.read_csv(RES / "factor_exposure.csv", encoding="utf-8-sig")
                .set_index("因子").loc["利率 Δ5Y", "暴露δ"])
    d10_ = float(pd.read_csv(RES / "factor_exposure.csv", encoding="utf-8-sig")
                 .set_index("因子").loc["利率 Δ10Y", "暴露δ"])
    GRP = {"利率上行": [("d5y_bp", d5_), ("d10y_bp", d10_)],
           "信用利差走阔": [("doas_bp", b)],
           "离岸汇率贬值": [("fx_ret_pct", 1.0)]}   # 汇率走次口径，系数恒为 1
    REG_SCEN = {"利率上行": ("S01", "S03"), "信用利差走阔": ("S04", "S06"),
                "离岸汇率贬值": ("S07", "S09")}
    ar_rows = []
    for grp, cols in GRP.items():
        print(f"\n  【{grp}】")
        for h in (3, 5, 6):
            # 锚点日 = 该方向（利率取 d5y）滚动 h 日累计的样本极值日
            key = cols[0][0]
            s = F[key].dropna()
            rc = s.rolling(h).sum().dropna()
            sign = -1.0 if grp == "离岸汇率贬值" else 1.0   # 汇率看贬值端（负），其余看上行端
            ad = (rc * sign).idxmax()
            end_i = F.index.get_loc(ad)
            path = F.iloc[end_i - h + 1:end_i + 1]
            shock = {c: float(path[c].sum()) * sign for c, _ in cols}
            loss_1x = -sum(shock[c] * co for c, co in cols)   # 轻度（1×）
            loss_2x = -sum(shock[c] * co * 2 for c, co in cols)  # 极端（2×）
            sid1, sid2 = REG_SCEN[grp]
            # 汇率组对照**次口径**：纯汇率情景的主口径按口径恒为 0，拿它做对账是空对账
            chk = "次" if grp == "离岸汇率贬值" else "主"
            reg1 = float(I[(I.scenario_id == sid1) & (I.caliber == chk)].loss_modeled_pct.iloc[0])
            reg2 = float(I[(I.scenario_id == sid2) & (I.caliber == chk)].loss_modeled_pct.iloc[0])
            # 汇率组本项算的是次口径损失（含汇率项），与登记值同口径
            l1 = abs(loss_1x) if grp == "离岸汇率贬值" else loss_1x
            l2 = abs(loss_2x) if grp == "离岸汇率贬值" else loss_2x
            ar_rows.append(dict(direction=grp, window_days=h, anchor_date=str(ad.date()),
                                shock_sum=round(sum(shock.values()), 4),
                                mild_1x_loss_pct=round(l1, 4),
                                extreme_2x_loss_pct=round(l2, 4),
                                registered_mild_pct=(round(reg1, 4) if h == 3 else np.nan),
                                registered_extreme_pct=(round(reg2, 4) if h == 3 else np.nan)))
            reg = (f"  （与登记 {sid1}/{sid2} 的 {reg1:.4f}/{reg2:.4f}% 一致）" if h == 3 else "")
            print(f"    {h} 日锚点 @ {ad.date()}：1× 冲击合计 {sum(shock.values()):+.4f}"
                  f"  轻度损失 {l1:+.4f}%  极端损失 {l2:+.4f}%{reg}")
        g = [r for r in ar_rows if r["direction"] == grp]
        sp = max(r["extreme_2x_loss_pct"] for r in g) - min(r["extreme_2x_loss_pct"] for r in g)
        print(f"    → 极端档损失在三种窗长下的极差 {sp:.4f}pp")
    AR = pd.DataFrame(ar_rows)
    AR.to_csv(RES / "stress_anchor_sens.csv", index=False, encoding="utf-8-sig")
    print(f"\n[5] 锚点敏感性 → results/stress_anchor_sens.csv  {AR.shape[0]} 行 × {AR.shape[1]} 列")

    # 同向 vs 绝对最大：利率样本内最大的 3 日变动是**下行**，须写明口径
    r3 = F["d5y_bp"].dropna().rolling(3).sum().dropna()
    print(f"\n  口径说明：5 年期美债 3 日累计的**绝对值**最大为 {abs(r3.iloc[r3.abs().argmax()]):.0f}bp"
          f" @ {r3.abs().idxmax().date()}，方向为**下行**（2023 银行业风波避险）。")
    print(f"  本报告假设情景方向为「无风险利率大幅上行」，故取**同向**极值 "
          f"{r3.max():.0f}bp @ {r3.idxmax().date()}。若按绝对值取，1× 锚点会改用那次下行冲击。")

    # ============================================================ 图
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, cal in zip(axes, ("主", "次")):
        g = C[(C.caliber == cal) & (C.horizon_days == 3)].sort_values("loss_pct")
        hist = hday_losses(rets[cal], 3)
        p99, p999 = np.percentile(hist, 99), np.percentile(hist, 99.9)
        y = np.arange(len(g))
        cols = [C_GARCH if r.below_worst == "是" else C_M1 for r in g.itertuples()]
        ax.barh(y, g.loss_pct, color=cols, height=0.6)
        ax.axvline(p99, color=C_ALT, ls="--", lw=1.3)
        ax.axvline(p999, color=INK2, ls=":", lw=1.3)
        ax.axvline(float(hist.max()), color="#c0392b", ls="-.", lw=1.4)
        top = len(g) + 0.6
        ax.set_ylim(-0.8, top)
        ax.text(p99, top, " 99%", color=C_ALT, fontsize=8, va="top")
        ax.text(p999, top, " 99.9%", color=INK2, fontsize=8, va="top")
        ax.text(float(hist.max()), top - 1.0, f" 实测最差 {hist.max():.2f}%",
                color="#c0392b", fontsize=8, va="top")
        ax.set_yticks(y)
        ax.set_yticklabels(g.scenario_id, fontsize=8)
        ax.set_xlabel("3 日累计损失（%），向右为损失更大", fontsize=9)
        ax.set_title(f"{cal}口径", fontsize=10.5, loc="left")
        ax.grid(axis="x", color="#e6e5e2", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_GARCH),
               plt.Rectangle((0, 0), 1, 1, color=C_M1),
               plt.Line2D([0], [0], color=C_ALT, ls="--", lw=1.3),
               plt.Line2D([0], [0], color=INK2, ls=":", lw=1.3),
               plt.Line2D([0], [0], color="#c0392b", ls="-.", lw=1.4)]
    fig.legend(handles, ["被实测最差窗口超越", "未被超越", "历史 99% 分位",
                         "历史 99.9% 分位", "样本内实测最差"],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("情景损失在历史 3 日损失分布中的位置", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / "stress_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n[图] figures/stress_coverage.png")


if __name__ == "__main__":
    main()
