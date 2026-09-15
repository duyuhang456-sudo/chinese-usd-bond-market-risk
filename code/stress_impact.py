"""
阶段三 Day2–Day3（9/22–9/23）：三路径情景映射 + 压力测试测算

消费 `results/stress_scenarios.csv`（9/21 产出），把每个情景的因子冲击映射为组合损失，
并按需求文档第四章第三阶段完成三项测算：潜在损失与最大回撤、因子贡献度分析、
风险承受能力评估与尾部风险点识别。

------------------------------------------------------------------------------
一、三条映射路径（**不得共用同一套 δ**）
------------------------------------------------------------------------------
① 利率路径 —— 沿用基准 δ 线性映射
   系数取自 `results/factor_exposure.csv`（主口径 USD 复权：Δ5Y −0.007714、Δ10Y −0.029303 %/bp，
   等效久期 3.70 年，R² = 0.757）。
   **可外推的依据**：9/18 实测偏差与窗口内利率冲击幅度的相关系数仅 +0.151
   （`docs/phase2_model_validation.md` §7.3），即偏差不随冲击变大而系统性恶化，
   没有证据表明线性传导会在更极端冲击下加速失效。**该性质支持 1.5 倍与 2 倍梯度**。
   不使用 GARCH 做多期外推：α+β 末期 0.99597（主）/ 0.99682（次）、
   全样本中位数 0.99212 / 0.99399，均近 IGARCH，长期方差不存在。

② 信用路径 —— **另建，不沿用利率那套 δ**
   现行 δ 的第三项（信用利差代理）是利率双因子回归的**残差本身**、系数恒为 1，属代数恒等式
   （实测 max|估算−实际| = 8.79e-05 pp），不构成可外推的暴露。故本脚本**重新估计**一个
   利率 + ΔOAS 三因子模型（对应 9/18 的 M3），取其 ΔOAS 系数。
   **必须随点估计一并给出区间**：该系数只在 OAS 可得的子样本（2023-09-05 起）上估出，
   而该子样本**不含 2022 年信用冲击**（偏差最大的事件全部落在 2022-03 与 2022-11）——
   给单点估计等于把一段平静期的系数外推到危机，是把「未验证」说成「已验证」。
   区间取 OLS 的 95% 置信区间，并显式标注为「假设性传导」。

③ 汇率路径 —— 口径搬运，不是估计
   次口径汇率系数恒为 **+1**：`r_rmb ≡ r_tr + fx_ret`，故 ∂r_rmb/∂fx = 1，无需估计、无估计误差。
   主口径 δ_fx = 0（美元计价，汇率不影响组合的 USD 收益）。
   **汇率冲击不施加偏差缓冲**——9/18 实测主次口径传导偏差的最大逐值差 = 0.000e+00
   （汇率项在偏差中精确对消）；对恒等式再叠一层缓冲是无依据的加码。
   但**次口径必须给汇率情景**：汇率占次口径方差 44.23%（`results/var_attribution.csv`），
   漏掉等于少算近一半风险来源。
   **纯汇率情景下主口径损失恒为 0**——这不是「没算」，而是口径的事实：美元计价组合的
   USD 收益不含汇率项。表中如实保留 0 并标注，不用缓冲把它凑成非零。

------------------------------------------------------------------------------
二、亏损侧偏差缓冲（量级来自 9/18 实测，不是拍脑袋留的安全垫）
------------------------------------------------------------------------------
`results/delta_transmission_summary.csv` 中「主口径 × 3日(主) × 事件前δ」行的 **90 分位绝对偏差**：
  M2（利率双因子）= **0.6548 pp**  → 利率路径
  M3（利率+ΔOAS）  = **0.2927 pp**  → **不采用**，理由见下

**信用路径为什么不用 M3 自己那个更小的数**：M3 的偏差确实更小（0.2927 vs 0.6548），
但这个改善只在 2023-09 之后的平静子样本上成立。信用情景恰恰是**外推到这个子样本从未见过的
冲击幅度**，用一个「没见过危机的样本」估出的偏差去给危机情景定缓冲，方向是反的。
故信用路径与利率路径同取 **0.6548 pp**（更保守的一侧）。

**缓冲按情景施加一次，不按路径累加**：它是「δ 估算值 vs 实际损失」这一整体偏差的保证金，
不是每条因子路径各留一份。故 `buffer = max(适用路径的缓冲)`；纯汇率情景无 δ 估计，缓冲为 0。

取 90 分位而非均值/中位数的理由（`delta_transmission_summary.csv` 对应行 + 阶段二报告 §7.3）：
全 85 个事件窗上偏差中位数仅 **+0.0550pp**、均值 **+0.0896pp**——正负两侧相互抵消，
**用中心趋势会把真实误差量级抹掉**。而收敛到**实际发生亏损的 33 个事件窗**上，
偏差分布明显偏向一侧：**58% 低估损失，平均低估 0.1219pp**（全样本该行 `under_share` 40%、
`under_mean` −0.2301pp）。压力测试只关心亏损侧，故取**亏损侧更保守**的 90 分位绝对偏差
0.6548pp，而不用中心趋势那个近乎为零的数。

------------------------------------------------------------------------------
三、最大回撤与因子贡献度的口径
------------------------------------------------------------------------------
**最大回撤 ≠ 潜在损失**（前者是路径峰谷差、后者是情景终点损失），本脚本分别计算：
  · 历史情景与补充情景：用窗内**实际逐日路径**；
  · 假设情景：把该方向锚点日的**实际 3 日因子路径**按倍数放大后映射回组合收益路径。
    意义在于——假设情景若只用单点冲击，路径退化成一个点、回撤恒等于损失，指标就没有信息量。
收益是 log 收益（`max|rmb − tr − fx| = 2.2e-16`），故**累计收益 = 直接求和**，
回撤在 exp(累计) 上算，且**必须含事件前起点 NAV = 1.0**
（只从窗内首个收益起算会漏掉「事件首日即最高点」，实测把 H3 的 MDD 低估一半以上）。

**因子贡献度必须单列「未解释残差」**：历史情景的实际损失与 δ 模型隐含损失**不相等**——
9/18 实测偏差的 88.1% 来自利率双因子之外的信用维度。若只把损失拆成「Δ5Y + Δ10Y + 汇率」，
残差会被摊进各因子的份额里，等于把模型拟合误差伪装成因子贡献。
故贡献度分解为：Δ5Y / Δ10Y / ΔOAS / 汇率 / **未解释残差**（历史情景；假设情景该列恒为 0，
因其损失本身就是模型输出）。

------------------------------------------------------------------------------
四、与基准口径对标的口径警告
------------------------------------------------------------------------------
基准 VaR/ES 是 **1 日** horizon（`results/backtest_results.csv`，HS250 @ common773）；
情景损失是 **3 日及以上** horizon。故 `x_var95` / `x_var99` / `x_es99` 是
**「相当于多少倍 1 日 VaR」的严重度参照，不是同期同类比较**。
不做 √t 缩放对齐：本样本近 IGARCH 且存在波动聚集，√t 的 iid 前提不成立，
缩放会把一个未经验证的换算当成可比性依据。报告引用这三个倍数时必须带此说明。

------------------------------------------------------------------------------
产出：results/stress_impact.csv           逐情景 × 口径的损失/回撤/缓冲/对标（长表）
      results/stress_factor_contrib.csv   因子贡献度分解（含未解释残差）
      figures/stress_mapping.png          三路径映射与适用边界
      figures/stress_loss_vs_var.png      情景损失 vs 基准口径 HS250 的 VaR/ES
      figures/stress_factor_contrib.png   因子贡献度
用法： ./.venv/bin/python code/stress_impact.py
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

from common import REPO, write_table
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2

FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 常量
BUFFER_RATE = 0.6548        # pp，M2 × 主口径 × 3日 × 事件前δ 的 90 分位绝对偏差
BUFFER_CREDIT = 0.6548      # pp，同取利率侧保守值，理由见 docstring 第二节
BUFFER_FX = 0.0             # 口径搬运，恒等式不加缓冲

BASE_SCOPE = "common773"    # 基准样本（跨模型横比禁用 full）
BASE_MODEL = "HS250"        # 9/18 定案的唯一基准口径
RATE_COLS = ["d5y_bp", "d10y_bp"]

FACTOR_CN = {"d5y_bp": "5年期美债收益率", "d10y_bp": "10年期美债收益率",
             "doas_bp": "新兴市场IG信用利差", "fx_ret_pct": "人民币兑美元汇率",
             "residual": "未解释残差（信用维度）"}


def horizon_es(r: pd.Series, h: int) -> dict:
    """经验 h 日累计**损失**分布的分位数与 ES（重叠窗口）。

    为什么不把 1 日 VaR 乘 √h：本项目近 IGARCH（α+β≈0.996）且存在波动聚集，
    √t 的 iid 前提不成立，缩放会把一个未经验证的换算当成可比性依据。
    直接取历史 h 日累计损失的经验分布，无分布假设，与情景损失同为「h 日收益之和」，
    是同类比较。代价是重叠窗口自相关、有效样本少于名义条数，故只作参照线不做检验。
    """
    r = pd.Series(r).dropna().reset_index(drop=True)
    losses = -r.rolling(h).sum().dropna()
    out = {}
    for q in (95, 99):
        var = float(np.percentile(losses, q))
        out[q] = dict(var=var, es=float(losses[losses >= var].mean()))
    return out


def path_cum_mdd(r: pd.Series) -> tuple[float, float]:
    """log 收益路径 → (累计收益%, 最大回撤%)，回撤含事件前起点 NAV=1.0。"""
    r = pd.Series(r).dropna().reset_index(drop=True)
    cum = r.cumsum() / 100.0
    eq = pd.concat([pd.Series([1.0]), np.exp(cum)], ignore_index=True)
    return float(r.sum()), float((eq / eq.cummax() - 1.0).min() * 100.0)


def main() -> None:
    print("=" * 78)
    print("阶段三 Day2–Day3：三路径映射 + 压力测试测算")
    print("=" * 78)

    # ---------------------------------------------------------- 1) 载入
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    S = pd.read_csv(RES / "stress_scenarios.csv", encoding="utf-8-sig")
    E = pd.read_csv(RES / "factor_exposure.csv", encoding="utf-8-sig")
    B = pd.read_csv(RES / "backtest_results.csv", encoding="utf-8-sig")

    d5 = float(E.loc[E["因子"] == "利率 Δ5Y", "暴露δ"].iloc[0])
    d10 = float(E.loc[E["因子"] == "利率 Δ10Y", "暴露δ"].iloc[0])
    print(f"[输入] stress_scenarios.csv  {S.shape[0]} 行 / {S.scenario_id.nunique()} 情景")
    print(f"[输入] factor_exposure.csv   δ(Δ5Y)={d5:+.6f}  δ(Δ10Y)={d10:+.6f} %/bp")

    # ---------------------------------------------------------- 2) 信用系数
    print("\n" + "=" * 78)
    print("信用路径系数：重新估计 M3（利率 + ΔOAS），并给 95% 区间")
    print("=" * 78)
    sub = F[["etf_ret_tr_pct", "d5y_bp", "d10y_bp", "doas_bp"]].dropna()
    m3 = sm.OLS(sub["etf_ret_tr_pct"],
                sm.add_constant(sub[["d5y_bp", "d10y_bp", "doas_bp"]])).fit()
    b_oas = float(m3.params["doas_bp"])
    ci_lo, ci_hi = (float(v) for v in m3.conf_int().loc["doas_bp"])
    print(f"  子样本 n={len(sub)}  {sub.index.min().date()} ~ {sub.index.max().date()}")
    print(f"  δ(ΔOAS) = {b_oas:+.6f} %/bp  95%CI=[{ci_lo:+.6f}, {ci_hi:+.6f}]  R²={m3.rsquared:.4f}")
    print(f"  !! 子样本不含 2022 年信用冲击 → 只作**假设性传导**，报告一律给区间")
    print(f"  信用缓冲取 {BUFFER_CREDIT:.4f} pp（利率侧保守值，非 M3 自身的 0.2927）")

    # ---------------------------------------------------------- 3) 基准口径
    bb = B[(B["model"] == BASE_MODEL) & (B["scope"] == BASE_SCOPE)]
    base = {}
    for cal in ("主", "次"):
        for conf in (95, 99):
            r = bb[(bb["caliber"] == cal) & (bb["conf"] == conf)].iloc[0]
            base[(cal, conf)] = dict(var=float(r["mean_var"]), es=float(r["es"]))
    print(f"\n[输入] 基准口径 {BASE_MODEL} @ {BASE_SCOPE}（**1 日** horizon）")
    for k, v in base.items():
        print(f"       {k[0]}口径 {k[1]}%  VaR={v['var']:.4f}%  ES={v['es']:.4f}%")
    print("       注意：情景损失是 3 日及以上 horizon，对标倍数只作严重度参照")

    # 硬断言对账：基准数字已登记在阶段二交付报告（docs/phase2_model_validation.md §7.2 表），
    # 上游若漂移必须在此处暴露，不允许静默沿用。
    REGISTERED = {("主", 95): (0.3544, 0.4473), ("主", 99): (0.5361, 0.6659),
                  ("次", 95): (0.4454, 0.5554), ("次", 99): (0.7260, 0.7024)}
    for k, (rv, re_) in REGISTERED.items():
        assert abs(base[k]["var"] - rv) < 5e-5 and abs(base[k]["es"] - re_) < 5e-5, (
            f"基准对账失败 {k}：CSV 给 VaR={base[k]['var']:.4f}/ES={base[k]['es']:.4f}，"
            f"阶段二报告登记 VaR={rv}/ES={re_}")
    print("       [对账] 四个基准配置与阶段二报告 §7.2 登记值一致")

    # ---------------------------------------------------------- 4) 逐情景测算
    print("\n" + "=" * 78)
    print("逐情景测算（损失为正；历史/补充用实际路径，假设用锚点日 3 日路径×倍数）")
    print("=" * 78)

    # 同期限经验分布所需的收益序列（主 = 美元复权，次 = 人民币复权）
    ret_series = {"主": F["etf_ret_tr_pct"], "次": F["etf_ret_rmb_pct"]}

    rows, crows = [], []

    def add(sid, cls, name, direction, sev, anchor_date,
            rate_path, credit_path, fx_path,
            realized_main=None, realized_rmb=None, horizon=3):
        def col_of(df, c):
            if df is None or c not in df.columns:
                return pd.Series(dtype=float)
            return pd.Series(df[c].values, dtype=float)

        a5, a10 = col_of(rate_path, "d5y_bp"), col_of(rate_path, "d10y_bp")
        ao = col_of(credit_path, "doas_bp")
        afx = col_of(fx_path, "fx_ret_pct")
        n = max(len(a5), len(a10), len(ao), len(afx), 1)
        a5 = a5.reindex(range(n), fill_value=0.0)
        a10 = a10.reindex(range(n), fill_value=0.0)
        ao = ao.reindex(range(n), fill_value=0.0)
        afx = afx.reindex(range(n), fill_value=0.0)

        r_rate = a5 * d5 + a10 * d10
        r_cr_pt, r_cr_lo, r_cr_hi = ao * b_oas, ao * ci_lo, ao * ci_hi

        has_rate = float(a5.abs().sum() + a10.abs().sum()) > 1e-12
        has_cr = float(ao.abs().sum()) > 1e-12
        has_fx = float(afx.abs().sum()) > 1e-12
        # 缓冲**只给假设情景**：历史情景用的是窗内实际收益，不含 δ 估算误差，
        # 给它加偏差缓冲等于对已知结果重复计一次模型误差。
        # 假设情景的损失本身由 δ 映射得出，才需要亏损侧偏差保证金。
        # 施加上按情景一次（取适用路径中较大者），不按路径累加。
        buf = (max(BUFFER_RATE if has_rate else 0.0,
                   BUFFER_CREDIT if has_cr else 0.0,
                   BUFFER_FX if has_fx else 0.0) if cls == "假设" else 0.0)

        for cal in ("主", "次"):
            fx_term = afx if cal == "次" else afx * 0.0
            r_model = r_rate + r_cr_pt + fx_term
            realized = realized_main if cal == "主" else realized_rmb
            hist = realized is not None and len(realized) > 0
            r_used = pd.Series(realized).dropna().reset_index(drop=True) if hist else r_model

            loss = -path_cum_mdd(r_used)[0]
            mdd = path_cum_mdd(r_used)[1]
            loss_model = -float(r_model.sum())
            residual = loss - loss_model if hist else 0.0

            lo_loss = -path_cum_mdd(r_rate + r_cr_lo + fx_term)[0]
            hi_loss = -path_cum_mdd(r_rate + r_cr_hi + fx_term)[0]

            v95, v99 = base[(cal, 95)], base[(cal, 99)]
            net = loss + buf
            # 同期限经验分布：与情景损失同为「h 日 log 收益之和」，是**可比**的参照；
            # 1 日基准只作「相当于几倍日 VaR」的严重度直觉，不作判定。
            hd = horizon_es(ret_series[cal], horizon)
            rows.append(dict(
                scenario_id=sid, scenario_class=cls, scenario_name=name,
                direction=direction, severity=sev, anchor_date=anchor_date, caliber=cal,
                horizon_days=horizon,
                loss_realized_pct=round(loss, 4),
                loss_modeled_pct=round(loss_model, 4),
                residual_pct=round(residual, 4),
                buffer_pct=round(buf, 4), loss_after_buffer_pct=round(net, 4),
                buffer_applied=("假设情景 δ 映射，加亏损侧偏差缓冲" if buf > 0
                                else ("历史情景取实际路径，无模型误差不加缓冲" if cls != "假设"
                                      else "纯汇率情景口径搬运，恒等式不加缓冲")),
                mdd_pct=round(mdd, 4),
                credit_range_lo_pct=round(lo_loss, 4) if has_cr else np.nan,
                credit_range_hi_pct=round(hi_loss, 4) if has_cr else np.nan,
                var_1d_95_pct=round(v95["var"], 4), var_1d_99_pct=round(v99["var"], 4),
                es_1d_99_pct=round(v99["es"], 4),
                var_hd_95_pct=round(hd[95]["var"], 4), var_hd_99_pct=round(hd[99]["var"], 4),
                es_hd_99_pct=round(hd[99]["es"], 4),
                x_var95=round(net / v95["var"], 3), x_var99=round(net / v99["var"], 3),
                x_es99=round(net / v99["es"], 3),
                x_hd_es99=round(net / hd[99]["es"], 3),
                tail_flag_hd="是" if net > hd[99]["es"] else "否",
                tail_flag="是" if net > v99["es"] else "否",
                # 「纯」汇率情景 = 汇率是**唯一**驱动。历史情景窗内也含汇率项，
                # 若只用 has_fx 判定，H1/H2/H3/X1/X2 的主口径行会被误贴该注释。
                fx_horizon_note=("纯汇率情景，主口径按口径恒为 0"
                                 if (has_fx and not has_rate and not has_cr and cal == "主")
                                 else ""),
            ))

            # 因子贡献度：模型隐含部分按因子拆，实际与模型的差单列残差
            c5, c10 = -float((a5 * d5).sum()), -float((a10 * d10).sum())
            ccr, cfx = -float(r_cr_pt.sum()), -float(fx_term.sum())
            parts = [("d5y_bp", c5), ("d10y_bp", c10), ("doas_bp", ccr),
                     ("fx_ret_pct", cfx), ("residual", residual)]
            tot = sum(abs(v) for _, v in parts)
            for fac, val in parts:
                crows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                                  direction=direction, severity=sev, caliber=cal,
                                  factor=fac, factor_cn=FACTOR_CN[fac],
                                  contribution_pct=round(val, 4),
                                  share_pct=round(abs(val) / tot * 100, 2) if tot > 1e-12 else 0.0))

        # 进度行（与落表口径一致：历史走实际路径，假设走映射路径）
        p_main = (pd.Series(realized_main).dropna().reset_index(drop=True)
                  if (realized_main is not None and len(realized_main) > 0) else r_rate + r_cr_pt)
        p_rmb = (pd.Series(realized_rmb).dropna().reset_index(drop=True)
                 if (realized_rmb is not None and len(realized_rmb) > 0)
                 else r_rate + r_cr_pt + afx)
        l_main, d_main = path_cum_mdd(p_main)
        l_rmb = path_cum_mdd(p_rmb)[0]
        print(f"  [{sid}] {name:26s} 主口径={-l_main:+7.4f}%  次口径={-l_rmb:+7.4f}%"
              f"  回撤={d_main:+7.4f}%  缓冲={buf:.4f}pp")

    # ---- 历史与补充情景：实际路径
    # 情景清单从 CSV 读，不在此硬编码——否则情景库扩容（如 9/24 补入 X2）会被静默漏算。
    HIST_IDS = (S[(S.measure == "cum_window") & (S.scenario_class.isin(["历史", "补充"]))]
                .scenario_id.drop_duplicates().tolist())
    for sid in HIST_IDS:
        g = S[(S.scenario_id == sid) & (S.measure == "cum_window")]
        a, b = g.window_start.iloc[0], g.window_end.iloc[0]
        W = F.loc[a:b]
        add(sid, g.scenario_class.iloc[0], g.scenario_name.iloc[0],
            g.lead_direction.iloc[0], "-", str(W.index.max().date()),
            pd.DataFrame({"d5y_bp": W["d5y_bp"].values, "d10y_bp": W["d10y_bp"].values}),
            pd.DataFrame({"doas_bp": W["doas_bp"].fillna(0.0).values}),
            pd.DataFrame({"fx_ret_pct": W["fx_ret_pct"].fillna(0.0).values}),
            realized_main=W["etf_ret_tr_pct"].dropna(),
            realized_rmb=W["etf_ret_rmb_pct"].dropna(),
            horizon=len(W))

    # ---- 假设情景：锚点日 3 日路径 × 倍数
    for sid in sorted(S[S.scenario_class == "假设"].scenario_id.unique()):
        g = S[S.scenario_id == sid]
        nm, lead = g.scenario_name.iloc[0], g.lead_direction.iloc[0]
        mult, sev = float(g.multiple.iloc[0]), g.scenario_name.iloc[0].split("·")[-1]
        paths, ad = {}, []
        for _, r in g.iterrows():
            col, dt = r["factor"], pd.Timestamp(r["anchor_date"])
            idx = F.index[F.index <= dt][-3:]
            paths[col] = pd.Series(F.loc[idx, col].values * mult)
            ad.append(str(dt.date()))
        add(sid, "假设", nm, lead, sev, "/".join(sorted(set(ad))),
            pd.DataFrame({c: paths[c] for c in RATE_COLS if c in paths}),
            pd.DataFrame({"doas_bp": paths["doas_bp"]}) if "doas_bp" in paths else None,
            pd.DataFrame({"fx_ret_pct": paths["fx_ret_pct"]}) if "fx_ret_pct" in paths else None)

    I = pd.DataFrame(rows)
    C = pd.DataFrame(crows)
    # 完整性断言：情景库里的每个情景都必须被测算到，缺一个即终止。
    # 该断言的存在理由：情景清单曾在第 4 节里被硬编码，导致 9/24 补入的 X2 静默漏算，
    # 覆盖度检验才发现。让「漏算」无法再静默通过。
    _missing = set(S.scenario_id.unique()) - set(I.scenario_id.unique())
    assert not _missing, (f"情景库有 {S.scenario_id.nunique()} 个情景，"
                          f"以下未进入测算：{sorted(_missing)}")
    print(f"\n  [对账] 情景库 {S.scenario_id.nunique()} 个情景全部进入测算（无静默漏算）")
    write_table(I, RES / "stress_impact.csv")
    write_table(C, RES / "stress_factor_contrib.csv")
    print(f"\n[1] 测算表 → results/stress_impact.csv  {I.shape[0]} 行 × {I.shape[1]} 列")
    print(f"[2] 因子贡献度 → results/stress_factor_contrib.csv  {C.shape[0]} 行 × {C.shape[1]} 列")

    # ---------------------------------------------------------- 5) 尾部风险点
    print("\n" + "=" * 78)
    print("风险承受能力评估（1/2）：同期限经验分布作参照线")
    print("=" * 78)
    for cal in ("主", "次"):
        for h in sorted(I.horizon_days.unique()):
            g = I[(I.caliber == cal) & (I.horizon_days == h)]
            if not len(g):
                continue
            print(f"  {cal}口径 · {h} 日：99%VaR={g.var_hd_99_pct.iloc[0]:.4f}%  "
                  f"99%ES={g.es_hd_99_pct.iloc[0]:.4f}%  （95%VaR={g.var_hd_95_pct.iloc[0]:.4f}%）")

    print("\n" + "=" * 78)
    print("风险承受能力评估（2/2）：损失（含缓冲）超过**同期限** 99% ES —— 尾部风险点")
    print("=" * 78)
    T = I[(I.tail_flag_hd == "是") & (I.loss_after_buffer_pct > 0)][
        ["scenario_id", "scenario_name", "caliber", "horizon_days",
         "loss_after_buffer_pct", "es_hd_99_pct", "x_hd_es99", "mdd_pct"]]
    print(T.to_string(index=False) if len(T) else "  （无）")
    # 分母从测算表取，不写死——情景库扩容后写死的分母会静默变成错的
    print(f"\n  超出同期限 99% ES 的情景 {len(T)} / {len(I)} 条（口径×情景，"
          f"{I.scenario_id.nunique()} 个情景）")
    print(f"  对照：若改用 1 日 99% ES 作阈值，超出 {int(((I.tail_flag == '是') & (I.loss_after_buffer_pct > 0)).sum())} 条 ——"
          f" 期限不匹配会把几乎所有情景判为超限，判定失去区分度，故不作结论")
    n_neg = int((I.loss_realized_pct < 0).sum())
    print(f"\n  反向（收益）情景 {n_neg} 条 —— 已如实保留，不并入损失统计")

    # ---------------------------------------------------------- 6) 三张图
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    for ax, (col, cn, c) in zip(axes, [("d5y_bp", "5年期美债收益率", C_M1),
                                       ("d10y_bp", "10年期美债收益率", C_GARCH),
                                       ("doas_bp", "新兴市场IG信用利差", C_ALT)]):
        grid = np.linspace(0, max(F[col].dropna().abs().max() * 2.2, 1.0), 50)
        if col == "doas_bp":
            ax.plot(grid, grid * b_oas, color=c, lw=2.0, label="M3 点估计")
            ax.fill_between(grid, grid * ci_lo, grid * ci_hi, color=c, alpha=0.18,
                            label="95% 区间（假设性传导）")
        else:
            ax.plot(grid, grid * (d5 if col == "d5y_bp" else d10), color=c, lw=2.0,
                    label="基准 δ 线性映射")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"{cn} 冲击（bp）", fontsize=9)
        ax.set_ylabel("组合收益影响（%）", fontsize=9)
        ax.set_title(cn, fontsize=10.5, loc="left")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("三条传导路径：利率沿用 δ、信用给区间、汇率口径搬移", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "stress_mapping.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n[图] figures/stress_mapping.png")

    # 图 2：情景损失 vs 基准 VaR/ES（只画损失为正的情景）
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, cal in zip(axes, ("主", "次")):
        sb = I[(I.caliber == cal) & (I.loss_after_buffer_pct > 0)].copy()
        sb = sb.sort_values("loss_after_buffer_pct", ascending=False).head(14)
        y = np.arange(len(sb))[::-1]
        ax.barh(y, sb["loss_after_buffer_pct"], color=C_M1, height=0.62, label="含偏差缓冲")
        ax.barh(y, sb["loss_realized_pct"], color=C_GREY, height=0.62, label="δ 映射/实际损失")
        # 参照线**逐条**：每个情景的窗口长度不同（3/5/6 日），用各自同期限的经验 99% ES
        ax.scatter(sb["es_hd_99_pct"], y, marker="D", s=26, color=C_GARCH, zorder=5,
                   label="同期限历史 99% ES")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.scenario_id} {r.scenario_name}（{r.horizon_days}日）"
                            for r in sb.itertuples()], fontsize=8)
        ax.set_xlabel("损失（%），向右为损失更大", fontsize=9)
        ax.set_title(f"{cal}口径", fontsize=10.5, loc="left")
        ax.grid(axis="x", color="#e6e5e2", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"情景损失 vs 同期限历史 99% ES（菱形为各自窗口长度的经验分布分位；{BASE_MODEL} @ {BASE_SCOPE}）",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(FIG / "stress_loss_vs_var.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[图] figures/stress_loss_vs_var.png")

    # 图 3：因子贡献度
    # 不共享 y 轴：主口径下纯汇率情景损失恒为 0、不进本图，两个面板的情景集合不同，
    # sharey 会让刻度标签被其中一个面板接管、条形与标签错位。
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    colmap = {FACTOR_CN["d5y_bp"]: C_M1, FACTOR_CN["d10y_bp"]: C_GARCH,
              FACTOR_CN["doas_bp"]: C_ALT, FACTOR_CN["fx_ret_pct"]: C_GREY,
              FACTOR_CN["residual"]: "#c9c8c4"}
    for ax, cal in zip(axes, ("主", "次")):
        sb = C[(C.caliber == cal) & (C.scenario_id.isin(
            I[(I.caliber == cal) & (I.loss_after_buffer_pct > 0)].scenario_id))]
        piv = sb.pivot_table(index="scenario_id", columns="factor_cn",
                             values="share_pct", aggfunc="sum").fillna(0.0)
        order = I[I.caliber == cal].set_index("scenario_id")["loss_after_buffer_pct"]
        piv = piv.reindex([s for s in order.sort_values(ascending=False).index if s in piv.index])
        bottom = np.zeros(len(piv))
        for c in piv.columns:
            # 残差不是因子，用纹理而非又一个灰色区分——颜色上它不该和汇率抢同一个槽位
            if c == FACTOR_CN["residual"]:
                ax.barh(np.arange(len(piv))[::-1], piv[c], left=bottom, height=0.62,
                        facecolor="#e8e7e3", edgecolor="#8f8e8a", hatch="///", lw=0.5, label=c)
            else:
                ax.barh(np.arange(len(piv))[::-1], piv[c], left=bottom, height=0.62,
                        color=colmap.get(c, C_GREY), label=c)
            bottom = bottom + piv[c].values
        ax.set_yticks(np.arange(len(piv))[::-1])
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_ylim(-0.7, max(len(piv), 1) - 0.3)
        ax.set_xlabel("贡献占比（%）", fontsize=9)
        ax.set_title(f"{cal}口径", fontsize=10.5, loc="left")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color="#e6e5e2", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955),
               fontsize=8)
    fig.suptitle("因子贡献度分解（按 |贡献| 归一；残差单列以免把拟合误差算作因子）",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    fig.savefig(FIG / "stress_factor_contrib.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[图] figures/stress_factor_contrib.png")


if __name__ == "__main__":
    main()
