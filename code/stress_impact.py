"""三路径情景映射 + 压力测试测算（阶段三 Day2–Day3 · 9/22–9/23）

消费：results/stress_scenarios.csv
产出：results/stress_impact.csv / results/stress_factor_contrib.csv
      figures/stress_mapping.png / stress_loss_vs_var.png / stress_factor_contrib.png
口径：三条路径不共用同一套 δ：利率沿用基准 δ 线性映射，信用另估「利率 + ΔOAS」三因子
      模型并随点估计给 OLS 95% 区间，汇率是口径搬运（次口径系数恒 +1，主口径 δ_fx=0）。
      缓冲按情景施加一次、取适用路径的最大值，信用区间上端不叠加。符号统一正值 = 损失，
      唯一例外 loss_realized_pct（正值 = 收益）。headline 取 1 日：历史/补充取窗内最差
      单日、假设情景取 δ 映射损失，贡献度分解落在该口径的最差单日当天。
边界：δ(ΔOAS) 只在 OAS 可得的子样本（2023-09-05 起）上估出，不含 2022 年信用冲击，
      其 95% 区间只反映估计误差，不反映「平静期系数外推到危机」。H1/H2/X1/X2 窗内
      doas_bp 缺数据，信用项按 0 计入后落进残差列，故这些情景的残差 ≠ CR_2022_RESID。
      纯汇率情景主口径损失恒为 0，是口径事实而非漏算。
用法：./.venv/bin/python code/stress_impact.py
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

from common import FACT, RES, FIG, ensure_dirs, write_table
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2

ensure_dirs()

# ---- 常量 ----
# 持有期口径的唯一来源，别处一律引用本常量。
HORIZON_DAYS = 1            # 1 日，与阶段二 VaR 口径对齐（老师 9 月口头指示）

BUFFER_RATE = 0.3365        # pp，M2 × 主口径 × 1日 × 事件前δ 的 90 分位绝对偏差
BUFFER_CREDIT = 0.3365      # pp，取利率侧保守值，不用 M3 自身的 1 日档 0.1997
BUFFER_FX = 0.0             # 汇率是口径搬运（次口径 +1、主口径 0），不加缓冲

# 信用区间上限的外推标定量：2022 年事件窗上「未解释残差」的最大绝对值。
# 平静样本估出的 δ(ΔOAS) 不含 2022 年信用冲击，其 95% 区间只反映估计误差、
# 不反映「平静期系数外推到危机」，故以 2022 年的实际残差幅度作为后者的代理，
# 加到「平静样本强传导」一端构成区间上限。
# 口径与信用情景同期限：信用情景是 1 日，此处即取 1 日窗。
# 三个窗长下 dev_credit_pct 的最大值都落在 2022-11-14，该量因此也等于全样本最大，
# 含义是「1 日窗全样本最大偏差（恰落在 2022 年）」，不是「2022 年极值」。
# 与常量的对账见下方 load_2022_credit_resid()。
CR_2022_RESID = 1.1147      # pp，1 日窗，2022-11-14（防控优化二十条+金融16条）

BASE_SCOPE = "common773"    # 基准样本；跨模型横比用公共样本，不用 full（各模型在 full 上长度不同）
BASE_MODEL = "HS250"        # 9/18 定案的唯一基准口径
RATE_COLS = ["d5y_bp", "d10y_bp"]

FACTOR_CN = {"d5y_bp": "5年期美债收益率", "d10y_bp": "10年期美债收益率",
             "doas_bp": "新兴市场IG信用利差", "fx_ret_pct": "人民币兑美元汇率",
             "residual": "未解释残差（信用维度）"}


def load_2022_credit_resid() -> float:
    """从逐事件传导表取 2022 年「未解释残差」的最大绝对值，作为信用区间上端的外推标定量。

    参数：
        无。

    返回：
        float，单位 pp：2022 年、主口径、M2 利率双因子、B 事件前δ、1 日窗各行的
        dev_credit_pct 绝对值最大值，即常量 CR_2022_RESID 登记的 1.1147pp（2022-11-14）。

    备注：
        `doas_bp` 序列 2022 年全部缺失（首值 2023-09-06），当年的残差无法拆解为纯信用
        维度，只能用「未解释残差（信用维度）」这个代理量，不能写成「已用 2022 信用维度
        残差标定」。口径与信用情景同为 1 日窗。该极值同时是 1 日窗全样本 85 事件窗的
        最大绝对偏差，故其含义是「1 日窗全样本最大偏差（恰落在 2022 年）」而非
        「2022 年极值」。
    """
    ev = pd.read_csv(RES / "delta_transmission_events.csv", encoding="utf-8-sig")
    d = ev[(ev.model == "M2 利率双因子") & (ev.caliber == "主")
           & (ev.delta_src == "B 事件前δ") & (ev.window == "1日")
           & (pd.to_datetime(ev.event_date).dt.year == 2022)]
    return float(d.dev_credit_pct.abs().max())


def horizon_es(r: pd.Series, h: int) -> dict:
    """用重叠窗口算经验 h 日累计损失分布的分位数与 ES。

    参数：
        r: 组合收益序列（%），主口径传 etf_ret_tr_pct，次口径传 etf_ret_rmb_pct。
        h: 持有期天数，收益按 h 日滚动求累计。

    返回：
        dict，键为 95 与 99 两个分位点，值为 dict(var=该分位的损失, es=超过该分位的损失的
        均值)，均为 float、单位 %、损失为正。

    备注：
        不把 1 日 VaR 乘 √h：本项目近 IGARCH（α+β≈0.996）且存在波动聚集，√t 的 iid 前提
        不成立，缩放会把一个未经验证的换算当成可比性依据。直接取历史 h 日累计损失的经验分布，
        与情景损失同为「h 日收益之和」，是同类比较。代价是重叠窗口自相关、有效样本少于名义
        条数，故只作参照线不做检验。
    """
    r = pd.Series(r).dropna().reset_index(drop=True)
    losses = -r.rolling(h).sum().dropna()
    out = {}
    for q in (95, 99):
        var = float(np.percentile(losses, q))
        out[q] = dict(var=var, es=float(losses[losses >= var].mean()))
    return out


def path_cum_mdd(r: pd.Series) -> tuple[float, float]:
    """把 log 收益路径折算为累计收益与最大回撤。

    参数：
        r: 组合收益序列（%），缺失值先剔除。

    返回：
        (cum, mdd) 二元组，单位 %：cum 为累计收益（正 = 收益），
        mdd 为最大回撤（≤ 0，含事件前起点 NAV=1.0 的净值口径）。
    """
    r = pd.Series(r).dropna().reset_index(drop=True)
    cum = r.cumsum() / 100.0
    eq = pd.concat([pd.Series([1.0]), np.exp(cum)], ignore_index=True)
    return float(r.sum()), float((eq / eq.cummax() - 1.0).min() * 100.0)


def main() -> None:
    """把情景库的因子冲击映射为组合损失，完成测算、贡献度分解与尾部风险判定，落盘三表三图。

    脚本契约：
        消费：factors/factor_table_nav_tr.csv；results/stress_scenarios.csv、
              results/factor_exposure.csv（基准 δ(Δ5Y)、δ(Δ10Y)）、
              results/backtest_results.csv（HS250 @ common773 基准 VaR/ES）、
              results/delta_transmission_events.csv（经 load_2022_credit_resid 取标定量）。
        产出：results/stress_impact.csv（逐情景 × 口径的损失/回撤/缓冲/对标长表）、
              results/stress_factor_contrib.csv（因子贡献度，含未解释残差）、
              figures/stress_mapping.png、figures/stress_loss_vs_var.png、
              figures/stress_factor_contrib.png。
        断言/边界：五处断言，三处对账——2022 残差标定量与 CR_2022_RESID（容差 5e-5pp）、
        主次口径四个基准配置的 VaR/ES 与阶段二报告 §7.2 登记值（容差 5e-5）、情景库每个
        情景都进入测算（scenario_id 集合相减不为空即终止，因 9/24 补入的 X2 曾被漏算而设）。
        另两处为回归守卫：历史情景的信用区间下端不等于窗口累计量（相等说明退回窗口级
        计算），假设情景的 loss_modeled_pct 与 worst_day_modeled_pct 重合（证明路径确为
        单日）。
        已知边界：纯汇率情景主口径损失恒为 0，是美元计价的口径事实；判定线取
        es_1d_99_pct，与并列输出的 es_hd_99_pct 不是同一个统计量；√t 缩放列只作参考；
        历史情景不加偏差缓冲。
    """
    print("=" * 78)
    print("阶段三 Day2–Day3：三路径映射 + 压力测试测算")
    print("=" * 78)

    # ---- 1) 载入 ----
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    S = pd.read_csv(RES / "stress_scenarios.csv", encoding="utf-8-sig")
    E = pd.read_csv(RES / "factor_exposure.csv", encoding="utf-8-sig")
    B = pd.read_csv(RES / "backtest_results.csv", encoding="utf-8-sig")

    d5 = float(E.loc[E["因子"] == "利率 Δ5Y", "暴露δ"].iloc[0])
    d10 = float(E.loc[E["因子"] == "利率 Δ10Y", "暴露δ"].iloc[0])
    print(f"[输入] stress_scenarios.csv  {S.shape[0]} 行 / {S.scenario_id.nunique()} 情景")
    print(f"[输入] factor_exposure.csv   δ(Δ5Y)={d5:+.6f}  δ(Δ10Y)={d10:+.6f} %/bp")

    # ---- 2) 信用系数 ----
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
    print(f"  信用缓冲取 {BUFFER_CREDIT:.4f} pp（利率侧保守值，非 M3 自身的 0.1997）")

    # 区间上限的外推标定量：平静样本 CI 只反映估计误差，不反映「平静期系数外推到危机」，
    # 以 2022 年的实际残差幅度代理。此处重算并与常量对账，上游漂移在此暴露。
    cr22 = load_2022_credit_resid()
    assert abs(cr22 - CR_2022_RESID) < 5e-5, (
        f"2022 残差标定量对账失败：逐事件表给 {cr22:.4f}pp，常量登记 {CR_2022_RESID:.4f}pp")
    print(f"  区间上限外推标定量：平静强传导 + 2022 年残差极值 {cr22:.4f} pp"
          f"（1 日窗，代理量，非纯信用维度）")
    print(f"  [对账] 2022 残差极值与常量登记值 {CR_2022_RESID:.4f}pp 一致")

    # ---- 3) 基准口径 ----
    bb = B[(B["model"] == BASE_MODEL) & (B["scope"] == BASE_SCOPE)]
    base = {}
    for cal in ("主", "次"):
        for conf in (95, 99):
            r = bb[(bb["caliber"] == cal) & (bb["conf"] == conf)].iloc[0]
            base[(cal, conf)] = dict(var=float(r["mean_var"]), es=float(r["es"]))
    print(f"\n[输入] 基准口径 {BASE_MODEL} @ {BASE_SCOPE}（**1 日** horizon）")
    for k, v in base.items():
        print(f"       {k[0]}口径 {k[1]}%  VaR={v['var']:.4f}%  ES={v['es']:.4f}%")
    print(f"       注意：情景损失本版统一为 {HORIZON_DAYS} 日，与基准**同期限**，倍数可直接比较")

    # 基准数字已登记在阶段二交付报告（docs/phase2_model_validation.md §7.2 表），
    # 上游漂移在此处暴露。
    REGISTERED = {("主", 95): (0.3544, 0.4473), ("主", 99): (0.5361, 0.6659),
                  ("次", 95): (0.4454, 0.5554), ("次", 99): (0.7260, 0.7024)}
    for k, (rv, re_) in REGISTERED.items():
        assert abs(base[k]["var"] - rv) < 5e-5 and abs(base[k]["es"] - re_) < 5e-5, (
            f"基准对账失败 {k}：CSV 给 VaR={base[k]['var']:.4f}/ES={base[k]['es']:.4f}，"
            f"阶段二报告登记 VaR={rv}/ES={re_}")
    print("       [对账] 四个基准配置与阶段二报告 §7.2 登记值一致")

    # ---- 4) 逐情景测算 ----
    print("\n" + "=" * 78)
    print(f"逐情景测算（损失为正；历史/补充取窗内最差单日，假设用锚点日 {HORIZON_DAYS} 日路径×倍数）")
    print("=" * 78)

    # 同期限经验分布所需的收益序列（主 = 美元复权，次 = 人民币复权）
    ret_series = {"主": F["etf_ret_tr_pct"], "次": F["etf_ret_rmb_pct"]}

    rows, crows = [], []

    def add(sid, cls, name, direction, sev, anchor_date,
            rate_path, credit_path, fx_path,
            realized_main=None, realized_rmb=None, horizon=HORIZON_DAYS,
            event_window_days=HORIZON_DAYS, worst_pos=None):
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
        # 缓冲只给假设情景：历史情景用窗内实际收益，不含 δ 估算误差，加缓冲等于对
        # 已知结果重复计一次模型误差；假设情景的损失由 δ 映射得出，才需要亏损侧
        # 偏差保证金。施加上按情景一次（取适用路径中较大者），不按路径累加。
        buf = (max(BUFFER_RATE if has_rate else 0.0,
                   BUFFER_CREDIT if has_cr else 0.0,
                   BUFFER_FX if has_fx else 0.0) if cls == "假设" else 0.0)

        for cal in ("主", "次"):
            fx_term = afx if cal == "次" else afx * 0.0
            r_model = r_rate + r_cr_pt + fx_term
            realized = realized_main if cal == "主" else realized_rmb
            hist = realized is not None and len(realized) > 0
            r_used = pd.Series(realized).dropna().reset_index(drop=True) if hist else r_model

            # 四个损失口径各司其职，不混用：
            #   loss   窗内累计损失（历史=实现，假设=δ 映射），方向性质与残差用
            #   mdd    窗内最大回撤（需求文档要求的伴随列，正值 = 回撤/损失）
            #   wd     窗内最差单日的损失，历史/补充情景的基准
            #   bench  基准损失：历史取窗内最差单日，假设取 δ 映射的 1 日值
            # 累计损失与残差仍按累计口径保留：残差度量 δ 模型误差，基准度量 1 日情景
            # 严重度，两者口径不同是有意的；把残差也改成「基准 − 模型」会把「单日」与
            # 「窗末累计」两个口径混算。
            # 符号统一：`path_cum_mdd` 按「净值回撤」原义返回负值，此处取负写入，
            # 使本表所有损失类列一律正值 = 损失（与 loss_* / worst_day 一致，
            # 也与已发布文档中「窗内最深 2.6784%」的写法一致）。此前 mdd_pct 是全表
            # 唯一以负值表示损失的列，容易与同行的正值损失列对错。
            loss = -path_cum_mdd(r_used)[0]
            mdd = -path_cum_mdd(r_used)[1]
            wd = -float(np.nanmin(r_used.values)) if hist else loss
            # 最差单日的日期随口径而变（主口径看 USD 复权、次口径另叠汇率项），
            # 故按口径各自记录，不共用 anchor_date。
            wd_dt = (str(pd.Series(realized).dropna().idxmin().date())
                     if hist else anchor_date)
            loss_model = -float(r_model.sum())
            # 模型路径的窗内最差单日（1 日口径）。信用区间已改为 1 日量，与此列比对
            # 才是同期限；拿窗末累计的 loss_model 去比 1 日区间是「累计 vs 单日」口径
            # 错配。假设情景路径即 1 日，两列重合。
            wd_model = -float(np.nanmin(r_model.values)) if hist else loss_model
            bench = wd if hist else loss
            bench_src = "窗内最差单日" if hist else "δ 映射"
            residual = loss - loss_model if hist else 0.0

            # 信用传导区间：下端取平静样本弱传导，上端取「平静样本强传导 + 2022 年
            # 残差极值」。上端不再叠加缓冲：2022 残差标定本身就是用实际偏差修正上限，
            # 再叠一次 0.3365pp 是重复计算。
            # 期限与基准损失列一致：历史/补充情景取窗内最差单日，假设情景的 δ 映射
            # 路径本身即 HORIZON_DAYS 日，两者同为单日量。此前一律取窗末累计，使历史
            # 情景的信用区间成为窗口级量、与 1 日基准不可比（`docs/week3_report.md`
            # §6.5 第 1 条，已修复）。
            cr_lo_path = r_rate + r_cr_hi + fx_term
            cr_hi_path = r_rate + r_cr_lo + fx_term
            cr_lo_win = -path_cum_mdd(cr_lo_path)[0]      # 窗口级（旧口径，留作断言对照）
            cr_hi_win = -path_cum_mdd(cr_hi_path)[0]
            cr_lower = (-float(np.nanmin(cr_lo_path.values)) if hist else cr_lo_win)
            cr_upper = (-float(np.nanmin(cr_hi_path.values))
                        if hist else cr_hi_win) + CR_2022_RESID
            # 历史情景窗长 > 1 日时，1 日量必与窗口累计量不同；两者相等说明上面又退回
            # 窗口级计算（本挂账项的回归）。比对输出列发现不了这种回归——旧窗口级值
            # 同样满足「下端 ≤ 点估计」。
            if hist and has_cr and len(cr_lo_path) > 1:
                assert abs(cr_lower - cr_lo_win) > 1e-12, (
                    f"{sid} {cal} 信用区间下端与窗口累计量相同——历史情景疑似退回窗口级计算")

            v95, v99 = base[(cal, 95)], base[(cal, 99)]
            net = bench + buf
            # 本脚本在全样本上算的无条件经验分布。1 日口径下它与阶段二基准
            # es_1d_99_pct 是两个不同统计量（样本起点、条件/无条件均不同），
            # 故并列输出作核对，判定用 es_1d_99_pct。
            hd = horizon_es(ret_series[cal], horizon)
            rows.append(dict(
                scenario_id=sid, scenario_class=cls, scenario_name=name,
                direction=direction, severity=sev, anchor_date=anchor_date, caliber=cal,
                horizon_days=horizon, event_window_days=event_window_days,
                worst_day_loss_pct=round(wd, 4), worst_day_date=wd_dt,
                loss_realized_pct=round(loss, 4),
                loss_modeled_pct=round(loss_model, 4),
                worst_day_modeled_pct=round(wd_model, 4),
                residual_pct=round(residual, 4),
                benchmark_loss_pct=round(bench, 4),
                benchmark_source=bench_src,
                buffer_pct=round(buf, 4), loss_after_buffer_pct=round(net, 4),
                buffer_applied=("假设情景 δ 映射，加亏损侧偏差缓冲" if buf > 0
                                else ("历史情景取实际路径，无模型误差不加缓冲" if cls != "假设"
                                      else "纯汇率情景口径搬运，恒等式不加缓冲")),
                mdd_pct=round(mdd, 4),
                # 1 日锚点路径只有一天，回撤退化为损失的单调换形，不含额外信息。
                # 需求文档要求给回撤，故保留该列并显式标注退化，不参与严重度解读。
                mdd_note=("1 日锚点路径为单点，回撤 = 1 − exp(损失)，与基准同信息、不含额外信息"
                          if cls == "假设" else ""),
                # 区间下端=平静样本弱传导；上端=平静强传导+2022 残差极值，不含缓冲
                credit_range_lo_pct=round(cr_lower, 4) if has_cr else np.nan,
                credit_range_hi_pct=round(cr_upper, 4) if has_cr else np.nan,
                credit_range_note=("参考项：上限由 2022 年未解释残差极值标定，"
                                   "线性外推有偏差，不作信用风险可控的判断"
                                   if has_cr else ""),
                var_1d_95_pct=round(v95["var"], 4), var_1d_99_pct=round(v99["var"], 4),
                es_1d_99_pct=round(v99["es"], 4),
                var_hd_95_pct=round(hd[95]["var"], 4), var_hd_99_pct=round(hd[99]["var"], 4),
                es_hd_99_pct=round(hd[99]["es"], 4),
                x_var95=round(net / v95["var"], 3), x_var99=round(net / v99["var"], 3),
                x_es99=round(net / v99["es"], 3),
                x_hd_es99=round(net / hd[99]["es"], 3),
                tail_flag_hd="是" if net > hd[99]["es"] else "否",
                tail_flag="是" if net > v99["es"] else "否",
                # 「纯」汇率情景指汇率是唯一驱动。历史情景窗内也含汇率项，
                # 只用 has_fx 判定会把 H1/H2/H3/X1/X2 的主口径行误贴该注释。
                fx_horizon_note=("纯汇率情景，主口径按口径恒为 0"
                                 if (has_fx and not has_rate and not has_cr and cal == "主")
                                 else ""),
            ))

            # 因子贡献度：分解对象是基准损失那一口径（历史 = 最差单日，假设 = 1 日映射）。
            # 故历史情景取该口径最差日当天的因子变动，而非整个事件窗的累计变动，
            # 否则会出现「用窗口级因子变动解释单日损失」的错配，share_pct 失去意义。
            # 最差日随口径而变，贡献度按口径分别计算。
            wp = (worst_pos or {}).get(cal)
            if hist:
                s5, s10 = float(a5.iloc[wp]), float(a10.iloc[wp])
                so, sfx = float(ao.iloc[wp]), float(afx.iloc[wp])
            else:
                s5, s10 = float(a5.sum()), float(a10.sum())
                so, sfx = float(ao.sum()), float(afx.sum())
            c5, c10 = -(s5 * d5), -(s10 * d10)
            ccr = -(so * b_oas)
            cfx = -(sfx if cal == "次" else 0.0)
            # 残差口径与分解对象一致：历史情景 = 最差单日实际损失 − 当天因子映射之和；
            # 假设情景损失本身就是模型输出，恒为 0。
            # H1/H2/X1/X2 窗内 doas_bp 缺数据，当天信用项按 0 计入、影响落进本列，
            # 故这些情景的残差 ≠ CR_2022_RESID。
            c_resid = (wd - (c5 + c10 + ccr + cfx)) if hist else 0.0
            parts = [("d5y_bp", c5), ("d10y_bp", c10), ("doas_bp", ccr),
                     ("fx_ret_pct", cfx), ("residual", c_resid)]
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
        _bench = -float(np.nanmin(p_main.values)) if cls != "假设" else -l_main
        print(f"  [{sid}] {name:26s} 主口径={-l_main:+7.4f}%  次口径={-l_rmb:+7.4f}%"
              f"  回撤={-d_main:+7.4f}%  基准(最差单日)={_bench:+7.4f}%  缓冲={buf:.4f}pp")

    # ---- 历史与补充情景：实际路径 ----
    # 情景清单从 CSV 读，不硬编码：否则情景库扩容（如 9/24 补入 X2）会被漏算。
    HIST_IDS = (S[(S.measure == "cum_window") & (S.scenario_class.isin(["历史", "补充"]))]
                .scenario_id.drop_duplicates().tolist())
    for sid in HIST_IDS:
        g = S[(S.scenario_id == sid) & (S.measure == "cum_window")]
        a, b = g.window_start.iloc[0], g.window_end.iloc[0]
        W = F.loc[a:b]
        # 最差单日在路径中的位置：主口径看 USD 复权、次口径另叠汇率项，两者可以不同日，
        # 故按口径分别给出，供贡献度分解取当天因子值。
        # np.nanargmin 作用在全窗（未 dropna）列上，位置即与上面三个路径 DataFrame 对齐。
        worst_pos = {"主": int(np.nanargmin(W["etf_ret_tr_pct"].values)),
                     "次": int(np.nanargmin(W["etf_ret_rmb_pct"].values))}
        add(sid, g.scenario_class.iloc[0], g.scenario_name.iloc[0],
            g.lead_direction.iloc[0], "-", str(W.index.max().date()),
            pd.DataFrame({"d5y_bp": W["d5y_bp"].values, "d10y_bp": W["d10y_bp"].values}),
            pd.DataFrame({"doas_bp": W["doas_bp"].fillna(0.0).values}),
            pd.DataFrame({"fx_ret_pct": W["fx_ret_pct"].fillna(0.0).values}),
            realized_main=W["etf_ret_tr_pct"].dropna(),
            realized_rmb=W["etf_ret_rmb_pct"].dropna(),
            horizon=HORIZON_DAYS, event_window_days=len(W), worst_pos=worst_pos)

    # ---- 假设情景：锚点日 1 日路径 × 倍数 ----
    for sid in sorted(S[S.scenario_class == "假设"].scenario_id.unique()):
        # 过滤 measure 不能省：假设情景同时存有 peak_1d（主口径）与 peak_3d（附录对照）
        # 两组行，不过滤会两套行都进循环。下面 paths[col] 是按因子名赋值的字典，
        # 后写入者覆盖前者，会把 3 日锚点当成 1 日锚点用且不报错。
        g = S[(S.scenario_id == sid) & (S.measure == "peak_1d")]
        nm, lead = g.scenario_name.iloc[0], g.lead_direction.iloc[0]
        mult, sev = float(g.multiple.iloc[0]), g.scenario_name.iloc[0].split("·")[-1]
        paths, ad = {}, []
        for _, r in g.iterrows():
            col, dt = r["factor"], pd.Timestamp(r["anchor_date"])
            idx = F.index[F.index <= dt][-HORIZON_DAYS:]
            paths[col] = pd.Series(F.loc[idx, col].values * mult)
            ad.append(str(dt.date()))
        add(sid, "假设", nm, lead, sev, "/".join(sorted(set(ad))),
            pd.DataFrame({c: paths[c] for c in RATE_COLS if c in paths}),
            pd.DataFrame({"doas_bp": paths["doas_bp"]}) if "doas_bp" in paths else None,
            pd.DataFrame({"fx_ret_pct": paths["fx_ret_pct"]}) if "fx_ret_pct" in paths else None)

    I = pd.DataFrame(rows)
    C = pd.DataFrame(crows)
    # 完整性断言：情景库里的每个情景都要进测算，缺一个即终止。
    # 情景清单曾在上一步里被硬编码，9/24 补入的 X2 因此漏算，直到覆盖度检验才发现。
    _missing = set(S.scenario_id.unique()) - set(I.scenario_id.unique())
    assert not _missing, (f"情景库有 {S.scenario_id.nunique()} 个情景，"
                          f"以下未进入测算：{sorted(_missing)}")
    print(f"\n  [对账] 情景库 {S.scenario_id.nunique()} 个情景全部进入测算（无静默漏算）")

    # 信用区间口径一致性对账。
    # 那次修订只应改动历史与补充情景的区间：假设情景的 δ 映射路径本来就是单日，
    # 其区间前后应逐值不变。故下面的断言把「假设情景路径确为单日」钉住：
    # loss_modeled_pct 与 worst_day_modeled_pct 重合即证明路径长度为 1，其区间
    # 天然是 1 日量。历史情景的分支是否真的生效由 add() 内的回归断言负责，
    # 输出列本身区分不了两种口径。
    _cr = I[I.credit_range_lo_pct.notna()]
    _hy = _cr[_cr.scenario_class == "假设"]
    if len(_hy):
        assert (_hy.loss_modeled_pct - _hy.worst_day_modeled_pct).abs().max() < 1e-4, (
            "假设情景路径应为单日，loss_modeled_pct 与 worst_day_modeled_pct 应重合")
    _h = _cr[_cr.scenario_class != "假设"]
    print(f"  [对账] 信用区间两列已统一为 1 日量：假设情景 {len(_hy)} 行（路径本即单日，"
          f"值不变）、历史与补充情景 {len(_h)} 行（由窗口级量重算）")
    write_table(I, RES / "stress_impact.csv")
    write_table(C, RES / "stress_factor_contrib.csv")
    print(f"\n[1] 测算表 → results/stress_impact.csv  {I.shape[0]} 行 × {I.shape[1]} 列")
    print(f"[2] 因子贡献度 → results/stress_factor_contrib.csv  {C.shape[0]} 行 × {C.shape[1]} 列")

    # ---- 5) 尾部风险点 ----
    print("\n" + "=" * 78)
    print(f"风险承受能力评估（1/2）：{HORIZON_DAYS} 日期限上的两条参照线")
    print("=" * 78)
    for cal in ("主", "次"):
        g = I[I.caliber == cal]
        if not len(g):
            continue
        print(f"  {cal}口径：判定线 = 阶段二基准 {BASE_MODEL}@{BASE_SCOPE} "
              f"99%VaR={g.var_1d_99_pct.iloc[0]:.4f}%  99%ES={g.es_1d_99_pct.iloc[0]:.4f}%"
              f"（95%VaR={g.var_1d_95_pct.iloc[0]:.4f}%）")
        print(f"          核对线 = 本脚本全样本无条件经验分布 "
              f"99%VaR={g.var_hd_99_pct.iloc[0]:.4f}%  99%ES={g.es_hd_99_pct.iloc[0]:.4f}%")

    print("\n" + "=" * 78)
    print(f"风险承受能力评估（2/2）：损失（含缓冲）超过 1 日 99% ES —— 尾部风险点")
    print("=" * 78)
    T = I[(I.tail_flag == "是") & (I.loss_after_buffer_pct > 0)][
        ["scenario_id", "scenario_name", "caliber", "worst_day_loss_pct",
         "buffer_pct", "loss_after_buffer_pct", "es_1d_99_pct", "x_es99"]]
    print(T.to_string(index=False) if len(T) else "  （无）")
    # 分母从测算表取，不写死：情景库扩容后写死的分母会变成错的
    n_hit = int(((I.tail_flag == "是") & (I.loss_after_buffer_pct > 0)).sum())
    n_hit_emp = int(((I.tail_flag_hd == "是") & (I.loss_after_buffer_pct > 0)).sum())
    print(f"\n  超出 1 日 99% ES 的情景 {n_hit} / {len(I)} 条（口径×情景，"
          f"{I.scenario_id.nunique()} 个情景）")
    print(f"  **判定对阈值口径敏感**：同一份损失若改用全样本无条件经验 ES"
          f"（样本更长、值更大），命中数变为 {n_hit_emp} 条，相差 {n_hit - n_hit_emp} 条 —— "
          f"该敏感性须随结论一并披露，不让读者自行发现")
    # 方向性质按窗内累计口径判定，与基准损失口径分开：
    # H2/X1 的窗内累计为正（收益），但其基准损失取窗内最差单日，两者口径不同，
    # 前者说明情景性质、后者进入严重度统计，不能互相替代。
    n_neg = int((I.loss_realized_pct < 0).sum())
    n_bench_pos = int(((I.loss_realized_pct < 0) & (I.benchmark_loss_pct > 0)).sum())
    print(f"\n  窗内累计为收益的情景 {n_neg} 条 —— 累计口径的收益性质如实保留")
    print(f"  其中基准损失（窗内最差单日）为正、即仍进入损失统计的 {n_bench_pos} 条 —— "
          f"两口径含义不同，不可互相替代")
    # 命中率过高本身要说明：压力情景按构造就该比 VaR 更极端，二值「是否超限」
    # 因此在 1 日口径下几乎失去区分度，正文改以倍数为主要统计量。
    print(f"  说明：1 日口径下命中率 {n_hit}/{len(I)}，压力情景按构造即应超过 VaR；"
          f"二值判定的区分度下降，正文以「相当于几倍 99% ES」为主导统计量")

    # ---- 6) 三张图 ----
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
        # 两个条形都以基准损失为基：历史情景取窗内最差单日、假设情景取 δ 映射值。
        # 历史情景不施缓冲（无 δ 估计误差），故两根条形重合；假设情景的差即缓冲。
        ax.barh(y, sb["loss_after_buffer_pct"], color=C_M1, height=0.62, label="基准损失 + 偏差缓冲")
        ax.barh(y, sb["benchmark_loss_pct"], color=C_GREY, height=0.62,
                label="基准损失（历史=窗内最差单日，假设=δ 映射）")
        # 判定线：全部情景同为 1 日口径，故是一条共用的竖线（阶段二 1 日 99% ES），
        # 不逐情景取各自窗口长度的分位。
        ax.scatter(sb["es_1d_99_pct"], y, marker="D", s=26, color=C_GARCH, zorder=5,
                   label="1 日 99% ES（阶段二基准）")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.scenario_id} {r.scenario_name}" for r in sb.itertuples()],
                           fontsize=8)
        ax.set_xlabel("损失（%），向右为损失更大", fontsize=9)
        ax.set_title(f"{cal}口径", fontsize=10.5, loc="left")
        ax.grid(axis="x", color="#e6e5e2", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"情景 1 日损失 vs 1 日 99% ES（全库同期限；菱形线为 {BASE_MODEL} @ {BASE_SCOPE} 基准）",
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
            # 残差不是因子，用纹理而非又一个灰色区分：颜色上不应和汇率抢同一个槽位
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
    # 标题自带两条边界：残差是拟合误差不是信用风险贡献；纯汇率情景的 100% 是口径定义。
    # 这张图会被单独抽进汇报材料，脱离正文时这两条要跟着图走。
    fig.suptitle("因子贡献度分解（按 |贡献| 归一；残差单列为拟合误差、非信用风险贡献；"
                 "纯汇率情景汇率 100% 系口径定义）",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    fig.savefig(FIG / "stress_factor_contrib.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[图] figures/stress_factor_contrib.png")


if __name__ == "__main__":
    main()
