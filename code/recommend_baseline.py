"""基准口径定案：为两个口径各定一版基准模型，供阶段三压力测试对标（阶段二补充 · 9/18）

消费：results/backtest_results.csv（9/17 的 262 行检验长表）。本脚本不重算任何检验，只做聚合
      与排序，以保证与回测报告逐字一致。
产出：results/model_scorecard.csv、results/baseline_decision.csv
      figures/baseline_model_tradeoff.png
口径：跨模型横比一律在 common773 上做（主 773 日 / 次 768 日，2023-08-03 起）——full 下各模型
      样本长度不同，直接比平均 VaR 会把样本构成差异读成模型差异。门槛与排序规则写死在本文件
      常量里：可达的核心范围 Kupiec 全不拒绝；无显著风险低估、基准样本失败率不超名义 +10%，
      且两档同时满足；过门槛者按两档校准偏差均值升序取第一名，相差 <10% 时取资本占用更低者。
      稳定性否决（落差 >2.0pp 且高波动段 T >= 250、聚集存疑 >= 2 个范围）只降为「备选」。
边界：M1g-t 是对照、M1d 是恒等式，读入后即排除，不参与定案。HS750 在 common773 上只剩 523
      天，不参与横比排序。同一口径内 95% 与 99% 须为同一版模型，否则两档损失不可比。
用法：./.venv/bin/python code/recommend_baseline.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import RES, FIG, ensure_dirs

ensure_dirs()

BASE_SCOPE = "common773"
CORE_SCOPES = ["full", "common773", "common523", "calm", "highvol"]
MODELS = ["M1", "M1g", "M1e", "HS250", "HS500", "HS750", "FHS-E", "FHS-G"]
CONFS = (95, 99)
NOMINAL = {95: 5.0, 99: 1.0}
CA = {"主": "#2a78d6", "次": "#eb6834"}

ALPHA = 0.05
PRUDENCE = 1.10        # 门槛 2②：相对名义水平的容许偏离
CAL_TIE = 0.10         # 并列裁决：两档校准偏差相差 <10% 视为并列
GAP_BAR = 2.0          # 条件覆盖落差否决门槛（pp）
N_IND_BAR = 2          # 聚集性存疑计数否决门槛
T_HIGHVOL_MIN = 250    # 启用落差否决所需的高波动段最小样本量


def build_scorecard(R: pd.DataFrame) -> pd.DataFrame:
    """把逐 (口径, 置信度, 模型, scope) 的回测长表聚合成模型评分表。

    参数：
        R: 回测长表，每个口径 × 置信度 × 模型 × scope 一行。用到的列为 caliber、conf、
            model、scope、T、x、exp_x_eff、rate_pct、p_uc、p_ind_mc_cond、mean_var、es。
            基准样本由 BASE_SCOPE 写死为 common773，只在该 scope 上有记录的模型才进表。

    返回：
        DataFrame，每行一个 (model, caliber, conf) 组合，列为
        n_pass_uc_core / n_core_scopes: 该模型可达的核心范围数，以及其中 Kupiec 不拒绝的个数；
        pass_core: 门槛 1，可达核心范围是否全部通过（bool）；
        under_coverage: 门槛 2①，是否存在 Kupiec 拒绝且 x > exp_x_eff 的范围（bool）；
        prudent: 门槛 2②，基准样本失败率是否不超过名义水平的 110%（bool）；
        rate_base_pct / nominal_pct / rate_over_nominal / cal_dev: 基准样本失败率、名义水平、
            两者之比、以及 |比值 - 1| 的校准偏差；
        cap_idx_vs_M1 / mean_var_base_pct: 基准样本平均 VaR，以及它相对同口径同置信度 M1 的倍数；
        T_base: 基准样本观测数；
        es_base_pct: 基准样本 ES，非有限值时记 NaN；
        rate_full_pct: full 范围的失败率；
        rate_highvol_pct / rate_calm_pct / T_highvol: 高波动段与平稳段的失败率及高波动段样本量，
            对应 scope 缺失时记 NaN 与 0；
        stab_scope_pp: 核心范围上失败率的标准差；
        gap_regime_pp / signed_gap_pp / gap_evaluable: 高波动与平稳段的失败率落差（绝对值与
            带符号值），以及该落差是否可评估（高波动段 T >= T_HIGHVOL_MIN）；
        n_ind_suspect: 全部 scope 中条件蒙特卡洛 p_ind_mc_cond < 0.05 的个数。

    备注：
        资本占用指数以 M1 在基准样本上的平均 VaR 为分母，同口径同置信度各取一个参照值，
        函数开头的 print 会把它先打出来。tier 不在这里给，由 decide 判定后在 main 里并回。
    """
    ref_var = {}
    for cal in ("主", "次"):
        for conf in CONFS:
            v = R[(R.caliber == cal) & (R.conf == conf) & (R.model == "M1")
                  & (R.scope == BASE_SCOPE)]["mean_var"]
            ref_var[(cal, conf)] = float(v.iloc[0])
    print(f"[基准样本 {BASE_SCOPE}] M1 平均 VaR：主 95%={ref_var[('主',95)]:.4f}% "
          f"99%={ref_var[('主',99)]:.4f}%；次 95%={ref_var[('次',95)]:.4f}% "
          f"99%={ref_var[('次',99)]:.4f}%")

    rows = []
    for cal in ("主", "次"):
        for conf in CONFS:
            g = R[(R.caliber == cal) & (R.conf == conf)]
            nom = NOMINAL[conf]
            for m in MODELS:
                sub = g[g["model"] == m].set_index("scope")
                if len(sub) == 0 or BASE_SCOPE not in sub.index:
                    continue
                core = [s for s in CORE_SCOPES if s in sub.index]

                # 门槛 1：可达的核心范围上 Kupiec 全部通过
                n_pass = int((sub.loc[core, "p_uc"] >= ALPHA).sum())
                pass_core = bool(n_pass == len(core))

                # 门槛 2①：是否存在显著风险低估
                rej = sub[sub["p_uc"] < ALPHA]
                under = bool((rej["x"] > rej["exp_x_eff"]).any())

                # 门槛 2②：基准样本失败率不超名义水平乘 PRUDENCE
                rate_base = float(sub.loc[BASE_SCOPE, "rate_pct"])
                rel = rate_base / nom
                prudent = bool(rel <= PRUDENCE)
                cal_dev = abs(rel - 1.0)

                # 资本占用（基准样本，相对 M1）
                cap = float(sub.loc[BASE_SCOPE, "mean_var"]) / ref_var[(cal, conf)]

                # 稳定性
                stab_scope = float(sub.loc[core, "rate_pct"].std())
                sg = (float(sub.loc["highvol", "rate_pct"]) - float(sub.loc["calm", "rate_pct"])
                      if {"highvol", "calm"} <= set(sub.index) else np.nan)
                gap = abs(sg) if np.isfinite(sg) else np.nan
                t_hi = int(sub.loc["highvol", "T"]) if "highvol" in sub.index else 0
                gap_eval = bool(t_hi >= T_HIGHVOL_MIN)
                n_ind = int((sub["p_ind_mc_cond"] < ALPHA).sum())

                rows.append({
                    "model": m, "caliber": cal, "conf": conf,
                    "n_pass_uc_core": n_pass, "n_core_scopes": len(core),
                    "pass_core": pass_core, "under_coverage": under, "prudent": prudent,
                    "rate_base_pct": round(rate_base, 4), "nominal_pct": nom,
                    "rate_over_nominal": round(rel, 4), "cal_dev": round(cal_dev, 4),
                    "cap_idx_vs_M1": round(cap, 4),
                    "mean_var_base_pct": round(float(sub.loc[BASE_SCOPE, "mean_var"]), 4),
                    "T_base": int(sub.loc[BASE_SCOPE, "T"]),
                    "es_base_pct": round(float(sub.loc[BASE_SCOPE, "es"]), 4)
                                   if np.isfinite(sub.loc[BASE_SCOPE, "es"]) else np.nan,
                    "rate_full_pct": round(float(sub.loc["full", "rate_pct"]), 4),
                    "rate_highvol_pct": round(float(sub.loc["highvol", "rate_pct"]), 4)
                                        if "highvol" in sub.index else np.nan,
                    "rate_calm_pct": round(float(sub.loc["calm", "rate_pct"]), 4)
                                     if "calm" in sub.index else np.nan,
                    "T_highvol": t_hi,
                    "stab_scope_pp": round(stab_scope, 4),
                    "gap_regime_pp": round(gap, 4) if np.isfinite(gap) else np.nan,
                    "signed_gap_pp": round(sg, 4) if np.isfinite(sg) else np.nan,
                    "gap_evaluable": gap_eval, "n_ind_suspect": n_ind,
                })
    return pd.DataFrame(rows)


def decide(S: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按门槛 1/2、稳定性否决、跨置信度自洽、排序与并列裁决逐层定级。

    参数：
        S: build_scorecard 返回的评分表。

    返回：
        (D, picks) 二元组。
        D 为定案表，每行一个 (model, caliber)：g1_pass_rate / g2_prudent 为两条门槛的通过情况，
        cal_dev_mean / cal_dev_95 / cal_dev_99 为两档校准偏差及其均值，
        cap_idx_95 / cap_idx_99、rate_95 / rate_99 为两档的资本占用指数与基准样本失败率，
        gap_95 / gap_99 / gap_95_eval 为两档的条件覆盖落差及是否可评估，n_ind_max 为聚集性存疑
        计数的两档最大值，tier 取「推荐」「备选」「淘汰」，reason 为淘汰或降级原因（多条用「；」
        连接，无原因记 "—"）。
        picks 为 dict，键 (口径, "model") 给出该口径的基准模型名，(口径, "rule") 给出定选规则
        文本；某口径没有任何模型进「推荐」档时，只写 (口径, "model") 且值为 None，
        不写 (口径, "rule") 键。

    备注：
        模型须在 95% 与 99% 两档上都出现在 S 中才参与判定，故 g1/g2 天然是跨两档的合取。
        稳定性否决只把模型降为「备选」，不淘汰；排名只在「推荐」档内按校准偏差做。
    """
    rec = []
    for cal in ("主", "次"):
        for m in MODELS:
            sub = S[(S.caliber == cal) & (S.model == m)]
            if len(sub) < len(CONFS):
                continue
            sub = sub.set_index("conf")
            n_core = int(sub.loc[95, "n_core_scopes"])

            # 门槛 1 & 2
            g1 = bool(sub["pass_core"].all())
            g2 = bool((~sub["under_coverage"]).all() and sub["prudent"].all())
            # 稳定性否决
            vetoes = []
            if len(sub[sub["gap_evaluable"] & (sub["gap_regime_pp"] > GAP_BAR)]):
                vetoes.append(f"条件覆盖落差 >{GAP_BAR:.0f}pp")
            n_ind_max = int(sub["n_ind_suspect"].max())
            if n_ind_max >= N_IND_BAR:
                vetoes.append(f"聚集性存疑 ×{n_ind_max}")

            if not g1 or not g2:
                tier = "淘汰"
            elif vetoes:
                tier = "备选"
            else:
                tier = "推荐"

            # 淘汰原因（只在淘汰档生成）
            reasons = []
            for conf in CONFS:
                r = sub.loc[conf]
                if not r["pass_core"]:
                    reasons.append(f"{conf}% Kupiec 未过 {int(r['n_core_scopes']-r['n_pass_uc_core'])}"
                                   f"/{int(r['n_core_scopes'])} 个核心范围")
                if r["under_coverage"]:
                    reasons.append(f"{conf}% 出现显著风险低估")
                if not r["prudent"]:
                    reasons.append(f"{conf}% 失败率 {r['rate_base_pct']:.2f}% 超名义 "
                                   f"{r['nominal_pct']:.0f}% 的 {(r['rate_over_nominal']-1)*100:.0f}%")
            reasons += vetoes
            # 两档原因相同时只留一条
            seen, uniq = set(), []
            for t in reasons:
                k = t.split(" ", 1)[1] if " " in t else t
                if k not in seen:
                    seen.add(k)
                    uniq.append(t)
            reasons = uniq

            rec.append({
                "model": m, "caliber": cal,
                "g1_pass_rate": g1, "g2_prudent": g2,
                "cal_dev_mean": round(float(sub["cal_dev"].mean()), 4),
                "cal_dev_95": round(float(sub.loc[95, "cal_dev"]), 4),
                "cal_dev_99": round(float(sub.loc[99, "cal_dev"]), 4),
                "cap_idx_95": round(float(sub.loc[95, "cap_idx_vs_M1"]), 4),
                "cap_idx_99": round(float(sub.loc[99, "cap_idx_vs_M1"]), 4),
                "rate_95": round(float(sub.loc[95, "rate_base_pct"]), 4),
                "rate_99": round(float(sub.loc[99, "rate_base_pct"]), 4),
                "gap_95": float(sub.loc[95, "gap_regime_pp"]),
                "gap_99": float(sub.loc[99, "gap_regime_pp"]),
                "gap_95_eval": bool(sub.loc[95, "gap_evaluable"]),
                "n_ind_max": n_ind_max,
                "tier": tier, "reason": "；".join(reasons) if reasons else "—",
            })
    D = pd.DataFrame(rec)

    # 排序与并列裁决
    picks = {}
    for cal in ("主", "次"):
        c = D[(D.caliber == cal) & (D.tier == "推荐")].sort_values("cal_dev_mean")
        if len(c) == 0:
            picks[(cal, "model")] = None
            continue
        head = c.iloc[0]
        tied = c[c["cal_dev_mean"] <= head["cal_dev_mean"] * (1 + CAL_TIE)]
        if len(tied) > 1:
            pick = tied.sort_values(["cap_idx_95", "cap_idx_99"]).iloc[0]
            picks[(cal, "rule")] = (f"并列裁决：{'、'.join(tied['model'])} 两档校准偏差相差 "
                                    f"<{CAL_TIE:.0%}，取资本占用更低者")
        else:
            pick, picks[(cal, "rule")] = head, "过门槛者中两档校准偏差最小"
        picks[(cal, "model")] = pick["model"]
    return D, picks


def main() -> None:
    """聚合 9/17 的回测长表，为两个口径各定一版基准模型，并出评分表与取舍平面图。

    脚本契约：
        消费：results/backtest_results.csv，只保留 MODELS 里的模型行（M1g-t 是对照、
              M1d 是恒等式，都在读入后即排除）。
        产出：results/model_scorecard.csv（评分表，在 build_scorecard 的列后并入 verdict）、
              results/baseline_decision.csv（定案表，逐模型给出 tier 与淘汰/降级原因）、
              figures/baseline_model_tradeoff.png（95% 与 99% 两张散点图，横轴为相对 M1 的资本
              占用指数、纵轴为基准样本失败率，标出推荐模型并画出名义水平与审慎条款上限）。
        断言/边界：无硬断言。脚本不重算任何检验，只做聚合与排序，以保证与 9/17 回测报告逐字一致。
              基准样本写死为 common773（主口径 773 天 / 次口径 768 天）；HS750 窗宽 750 天，
              在该样本上只剩 523 天。主次口径各定一版，但每个口径内 95% 与 99% 必须是同一版模型，
              否则两档损失不可比。
    """
    R = pd.read_csv(RES / "backtest_results.csv")
    R = R[R["model"].isin(MODELS)].copy()     # 排除 M1g-t（对照）与 M1d（恒等式）
    print(f"[输入] backtest_results.csv 主模型行数={len(R)}")

    S = build_scorecard(R)
    D, picks = decide(S)

    S = S.merge(D[["model", "caliber", "tier"]].rename(columns={"tier": "verdict"}),
                on=["model", "caliber"], how="left")
    S["verdict"] = S["verdict"].fillna("淘汰")

    S.to_csv(RES / "model_scorecard.csv", index=False, encoding="utf-8-sig")
    D.to_csv(RES / "baseline_decision.csv", index=False, encoding="utf-8-sig")
    print(f"[1] 评分表 → results/model_scorecard.csv          {S.shape[0]} 行 × {S.shape[1]} 列")
    print(f"[2] 定案表 → results/baseline_decision.csv        {D.shape[0]} 行 × {D.shape[1]} 列")

    print("\n" + "=" * 118)
    print("逐模型判定（基准样本 common773；校准偏差=两档 |rate/名义−1| 均值，越小越准）")
    print("=" * 118)
    cols = ["model", "caliber", "g1_pass_rate", "g2_prudent", "cal_dev_95", "cal_dev_99",
            "cal_dev_mean", "cap_idx_95", "cap_idx_99", "gap_95", "gap_99", "n_ind_max", "tier"]
    print(D.sort_values(["caliber", "tier", "cal_dev_mean"])
          .replace({True: "Y", False: "N"})[cols].to_string(index=False))

    print("\n淘汰 / 降级原因")
    for _, r in D[D.tier != "推荐"].sort_values(["caliber", "model"]).iterrows():
        print(f"  [{r['tier']}] {r['model']}（{r['caliber']}口径）：{r['reason']}")

    print("\n" + "=" * 118)
    for cal in ("主", "次"):
        m = picks.get((cal, "model"))
        if m is None:
            print(f"[定案] {cal}口径：无模型通过门槛 1 + 门槛 2")
            continue
        r = D[(D.caliber == cal) & (D.model == m)].iloc[0]
        s3 = S[(S.caliber == cal) & (S.model == m)].set_index("conf")
        print(f"[定案] {cal}口径 基准 = {m}（95% 与 99% 同一版模型）")
        print(f"       通过率   Kupiec 核心范围 "
              f"{int(s3.loc[95, 'n_pass_uc_core'])}/{int(s3.loc[95, 'n_core_scopes'])}（95%）、"
              f"{int(s3.loc[99, 'n_pass_uc_core'])}/{int(s3.loc[99, 'n_core_scopes'])}（99%）全过")
        print(f"       校准     95% 失败率 {r['rate_95']:.2f}%（名义 5%，相对 "
              f"{r['rate_95']/5:.3f}）；99% 失败率 {r['rate_99']:.2f}%（名义 1%，相对 "
              f"{r['rate_99']/1:.3f}）；两档校准偏差均值 {r['cal_dev_mean']:.4f}")
        print(f"       经济性   资本占用 {r['cap_idx_95']:.3f}×M1（95%）/ {r['cap_idx_99']:.3f}×M1（99%），"
              f"即较参数法基准省 {(1-r['cap_idx_95'])*100:.1f}% / {(1-r['cap_idx_99'])*100:.1f}%")
        print(f"       稳定性   条件覆盖落差 {r['gap_95']:.2f}pp（95%）/ {r['gap_99']:.2f}pp（99%），"
              f"均在 {GAP_BAR:.0f}pp 门槛内；聚集性存疑 {int(r['n_ind_max'])} 处")
        print(f"       规则     {picks[(cal, 'rule')]}")
        alt = D[(D.caliber == cal) & (D.tier == "推荐") & (D.model != m)] \
            .sort_values("cal_dev_mean")
        for _, a in alt.iterrows():
            print(f"       并列项   {a['model']}：校准偏差 {a['cal_dev_mean']:.4f}"
                  f"（较基准差 {(a['cal_dev_mean']/r['cal_dev_mean']-1)*100:.0f}%），"
                  f"资本占用 {a['cap_idx_95']:.3f}×M1")
    print("=" * 118)

    print(f"\n[摘要] 基准口径在基准样本（{BASE_SCOPE}）上的 VaR / ES 水平")
    for cal in ("主", "次"):
        m = picks.get((cal, "model"))
        if m is None:
            continue
        sub = S[(S.caliber == cal) & (S.model == m)].set_index("conf")
        print(f"  {cal}口径 {m}："
              + "；".join(f"{c}% VaR {sub.loc[c, 'mean_var_base_pct']:.4f}% / "
                          f"ES {sub.loc[c, 'es_base_pct']:.4f}%"
                          f"（超损倍数 {sub.loc[c, 'es_base_pct']/sub.loc[c, 'mean_var_base_pct']:.3f}）"
                          for c in CONFS))

    # ---- 图：基准样本上的资本占用 × 覆盖率 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, conf in zip(axes, CONFS):
        nom = NOMINAL[conf]
        sc = S[S["conf"] == conf].sort_values(["rate_base_pct", "cap_idx_vs_M1"])
        for _, r_ in sc.iterrows():
            v = r_["verdict"]
            if v == "推荐":
                ax.scatter(r_["cap_idx_vs_M1"], r_["rate_base_pct"], s=78, marker="o",
                           color=CA[r_["caliber"]], edgecolor="white", linewidth=0.9, zorder=3)
            elif v == "备选":
                ax.scatter(r_["cap_idx_vs_M1"], r_["rate_base_pct"], s=58, marker="D",
                           color=CA[r_["caliber"]], alpha=0.5, edgecolor="white",
                           linewidth=0.8, zorder=3)
            else:
                ax.scatter(r_["cap_idx_vs_M1"], r_["rate_base_pct"], s=42, marker="x",
                           color="#9a9a96", alpha=0.75, linewidth=1.2, zorder=2)
        # 只给过门槛的模型直标，并按顺序交替上下偏移防重叠
        lab = sc[sc["verdict"] == "推荐"]
        for k, (_, r_) in enumerate(lab.iterrows()):
            dy = 7 if k % 2 == 0 else -13
            ax.annotate(f"{r_['model']}({r_['caliber']})",
                        (r_["cap_idx_vs_M1"], r_["rate_base_pct"]),
                        textcoords="offset points", xytext=(8, dy),
                        fontsize=8.5, color="#3a3a38", zorder=5)
        ax.axhspan(nom, nom * PRUDENCE, color="#9a9a96", alpha=0.13, zorder=0)
        ax.axhline(nom, color="#9a9a96", ls="--", lw=1.1, zorder=1)
        bb = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85)
        ax.text(0.02, nom, f"名义 {nom:.0f}%", va="bottom", ha="left", bbox=bb,
                transform=ax.get_yaxis_transform(), fontsize=8.5, color="#3a3a38", zorder=6)
        ax.text(0.02, nom * PRUDENCE, f"审慎条款上限 +{(PRUDENCE-1)*100:.0f}%",
                va="bottom", ha="left", bbox=bb,
                transform=ax.get_yaxis_transform(), fontsize=7.5, color="#8a8a86", zorder=6)
        ax.set_ylim(0, max(sc["rate_base_pct"].max(), nom * PRUDENCE) * 1.18)
        ax.set_axisbelow(True)
        ax.set_title(f"{conf}% 置信度", fontsize=10.5)
        ax.set_xlabel(f"资本占用指数（{BASE_SCOPE} 平均 VaR ÷ 同口径同置信度 M1）")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"基准样本（{BASE_SCOPE}）失败率 (%)")
    h = [plt.Line2D([], [], marker="o", ls="", color=CA["主"], label="主口径"),
         plt.Line2D([], [], marker="o", ls="", color=CA["次"], label="次口径"),
         plt.Line2D([], [], marker="D", ls="", color="#9a9a96", alpha=0.5, label="备选（稳定性降级）"),
         plt.Line2D([], [], marker="x", ls="", color="#9a9a96", label="淘汰")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("基准口径定案：资本占用与覆盖率的取舍平面", fontsize=12.5, y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG / "baseline_model_tradeoff.png", dpi=150, bbox_inches="tight")
    print("[图] figures/baseline_model_tradeoff.png")


if __name__ == "__main__":
    main()
