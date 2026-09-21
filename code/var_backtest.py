"""滚动窗口回测：失败率 + Kupiec + Christoffersen 检验，出违规时间线与覆盖率图。

产出 results/backtest_results.csv、results/backtest_exceptions.csv 与两张图。

**执行入口**：模块级只做导入与常量定义，全部测算在 `main()` 内、由 `__main__`
守卫触发——`import var_backtest` 不产生任何输出、不写任何文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import RES, FIG, EVENTS_CSV, FACT, ensure_dirs
from var_common import (C_M1, C_GARCH, C_ALT, C_GREY, INK2, WIN,
                        hit_series, kupiec_lr, christoffersen_lr, christoffersen_from_counts,
                        accept_region, lr_null, lr_ind_null_conditional, p_mc, lr_cc)


B_MC = 20000            # 蒙特卡洛重复数（spec §7.2 L261 的「检验力有限」量化所需）
P_NOM = {"95": 0.05, "99": 0.01}

# ---------------------------------------------------------------- 模型注册表
# (tag, role, 源, 主口径列模板, 次口径列模板)。次口径 None = 该变体无次口径序列
# （GARCH-t 只有主口径，9/15 未生成 rmb_garch_t_*，此处如实反映，不臆造）。
MODELS = [
    ("M1",     "主",                 "par", "m1_var_{c}",      "rmb_m1_var_{c}"),
    ("M1g",    "主",                 "par", "garch_var_{c}",   "rmb_garch_var_{c}"),
    ("M1e",    "主",                 "par", "ewma_var_{c}",    "rmb_ewma_var_{c}"),
    # M1g-t 实为「t 似然估的 σ_t」+「正态分位」——实测 garch_t_var_{95,99}/sig_garch_t
    # 恒为 1.6449/2.3263（1020 个值只在浮点末位抖动），**不是** t 分位 VaR。
    # role 直接写明，防日后被读成「已做 Student-t 分位数」。
    ("M1g-t",  "对照-t估σ+正态分位",  "par", "garch_t_var_{c}", None),
    ("M1d",    "恒等式-不作独立证据",  "par", "dnorm_var_{c}",   "rmb_dnorm_var_{c}"),
    ("HS250",  "主",                 "his", "hs250_var_{c}",   "rmb_hs250_var_{c}"),
    ("HS500",  "主",                 "his", "hs500_var_{c}",   "rmb_hs500_var_{c}"),
    ("HS750",  "主",                 "his", "hs750_var_{c}",   "rmb_hs750_var_{c}"),
    ("FHS-E",  "主",                 "his", "fhs_e_var_{c}",   "rmb_fhs_e_var_{c}"),
    ("FHS-G",  "主",                 "his", "fhs_g_var_{c}",   "rmb_fhs_g_var_{c}"),
]
CAL_COL = {"主": "ret_tr_pct", "次": "ret_rmb_pct"}
# 图只画 8 个主模型：M1d 与 M1 的违规集合完全相同（恒等式），M1g-t 是分布变体对照，
# 三者进 CSV 与检验表，但不进图，避免「多一个点就当多一个模型」的误读。
PLOT_MODELS = ["M1", "M1g", "M1e", "HS250", "HS500", "HS750", "FHS-E", "FHS-G"]
FAMILY = {"M1": "参数法", "M1g": "参数法", "M1e": "参数法",
          "HS250": "经典 HS", "HS500": "经典 HS", "HS750": "经典 HS",
          "FHS-E": "Filtered-HS", "FHS-G": "Filtered-HS"}
FAM_COLOR = {"参数法": C_M1, "经典 HS": C_ALT, "Filtered-HS": C_GARCH}

# spec §6 L222-224 登记的 VaR 跳变点（大额亏损日进出窗驱动，非模型失效）；
# 「9/17 凡遇违规日，先查是否落在该类跳变点附近再归因」
GHOST_JUMPS = [pd.Timestamp(x) for x in
               ["2022-09-27", "2023-10-04", "2025-04-08", "2026-04-09"]]

# 9/16 已登记的计数（硬断言基线，防 9/16 与 9/17 口径分叉）
BASELINE = {
    ("M1", "95"): (29, 1023), ("M1g", "95"): (30, 1023), ("M1e", "95"): (45, 1023),
    ("HS250", "95"): (44, 1023), ("HS500", "95"): (20, 773), ("HS750", "95"): (9, 523),
    ("FHS-E", "95"): (44, 773), ("FHS-G", "95"): (42, 773),
    ("M1", "99"): (8, 1023), ("M1g", "99"): (9, 1023), ("M1e", "99"): (15, 1023),
    ("HS250", "99"): (9, 1023), ("HS500", "99"): (4, 773), ("HS750", "99"): (1, 523),
}
BASELINE_COMMON523 = {("HS250", "95"): 28, ("HS500", "95"): 15, ("HS750", "95"): 9,
                      ("HS250", "99"): 5, ("HS500", "99"): 3, ("HS750", "99"): 1}
BASELINE_COMMON773 = {("HS250", "95"): 38, ("FHS-E", "95"): 44, ("FHS-G", "95"): 42,
                      ("HS250", "99"): 8, ("FHS-E", "99"): 11, ("FHS-G", "99"): 11}




def sec(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")

def valid_index(tag: str, src: str, caliber: str, conf: str) -> pd.DatetimeIndex:
    """某模型在某口径/置信度下的有效日（收益与 VaR 均非 NaN）。"""
    rcol = CAL_COL[caliber]
    col = next((m if caliber == "主" else rm)
               for t, _, s, m, rm in MODELS if t == tag and s == src)
    if col is None:
        return pd.DatetimeIndex([])
    d = pd.concat([SRC[src][rcol].rename("r"),
                   SRC[src][col.format(c=conf)].rename("v")], axis=1).dropna()
    return d.index

def transition_counts(idx: pd.DatetimeIndex, hit: np.ndarray, drop) -> tuple:
    """按**剔除前**的原始日历相邻性数转移对：任一端点为剔除日的 `(t−1, t)` 对整对丢弃。

    不把 `d−1` 与 `d+1` 重新拼成一对——聚集性说的就是**日历相邻**，跨越被剔除日重拼
    会凭空造一条不存在的邻接关系。这是 9/17 含/不含对照里唯一容易被写错的地方。

    传入 `idx`/`hit` 必须是**未剔除**的完整序列；返回 (n00, n01, n10, n11, 丢弃对数)。
    """
    a, b = idx[:-1], idx[1:]
    ha, hb = hit[:-1], hit[1:]
    if drop is not None and len(drop):
        keep = ~a.isin(drop) & ~b.isin(drop)
    else:
        keep = np.ones(len(a), dtype=bool)
    ha, hb = ha[keep], hb[keep]
    return (int(np.sum(~ha & ~hb)), int(np.sum(~ha & hb)),
            int(np.sum(ha & ~hb)), int(np.sum(ha & hb)), int((~keep).sum()))

def evaluate(tag: str, role: str, src: str, caliber: str, conf: str,
             index: pd.DatetimeIndex | None, drop=None) -> dict | None:
    """对一个 (模型, 口径, 置信度, 范围) 组合出具完整检验行。"""
    rcol = CAL_COL[caliber]
    _, _, _, m, rm = next(r for r in MODELS if r[0] == tag)
    col = m if caliber == "主" else rm
    if col is None:
        return None
    d0 = pd.concat([SRC[src][rcol].rename("r"),
                    SRC[src][col.format(c=conf)].rename("v")], axis=1).dropna()
    if index is not None:
        d0 = d0.loc[d0.index.intersection(index)]
    if len(d0) < 30:
        return None
    # 转移对按**剔除前**的完整序列定相邻关系（见 transition_counts）
    hit0 = (d0["r"] < -d0["v"]).values
    n00, n01, n10, n11, n_pair_dropped = transition_counts(d0.index, hit0, drop)

    d = d0 if not (drop is not None and len(drop)) else d0.loc[~d0.index.isin(drop)]
    n_dropped = len(d0) - len(d)
    # 防「静默空操作」：剔除集合与序列索引若因类型不匹配而完全不相交，np.isin/pandas.isin
    # 会给出一片 False，剔除数为 0，于是「两版结果相同」变成一个假结论——本样本上
    # 2025-03-24/2025-01-21 确实命中违规，真剔除数必 > 0。故按预期条数硬断言。
    if drop is not None and len(drop):
        expect = len([x for x in drop if x in d0.index])
        assert n_dropped == expect, f"剔除数 {n_dropped} ≠ 预期 {expect}——剔除未生效（静默空操作）"
    if len(d) < 30:
        return None
    p = P_NOM[conf]
    h = d["r"] < -d["v"]
    n, x = len(h), int(h.sum())

    lr_uc, p_uc = kupiec_lr(x, n, p)
    lr_ind, p_ind, n00, n01, n10, n11 = christoffersen_from_counts(n00, n01, n10, n11)
    n_pairs = n00 + n01 + n10 + n11
    lr_cc_v, p_cc = lr_cc(lr_uc, lr_ind)

    # 有限样本 p：无条件零分布（与 size 表同一份模拟）+ 条件零分布（给定观测违规数）
    nul = lr_null(n, p, B=B_MC)
    p_uc_mc = p_mc(nul["lr_uc"], lr_uc)
    p_ind_mc = p_mc(nul["lr_ind"], lr_ind)
    p_ind_mc_cond = p_mc(lr_ind_null_conditional(n, x, B=B_MC), lr_ind)

    x_lo, x_hi = accept_region(n, p)
    viol = d.loc[h, "r"]
    mean_var = float(d["v"].mean())
    es = float(-viol.mean()) if len(viol) else float("nan")
    # 零收益日（NAV 原样结转）机械上不可能构成 r < −VaR，故名义期望 N·p 偏高。
    # 并列给出「有效可违规日」口径的期望，供报告引用时二选一（本表两列都给）。
    n_zero = int((d["r"].abs() < 1e-12).sum())
    return dict(model=tag, role=role, caliber=caliber, conf=conf, scope="", T=n, x=x,
                rate_pct=x / n * 100.0, exp_x=n * p, n_zero=n_zero,
                exp_x_eff=(n - n_zero) * p, lr_uc=lr_uc, p_uc=p_uc,
                p_uc_mc=p_uc_mc, lr_ind=lr_ind, p_ind=p_ind, p_ind_mc=p_ind_mc,
                p_ind_mc_cond=p_ind_mc_cond, lr_cc=lr_cc_v, p_cc=p_cc,
                n00=n00, n01=n01, n10=n10, n11=n11, n_pairs=n_pairs,
                x_lo=x_lo, x_hi=x_hi, mean_var=mean_var, es=es,
                es_ratio=(es / mean_var if mean_var and np.isfinite(es) else float("nan")),
                size_uc=nul["size_uc"], size_ind=nul["size_ind"],
                n_pair_dropped=n_pair_dropped)

def _near_event(d):
    m = EVT[(EVT["date"] >= d - pd.Timedelta(days=6)) & (EVT["date"] <= d + pd.Timedelta(days=6))]
    return "；".join(f"{r['date'].date()} {r['event_cn']}" for _, r in m.iterrows())

def main() -> None:
    """滚动窗口回测：失败率 + Kupiec + Christoffersen 检验，出违规时间线与覆盖率图。…（完整说明见模块 docstring）"""
    global SRC, EVT
    ensure_dirs()

    # ---------------------------------------------------------------- 载入
    sec("载入 9/15 / 9/16 产物与回测辅助数据")
    VP = pd.read_csv(RES / "var_parametric.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    VH = pd.read_csv(RES / "var_historical.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    REG = pd.read_csv(RES / "regimes.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    DBT = pd.read_csv(RES / "doubtful_days.csv", encoding="utf-8-sig", parse_dates=["date"])
    EVT = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig", parse_dates=["date"])
    B31 = pd.read_csv(FACT / "factor_table_3141HK.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    SRC = {"par": VP, "his": VH}
    OOS = VP.index                                   # 样本外 1023 日
    print(f"  样本外 {len(OOS)} 日：{OOS[0].date()} ~ {OOS[-1].date()}")
    assert VP.index.equals(VH.index), "两个 VaR 产物的日期向量不一致——口径已分叉，停止"

    # 存疑日按口径分集合（spec §8.1 L281：清单是**市价口径**识别，映射到 NAV 收益逐条核对）
    ETF_D = list(DBT.loc[DBT["series"] == "ETF_ret_pct", "date"])
    FX_D = list(DBT.loc[DBT["series"] == "FX_ret_pct", "date"])
    OAS_D = list(DBT.loc[DBT["series"] == "dOAS_bp", "date"])
    DROP = {"主": [d for d in ETF_D if d in OOS],
            "次": [d for d in ETF_D + FX_D if d in OOS]}
    DROP_ALL = [d for d in list(DBT["date"]) if d in OOS]      # 放宽版敏感性：不打折扣全剔
    print(f"  存疑日 16 条：ETF {len(ETF_D)} / dOAS {len(OAS_D)} / FX {len(FX_D)}")
    print(f"  落样本外并可剔除：主口径 {len(DROP['主'])} 条、次口径 {len(DROP['次'])} 条、放宽版 {len(DROP_ALL)} 条")
    print(f"  dOAS 类 {len(OAS_D)} 条**不剔除**：标注的是 OAS 序列异常，未改动 ETF 收益，而 VaR 的判定对象是 ETF 收益")

    # 零收益日：NAV 原样结转（港股休市或数据陈旧），r=0 机械上不可能构成违规
    _nz = int((VP["ret_tr_pct"].reindex(OOS).abs() < 1e-12).sum())
    print(f"\n  【口径提示】主口径样本外有 {_nz}/{len(OOS)} 日（{_nz / len(OOS) * 100:.2f}%）复权收益恰为 0.0000%"
          f"——这些日子**机械上不可能违规**。")
    print(f"  故名义期望违规数须分两个口径报：含零收益日 95% {len(OOS) * .05:.1f} / 99% {len(OOS) * .01:.1f}；"
          f"剔除后 95% {(len(OOS) - _nz) * .05:.1f} / 99% {(len(OOS) - _nz) * .01:.1f}。")
    print("  长表同时给 exp_x（含零收益日）与 exp_x_eff（剔除）两列，报告引用时须二选一并声明。")


    # 跨模型公共样本：口径 → 置信度 → 日期交集
    COMMON = {}
    for caliber in ("主", "次"):
        for conf in ("95", "99"):
            idxs = {}
            for tag, role, src, m, rm in MODELS:
                if caliber == "次" and rm is None:
                    continue
                i = valid_index(tag, src, caliber, conf)
                if len(i):
                    idxs[tag] = i
            c523 = None
            for i in idxs.values():
                c523 = i if c523 is None else c523.intersection(i)
            # 773 组：剔除最宽的 HS750 后的交集
            c773 = None
            for t, i in idxs.items():
                if t == "HS750":
                    continue
                c773 = i if c773 is None else c773.intersection(i)
            COMMON[(caliber, conf)] = {"common523": c523, "common773": c773}
    print(f"  公共样本：523 组 {len(COMMON[('主','95')]['common523'])} 日、"
          f"773 组 {len(COMMON[('主','95')]['common773'])} 日")


    sec("逐配置回测")
    ROWS, EXC = [], []
    SCOPES = ["full", "common773", "common523", "calm", "highvol", "ex_doubtful", "ex_doubtful_all"]
    for caliber in ("主", "次"):
        for conf in ("95", "99"):
            reg = REG["chronic_high"].reindex(OOS).fillna(False).astype(bool)
            for tag, role, src, m, rm in MODELS:
                if caliber == "次" and rm is None:
                    continue
                base = valid_index(tag, src, caliber, conf)
                if not len(base):
                    continue
                for scope in SCOPES:
                    if scope == "full":
                        ix, dr = base, None
                    elif scope in ("common773", "common523"):
                        ix, dr = COMMON[(caliber, conf)][scope], None
                    elif scope == "calm":
                        ix, dr = base[~reg.reindex(base).fillna(False).values], None
                    elif scope == "highvol":
                        ix, dr = base[reg.reindex(base).fillna(False).values], None
                    elif scope == "ex_doubtful":
                        ix, dr = base, DROP[caliber]
                    else:
                        ix, dr = base, DROP_ALL
                    row = evaluate(tag, role, src, caliber, conf, ix, dr)
                    if row is None:
                        continue
                    row["scope"] = scope
                    ROWS.append(row)
                    if scope == "full":
                        ser = pd.concat([SRC[src][CAL_COL[caliber]].rename("r"),
                                         SRC[src][(m if caliber == "主" else rm).format(c=conf)].rename("v")],
                                        axis=1).dropna()
                        for dt in ser.index[ser["r"] < -ser["v"]]:
                            EXC.append(dict(model=tag, caliber=caliber, conf=conf, date=dt,
                                            ret=float(ser.loc[dt, "r"]), var=float(ser.loc[dt, "v"]),
                                            exceed_ratio=float(-ser.loc[dt, "r"] / ser.loc[dt, "v"])))
    BT = pd.DataFrame(ROWS)
    EX = pd.DataFrame(EXC)
    print(f"  检验行 {len(BT)} 条；违规记录 {len(EX)} 条")


    # ---------------------------------------------------------------- 硬断言：与 9/16 已登记数字对账
    sec("口径对账（硬断言：9/16 已登记的每一个数字都必须复现）")
    bad = []
    for (tag, conf), (ex_x, ex_n) in BASELINE.items():
        r = BT[(BT.model == tag) & (BT.caliber == "主") & (BT.conf == conf) & (BT.scope == "full")]
        if not len(r):
            bad.append(f"{tag}/{conf}: 缺行"); continue
        if (int(r["x"].iloc[0]), int(r["T"].iloc[0])) != (ex_x, ex_n):
            bad.append(f"{tag}/{conf}: 实得 {int(r['x'].iloc[0])}/{int(r['T'].iloc[0])} ≠ 登记 {ex_x}/{ex_n}")
    for nm, base in (("common523", BASELINE_COMMON523), ("common773", BASELINE_COMMON773)):
        for (tag, conf), ex_x in base.items():
            r = BT[(BT.model == tag) & (BT.caliber == "主") & (BT.conf == conf) & (BT.scope == nm)]
            if not len(r) or int(r["x"].iloc[0]) != ex_x:
                got = int(r["x"].iloc[0]) if len(r) else "NA"
                bad.append(f"{nm} {tag}/{conf}: 实得 {got} ≠ 登记 {ex_x}")
    if bad:
        for b in bad:
            print("  [不一致] " + b)
        raise AssertionError("口径对账失败——9/17 与 9/16 已分叉，停止出具结论")
    print(f"  主口径 full 8 模型 × 2 置信度 + 公共 523/773 组：全部与 9/16 登记一致 ✓")
    # 恒等式断言：M1δ 与 M1 的违规集合必须逐个相同（spec §4 L102-103）
    for conf in ("95", "99"):
        a = EX[(EX.model == "M1") & (EX.caliber == "主") & (EX.conf == conf)]["date"].sort_values().tolist()
        b = EX[(EX.model == "M1d") & (EX.caliber == "主") & (EX.conf == conf)]["date"].sort_values().tolist()
        assert a == b, f"M1δ 与 M1 违规日集合不同（{conf}）——恒等式被破坏"
    print("  M1δ ≡ M1 违规日集合逐一相同 ✓（恒等式，不作独立证据）")


    # ---------------------------------------------------------------- 存疑日含/不含 + 3141 核对
    sec("存疑日「含/不含」两版对照（spec §8.1：必做，不作可选项）")
    # 先列实际差异：哪些存疑日真的被判成违规
    hits_on_doubt = EX[EX["date"].isin(DBT["date"])].copy()
    if len(hits_on_doubt):
        print(f"  落入存疑清单的违规记录 {len(hits_on_doubt)} 条，涉及 "
              f"{hits_on_doubt['date'].nunique()} 个日期：")
        for dt, g in hits_on_doubt.groupby("date"):
            ser = DBT.loc[DBT["date"] == dt, "series"].iloc[0]
            print(f"    {dt.date()} [清单类别 {ser}] 被 {len(g)} 个 模型×口径×置信度 组合判为违规")
    else:
        print("  无违规日落入存疑清单")

    print("\n  --- 涉事日期的 3141.HK 同向性 + 事件时间线核对（spec §8.1 L280-281）---")
    for dt in sorted(hits_on_doubt["date"].unique()):
        dt = pd.Timestamp(dt)
        tr = VP.loc[dt, "ret_tr_pct"]; rmb = VP.loc[dt, "ret_rmb_pct"]
        b31 = B31.loc[dt, "etf_ret_pct"] if dt in B31.index else np.nan
        ev_near = EVT[(EVT["date"] >= dt - pd.Timedelta(days=9)) & (EVT["date"] <= dt + pd.Timedelta(days=9))]
        gh = [g for g in GHOST_JUMPS if abs((g - dt).days) <= 6]
        print(f"  · {dt.date()}  9141复权 {tr:+.4f}%  9141次口径 {rmb:+.4f}%  3141 {b31:+.4f}%  "
              f"同向={bool(np.sign(tr) == np.sign(b31)) if np.isfinite(b31) else 'NA'}")
        ev_txt = "；".join(f"{r['date'].date()} {r['event_cn']}" for _, r in ev_near.iterrows())
        print(f"      跳变点邻近：{('是 ' + str([str(g.date()) for g in gh])) if gh else '否'}"
              f"   事件(±9 日)：{ev_txt or '无'}")

    print("\n  --- 两版对照（仅从**评估样本**剔除；VaR 与收益序列一字不动）---")
    cmp_rows = []
    for (tag, role, src, m, rm), caliber in [(r, c) for r in MODELS for c in ("主", "次")]:
        if caliber == "次" and rm is None:
            continue
        for conf in ("95", "99"):
            inc = BT[(BT.model == tag) & (BT.caliber == caliber) & (BT.conf == conf) & (BT.scope == "full")]
            exc = BT[(BT.model == tag) & (BT.caliber == caliber) & (BT.conf == conf) & (BT.scope == "ex_doubtful")]
            alx = BT[(BT.model == tag) & (BT.caliber == caliber) & (BT.conf == conf) & (BT.scope == "ex_doubtful_all")]
            if not len(inc) or not len(exc):
                continue
            cmp_rows.append(dict(model=tag, caliber=caliber, conf=conf,
                                 含_x=int(inc["x"].iloc[0]), 不含_x=int(exc["x"].iloc[0]),
                                 全剔_x=int(alx["x"].iloc[0]) if len(alx) else -1,
                                 含_T=int(inc["T"].iloc[0]), 不含_T=int(exc["T"].iloc[0]),
                                 含_p=inc.p_uc.iloc[0], 不含_p=exc.p_uc.iloc[0]))
    CMP = pd.DataFrame(cmp_rows)
    d1 = CMP[CMP["含_x"] != CMP["不含_x"]]
    d2 = CMP[CMP["含_x"] != CMP["全剔_x"]]
    print(f"  按预注册规则剔除后计数**发生变化的配置**：{len(d1)} 条")
    if len(d1):
        print(d1[["model", "caliber", "conf", "含_x", "不含_x", "含_T", "不含_T"]].to_string(index=False))
    print(f"  放宽版（剔除全部 16 条存疑日）下发生变化的配置：{len(d2)} 条")
    if len(d2):
        print(d2[["model", "caliber", "conf", "含_x", "全剔_x"]].to_string(index=False))
    print("\n  结论：spec §6 L229 预注册的「两版结果相同是预期结论」**未成立**——")
    print("  预注册的依据是「ETF 类存疑日 |r| 在全样本排名最靠前 223/1273，不进任何 W 的尾部 1%」，")
    print("  但 95% 口径的尾部宽达 5%，「不进 1% 尾部」推不出「不违规」。2025-03-24 在 W=250 窗内升序排第 11/250")
    print("  （约 4.4% 分位）——进得了 95% 尾部、进不了 99% 尾部，故只在 95% 上构成违规。")


    # ---------------------------------------------------------------- 分段
    sec("分段评估（spec §7.3）—— 平稳 / 高波动")
    seg = BT[BT.scope.isin(["calm", "highvol"])]
    piv = seg.pivot_table(index=["model", "caliber", "conf"], columns="scope",
                          values=["T", "x", "rate_pct", "mean_var", "es"], aggfunc="first")
    print(piv.round(4).to_string())
    cw = REG["crisis_window"].reindex(OOS).fillna(False).astype(bool)
    print(f"\n  【关键发现】急性危机窗（spec §7.3：2022-06-06~06-22）在样本外命中 "
          f"{int(cw.sum())} 天 —— 该窗整体落在估计窗内，样本外**无危机日**，故不建危机段检验表。")
    print(f"  分段标志口径：含当日的 60 日滚动 σ（事后同期标记），**不是**事前可知的高波动段（spec §7.3 L267-269）。")
    hv99 = BT[(BT.scope == "highvol") & (BT.conf == "99") & (BT.caliber == "主")]
    if len(hv99):
        print(f"  高波动段仅 {int(hv99['T'].iloc[0])} 天 → 99% 期望违规仅 {hv99['exp_x'].iloc[0]:.1f} 次，"
              f"该段 LR 检验**基本没有检验力**，结论只能作描述性对比。")


    # ---------------------------------------------------------------- 极端日反应（周计划 9/17 下午 4 之「谁先反应」）
    sec("极端日反应：最差 |r| 日各模型 VaR 在自身 250 日历史中的分位")
    print("  读法：分位 = 当日 VaR 在该模型前 250 日 VaR 分布中的百分位。")
    print("  分位低 ⇒ 当天模型**还没把 VaR 抬起来**（未先反应）；分位高 ⇒ 已反应。")
    worst = VP["ret_tr_pct"].reindex(OOS).nsmallest(6)
    ext_rows = []
    for dt in worst.index:
        for mdl in PLOT_MODELS:
            _, _, src, m, rm = next(r for r in MODELS if r[0] == mdl)
            col = m.format(c="99")
            ser = SRC[src][col] if col in SRC[src].columns else None
            if ser is None or dt not in ser.index or not np.isfinite(ser.loc[dt]):
                continue
            hist = ser.loc[:dt].iloc[-WIN - 1:-1].dropna()
            if len(hist) < 50:
                continue
            pct = float((hist < ser.loc[dt]).mean() * 100)
            v95 = SRC[src][m.format(c="95")].loc[dt]
            ext_rows.append(dict(date=dt, model=mdl, ret=VP.loc[dt, "ret_tr_pct"],
                                 var99=float(ser.loc[dt]), pct99=pct,
                                 viol99=bool(VP.loc[dt, "ret_tr_pct"] < -ser.loc[dt]),
                                 viol95=bool(VP.loc[dt, "ret_tr_pct"] < -float(v95))))
    EXT = pd.DataFrame(ext_rows)
    for dt, ret in worst.items():
        print(f"\n  {dt.date()}  主口径复权收益 {ret:+.4f}%"
              f"{'  【零收益日：NAV 原样结转，主口径机械上不可能违规】' if abs(ret) < 1e-12 else ''}")
        g = EXT[EXT["date"] == dt] if len(EXT) else EXT
        if not len(g):
            t = OOS.get_loc(dt)
            print(f"    → 无可用分位：该日距样本外起点仅 {t + 1} 日，各模型 VaR 序列不足 250 日历史，"
                  f"无法在自身分布中定位。")
            continue
        print("    " + "  ".join(f"{r.model}:{r.pct99:.0f}分位{'✗违规' if r.viol99 else ''}" for r in g.itertuples()))
        gj = [x for x in GHOST_JUMPS if (OOS.get_loc(x) - OOS.get_loc(dt) == 1) if x in OOS]
        print(f"    → 99% 违规 {int(g['viol99'].sum())} 个模型；各模型 VaR 分位中位数 {g['pct99'].median():.0f}"
              f"（<50 说明多数模型当天尚未抬升 VaR）")
        if gj:
            print(f"      注意：该日是登记跳变点 {gj[0].date()} 的**前一交易日**——大额亏损次日入窗会把 VaR 抬上去，"
                  f"故此处低分位是 ghost effect 的机械结果，不能读成「模型没反应」。")

    # ---------------------------------------------------------------- 自检 + size 表
    sec("自检与有限样本水平（χ²(1) 近似的实际拒绝率）")
    sz = []
    for n, p in [(1023, .05), (773, .05), (523, .05), (299, .05),
                 (1023, .01), (773, .01), (523, .01), (299, .01), (133, .01)]:
        d = lr_null(n, p, B=B_MC)
        sz.append(dict(N=n, p=p, size_uc_pct=d["size_uc"] * 100, size_ind_pct=d["size_ind"] * 100,
                       q95_uc=float(np.quantile(d["lr_uc"], .95)), q95_ind=float(np.quantile(d["lr_ind"], .95))))
    SZ = pd.DataFrame(sz)
    print(SZ.round(3).to_string(index=False))
    print("  χ²(1) 95% 临界值 = 3.8415。95% 口径下 LR_ind 名义 5% 实际约 7~8%（偏宽松）；")
    print("  99% 口径下实际仅约 1.3~1.9%（偏保守）——**不显著是检验力不足下的弱结论，不等于证明无聚集**。")

    print("\n  --- 检验实现自检 ---")
    l7, p7 = kupiec_lr(7, 250, 0.01)
    print(f"  教材例 T=250, x=7, p=1%：LR_uc={l7:.5f}（手算 5.4974）、p={p7:.5f}（0.01905）")
    for lbl, h in {"全 0": np.zeros(50, bool), "全 1": np.ones(50, bool), "单次": np.r_[np.zeros(5, bool), True, np.zeros(4, bool)]}.items():
        print(f"  LR_ind {lbl:<4} = {christoffersen_lr(h)[0]:.4f}（全 0/全 1 应为 0）")
    a, b = 3.2, 1.7
    print(f"  LR_cc 恒等：{lr_cc(a, b)[0]:.4f} = {a}+{b}")
    msk = np.random.default_rng(0).random(1023) < 0.03
    lr_v = christoffersen_lr(msk)[0]
    from var_common import _lr_vec
    print(f"  向量化 vs 标量 LR_ind：{_lr_vec(msk[None, :])[1][0]:.10f} vs {lr_v:.10f} → "
          f"{'一致' if abs(_lr_vec(msk[None, :])[1][0] - lr_v) < 1e-10 else '不一致'}")

    print("\n  --- 前视复核：随机抽 3 个违规日手工重算 VaR（沿用 9/16 做法）---")
    rng = np.random.default_rng(20260917)
    pool = [d for d in EX[(EX.model == "HS250") & (EX.caliber == "主")
                          & (EX.conf == "99")]["date"].values
            if OOS.get_loc(pd.Timestamp(d)) >= WIN]        # 窗未满的日子没有可比的手工值
    for dt in rng.choice(np.array(pool, dtype="datetime64[ns]"), 3, replace=False):
        dt = pd.Timestamp(dt)
        t = int(OOS.get_loc(dt))
        w = VP["ret_tr_pct"].iloc[t - WIN:t].dropna()
        manual = -np.quantile(w, 0.01)
        got = VH.loc[dt, "hs250_var_99"]
        print(f"    {dt.date()}  手工 {manual:.6f}  产物 {got:.6f}  差 {abs(manual - got):.2e}")


    # ---------------------------------------------------------------- 出表
    sec("落盘")
    R = BT.copy()
    R["rate_pct"] = R["rate_pct"].round(6); R["exp_x"] = R["exp_x"].round(4)
    for c in ["lr_uc", "p_uc", "p_uc_mc", "lr_ind", "p_ind", "p_ind_mc", "p_ind_mc_cond",
              "lr_cc", "p_cc", "size_uc", "size_ind"]:
        R[c] = R[c].round(6)
    for c in ["mean_var", "es", "es_ratio"]:
        R[c] = R[c].round(6)
    cols = ["model", "role", "caliber", "conf", "scope", "T", "x", "rate_pct", "exp_x",
            "n_zero", "exp_x_eff",
            "lr_uc", "p_uc", "p_uc_mc", "lr_ind", "p_ind", "p_ind_mc", "p_ind_mc_cond",
            "lr_cc", "p_cc", "n00", "n01", "n10", "n11", "n_pairs", "n_pair_dropped",
            "x_lo", "x_hi", "mean_var", "es", "es_ratio", "size_uc", "size_ind"]
    R[cols].to_csv(RES / "backtest_results.csv", index=False, encoding="utf-8-sig")
    print(f"  [表] results/backtest_results.csv  {len(R)} 行 × {len(cols)} 列")

    EX["is_doubtful"] = EX["date"].isin(DBT["date"])
    EX["doubtful_series"] = EX["date"].map(lambda d: (DBT.loc[DBT["date"] == d, "series"].iloc[0]
                                                      if (DBT["date"] == d).any() else ""))
    EX["near_ghost"] = EX["date"].apply(lambda d: any(abs((g - d).days) <= 6 for g in GHOST_JUMPS))
    EX["near_event"] = EX["date"].apply(_near_event)
    EX["co_3141_pct"] = EX["date"].map(lambda d: float(B31.loc[d, "etf_ret_pct"]) if d in B31.index else np.nan)
    EX = EX.sort_values(["caliber", "conf", "date", "model"])
    EX.to_csv(RES / "backtest_exceptions.csv", index=False, encoding="utf-8-sig")
    print(f"  [表] results/backtest_exceptions.csv  {len(EX)} 行")


    # ---------------------------------------------------------------- 图 1
    sec("出图 1：违规时间线 + 市场事件标注")
    ret95 = VP["ret_tr_pct"].reindex(OOS)
    ret99 = VP["ret_rmb_pct"].reindex(OOS)
    # 事件标注**数据驱动**：只标注落在任一违规日 ±3 交易日内的既有事件（避免手工挑事件的选择性
    # 偏差），再把交易日相邻（间隔 <= 3 个交易日）的事件并成「事件簇」——2025-04-08/09/10/11 这类
    # 连续冲击本就是同一段行情，逐日标注只会互相压叠。簇按命中违规条数排序取前 6，逐一编号。
    ev_hits = {}
    for d in EX["date"].unique():
        t = OOS.get_loc(d)
        lo, hi = OOS[max(0, t - 3)], OOS[min(len(OOS) - 1, t + 3)]
        for _, r in EVT[(EVT["date"] >= lo) & (EVT["date"] <= hi)].iterrows():
            ev_hits.setdefault((r["date"], r["event_cn"]), 0)
            ev_hits[(r["date"], r["event_cn"])] += 1
    _cl = []
    for (d, name), c in sorted(ev_hits.items(), key=lambda kv: kv[0][0]):
        t = int(OOS.searchsorted(d))          # 非交易日事件用插入位置排序，仍参与成簇
        if _cl and t - _cl[-1]["t1"] <= 3:
            cl = _cl[-1]
            cl["t1"] = t
            cl["dates"].append(d)
            cl["n_ev"] += 1
            cl["hits"] += c
            if c > cl["top_hits"]:
                cl["top_hits"], cl["top_date"], cl["top_name"] = c, d, name
        else:
            _cl.append({"t1": t, "dates": [d], "n_ev": 1, "hits": c,
                        "top_hits": c, "top_date": d, "top_name": name})
    NUM = "①②③④⑤⑥"
    top_ev = sorted(_cl, key=lambda cl: -cl["hits"])[:6]
    for i, cl in enumerate(top_ev):
        ds = sorted(set(cl["dates"]))
        in_oos = [d for d in ds if d in OOS]
        # 每个簇只画一条线；锚点优先取命中最多的那天，其次取簇内首个交易日
        cl["anchor"] = (cl["top_date"] if cl["top_date"] in OOS
                        else (in_oos[0] if in_oos else None))
        rng = ds[0].strftime("%Y-%m-%d")
        if len(ds) > 1:
            rng += "~" + ds[-1].strftime("%m-%d")
        # 事件名截断取**自然断点**（空格/括号/顿号），避免把词切成半截；无断点则硬截 18 字
        _brk = [p for p in (cl["top_name"].find(c) for c in " (（、") if p >= 4]
        short = cl["top_name"][:min(_brk)] if _brk else cl["top_name"][:18]
        cl["label"] = (f"{NUM[i]} {rng}　{short}"
                       + (f" 等 {cl['n_ev']} 事件" if cl["n_ev"] > 1 else "")
                       + f"（命中 {cl['hits']} 条违规）")
    # 编号分层：日期相近的簇若同层会把编号叠在一起（① 2025-04-08 与 ③ 2025-05-07 只隔 1 个月），
    # 故按交易日间隔贪心分层——同层内相邻锚点至少隔 90 个交易日，最多两层。
    _LVL_Y, _LVL_GAP = (7.55, 8.02), 90
    _lvl_last = [None] * len(_LVL_Y)
    for cl in sorted(top_ev, key=lambda c: c["anchor"] or OOS[0]):
        if cl["anchor"] is None:
            cl["lvl"] = 0
            continue
        t = int(OOS.get_loc(cl["anchor"]))
        cl["lvl"] = next((i for i, last in enumerate(_lvl_last) if last is None or t - last >= _LVL_GAP), 0)
        _lvl_last[cl["lvl"]] = t
    print("  被违规日命中最多的事件簇（数据驱动，非手工挑选；相邻交易日的事件已并簇）：")
    for cl in top_ev:
        note = "　[锚点非交易日，仅列名不画线]" if cl["anchor"] is None else f"　[编号层 {cl['lvl']}]"
        print("    " + cl["label"] + note)

    fig = plt.figure(figsize=(15.5, 9.2))
    # 不在 GridSpec 上设 hspace/wspace：tight_layout 会覆盖它们并发出「Axes not compatible」警告，
    # 行距改由 tight_layout 的 h_pad/w_pad 控制
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.15])
    for ci, caliber in enumerate(("主", "次")):
        ret = ret95 if caliber == "主" else ret99
        for ri, conf in enumerate(("95", "99")):
            ax = fig.add_subplot(gs[ri, ci])
            for k, mdl in enumerate(PLOT_MODELS):
                dd = EX[(EX.model == mdl) & (EX.caliber == caliber) & (EX.conf == conf)]["date"]
                if len(dd):
                    ax.scatter(dd, np.full(len(dd), k), s=17, marker="|",
                               color=FAM_COLOR[FAMILY[mdl]], linewidths=1.5, zorder=3)
            ax.set_yticks(range(len(PLOT_MODELS)))
            ax.set_yticklabels(PLOT_MODELS if ci == 0 else [""] * len(PLOT_MODELS), fontsize=8.5)
            # 底部留白给事件编号；95% 与 99% 两行共用同一 ylim，保证同名模型上下对齐可比
            ax.set_ylim(-0.7, 8.35)
            ax.invert_yaxis()
            ax.grid(axis="x", color="#e6e5e2", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            nom = 51 if conf == "95" else 10
            ax.set_title(f"{'主口径（USD 复权）' if caliber == '主' else '次口径（人民币视角）'} · "
                         f"{conf}% VaR 违规日（名义期望约 {nom} 次 / 1023 日）", fontsize=9.5, loc="left")
            for cl in top_ev:
                if cl["anchor"] is not None:
                    ax.axvline(cl["anchor"], color=INK2, lw=0.8, ls=":", alpha=0.5, zorder=1)
            if ri == 0:
                # 编号贴在 95% 面板底部留白带：不压标题、不压最下一行模型标记、也不压下方收益条
                for i, cl in enumerate(top_ev):
                    if cl["anchor"] is not None:
                        ax.text(cl["anchor"], _LVL_Y[cl["lvl"]], NUM[i], fontsize=9,
                                ha="center", va="center", color=INK2, zorder=4)
            if ri == 1:
                ax.set_xlabel("")
            else:
                ax.set_xticklabels([])
        axr = fig.add_subplot(gs[2, ci])
        axr.bar(ret.index, ret.values, width=1.0, color=C_GREY, zorder=2)
        axr.set_ylabel("日收益 %", fontsize=8)
        axr.set_ylim(-2.0, 2.0)
        axr.grid(axis="x", color="#e6e5e2", lw=0.6, zorder=0)
        axr.set_axisbelow(True)
        for s in ("top", "right"):
            axr.spines[s].set_visible(False)
        for cl in top_ev:
            if cl["anchor"] is not None:
                axr.axvline(cl["anchor"], color=INK2, lw=0.8, ls=":", alpha=0.5, zorder=1)
        axr.xaxis.set_major_locator(mdates.YearLocator())
        axr.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axr.tick_params(labelsize=8)
    handles = [Line2D([], [], marker="|", ls="", color=FAM_COLOR[f], markersize=10, markeredgewidth=1.6, label=f)
               for f in ("参数法", "经典 HS", "Filtered-HS")]
    handles.append(Line2D([], [], color=INK2, ls=":", lw=1.0,
                          label="市场事件簇（仅标注违规日 ±3 交易日内的既有事件，相邻交易日者并簇；编号见下方对照表）"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.946))
    fig.suptitle("滚动回测违规时间线（样本外 2022-08 ~ 2026-09，估计窗 250 日 / 步长 1 日）\n"
                 "M1δ（恒等式，与 M1 违规日逐一相同）与 M1g-t（分布变体）不在图内",
                 fontsize=11, y=0.992)
    # 编号对照表放图下缘：放顶部会压到各面板标题（面板标题在坐标区之外，tight_layout 的 rect 管不到）
    for j, row in enumerate((top_ev[:3], top_ev[3:])):
        fig.text(0.5, 0.055 - 0.022 * j, "　　".join(cl["label"] for cl in row),
                 ha="center", va="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.082, 1, 0.902), h_pad=1.1, w_pad=0.6)
    fig.savefig(FIG / "backtest_exceptions.png", dpi=150)
    plt.close(fig)
    print("\n[图] figures/backtest_exceptions.png")

    # ---------------------------------------------------------------- 图 2
    sec("出图 2：覆盖率 + Kupiec 精确接受区间（检验力可视化）")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for ri, conf in enumerate(("95", "99")):
        # 同一置信度下主/次口径**共用纵轴**：各自 autoscale 会让两列尺度不同，横向比就成了错觉
        _b = []
        for _c in ("主", "次"):
            _s = BT[(BT.conf == conf) & (BT.caliber == _c) & (BT.scope == "full")
                    & (BT.model.isin(PLOT_MODELS))].set_index("model").reindex(PLOT_MODELS).dropna(subset=["x"])
            _b += list(_s["rate_pct"]) + list(_s["x_lo"] / _s["T"] * 100) + list(_s["x_hi"] / _s["T"] * 100)
        _pad = 0.07 * (max(_b) - min(_b))
        ylim = (max(0.0, min(_b) - _pad), max(_b) + _pad)     # 失败率非负，下界不越 0
        for ci, caliber in enumerate(("主", "次")):
            ax = axes[ri, ci]
            sub = BT[(BT.conf == conf) & (BT.caliber == caliber) & (BT.scope == "full")
                     & (BT.model.isin(PLOT_MODELS))].set_index("model").reindex(PLOT_MODELS).dropna(subset=["x"])
            xs = np.arange(len(sub))
            nom = P_NOM[conf] * 100
            lo = sub["x_lo"] / sub["T"] * 100
            hi = sub["x_hi"] / sub["T"] * 100
            for i, (mdl, r) in enumerate(sub.iterrows()):
                col = FAM_COLOR[FAMILY[mdl]]
                ax.plot([i, i], [lo.iloc[i], hi.iloc[i]], color=col, lw=2.0, alpha=0.45, zorder=2,
                        solid_capstyle="butt")
                ax.scatter(i, r.rate_pct, s=64, color=col, zorder=4,
                           edgecolor="white", linewidth=1.2,
                           marker="o" if r.p_uc >= 0.05 else "D")
            ax.set_ylim(*ylim)
            ax.axhline(nom, color=INK2, lw=1.4, ls="--", zorder=1)
            ax.text(len(sub) - 0.45, nom, f"名义 {nom:g}%", fontsize=8, color=INK2, va="bottom", ha="right")
            ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=35, ha="right", fontsize=8.5)
            ax.set_title(f"{'主口径（USD 复权）' if caliber == '主' else '次口径（人民币视角）'} · {conf}%", fontsize=9.5, loc="left")
            # 「95%」是**检验置信度**，两行都是它，与面板的 VaR 95%/99% 不是一回事，故写明「检验」
            ax.set_ylabel("实际失败率 %（点=估计，竖线=Kupiec 95% 检验接受区间）" if ci == 0 else "", fontsize=8.5)
            ax.grid(axis="y", color="#e6e5e2", lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=FAM_COLOR[f], markersize=8, label=f) for f in ("参数法", "经典 HS", "Filtered-HS")]
    handles += [Line2D([], [], marker="D", ls="", color=INK2, markersize=7, label="菱形 = Kupiec p<5%（拒绝）"),
                Line2D([], [], marker="o", ls="", color=INK2, markersize=8, mfc="none", label="圆圈 = 未拒绝"),
                Line2D([], [], color=INK2, ls="--", lw=1.4, label="名义失败率")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.938))
    fig.suptitle("失败率点估计 + Kupiec 精确接受区间：区间宽度即检验力——99% 面板的接受区间宽到几乎覆盖全图\n"
                 "落入区间内 = 覆盖率无可辨识偏差；区间外 = 拒绝（本样本的拒绝全部是「过度覆盖」方向）",
                 fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    fig.savefig(FIG / "backtest_coverage.png", dpi=150)
    plt.close(fig)
    print("[图] figures/backtest_coverage.png")

    sec("对比结论（供 9/18 报告取用，与 spec §7 补记一致）")
    _uc = R[(R.scope == "full") & (R.model.isin(PLOT_MODELS))]
    _rej = _uc[_uc.p_uc < 0.05]
    print(f"  1) 覆盖率：full scope 共 {len(_uc)} 个检验，Kupiec 拒绝 {len(_rej)} 处 ——")
    print(f"     全部是「过度覆盖」方向（x 小于期望 ⇒ VaR 偏高、资本浪费），**不是风险低估**；")
    print(f"     LR_uc 是双边检验，拒绝本身不等于模型危险。")
    _sig = R[R.p_ind_mc_cond < 0.05]
    _pre = R[(R.scope == "full") & (R.model.isin(PLOT_MODELS))]
    print(f"  2) 聚集：全表 {len(R)} 行中 p_ind(MC)<0.05 者 {len(_sig)} 行，"
          f"其中预注册主比较集（{len(_pre)} 次）内 {int((_pre.p_ind_mc_cond < 0.05).sum())} 处；")
    print(f"     11 行全部落在 95%、集中在小样本段与次口径 HS500，且六种 scope 高度重叠、非独立发现。")
    print(f"     Bonferroni：主比较集 0.05/{len(_pre)} = {0.05/len(_pre):.5f}；全表 0.05/{len(R)} = {0.05/len(R):.5f}；"
          f"全局最小 p_ind(MC) = {R.p_ind_mc_cond.min():.5f} —— 两种口径下均**不成立**。")
    print(f"     → 不得宣称「HS500 在人民币口径下会聚集」，也不得宣称「GARCH 消除了聚集」。")
    _b = R[(R.caliber == "主") & (R.scope == "full")]
    _b2 = R[(R.caliber == "次") & (R.scope == "full") & (R.model == "M1")]
    _m = _b[_b.model == "M1"]
    print(f"  3) 双口径：主口径 M1 95% 失败率 {float(_m[_m.conf.astype(str)=='95'].rate_pct.iloc[0]):.2f}%"
          f"（被 Kupiec 拒绝），次口径 M1 95% 失败率 {float(_b2[_b2.conf.astype(str)=='95'].rate_pct.iloc[0]):.2f}%"
          f"（不拒绝）——")
    print(f"     汇率成分降低收益自相关，使无条件覆盖更接近名义；是 9/15 归因表（汇率贡献 44.2%）的独立佐证。")
    print(f"  4) 边界解段（2022-08-03~2023-04-10，133 天）：N=133/p=0.01 下 LR_uc 实际水平仅 1.17%、LR_ind 0.82%，")
    print(f"     该段检验几乎不拒绝任何东西；任何「次口径 GARCH 在高波动段更优」的说法都不得引用这段。")

    sec("完成")
    print(f"  检验行 {len(R)}、违规记录 {len(EX)}")
    print(f"  违规判定表列：is_doubtful / doubtful_series / near_ghost / near_event / co_3141_pct")


if __name__ == "__main__":
    main()
