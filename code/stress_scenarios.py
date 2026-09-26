"""压力测试情景库构建：历史情景 + 假设情景（阶段三 Day1 · 9/21）

消费：factors/factor_table_nav_tr.csv
产出：results/stress_scenarios.csv（情景库长表，一行 = 情景 × 因子 × measure）
      figures/stress_scenario_library.png
口径：需求文档第四章第三阶段——历史情景取 3 次典型冲击期间风险因子的最大波动，假设情景按
      利率大幅上行 / 信用利差阶梯走阔 / 离岸汇率大幅波动三个方向各设轻度、中度、极端三档，
      幅度取历史极端值的 1 / 1.5 / 2 倍。梯度锚点取全样本单日极值（measure=peak_1d），
      历史情景基准取窗内最差单日（worst_day），两者同为 1 日口径，与阶段二 VaR 对齐；
      3 日累计峰值（peak_3d）留在表中作附录对照，不参与判定。风险方向：d5y_bp / d10y_bp /
      doas_bp 上行为风险，fx_ret_pct 下行为风险，取极值须按方向 max/min。
边界：oas_bp / doas_bp 仅自 2023-09-05 起有值，H1、H2、X1 三窗内信用利差不可用，表中标
      available=False，不插补也不用其他口径数值替代；fx_ret_pct 与 etf_ret_rmb_pct 尾部
      缺 5 日，次口径全样本 n=1268。
用法：./.venv/bin/python code/stress_scenarios.py
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

from common import FACT, RES, FIG, ensure_dirs
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2

ensure_dirs()

# ---- 情景定义 ----
# 历史与补充情景：编号 → (窗起, 窗止, 名称, 主导方向)
HIST = {
    "H1": ("2022-06-09", "2022-06-16", "2022 美联储快速加息", "利率上行"),
    "H2": ("2023-03-10", "2023-03-17", "2023 美国银行业风波", "避险(方向相反)"),
    "H3": ("2025-04-07", "2025-04-11", "2025 关税冲击", "利率+信用+汇率"),
    "X1": ("2022-11-29", "2022-12-05", "2022 离岸人民币贬值(补充)", "汇率贬值"),
    "X2": ("2022-03-08", "2022-03-15", "2022 俄乌与加息周期开启(补充)", "利率上行"),
}
HIST_ORDER = ["H1", "H2", "H3", "X1", "X2"]

# 因子 → (中文名, 单位, 风险方向符号)： +1 表示数值上行是风险，-1 表示下行是风险
FACTORS = {
    "d5y_bp":     ("5年期美债收益率变动", "bp", +1),
    "d10y_bp":    ("10年期美债收益率变动", "bp", +1),
    "doas_bp":    ("新兴市场IG信用利差变动", "bp", +1),
    "fx_ret_pct": ("人民币兑美元汇率收益", "%", -1),
}

# 假设情景：三类核心冲击方向 → 该方向下的因子
ASSUM_DIRS = {
    "利率上行":     ["d5y_bp", "d10y_bp"],
    "信用利差走阔": ["doas_bp"],
    "离岸汇率贬值": ["fx_ret_pct"],
}
SEVERITY = [("轻度", 1.0), ("中度", 1.5), ("极端", 2.0)]

# 持有期口径的唯一来源，假设情景锚点与历史情景基准都用它。
HORIZON_DAYS = 1      # 1 日，与阶段二 VaR 口径对齐（老师 9 月口头指示）
ANCHOR_WIN = HORIZON_DAYS   # 梯度锚点窗口 = 单日
ROLL_WIN = 3          # 附录对照用的 3 日窗长，只进 peak_3d 行，不参与判定
OAS_START = pd.Timestamp("2023-09-05")


# ---- 工具 ----
def risk_extreme(s: pd.Series, sign: int) -> float:
    """按风险方向取序列极值，作为该因子的情景冲击幅度。

    参数：
        s: 待取极值的序列，可为因子原值，也可为其滚动累计和。
        sign: 风险方向，+1 表示数值上行是风险（利率、信用利差），
              -1 表示数值下行是风险（人民币贬值）。

    返回：
        float，sign > 0 时取 s.max()，sign < 0 时取 s.min()。

    备注：
        利率上行与汇率贬值的滚动极值落在相反两端，故调用处一律传 FACTORS 的方向符号列，
        不能统一取 max。
    """
    return float(s.max() if sign > 0 else s.min())


def path_metrics(s: pd.Series) -> tuple[float, float]:
    """把组合收益路径折算为累计收益与最大回撤。

    参数：
        s: 组合收益序列（%），通常取事件窗内的 etf_ret_tr_pct 或 etf_ret_rmb_pct。

    返回：
        (cum, mdd) 二元组，均为 float、单位 %：cum 为窗内累计收益（正 = 收益），
        mdd 为窗内最大回撤（≤ 0，含事件前起点 NAV=1.0 的净值口径）。

    备注：
        两条口径都是硬约束，写错会失真。收益是 log 收益，etf_ret_rmb_pct = etf_ret_tr_pct
        + fx_ret_pct 严格可加（实测最大绝对偏差 2.2e-16），故累计收益直接求和，不是复利连乘；
        最大回撤必须含事件前起点 NAV=1.0，只从窗内首个收益起算会漏掉「事件首日就是最高点」的
        情形，实测把 H3 的 MDD 低估一半以上（不含起点 −1.14% vs 含起点 −2.26%）。
    """
    cum = s.cumsum() / 100.0                      # log 收益累计
    eq = pd.concat([pd.Series([1.0]), np.exp(cum)], ignore_index=True)
    mdd = float((eq / eq.cummax() - 1.0).min() * 100.0)
    return float(s.sum()), mdd


def main() -> None:
    """构建历史与假设压力情景库，落盘情景表与情景库图。

    脚本契约：
        消费：factors/factor_table_nav_tr.csv（复权因子表）。
        产出：results/stress_scenarios.csv（情景库长表，一行 = 情景 × 因子 × measure）、
              figures/stress_scenario_library.png（三方向 × 三梯度）。
        断言/边界：启动时断言 max|etf_ret_rmb_pct − etf_ret_tr_pct − fx_ret_pct| < 1e-9，
        失败即终止——整个双口径分析建立在这条 log 可加性上，不成立则「主次口径之差 =
        汇率折算项」与累计收益直接求和两处结论同时失效。已知边界：oas_bp / doas_bp 仅自
        2023-09-05 起有值，H1、H2、X1 三窗内信用利差不可用，表中标 available=False，
        不插补也不用其他口径数值替代；fx_ret_pct 与 etf_ret_rmb_pct 尾部缺 5 日，
        次口径全样本 n=1268。
    """
    print("=" * 78)
    print("阶段三 Day1：压力测试情景库构建")
    print("=" * 78)

    # ---- 1) 载入 ----
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    print(f"[输入] factor_table_nav_tr.csv  {F.shape[0]} 行 × {F.shape[1]} 列  "
          f"{F.index.min().date()} ~ {F.index.max().date()}")

    # 口径自检：log 收益可加性。整个双口径分析建立在此恒等式上，若这里不成立，
    # 「主次口径之差 = 汇率折算项」与累计收益直接求和两处结论同时失效。
    idn = (F["etf_ret_rmb_pct"] - F["etf_ret_tr_pct"] - F["fx_ret_pct"]).abs().max()
    print(f"       口径自检 max|rmb − tr − fx| = {idn:.3e}  → log 收益（累计=直接求和）")
    assert idn < 1e-9, (f"口径自检失败：max|rmb − tr − fx| = {idn:.3e}，"
                        f"收益非 log 可加，累计不可直接求和，双口径差值不再等于汇率项")

    # 数据可用性
    av = {c: (F[c].dropna().index.min(), int(F[c].notna().sum())) for c in FACTORS}
    print("       因子可用性：" + "  ".join(
        f"{c}=自 {v[0].date()} n={v[1]}" for c, v in av.items()))

    rows: list[dict] = []

    # ---- 2) 历史情景 ----
    print("\n" + "=" * 78)
    print("历史情景（需求文档点名的 3 次 + 1 个补充）")
    print("=" * 78)
    for sid in HIST_ORDER:
        a, b, name, lead = HIST[sid]
        W = F.loc[a:b]
        cls = "补充" if sid.startswith("X") else "历史"
        print(f"\n[{sid}] {name}  {W.index.min().date()} ~ {W.index.max().date()}"
              f"  ({len(W)} 交易日)  主导方向={lead}")

        for col, (cn, unit, sign) in FACTORS.items():
            s = W[col].dropna()
            if len(s) == 0:
                rows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                                 lead_direction=lead, window_start=a, window_end=b,
                                 factor=col, factor_cn=cn, unit=unit,
                                 measure="窗内实现", shock=np.nan, multiple=np.nan,
                                 available=False, anchor_date="",
                                 note=f"该窗内 {col} 无数据（序列自 {OAS_START.date()} 起）"))
                print(f"     {col:11s} 不可用")
                continue

            cum = float(s.sum())
            single = risk_extreme(s, sign)
            peak3 = risk_extreme(s.rolling(ROLL_WIN).sum().dropna(), sign)
            d_single = (s.idxmax() if sign > 0 else s.idxmin()).date()
            d_peak = (s.rolling(ROLL_WIN).sum().dropna().idxmax() if sign > 0
                      else s.rolling(ROLL_WIN).sum().dropna().idxmin()).date()

            for meas, val, dt in (("single_extreme", single, d_single),
                                  ("peak_3d", peak3, d_peak),
                                  ("cum_window", cum, W.index.max().date())):
                rows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                                 lead_direction=lead, window_start=a, window_end=b,
                                 factor=col, factor_cn=cn, unit=unit,
                                 measure=meas, shock=round(val, 4), multiple=np.nan,
                                 available=True, anchor_date=str(dt), note=""))
            print(f"     {col:11s} 单日={single:+8.4f} ({d_single})  "
                  f"3日峰值={peak3:+8.4f} ({d_peak})  累计={cum:+8.4f}  {unit}")

        # 组合路径
        for col, cn in (("etf_ret_tr_pct", "主口径ETF(USD复权)"),
                        ("etf_ret_rmb_pct", "次口径ETF(人民币)")):
            s = W[col].dropna()
            if len(s) == 0:
                continue
            cum, mdd = path_metrics(s)
            # 窗内最差单日：1 日重做后历史情景基准列的来源。逐日取最小，不等于各因子
            # `single_extreme` 的相加（各因子的极值落在不同日子），故须按组合收益序列
            # 单独算，不能由因子极值合成。
            wd = float(s.min())
            wd_dt = str(s.idxmin().date())
            for meas, val, dt in (("worst_day", wd, wd_dt),
                                  ("cum_window", cum, str(W.index.max().date())),
                                  ("max_drawdown", mdd, str(W.index.max().date()))):
                rows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                                 lead_direction=lead, window_start=a, window_end=b,
                                 factor=col, factor_cn=cn, unit="%",
                                 measure=meas, shock=round(val, 4), multiple=np.nan,
                                 available=True, anchor_date=dt, note=""))
            print(f"     {col:18s} 最差单日={wd:+8.4f}% ({wd_dt})  累计={cum:+8.4f}%  "
                  f"最大回撤={mdd:+8.4f}%")

    # ---- 3) 假设情景 ----
    print("\n" + "=" * 78)
    print(f"假设情景：三类冲击方向 × 轻度/中度/极端"
          f"（锚点 = 全样本 {ANCHOR_WIN} 日极值）")
    print("=" * 78)
    anchor: dict[str, tuple[float, str]] = {}      # 1 日锚点（主口径）
    anchor3: dict[str, tuple[float, str]] = {}     # 3 日锚点（附录对照，不参与判定）
    for col, (cn, unit, sign) in FACTORS.items():
        s = F[col].dropna()
        r1 = s.rolling(ANCHOR_WIN).sum().dropna()
        v = risk_extreme(r1, sign)
        dt = str((r1.idxmax() if sign > 0 else r1.idxmin()).date())
        anchor[col] = (v, dt)
        r3 = s.rolling(ROLL_WIN).sum().dropna()
        v3 = risk_extreme(r3, sign)
        dt3 = str((r3.idxmax() if sign > 0 else r3.idxmin()).date())
        anchor3[col] = (v3, dt3)
        print(f"  锚点 {col:11s} {ANCHOR_WIN}日={v:+9.4f} {unit:2s} ({dt})"
              f"   [附录对照] {ROLL_WIN}日={v3:+9.4f} ({dt3})")

    print()
    n_assum = 0
    for dir_name, cols in ASSUM_DIRS.items():
        print(f"  ── {dir_name} ──")
        for sev, mult in SEVERITY:
            n_assum += 1
            sid = f"S{n_assum:02d}"
            for col in cols:
                base, dt = anchor[col]
                base3, dt3 = anchor3[col]
                rows.append(dict(scenario_id=sid, scenario_class="假设",
                                 scenario_name=f"{dir_name}·{sev}",
                                 lead_direction=dir_name, window_start="", window_end="",
                                 factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                                 measure="peak_1d", shock=round(base * mult, 4),
                                 multiple=mult, available=True, anchor_date=dt,
                                 note=f"历史单日极值 {base:.4f} × {mult}"))
                # 3 日锚点的同梯度值：仅作附录多期近似的对照，不参与判定。
                rows.append(dict(scenario_id=sid, scenario_class="假设",
                                 scenario_name=f"{dir_name}·{sev}",
                                 lead_direction=dir_name, window_start="", window_end="",
                                 factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                                 measure="peak_3d", shock=round(base3 * mult, 4),
                                 multiple=mult, available=True, anchor_date=dt3,
                                 note=f"[附录对照，不参与判定] 历史 3 日累计峰值 {base3:.4f} × {mult}"))
                print(f"     {sid} {sev}({mult}×)  {col:11s} {base*mult:+9.4f} {FACTORS[col][1]}")
    # 组合档：利率与信用同时恶化（对齐 IMF WP/15/216「三种冲击同时发生」的做法）
    n_assum += 1
    sid = f"S{n_assum:02d}"
    print(f"  ── 利率+信用同时恶化（组合档）──")
    for col in ["d5y_bp", "d10y_bp", "doas_bp"]:
        base, dt = anchor[col]
        base3, dt3 = anchor3[col]
        rows.append(dict(scenario_id=sid, scenario_class="假设",
                         scenario_name="利率+信用同时恶化·极端",
                         lead_direction="组合", window_start="", window_end="",
                         factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                         measure="peak_1d", shock=round(base * 2.0, 4), multiple=2.0,
                         available=True, anchor_date=dt,
                         note="利率与信用同向极端（2×），对齐 IMF WP/15/216 多冲击同时发生"))
        rows.append(dict(scenario_id=sid, scenario_class="假设",
                         scenario_name="利率+信用同时恶化·极端",
                         lead_direction="组合", window_start="", window_end="",
                         factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                         measure="peak_3d", shock=round(base3 * 2.0, 4), multiple=2.0,
                         available=True, anchor_date=dt3,
                         note="[附录对照，不参与判定] 3 日锚点同梯度值"))
        print(f"     {sid} 极端(2×)  {col:11s} {base*2.0:+9.4f} {FACTORS[col][1]}")

    # ---- 4) 落盘 ----
    S = pd.DataFrame(rows)
    S.to_csv(RES / "stress_scenarios.csv", index=False, encoding="utf-8-sig")
    n_scen = S["scenario_id"].nunique()
    print(f"\n[1] 情景库 → results/stress_scenarios.csv  "
          f"{S.shape[0]} 行 × {S.shape[1]} 列（{n_scen} 个情景）")
    print(f"    历史 {S[S.scenario_class=='历史'].scenario_id.nunique()} 个 + "
          f"补充 {S[S.scenario_class=='补充'].scenario_id.nunique()} 个 + "
          f"假设 {S[S.scenario_class=='假设'].scenario_id.nunique()} 个")
    na = int((~S["available"]).sum())
    print(f"    不可用因子条目 {na} 条（均为 OAS 覆盖区间外，已显式标注、未插补）")

    # ---- 5) 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    labels = [s for s, _ in SEVERITY]
    x = np.arange(len(labels))

    panels = [("利率上行", ["d5y_bp", "d10y_bp"], [C_M1, C_GARCH], "美债收益率上行（bp）"),
              ("信用利差走阔", ["doas_bp"], [C_GARCH], "新兴市场 IG 利差走阔（bp）"),
              ("离岸汇率贬值", ["fx_ret_pct"], [C_ALT], "人民币贬值（%）")]
    for ax, (dir_name, cols, colors, ylab) in zip(axes, panels):
        w = 0.8 / len(cols)
        top = 0.0
        for i, (col, c) in enumerate(zip(cols, colors)):
            base = anchor[col][0]
            vals = [abs(base * m) for _, m in SEVERITY]
            pos = x - 0.4 + w * (i + 0.5)
            b = ax.bar(pos, vals, width=w * 0.86, color=c, label=FACTORS[col][0])
            top = max(top, max(vals))
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
        # 样本内最差单日参考线用图例项而非行内文字，避免与柱体重叠。1 日重做后这条线
        # 即梯度锚点本身，与「轻度」档柱顶齐平：轻度档 = 已实现的最差单日，中度/极端
        # 两档是它的 1.5 倍与 2 倍外推。这条线是锚点口径的可视证据。
        ref = max(abs(risk_extreme(F[c].dropna(), FACTORS[c][2])) for c in cols)
        ax.axhline(ref, color=INK2, ls="--", lw=1.1)
        ax.plot([], [], color=INK2, ls="--", lw=1.1,
                label=f"样本内最差单日 {ref:.1f}（=轻度档锚点）")
        ax.set_ylim(0, top * 1.30)          # 留出标注与图例空间
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n({m}×)" for s, m in SEVERITY], fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(dir_name, fontsize=10.5, loc="left")
        ax.grid(axis="y", color="#e6e5e2", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.6)

    fig.suptitle(f"假设情景三方向 × 三级梯度（锚点 = 全样本最差单日，{HORIZON_DAYS} 日持有期）",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "stress_scenario_library.png", dpi=150, bbox_inches="tight")
    print("[图] figures/stress_scenario_library.png")


if __name__ == "__main__":
    main()
