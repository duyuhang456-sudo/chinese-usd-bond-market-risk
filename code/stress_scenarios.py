"""
阶段三 Day1（9/21）：压力测试情景库构建（历史情景 + 假设情景）

需求文档第四章第三阶段：
  历史情景：筛选近 5 年 **3 次**典型市场冲击事件，提取各事件期间**风险因子的最大波动幅度**，
            作为情景冲击参数；
  假设情景：设计**三类核心冲击方向**——无风险利率大幅上行、信用利差阶梯式走阔、离岸汇率
            大幅波动；每类方向设置**轻度、中度、极端三个冲击梯度**，冲击幅度参考**历史极端值
            的 1 倍、1.5 倍、2 倍**设定。

------------------------------------------------------------------------------
一、情景编号与选型
------------------------------------------------------------------------------
历史情景（需求文档点名的 3 次）：
  H1  2022 美联储快速加息      2022-06-09 ~ 2022-06-16
  H2  2023 美国银行业风波      2023-03-10 ~ 2023-03-17
  H3  2025 关税冲击            2025-04-07 ~ 2025-04-11
补充情景（超出需求文档，**不参与主结论**，用于补上需求文档 3 次未覆盖到的情形）：
  X1  2022 离岸人民币贬值      2022-11-29 ~ 2022-12-05
  X2  2022 俄乌与加息周期开启  2022-03-08 ~ 2022-03-15

两条必须写明的选型事实（实测，见本脚本 [2] 段输出）：
  ① 需求文档举例的「2020 年疫情冲击」早于本项目因子表起点 2021-08-03，
     **落在数据范围之外**，无法提取该期间的风险因子波动。不伪造、不用近似事件冒充，
     以样本内同级冲击 H3 替代，并在情景库文档中记明该替代关系。
  ② H2（2023 银行业风波）在样本内是**避险行情而非损失**：主口径 ETF 复权累计 +2.00%。
     它是需求文档点名的 3 个之一，照实保留并如实标注方向，不硬凑成损失情景；
     真正的汇率损失样本由补充情景 X1 承担（该窗 fx 累计 −3.4785%，为全样本最剧 5 日贬值，
     且主口径 +1.4370% 的债券收益**不足抵消**，次口径落到 −2.0415%）。
  ③ X2 是**事后补入**的：做覆盖度检验（`stress_robustness.py`）时发现
     2022-03-08 ~ 2022-03-15 是**样本内实测最差的 6 日窗口**（主口径 −3.0008%、Δ5Y +39bp），
     其冲击与损失**双双超过**点名的 H1（−1.5368%、Δ5Y +32bp），
     而该事件本就在 `events/risk_events_timeline.csv` 中（2022-03-15 美联储首次加息 + 俄乌）。
     选型时只按「3 日利率峰值」挑窗（峰值确在 2022-06-14），漏掉了窗长更长时更差的一段。
     一个不含实测最差结果的压力情景库是有缺陷的，故补入并保留这一发现记录。

------------------------------------------------------------------------------
二、幅度口径（三套并列，不得混用）
------------------------------------------------------------------------------
  ① `single_extreme` 单日极值         —— 窗内/全样本最差单日
  ② `peak_3d`        3 日累计峰值     —— 滚动 3 日累计和在风险方向的极值  ← **梯度锚点**
  ③ `cum_window`     窗口累计变动     —— 窗首至窗末的净变动（历史情景专用）

**梯度锚点取 ②「3 日累计峰值」**，理由是它与阶段二 δ 传导验证的主窗口
（`verify_delta_transmission.py` 的 MAIN_WIN = "3日(主)"）同口径，偏差缓冲量级可直接沿用，
不必另做窗口换算。代价是「轻度」档（1×）已超过样本内任何已实现单日——如实写明，不淡化。

风险方向约定：`d5y_bp`/`d10y_bp`/`doas_bp` **上行**为风险，`fx_ret_pct` **下行**（人民币贬值）
为风险。滚动极值在 2022-06-14（利率上行）与 2022-12-05（汇率贬值）落在相反两端，取极值时
必须按方向分别取 max/min，不可一律取 max。

------------------------------------------------------------------------------
三、数据可用性（实测，不可静默插补）
------------------------------------------------------------------------------
`oas_bp`/`doas_bp` 仅自 **2023-09-05** 起有值（750 / 749 行），故：
  H1（2022-06）、H2（2023-03）、X1（2022-12）三个窗口内**信用利差数据完全不可用**，
  情景表中该因子一律标 `available=False`，**不插补、不用其他口径数值替代**。
同时 `fx_ret_pct` / `etf_ret_rmb_pct` 尾部缺 5 日（2026-08-31 ~ 09-04），次口径全样本 n=1268。

------------------------------------------------------------------------------
产出：results/stress_scenarios.csv   情景库长表（一行 = 情景 × 因子）
      figures/stress_scenario_library.png
用法： ./.venv/bin/python code/stress_scenarios.py
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

from common import REPO
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2

FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 情景定义
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

ROLL_WIN = 3          # 梯度锚点窗口：滚动 3 日累计和
OAS_START = pd.Timestamp("2023-09-05")


# ---------------------------------------------------------------- 工具
def risk_extreme(s: pd.Series, sign: int) -> float:
    """按风险方向取极值：sign=+1 取最大，sign=-1 取最小。"""
    return float(s.max() if sign > 0 else s.min())


def path_metrics(s: pd.Series) -> tuple[float, float]:
    """
    组合收益路径的累计收益与最大回撤。

    口径（两条都是硬约束，写错会失真）：
      1. 收益是 **log 收益**——`etf_ret_rmb_pct = etf_ret_tr_pct + fx_ret_pct` 严格可加
         （实测 max 绝对偏差 2.2e-16），故**累计收益 = 直接求和**，不是复利连乘。
      2. 最大回撤**必须含事件前起点 NAV=1.0**。只从窗内首个收益起算会漏掉
         「事件首日就是最高点」的情形，实测把 H3 的 MDD 低估一半以上
         （不含起点 −1.14% vs 含起点 −2.26%）。
    """
    cum = s.cumsum() / 100.0                      # log 收益累计
    eq = pd.concat([pd.Series([1.0]), np.exp(cum)], ignore_index=True)
    mdd = float((eq / eq.cummax() - 1.0).min() * 100.0)
    return float(s.sum()), mdd


def main() -> None:
    print("=" * 78)
    print("阶段三 Day1：压力测试情景库构建")
    print("=" * 78)

    # ---------------------------------------------------------- 1) 载入
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    print(f"[输入] factor_table_nav_tr.csv  {F.shape[0]} 行 × {F.shape[1]} 列  "
          f"{F.index.min().date()} ~ {F.index.max().date()}")

    # 口径自检：log 收益可加性（硬断言——整个双口径分析建立在此恒等式上，
    # 若这里不成立，则「主次口径之差 = 汇率折算项」与累计收益直接求和两处结论同时失效）
    idn = (F["etf_ret_rmb_pct"] - F["etf_ret_tr_pct"] - F["fx_ret_pct"]).abs().max()
    print(f"       口径自检 max|rmb − tr − fx| = {idn:.3e}  → log 收益（累计=直接求和）")
    assert idn < 1e-9, (f"口径自检失败：max|rmb − tr − fx| = {idn:.3e}，"
                        f"收益非 log 可加，累计不可直接求和，双口径差值不再等于汇率项")

    # 数据可用性
    av = {c: (F[c].dropna().index.min(), int(F[c].notna().sum())) for c in FACTORS}
    print("       因子可用性：" + "  ".join(
        f"{c}=自 {v[0].date()} n={v[1]}" for c, v in av.items()))

    rows: list[dict] = []

    # ---------------------------------------------------------- 2) 历史情景
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
            rows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                             lead_direction=lead, window_start=a, window_end=b,
                             factor=col, factor_cn=cn, unit="%",
                             measure="cum_window", shock=round(cum, 4), multiple=np.nan,
                             available=True, anchor_date=str(W.index.max().date()), note=""))
            rows.append(dict(scenario_id=sid, scenario_class=cls, scenario_name=name,
                             lead_direction=lead, window_start=a, window_end=b,
                             factor=col, factor_cn=cn, unit="%",
                             measure="max_drawdown", shock=round(mdd, 4), multiple=np.nan,
                             available=True, anchor_date=str(W.index.max().date()), note=""))
            print(f"     {col:18s} 累计={cum:+8.4f}%  最大回撤={mdd:+8.4f}%")

    # ---------------------------------------------------------- 3) 假设情景
    print("\n" + "=" * 78)
    print("假设情景：三类冲击方向 × 轻度/中度/极端（锚点 = 全样本 3 日累计峰值）")
    print("=" * 78)
    anchor: dict[str, tuple[float, str]] = {}
    for col, (cn, unit, sign) in FACTORS.items():
        s = F[col].dropna()
        r3 = s.rolling(ROLL_WIN).sum().dropna()
        v = risk_extreme(r3, sign)
        dt = str((r3.idxmax() if sign > 0 else r3.idxmin()).date())
        anchor[col] = (v, dt)
        print(f"  锚点 {col:11s} 3日峰值={v:+9.4f} {unit:2s}  发生日={dt}")

    print()
    n_assum = 0
    for dir_name, cols in ASSUM_DIRS.items():
        print(f"  ── {dir_name} ──")
        for sev, mult in SEVERITY:
            n_assum += 1
            sid = f"S{n_assum:02d}"
            for col in cols:
                base, dt = anchor[col]
                rows.append(dict(scenario_id=sid, scenario_class="假设",
                                 scenario_name=f"{dir_name}·{sev}",
                                 lead_direction=dir_name, window_start="", window_end="",
                                 factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                                 measure="peak_3d", shock=round(base * mult, 4),
                                 multiple=mult, available=True, anchor_date=dt,
                                 note=f"历史 3 日累计峰值 {base:.4f} × {mult}"))
                print(f"     {sid} {sev}({mult}×)  {col:11s} {base*mult:+9.4f} {FACTORS[col][1]}")
    # 组合档：利率与信用同时恶化（对齐 IMF WP/15/216「三种冲击同时发生」的做法）
    n_assum += 1
    sid = f"S{n_assum:02d}"
    print(f"  ── 利率+信用同时恶化（组合档）──")
    for col in ["d5y_bp", "d10y_bp", "doas_bp"]:
        base, dt = anchor[col]
        rows.append(dict(scenario_id=sid, scenario_class="假设",
                         scenario_name="利率+信用同时恶化·极端",
                         lead_direction="组合", window_start="", window_end="",
                         factor=col, factor_cn=FACTORS[col][0], unit=FACTORS[col][1],
                         measure="peak_3d", shock=round(base * 2.0, 4), multiple=2.0,
                         available=True, anchor_date=dt,
                         note="利率与信用同向极端（2×），对齐 IMF WP/15/216 多冲击同时发生"))
        print(f"     {sid} 极端(2×)  {col:11s} {base*2.0:+9.4f} {FACTORS[col][1]}")

    # ---------------------------------------------------------- 4) 落盘
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

    # ---------------------------------------------------------- 5) 图
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
        # 历史已实现最差单日参考线：**作为图例项**而非行内文字，避免与柱体重叠。
        # 这条线是「轻度档已超过样本内任何已实现单日」这一事实的可视证据，必须留。
        ref = max(abs(risk_extreme(F[c].dropna(), FACTORS[c][2])) for c in cols)
        ax.axhline(ref, color=INK2, ls="--", lw=1.1)
        ax.plot([], [], color=INK2, ls="--", lw=1.1,
                label=f"样本内最差单日 {ref:.1f}")
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

    fig.suptitle("假设情景三方向 × 三级梯度（锚点 = 全样本 3 日累计峰值）", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "stress_scenario_library.png", dpi=150, bbox_inches="tight")
    print("[图] figures/stress_scenario_library.png")


if __name__ == "__main__":
    main()
