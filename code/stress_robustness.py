"""
阶段三 Day4（9/24）：压力测试稳健性分析

对 `results/stress_impact.csv` 的测算结果做四项检验，回答「结论在多大程度上依赖于
我在构造与测算中做的那几个选择」。四项检验的共同目的是把「我选了什么」与
「数据支持什么」分开——凡是结论随选择大幅摆动的地方，都必须写明。

------------------------------------------------------------------------------
一、情景覆盖度检验（本脚本的存在理由）
------------------------------------------------------------------------------
压力情景库若不含样本内**实测最差**的结果，则「极端情景」这个说法没有依据。
本项检验取历史滚动 h 日累计损失的**前 10 名窗口**，与情景库的损失逐一对齐，输出两件事：
  ① 情景损失在历史分布中的**分位数**（比「超过 99% ES」信息量更大）；
  ② 情景库是否**被实测最差窗口超越**——若被超越，该情景的命名（轻/中/极端）就有误导性。

**口径（9/21 重做后）**：主口径全部落在 **h = 1 日**（与 VaR 口径对齐）。
情景侧的损失取 `stress_impact.csv` 的**基准列** `benchmark_loss_pct`
（历史/补充情景 = **窗内最差单日**，假设情景 = δ 映射的 1 日损失）；
历史侧取滚动 1 日累计损失——**即单日损失**。两者在 h = 1 上**是同一个统计量**
（都是「某一交易日的组合损失」），故分位数读数是直接的，不再需要口径换算。

这一点与重做前**不同，是本次重做带来的实质改善**：3 日口径下情景侧是「窗内最深」
（路径量）、历史侧是「3 日累计」（端点量），两侧不可直接比，本脚本当时只能把缺口
标注为「偏乐观的下界」并列为已知局限。主口径降到 1 日后该局限在主口径上消失。

**多期（3/5/6 日）仍保留为附录对照**，由 `HORIZONS_APPENDIX` 驱动，只做两件事：
  · `stress_worst_windows.csv` 里补出 3/5/6 日的最差窗口榜，使此前的选型可被复算；
  · `stress_shift_coverage.csv` 里给出各期限的**经验 99% ES 与 √t 缩放值对照**
    （§二末，这是「√t 系统性低估」的量化证据）。
附录行**不出情景侧的覆盖度结论**——1 日重做后情景侧已无 3/5/6 日的对应量，
硬拼出来的缺口是两个不同量纲之差，`caliber_note` 列会写明这一点。

**这项检验在重做前的执行中查出了选型遗漏**：2022-03-08 ~ 2022-03-15 曾是样本内实测最差的
6 日窗口，其因子冲击与组合损失双双超过点名的 H1，而该事件本就在
`events/risk_events_timeline.csv` 中。选型时只按「3 日利率峰值」挑窗（峰值确在 2022-06-14），
漏掉了窗长更长时更差的一段。已补入补充情景 X2 并沿用至今。

------------------------------------------------------------------------------
二、窗口敏感性
------------------------------------------------------------------------------
历史情景的窗口是人工选定的。本项把每个窗口的起止各平移 ±1、±2 个交易日，
重算窗内累计冲击与组合实际损失，**同时给出三个口径**：
  · `worst_day_pct` 窗内最差单日 ← **与 1 日基准列同口径**，2b 与对账断言用它；
  · `mdd_pct` 窗内最大回撤   ← 需求文档要求的伴随列，与重做前的读数逐值可比；
  · `loss_pct` 窗内累计损失   ← 原口径，保留以便追溯。
意义：若某情景的损失对窗口边界高度敏感（如 H3 的 ΔOAS 会被窗口截断一半），
则该情景的因子贡献度读数不稳健，报告必须带此说明。

2b 是**平移感知的库级覆盖度**：把每个情景（历史情景取其 ±2 日内最优对齐、
假设情景取原值）合在一起，看情景库在该期限上最深能到多少，与同期限实测最差比。
两侧在 1 日口径下同量纲：情景侧取历史情景的 `worst_day_pct` 最优值 / 假设情景的 1 日 δ 映射，
历史侧取滚动 1 日损失最大值。
为什么是库级：单个历史情景对应的是它自己那段事件，拿避险情景去「覆盖」利率上行的最差窗
是范畴错误，要问的是整个库够不够深。
该表另附 **√t 时间缩放参考列**（1 日 VaR/ES × √h）：它是**参考值、未做独立回测**，
不参与任何判定——判定阈值一律用阶段二登记的 1 日 ES。并列显示是为了让
「时间缩放的近似误差有多大」可被直接读出：主口径 √5 × 1 日 99% ES = 1.4890%，
而实测 5 日 99% ES = 2.0671%（次口径 1.5706% vs 1.7419%），
缩放值系统性低于经验值——这正是本报告不拿它当判定阈值的量化依据。

------------------------------------------------------------------------------
三、缓冲敏感性
------------------------------------------------------------------------------
缓冲（0.3365pp）是本报告自行选定的量级，取 M2 利率双因子 × 主口径 × B 事件前δ ×
**1 日** 行的 `absdev_p90`（`results/delta_transmission_summary.csv`），与主口径同期限。
本项在 5 档缓冲下重算尾部判定：
  0（不缓冲）/ 0.1997（M3 信用自身 1 日 90 分位）/ 0.3365（本报告取值）/
  0.8390（5 日 90 分位，跨期限对照档）/ 1.0000（+1pp 压力档）
输出每档下的超限情景数与被判定为尾部风险的情景清单。
**判定线与 headline 完全同源**（`es_1d_99_pct`，阶段二登记的 HS250@common773），
且「本报告取值」档须逐条复现 `stress_impact.csv` 的 `tail_flag`——下方有断言钉住。
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

from common import REPO, write_table
from var_common import C_M1, C_GARCH, C_ALT, C_GREY, INK2
# 信用区间上端的加项只在 stress_impact.py 中维护一份常量，此处引用同一个值，
# 避免两个脚本各写一个数字、日后只改一处导致口径分叉。stress_impact 的模块级代码
# 只有导入与常量定义（`main()` 有 `__main__` 守卫），import 不触发任何测算。
from stress_impact import CR_2022_RESID, horizon_es

FACT = REPO / "factors"
RES = REPO / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

DELTA_SRC = "B 事件前δ"
HORIZON_MAIN = "1日"        # 主口径持有期（与阶段二 VaR 对齐；老师 9 月指示）
# 附录对照用的多期。只驱动「样本内最差窗口榜」与附录取样，**不参与任何判定**。
HORIZONS_APPENDIX = (3, 5, 6)
# 缓冲档位的唯一来源：主口径档 0.3365 必须与 stress_impact.BUFFER_RATE 同值，
# 否则「本报告取值」那一档复现不出 headline（下方断言会失败）。
BUFFER = 0.3365
BUFFER_GRID = [("0.0000 不缓冲", 0.0), ("0.1997 信用 1 日", 0.1997),
               ("0.3365 本报告取值", 0.3365), ("0.8390 五日 90 分位", 0.8390),
               ("1.0000 +1pp 压力档", 1.0000)]
BUFFER_LABEL = "0.3365 本报告取值"   # 与 BUFFER_GRID 同处维护，断言按它定位基线行
# 事件窗**不在此处硬编码**：改为运行时从 results/stress_scenarios.csv 的
# (window_start, window_end) 读取，使事件窗在全仓库只有一个来源
# （code/stress_scenarios.py 的 HIST 字典）。重做前此文件是第二份硬编码拷贝，
# 两边改一处就会静默分叉。


def hday_losses(r: pd.Series, h: int) -> pd.Series:
    """历史滚动 h 日累计损失（正 = 损失），保留索引以便定位窗口终点。"""
    return (-pd.Series(r).dropna().rolling(h).sum().dropna())


def _window_measures(r: pd.Series) -> tuple[float, float]:
    """一段收益路径的（窗内累计损失%, 窗内最大回撤%），均正 = 损失。

    回撤定义与 `stress_impact.path_cum_mdd` 逐字对齐：log 收益累加 → exp 成净值，
    并以**窗起点 NAV = 1.0** 作为首个峰值候选，回撤 = min(eq / cummax(eq) − 1)。
    两处若不一致，「窗敏表的回撤」与「基准列」就对不上，故脚本内设了逐值对账断言。
    """
    v = pd.Series(r).dropna().values
    eq = np.exp(np.concatenate([[0.0], np.cumsum(v / 100.0)]))
    return (float(v.sum()), float(-(eq / np.maximum.accumulate(eq) - 1.0).min() * 100.0))


def path_worst_day_pct(r: pd.Series) -> float:
    """窗内**最差单日**损失（%，正 = 损失）——1 日基准列的窗口侧对应量。

    与 `stress_impact` 中历史情景的 `worst_day_loss_pct` 同定义：窗内 log 收益的最小值
    取负。窗长为 1 时与 `loss_pct` 恒等（单日窗只有一天）。
    """
    return float(-pd.Series(r).dropna().min())


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
    # 主期限由 stress_impact.csv 自己声明，不在此处再写一个字面量。重做后恒为 [1]；
    # 若哪天基准列换了期限而这里没跟上，下面的断言会把两处的不一致顶出来。
    HORIZONS = sorted(int(h) for h in I.horizon_days.unique())
    assert HORIZONS == [1], (
        f"主期限应为 1 日，实际读到 {HORIZONS}——stress_impact.csv 的口径与本脚本不同步")
    print(f"  主期限 h = {HORIZONS[0]} 日（来自 stress_impact.csv 的 horizon_days）"
          f"；附录对照期限 {HORIZONS_APPENDIX}")

    # 事件窗从情景表读取（唯一来源 = stress_scenarios.py 的 HIST 字典）
    _hw = S[S.scenario_class != "假设"].dropna(subset=["window_start", "window_end"])
    HIST_WINDOWS = {sid: (str(g.window_start.iloc[0])[:10], str(g.window_end.iloc[0])[:10])
                    for sid, g in _hw.groupby("scenario_id")}
    assert len(HIST_WINDOWS) == 5, f"历史/补充情景应为 5 个，实际 {sorted(HIST_WINDOWS)}"

    # 1a) 样本内实测最差窗口榜（每期限 × 每口径取前 10）。主期限 1 日为**判定口径**，
    #     3/5/6 日为附录——它们的作用不是提供结论，而是让既有的选型可被复算
    #     （X2 正是在 6 日榜上被查出来的）。
    print("\n  1a) 样本内滚动 h 日累计损失最差窗口榜")
    print("      （表中 loss_pct 为**实际发生**的损失，非模型估计，不含任何缓冲）")
    top = []
    for h in HORIZONS + list(HORIZONS_APPENDIX):
        for cal in ("主", "次"):
            hist = hday_losses(rets[cal], h)
            for rank, (d, v) in enumerate(hist.nlargest(10).items(), start=1):
                top.append(dict(horizon_days=h, is_main_horizon=(h in HORIZONS), caliber=cal,
                                rank=rank, window_end=str(d.date()),
                                window_start=str((hist.index[hist.index.get_loc(d) - (h - 1)]).date())
                                if hist.index.get_loc(d) >= h - 1 else "",
                                loss_pct=round(float(v), 4),
                                caliber_note=("" if h in HORIZONS else
                                              "[附录] 与 1 日基准列不同量纲，仅供追溯选型，"
                                              "不构成覆盖度结论")))
    T = pd.DataFrame(top)
    write_table(T, RES / "stress_worst_windows.csv")
    print(f"      → results/stress_worst_windows.csv  {T.shape[0]} 行 × {T.shape[1]} 列"
          f"（主期限 {len(HORIZONS)} + 附录 {len(HORIZONS_APPENDIX)} 个期限）")
    for h in HORIZONS:
        for cal in ("主", "次"):
            g = T[(T.horizon_days == h) & (T.caliber == cal)].head(5)
            s = "  ".join(f"{r.rank}.{r.loss_pct:.4f}%({r.window_end})" for r in g.itertuples())
            print(f"      {cal}口径 {h} 日：{s}")
    for h in HORIZONS_APPENDIX:
        g = T[(T.horizon_days == h) & (T.caliber == "主")].head(3)
        print(f"      [附录] 主口径 {h} 日最差前三："
              + "  ".join(f"{r.loss_pct:.4f}%({r.window_end})" for r in g.itertuples()))

    # 1b) 情景在历史分布中的分位数 + 是否被实测最差超越
    cov = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            hist = hday_losses(rets[cal], h)
            worst, worst_d = float(hist.max()), hist.idxmax()
            scen = I[(I.horizon_days == h) & (I.caliber == cal)]
            for _, r in scen.iterrows():
                # 情景侧取**基准列**（历史=窗内最差单日、假设=δ 映射的 1 日损失）。
                # h=1 时历史侧（滚动 1 日累计 = 单日损失）与情景侧**是同一个统计量**，
                # 故 hist_percentile 可直接读作「该情景的损失落在样本单日分布的哪个位置」。
                pct = float((hist < r.benchmark_loss_pct).mean() * 100)
                cov.append(dict(scenario_id=r.scenario_id, scenario_name=r.scenario_name,
                                scenario_class=r.scenario_class, caliber=cal, horizon_days=h,
                                loss_pct=round(r.benchmark_loss_pct, 4),
                                benchmark_source=r.benchmark_source,
                                loss_cum_pct=round(r.loss_realized_pct, 4),
                                mdd_pct=round(r.mdd_pct, 4),
                                hist_percentile=round(pct, 2),
                                hist_n_obs=int(hist.notna().sum()),
                                hist_worst_pct=round(worst, 4),
                                hist_worst_end=str(worst_d.date()),
                                # 正数 = 实测最差比该情景更差（情景被超越）
                                exceeded_by_pct=round(worst - r.benchmark_loss_pct, 4),
                                below_worst="是" if worst > r.benchmark_loss_pct else "否"))
    C = pd.DataFrame(cov)
    write_table(C, RES / "stress_coverage.csv")
    print(f"\n[1] 覆盖度 → results/stress_coverage.csv  {C.shape[0]} 行 × {C.shape[1]} 列")

    # 覆盖率只在 **loss_pct > 0** 上判定。该过滤的作用是剔掉零损失的纯汇率主口径行
    # （S07–S09 主口径按口径搬运恒为 0，被任何正损失「超越」都是同义反复）。
    # 副作用须写明：H2、X1 这两条**避险情景**的窗内最差单日（0.5620% / 0.3084%）也进入判定，
    # 它们必然被实测最差超越——这不是「情景选得不够深」，而是避险情景本就不该承担
    # 覆盖利率上行最差日的职责（见 2b 的库级口径说明）。
    print("\n  1b) 各口径：情景库最深损失（基准列） vs 样本内实测最差")
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
    # 只有**假设情景**带轻/中/极端这类严重度命名，被超越才构成「命名误导」；
    # 历史情景被另一个历史窗口超越是样本内的自然排序，不构成命名问题。
    bad_graded = bad[bad.scenario_class == "假设"]
    print(f"\n  **被样本内实测最差窗口超越的损失情景 {len(bad)} 条**，"
          f"其中带严重度命名的假设情景 {len(bad_graded)} 条"
          f"（只有后者涉及「命名是否有误导性」）：")
    if len(bad):
        print(bad[["scenario_id", "scenario_name", "scenario_class", "caliber",
                   "horizon_days", "loss_pct", "hist_worst_pct", "hist_worst_end",
                   "exceeded_by_pct"]].to_string(index=False))
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
                cum, mdd = _window_measures(W[col])
                wrows.append(dict(scenario_id=sid, shift_days=sh, caliber=cal,
                                  window_start=str(W.index.min().date()),
                                  window_end=str(W.index.max().date()),
                                  n_days=len(W),
                                  d5y_bp=round(float(W["d5y_bp"].sum()), 4),
                                  d10y_bp=round(float(W["d10y_bp"].sum()), 4),
                                  loss_pct=round(cum, 4),
                                  mdd_pct=round(mdd, 4),
                                  # 与 1 日基准列同口径的量：2b 的库级覆盖度与下面的对账都用它。
                                  worst_day_pct=round(path_worst_day_pct(W[col]), 4),
                                  worst_day_date=str(W[col].idxmin().date())))
    Ws = pd.DataFrame(wrows)
    # [对账 1] 平移 0 日（事件窗原样）的**窗内最差单日**必须与 stress_impact.csv 的基准列逐值一致。
    # 这是本次重做后 headline 的口径，对不上说明两个脚本的窗口约定已经分叉。
    for sid in HIST_WINDOWS:
        for cal in ("主", "次"):
            row = Ws[(Ws.scenario_id == sid) & (Ws.caliber == cal) & (Ws.shift_days == 0)]
            b = float(I[(I.scenario_id == sid) & (I.caliber == cal)]
                      .benchmark_loss_pct.iloc[0])
            w = float(row.worst_day_pct.iloc[0])
            assert abs(w - b) < 5e-4, (
                f"{sid} {cal} 口径最差单日对账失败：窗敏表 {w:.4f}% vs 基准列 {b:.4f}%")
            # [对账 2] 旧口径一并保留核对：需求文档要求的「最大回撤」伴随列，
            # 必须与 stress_impact.csv 的 mdd_pct 逐值一致（重做不改回撤定义）。
            m, bm = float(row.mdd_pct.iloc[0]), float(
                I[(I.scenario_id == sid) & (I.caliber == cal)].mdd_pct.iloc[0])
            assert abs(m - bm) < 5e-4, (
                f"{sid} {cal} 口径回撤对账失败：窗敏表 {m:.4f}% vs 伴随列 {bm:.4f}%")
            # [对账 3] 定义自洽性：窗内最大回撤（净值口径）不得浅于「最差单日按净值折算」
            # 的损失。两侧必须换算到**同一量纲**再比——`w` 是 log 收益、`m` 是净值回撤，
            # 直接写 `w <= m` 是错的：H2 主口径 w = 0.5620%、m = 0.5604%，
            # 单日损失数值上反而更大，而 1 − exp(−0.5620%) = 0.5604% 恰好相等，
            # 说明二者一致、只是 log 与净值两种表达的换算差（见 stress_impact docstring §三）。
            w_net = (1.0 - np.exp(-w / 100.0)) * 100.0
            assert m >= w_net - 5e-4, (
                f"{sid} {cal} 定义不自洽：净值回撤 {m:.4f}% 浅于最差单日的净值折算 "
                f"{w_net:.4f}%（由 log 损失 {w:.4f}% 换算）")
    print("    [对账] 5 个历史情景 × 2 口径：最差单日 = 基准列、最大回撤 = 伴随列，逐值一致")
    write_table(Ws, RES / "stress_window_sens.csv")
    print(f"[2] 窗口敏感性 → results/stress_window_sens.csv  {Ws.shape[0]} 行 × {Ws.shape[1]} 列\n")
    for sid in HIST_WINDOWS:
        for cal in ("主", "次"):
            g = Ws[(Ws.scenario_id == sid) & (Ws.caliber == cal)]
            bwd = g[g.shift_days == 0].worst_day_pct.iloc[0]
            bmdd = g[g.shift_days == 0].mdd_pct.iloc[0]
            print(f"    {sid} {cal}口径：")
            print(f"        最差单日 基准 {bwd:+.4f}%  "
                  f"平移区间 [{g.worst_day_pct.min():+.4f}, {g.worst_day_pct.max():+.4f}]  "
                  f"极差 {g.worst_day_pct.max() - g.worst_day_pct.min():.4f}pp   ← 与基准列同口径")
            print(f"        最大回撤 基准 {bmdd:+.4f}%  "
                  f"平移区间 [{g.mdd_pct.min():+.4f}, {g.mdd_pct.max():+.4f}]  "
                  f"极差 {g.mdd_pct.max() - g.mdd_pct.min():.4f}pp")
            print(f"        窗内累计 基准 {g[g.shift_days == 0].loss_pct.iloc[0]:+.4f}%  "
                  f"平移区间 [{g.loss_pct.min():+.4f}, {g.loss_pct.max():+.4f}]")
        g5 = Ws[(Ws.scenario_id == sid) & (Ws.caliber == "主")]
        print(f"        Δ5Y 窗内累计区间 [{g5.d5y_bp.min():+.0f}, {g5.d5y_bp.max():+.0f}] bp")

    # 2b) 平移感知的**库级**覆盖度（仅主期限）：把全部情景（历史情景取其 ±2 日内最优对齐、
    #     假设情景取原值）放在一起，看情景库最深能到多少，与样本内实测最差单日比。
    #     为什么是库级：单个历史情景对应的是它自己那段事件，拿它去「覆盖」别的方向的最差日
    #     （如拿避险情景 H2 去覆盖利率上行的最差日）是范畴错误。要问的是整个库够不够深。
    #     两侧同量纲：假设情景基准列 = 1 日 δ 映射，历史情景取平移窗的 worst_day_pct，
    #     历史侧 = 滚动 1 日损失最大值。重做前这里两侧是「窗内最深 vs 3 日累计」，不可比。
    print("\n  2b) 平移感知的库级覆盖度（主期限；历史情景取 ±2 日内最优对齐）")
    # 1 日参照线，按口径各取一份。var/es99 来自阶段二登记的 HS250@common773 基准
    # （stress_impact.csv 的 var_1d_*_pct / es_1d_99_pct，各期限内恒定）；
    # es_hd[h] 是本脚本现算的经验 h 日 ES，供附录的 √t 近似误差对照使用。
    ref_1d = {}
    for cal in ("主", "次"):
        g0 = I[I.caliber == cal].iloc[0]
        ref_1d[cal] = dict(
            var95=float(g0.var_1d_95_pct), var99=float(g0.var_1d_99_pct),
            es99=float(g0.es_1d_99_pct),
            es_hd={h: horizon_es(rets[cal], h)[99]["es"]
                   for h in HORIZONS + list(HORIZONS_APPENDIX)})
    sh_rows = []
    for h in HORIZONS:
        for cal in ("主", "次"):
            cands = []
            grp = I[(I.caliber == cal) & (I.horizon_days == h)]
            assert len(grp) > 0, f"{cal}口径在 {h} 日期限上没有情景行——情景表与基准表不同步"
            for _, r in grp.iterrows():
                if r.scenario_class == "假设":
                    cands.append((r.scenario_id, float(r.benchmark_loss_pct), 0))
                else:
                    g = Ws[(Ws.scenario_id == r.scenario_id) & (Ws.caliber == cal)]
                    k = g.worst_day_pct.idxmax()
                    cands.append((r.scenario_id, float(g.loc[k, "worst_day_pct"]),
                                  int(g.loc[k, "shift_days"])))
            best_sid, best_v, best_sh = max(cands, key=lambda t: t[1])
            worst_hist = float(hday_losses(rets[cal], h).max())
            sq = float(np.sqrt(h))
            sh_rows.append(dict(horizon_days=h, caliber=cal, is_main_horizon=True,
                                deepest_scenario=best_sid,
                                deepest_loss_shifted=round(best_v, 4),
                                best_shift_days=best_sh,
                                hist_worst_pct=round(worst_hist, 4),
                                residual_gap_pct=round(worst_hist - best_v, 4),
                                shift_covered="是" if best_v >= worst_hist - 1e-9 else "否",
                                sqrt_t_var95_ref_pct=round(float(ref_1d[cal]["var95"]) * sq, 4),
                                sqrt_t_var99_ref_pct=round(float(ref_1d[cal]["var99"]) * sq, 4),
                                sqrt_t_es99_ref_pct=round(float(ref_1d[cal]["es99"]) * sq, 4),
                                ref_note="√t 时间缩放参考值，未做独立回测，不参与判定"))
            print(f"      {cal}口径 {h} 日：库内最深（平移后）{best_sid} {best_v:+.4f}%"
                  f"（平移 {best_sh:+d} 日）  实测最差单日 {worst_hist:+.4f}%  "
                  f"残余缺口 {worst_hist - best_v:+.4f}pp  → "
                  f"{'覆盖' if best_v >= worst_hist - 1e-9 else '**仍有缺口**'}"
                  f"   [√t 参考 · 99%ES×√{h}={float(ref_1d[cal]['es99']) * sq:.4f}%"
                  f" vs 经验 {float(ref_1d[cal]['es_hd'][h]):.4f}%]")

    # 2c) **附录**：√t 时间缩放近似误差随期限的变化。这一块只为报告附录提供证据，
    #     不出覆盖度结论（情景侧已无 3/5/6 日的对应量，拼缺口是两个量纲之差）。
    print("\n  2c) [附录] √t 时间缩放 vs 同期限经验 99% ES（近似误差的量化）")
    for h in HORIZONS_APPENDIX:
        for cal in ("主", "次"):
            sq = float(np.sqrt(h))
            s_es = float(ref_1d[cal]["es99"]) * sq
            e_es = float(ref_1d[cal]["es_hd"][h])
            sh_rows.append(dict(horizon_days=h, caliber=cal, is_main_horizon=False,
                                deepest_scenario="", deepest_loss_shifted=np.nan,
                                best_shift_days=np.nan,
                                hist_worst_pct=round(float(hday_losses(rets[cal], h).max()), 4),
                                residual_gap_pct=np.nan, shift_covered="",
                                sqrt_t_var95_ref_pct=round(float(ref_1d[cal]["var95"]) * sq, 4),
                                sqrt_t_var99_ref_pct=round(float(ref_1d[cal]["var99"]) * sq, 4),
                                sqrt_t_es99_ref_pct=round(s_es, 4),
                                ref_note="[附录] √t 缩放 vs 经验 ES 的近似误差；出结论需"
                                         "情景侧的同期限损失，1 日重做后已无此量，故不作判定"))
            print(f"      {cal}口径 {h} 日：√t 99%ES {s_es:.4f}%  vs 经验 {e_es:.4f}%  "
                  f"低估 {e_es - s_es:+.4f}pp（{(e_es / s_es - 1) * 100:+.1f}%）"
                  f"   实测最差 h 日累计 {float(hday_losses(rets[cal], h).max()):.4f}%")
    write_table(pd.DataFrame(sh_rows), RES / "stress_shift_coverage.csv")
    print("      → results/stress_shift_coverage.csv")

    # ============================================================ 三、缓冲敏感性
    print("\n" + "=" * 78)
    print("三、缓冲敏感性：5 档缓冲下的尾部判定")
    print("=" * 78)
    brows = []
    # 纯汇率情景（S07–S09）走口径搬运的恒等式，无 δ 估计误差，报告基线给 0 缓冲。
    # 若在本网格里也给它加缓冲，标着「本报告取值」的那一档就复现不出报告本身。
    PURE_FX = {"离岸汇率贬值"}
    for lab, buf in BUFFER_GRID:
        for _, r in I.iterrows():
            # 缓冲只对假设情景生效；历史情景取实际路径，任何档位下都不加。
            # 基数取**基准列**（历史=窗内最差单日、假设=δ 映射的 1 日损失），
            # 判定线取 `es_1d_99_pct`（阶段二登记的 1 日 ES），与 headline 的
            # `tail_flag` 完全同源，故「本报告取值」那一档应逐条复现它（下方有断言）。
            applies = r.scenario_class == "假设" and r.direction not in PURE_FX
            net = r.benchmark_loss_pct + (buf if applies else 0.0)
            brows.append(dict(buffer_label=lab, buffer_value=buf,
                              scenario_id=r.scenario_id, scenario_name=r.scenario_name,
                              caliber=r.caliber, horizon_days=r.horizon_days,
                              loss_net_pct=round(net, 4),
                              es_1d_99_pct=round(float(r.es_1d_99_pct), 4),
                              tail_flag="是" if (net > r.es_1d_99_pct and net > 0) else "否"))
    Bs = pd.DataFrame(brows)
    # [对账] 标着「本报告取值」的那一档必须逐条复现 stress_impact.csv 的 headline——
    # **损失值与尾部判定都要对**。只对损失不对判定的话，缓冲网格可能在数值上与基准一致、
    # 在结论上却用了另一条阈值线，本项检验就答非所问了。
    for _, r in I.iterrows():
        b = Bs[(Bs.buffer_label == BUFFER_LABEL)
               & (Bs.scenario_id == r.scenario_id) & (Bs.caliber == r.caliber)]
        assert len(b) == 1, f"{r.scenario_id} {r.caliber} 在缓冲网格中行数异常：{len(b)}"
        assert abs(float(b.loss_net_pct.iloc[0]) - r.loss_after_buffer_pct) < 5e-5, (
            f"{r.scenario_id} {r.caliber} 口径缓冲对账失败："
            f"{float(b.loss_net_pct.iloc[0]):.4f} vs {r.loss_after_buffer_pct:.4f}")
        assert str(b.tail_flag.iloc[0]) == str(r.tail_flag), (
            f"{r.scenario_id} {r.caliber} 口径尾部判定对账失败："
            f"缓冲网格 {b.tail_flag.iloc[0]} vs headline {r.tail_flag}")
    assert abs(float(Bs[Bs.buffer_label == BUFFER_LABEL].buffer_value.iloc[0])
               - BUFFER) < 1e-12, "BUFFER_LABEL 指向的档位值与 BUFFER 不一致"
    print(f"    [对账] {BUFFER_LABEL} 档逐条复现 stress_impact.csv 的 "
          f"loss_after_buffer_pct 与 tail_flag")
    write_table(Bs, RES / "stress_buffer_sens.csv")
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
    # 口径纪律：信用区间是**模型路径**下算的，必须与 loss_modeled_pct 比，不能与历史情景的
    # 实际实现损失（loss_realized_pct）比——后者含残差，两者不是同一个量。
    # 区间方向（9/21 修订起）：下端 = 平静样本**弱传导**（损失小）；上端 = 平静样本
    # 强传导 + 2022 年未解释残差极值（损失大）。故包含关系是 lo ≤ 点估计 ≤ hi。
    # 上端的加项已是用实际偏差幅度修正过的量，**不再叠加缓冲**（叠加即重复计算）。
    cr = I[I.credit_range_lo_pct.notna()].copy()
    cr["in_range"] = cr.apply(
        lambda r: r.credit_range_lo_pct - 1e-9 <= r.loss_modeled_pct
        <= r.credit_range_hi_pct + 1e-9, axis=1)
    # 平静样本口径下的强传导端 = 上端减去 2022 残差加项，即修订前那个数，便于逐值对照
    cr["strong_quiet"] = cr.credit_range_hi_pct - CR_2022_RESID
    for _, r in cr.iterrows():
        tag = "" if r.in_range else "  **点估计落在区间外，检查口径**"
        hist = (f"  | 该情景为历史情景：窗内最差单日 {r.benchmark_loss_pct:+.4f}%、"
                f"窗内最大回撤 {r.mdd_pct:+.4f}%（窗内累计口径的残差 "
                f"{r.residual_pct:+.4f}%）"
                if r.scenario_class != "假设" else "")
        print(f"    {r.scenario_id} {r.caliber}口径（模型路径）："
              f"弱传导 {r.credit_range_lo_pct:+.4f}%  点估计 {r.loss_modeled_pct:+.4f}%  "
              f"平静强传导 {r.strong_quiet:+.4f}%  →＋2022残差 {r.credit_range_hi_pct:+.4f}%"
              f"  | 1日99%ES {r.es_1d_99_pct:.4f}%{hist}{tag}")

    # 只对**假设情景**判定：历史情景的实现损失另属已实现路径，不在此项做区间敏感性
    hy = cr[cr.scenario_class == "假设"]
    cr_only = hy[hy.scenario_id.str.startswith("S0") & ~hy.scenario_id.isin(["S10"])]
    worst_quiet = float((cr_only.strong_quiet + BUFFER).max())
    worst_extrap = float(cr_only.credit_range_hi_pct.max())
    # 判定线用 1 日 ES（阶段二登记基准），与 headline 的 tail_flag 同源。
    es_min = float(cr_only.es_1d_99_pct.min())
    hit_q = worst_quiet > es_min
    hit_e = worst_extrap > es_min
    print(f"\n  纯信用情景（S04–S06），1 日 99% ES 最小值 {es_min:.4f}%：")
    print(f"    ① 平静样本口径（强传导端 + 缓冲 {BUFFER:.4f}pp）最大损失 {worst_quiet:.4f}%"
          f"  → {'未触及尾部' if not hit_q else '**已触及尾部**'}")
    print(f"    ② 纳入 2022 残差标定后（区间上端，不加缓冲）最大损失 {worst_extrap:.4f}%"
          f"  → {'未触及尾部' if not hit_e else '**已触及尾部**'}")
    # 结论由数值推出，不写成固定字符串——1 日重做后 ① 的判定可能已经翻转，
    # 若这里还印着「结论相反」，脚本就会在数字变了的情况下继续说旧结论。
    if hit_q != hit_e:
        print("    → 两条口径的尾部判定**相反**，即该结论依赖「是否把 2022 年外推风险计入」，")
        print("      报告必须并列两个读数、不得只报其一。")
    elif hit_q and hit_e:
        print("    → 两条口径**都触及尾部**：平静样本口径已经不依赖于外推假设，")
        print("      信用维度的尾部风险在样本内即可观察到，**不再存在口径依赖**。")
    else:
        print("    → 两条口径**都未触及尾部**：信用维度单独不足以触及，")
        print("      即便把 2022 年外推风险计入也不足以。")
    print("    ② 是**参考项**，不参与尾部判定（判定用 ①）；它的作用是量化")
    print("    「平静期系数外推到危机」的幅度：把 2022 年的实际偏差纳入后，")
    print("    信用维度单独能走多远——这正是修订前 §5.4 自述的「区间不覆盖外推风险」")
    print("    被量化后的结果。")
    s10 = hy[hy.scenario_id == "S10"]
    print(f"  组合档 S10：平静强传导端 + 缓冲 "
          f"{float((s10.strong_quiet + BUFFER).max()):.4f}%，纳入 2022 残差后 "
          f"{float(s10.credit_range_hi_pct.max()):.4f}% —— 但 S10 同时含利率极端项，"
          f"其超限不可归因于信用维度")

    # ============================================================ 五、锚点敏感性
    print("\n" + "=" * 78)
    print(f"五、锚点口径敏感性：把 1× 的窗长从 1 日改成 3 / 5 / 6 日")
    print("=" * 78)
    # 需求文档只要求假设情景幅度取「历史极端值的 1/1.5/2 倍」，**未规定窗长**。
    # 1 日重做后本报告取**单日极值**为 1×，这是自选口径，必须量化它影响多大。
    # 现象：同一因子的极端值随窗长显著变化（汇率 5 日贬值 −3.4785% 比 1 日 −1.5854% 深 119.4%），
    # 故 1× 的绝对水平依赖于窗长选择，连带整条梯度平移；窗越长、梯度越高。
    # 此项取 1 日为**登记档**（与 stress_scenarios.py 的 peak_1d 同源），3/5/6 日为对照。
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
    # 登记档 h = 1 必须排在第一个：下面的对账断言只在它上面做，且它要与
    # stress_scenarios.csv 的 peak_1d 锚点、stress_impact.csv 的登记损失三处对上。
    H_ALL = HORIZONS + list(HORIZONS_APPENDIX)
    for grp, cols in GRP.items():
        print(f"\n  【{grp}】")
        for h in H_ALL:
            # 锚点日 = 该方向（利率取 d5y）滚动 h 日累计的样本极值日；h = 1 即单日极值日
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
            is_main = h in HORIZONS
            ar_rows.append(dict(direction=grp, window_days=h, is_main_horizon=is_main,
                                caliber=chk, anchor_date=str(ad.date()),
                                shock_sum=round(sum(shock.values()), 4),
                                mild_1x_loss_pct=round(l1, 4),
                                extreme_2x_loss_pct=round(l2, 4),
                                registered_mild_pct=(round(reg1, 4) if is_main else np.nan),
                                registered_extreme_pct=(round(reg2, 4) if is_main else np.nan)))
            reg = (f"  （与登记 {sid1}/{sid2} 的 {reg1:.4f}/{reg2:.4f}% 一致）" if is_main
                   else "  [附录对照，不参与判定]")
            print(f"    {h} 日锚点 @ {ad.date()}：1× 冲击合计 {sum(shock.values()):+.4f}"
                  f"  轻度损失 {l1:+.4f}%  极端损失 {l2:+.4f}%{reg}")
        g = [r for r in ar_rows if r["direction"] == grp]
        sp = max(r["extreme_2x_loss_pct"] for r in g) - min(r["extreme_2x_loss_pct"] for r in g)
        print(f"    → 极端档损失在 {len(H_ALL)} 种窗长下的极差 {sp:.4f}pp"
              f"（窗越长梯度越高，登记档取最短的 1 日 = 最保守的冲击幅度）")
    AR = pd.DataFrame(ar_rows)
    write_table(AR, RES / "stress_anchor_sens.csv")
    print(f"\n[5] 锚点敏感性 → results/stress_anchor_sens.csv  {AR.shape[0]} 行 × {AR.shape[1]} 列")

    # 同向 vs 绝对最大：利率样本内最大的**单日**变动是**下行**，须写明口径
    r1 = F["d5y_bp"].dropna()
    r3 = r1.rolling(3).sum().dropna()
    print(f"\n  口径说明：5 年期美债单日的**绝对值**最大为 {abs(r1.iloc[r1.abs().argmax()]):.0f}bp"
          f" @ {r1.abs().idxmax().date()}，方向为**下行**"
          f"（{'2023 银行业风波避险' if str(r1.abs().idxmax().date()).startswith('2023') else '避险事件'}）。")
    print(f"  本报告假设情景方向为「无风险利率大幅上行」，故取**同向**单日极值 "
          f"{r1.max():.0f}bp @ {r1.idxmax().date()}。若按绝对值取，1× 锚点会改用那次下行冲击。")
    print(f"  （3 日窗下同一比较为：绝对值最大 {abs(r3.iloc[r3.abs().argmax()]):.0f}bp"
          f" @ {r3.abs().idxmax().date()} vs 同向极值 {r3.max():.0f}bp @ {r3.idxmax().date()}）")

    # ============================================================ 图
    h = HORIZONS[0]                      # 图中的口径 = 主期限，与判定同源
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, cal in zip(axes, ("主", "次")):
        g = C[(C.caliber == cal) & (C.horizon_days == h)].sort_values("loss_pct")
        # 静默空图是这里最容易出的错（口径改名后选择子返回空表、图照画但一片空白），
        # 故显式断言，宁可报错也不出一张没有内容的图。
        assert len(g) > 0, f"{cal}口径在 {h} 日期限上没有情景行，图将为空"
        hist = hday_losses(rets[cal], h)
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
        # 标签放在横轴上方、最长的柱子之上（top − 0.5 仍在坐标区内而高于 S10 的柱顶），
        # 否则会压在排名靠前的柱子上——1 日口径下柱高与参照线距离很近，这一点尤其明显。
        ax.text(float(hist.max()), top - 0.5, f" 实测最差 {hist.max():.2f}%",
                color="#c0392b", fontsize=8, va="top")
        ax.set_yticks(y)
        ax.set_yticklabels(g.scenario_id, fontsize=8)
        ax.set_xlabel(f"{h} 日损失（%），向右为损失更大", fontsize=9)
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
    fig.legend(handles, ["被实测最差日超越", "未被超越", "历史 99% 分位",
                         "历史 99.9% 分位", "样本内实测最差"],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"情景损失在历史 {h} 日损失分布中的位置", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / "stress_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n[图] figures/stress_coverage.png")


if __name__ == "__main__":
    main()
