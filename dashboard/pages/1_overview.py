"""概览与业务结论。

看板第一页，两个动作：先用顶部口径开关选一个计价口径，页面上所有读数与大图跟着换；
再点下面结论表里的任意一行，紧挨着的那张图就换成那条结论的依据。图与表都占满一栏，
文字只留一句读数。所有数字都读自已落库的产物，本页不重算。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("概览与业务结论")

_tok = theme.tokens()

# 侧栏导航的六页，供 jump() 把「展开：…」渲染成页面名而不是文件名
PAGES = {
    "data_quality.py": "数据质量",
    "robustness.py": "稳健性检验",
    "2_data_factors.py": "数据与因子概览",
    "3_var_backtest.py": "VaR 模型对比与回测",
    "4_model_validation.py": "模型诊断与验证",
    "5_stress.py": "压力测试交互分析",
    "7_conclusions.py": "结论分级与已知边界",
}


def jump(path: str) -> None:
    """指到展开这条结论的那一页。

    看板跑在真正的导航里时渲染成可点的链接；脱离导航执行时（冒烟测试）
    `st.page_link` 会因为没有页面注册表而报错，这里退回成一句文字指引。
    """
    label = PAGES.get(path, path)
    try:
        st.page_link(f"pages/{path}", label=f"展开：{label}")
    except Exception:
        st.caption(f"→ 左侧导航「{label}」")


page.header(
    "中资投资级美元债市场风险计量与压力测试",
    "组合代理：华夏亚洲美元投资级债券 ETF（9141.HK，美元柜台）。",
    crumb="阶段四 · 概览与业务结论",
    span="因子表 2021-08-03 → 风险窗口 2026-09-04",
)

# ------------------------------------------------------------------ 四阶段进度
# 老师问「做到哪」时的第一屏：横向四段，每段完成态 + 代表 commit。
# 进度数据全部来自 labels.PHASES（已发生的 git 历史），本页只读展示、不重算。
with page.card("四阶段进度 · 均已交付",
               "进度来自 git 历史，`git log --oneline --reverse` 逐条可溯；"
               "左侧导航按这四阶段分组。看某一阶段的细节就进对应的页。"):
    _pcols = st.columns(len(labels.PHASES))
    for _p, _col in zip(labels.PHASES, _pcols):
        with _col:
            st.markdown(f"**阶段 {_p['no']} · {_p['name']}**")
            st.caption(f"`{_p['commit']}`")
            st.caption("已完成　·　" + _p["summary"])
    with st.expander("各阶段交付、覆盖表与对应页面"):
        for _p in labels.PHASES:
            st.markdown(f"**阶段 {_p['no']} · {_p['name']}**（{_p['span']} · `{_p['commit']}`）")
            st.write("交付：" + "；".join(_p["deliverables"]))
            st.caption("覆盖表：`" + "`、`".join(_p["tables"]) +
                       "`　·　对应页：" + "、".join(_p["pages"]))
        st.caption("口径：进度与交付均来自仓库 docs/ 与 git 历史，本页只读展示、不重算。")

# ------------------------------------------------------------------ 口径开关
# 这一条驱动本页其余全部：读数带、大图、以及结论表下面那张依据图。
_picked = st.segmented_control(
    "计价口径",
    options=["主", "次"],
    default="主",
    format_func=lambda k: labels.CALIBER_FULL[k],
    key="ov_caliber",
    help="主口径＝复权美元净值，次口径＝人民币计价。两个口径的差别是计价基准的差别，"
         "不是精度的差别：同一条情景在两条线上的读数不同，判定线也不同。",
)
caliber = _picked or "主"
cal_name = labels.CALIBER_FULL[caliber]

ES_LINE = labels.ES_1D_99[caliber]
RET_COL = {"主": "ret_tr_pct", "次": "ret_rmb_pct"}[caliber]
VAR_COL = {"主": "hs250_var_99", "次": "rmb_hs250_var_99"}[caliber]

# -------------------------------------------------------------------- 读数带
_impact = data.stress_impact()
_env = _impact[_impact["caliber"] == caliber].sort_values("x_es99", ascending=False)
_top = _env.iloc[0]

_attr = data.var_attribution()
_fx = _attr[(_attr["caliber"] == "次") & _attr["factor"].astype(str).str.contains("汇率")]
_fx_share = float(_fx["share_pct"].iloc[0]) if not _fx.empty else float("nan")

page.kpi_row([
    {"label": "日度基准模型", "value": "HS250",
     "help": "历史模拟法 250 日窗。95% / 99% 两档实测违规率贴近名义水平，"
             "美元与人民币两个口径下都成立。阶段二定案。"},
    {"label": "1 日 99% 期望损失判定线", "value": f"{ES_LINE:.4f} %",
     "help": f"{cal_name}。压力情景的越线判定一律以它为准。"
             "需求文档全文未出现「持有期」，1 日持有期来自口头指示。"},
    {"label": "最极端情景相当于判定线", "value": f"{float(_top['x_es99']):.2f} 倍",
     "help": f"{_top['scenario_id']} {_top['scenario_name']}，{cal_name}，全表最高。"
             "纯汇率情景不加缓冲，其倍数也不随缓冲档位变化。"},
    {"label": "人民币口径波动中汇率占",
     "value": f"{_fx_share:.1f} %" if caliber == "次" else "不进入分解",
     "help": "按因子协方差分解。美元口径下汇率不进入因子分解，这部分波动完全不被观测；"
             "44% 与压力测试那章「汇率实际贡献大致 20% 到 25%」不冲突："
             "前者是日常波动的来源分解，后者是特定历史冲击窗内累计损失的分摊。"},
])

# -------------------------------------------------------------- 大图：判定线
with page.card("组合日收益与 1 日 99% 期望损失判定线",
               f"{cal_name}。跌破判定线的那几天，就是回测里记下的违规日。"):
    _vh = data.var_historical().copy()
    _hero = charts.line_crosshair(
        _vh, y=RET_COL, height=360,
        y_title="日收益（%，负值 = 亏）",
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip(f"{RET_COL}:Q", title="日收益 %", format=".4f"),
                 alt.Tooltip(f"{VAR_COL}:Q", title="当日 99% VaR %", format=".4f")],
    )
    st.altair_chart(
        charts.add_rule(_hero, -ES_LINE,
                        label=f"−1 日 99% 期望损失 {ES_LINE:.4f}%"),
        use_container_width=True,
    )
    st.caption(
        f"样本 {len(_vh)} 天。线是日收益，另一条虚线是判定线；"
        "悬停任一天即可读出当天收益与当天的 VaR。"
    )
    charts.data_drawer(
        _vh[["date", RET_COL, VAR_COL]],
        key="ov_hero",
        title=f"数据抽屉 · {cal_name}的日收益与 HS250 99% VaR",
        height=280,
        fmt={RET_COL: "%.4f", VAR_COL: "%.4f"},
        labels={"date": "日期", RET_COL: "日收益 %", VAR_COL: "HS250 99% VaR %"},
        note="VaR 以正值存储，本页图中不画负号，虚线位置即 −VaR。",
    )

# ------------------------------------------------- 结论表 + 随选中切换的依据图
_bt = data.backtest_results()
_hs = _bt[(_bt["model"] == "HS250") & (_bt["caliber"] == "主")
          & (_bt["conf"].astype(int) == 99) & (_bt["scope"] == "common773")]

_shift = data.stress_shift_coverage()
_n_below = int((_shift["sqrt_t_es99_ref_pct"] < _shift["hist_worst_pct"]).sum())

_fx_extreme = _impact[(_impact["caliber"] == "次")
                      & (_impact["scenario_id"] == "S09")]["x_es99"]

CONCL = pd.DataFrame([
    {"业务结论": "日度基准模型选 HS250：真实违规率贴近名义水平，参数法是保守上限而非更准",
     "关键读数": f"99% 档真实违规率 {float(_hs['rate_pct'].iloc[0]):.2f}%（名义 1%）"},
    {"业务结论": "汇率是人民币口径下常态波动的主要来源，日常监控必须双口径并行",
     "关键读数": f"次口径波动中汇率占 {_fx_share:.1f}%"},
    {"业务结论": "压力情景与 VaR 同为 1 日持有期，可直接对接日度阈值；多日期限不能",
     "关键读数": f"√t 缩放值在 {_n_below}/{len(_shift)} 个组合上低于同期限实测最差"},
    {"业务结论": "严重度排序用连续倍数，不用超限与否的二值名单",
     "关键读数": f"极端汇率情景达判定线 {float(_fx_extreme.iloc[0]):.2f} 倍（次口径）"},
])

_pick = st.dataframe(
    CONCL, width="stretch", hide_index=True,
    on_select="rerun", selection_mode="single-row", key="ov_conclusion",
    height=38 * len(CONCL) + 40,
)
_rows = getattr(getattr(_pick, "selection", None), "rows", None) or []
_sel = _rows[0] if _rows else 0
_sel = _sel if 0 <= _sel < len(CONCL) else 0

TITLES = [
    "依据一 · 各模型的真实违规率与 Kupiec 接受区间",
    "依据二 · 组合波动的因子来源分解",
    "依据三 · 把 1 日阈值按 √t 缩放会得到什么",
    "依据四 · 情景库按判定线倍数排序",
]
NOTES = [
    "点落在名义线上方＝覆盖充分（保守），落在下方＝低估。两个标「对照」的模型不参与评选。",
    "100% 堆叠条，看的是「这部分波动由谁贡献」。",
    "两线的差距就是缩放法丢掉的那部分。",
    "同一条情景在不同缓冲档下倍数不变，这正是它比二值名单稳的地方。",
]
CAPTIONS = [
    "点 = 真实违规率，灰杆 = 接受区间，虚线 = 名义水平；换回测范围必须换标签。"
    "M1g-t 只在主口径拟合，故次口径少一个点。",
    "美元口径下汇率一栏根本不出现——那部分波动不被观测。",
    "√t 参考值在每一个期限上都低于同期限实测最差，多日期限的阈值不能靠缩放得到。",
    "纯汇率三档在美元口径下恒为 0，在人民币口径下最高的一档是判定线的 4.51 倍。",
]
JUMPS = ["3_var_backtest.py", "2_data_factors.py",
         "robustness.py", "5_stress.py"]

with page.card(TITLES[_sel], NOTES[_sel]):
    if _sel == 0:
        _sub = _bt[(_bt["caliber"] == caliber) & (_bt["conf"].astype(int) == 99)
                   & (_bt["scope"] == "common773")].copy()
        # 十个模型一个不落：M1d 与 M1g-t 是恒等式与分布分位对照，标「对照」后并排画出。
        # 只用 MODEL_ORDER 会把这两个滤成 Categorical 的 NaN，轴上多出一个 nan 类别。
        _order = [m for m in labels.MODEL_ORDER if m in set(_sub["model"])] + \
                 [m for m in labels.NON_EVIDENCE_MODELS if m in set(_sub["model"])]
        _sub["model"] = pd.Categorical(_sub["model"], categories=_order, ordered=True)
        _sub = _sub.sort_values("model")
        # 接受区间是违规「次数」的上下沿（x_lo / x_hi），换算成率才能与点同轴
        _sub["lo_pct"] = _sub["x_lo"] / _sub["T"] * 100
        _sub["hi_pct"] = _sub["x_hi"] / _sub["T"] * 100
        _sub["lab"] = _sub["model"].astype(str).map(
            lambda m: f"{m} 对照" if m in labels.NON_EVIDENCE_MODELS else m)
        st.altair_chart(
            charts.point_interval(
                _sub, cat="lab", val="rate_pct", lo="lo_pct", hi="hi_pct",
                nominal=labels.NOMINAL_RATE["99"], height=340,
                y_title="真实违规率（%）", x_title="",
            ),
            use_container_width=True,
        )
        charts.data_drawer(
            _sub[["model", "role", "T", "x", "rate_pct", "lo_pct", "hi_pct",
                  "p_uc", "es", "es_ratio"]],
            key="ov_drawer_model",
            title="数据抽屉 · 各模型的违规率与 Kupiec 接受区间（99% 检验水平）",
            height=280,
            fmt={"T": "%.0f", "x": "%.0f", "rate_pct": "%.4f", "lo_pct": "%.4f",
                 "hi_pct": "%.4f", "p_uc": "%.4f", "es": "%.4f", "es_ratio": "%.3f"},
            labels={"model": "模型", "role": "角色", "T": "样本天数", "x": "违规次数",
                    "rate_pct": "真实违规率 %", "lo_pct": "区间下沿 %", "hi_pct": "区间上沿 %",
                    "p_uc": "Kupiec p", "es": "平均期望损失 %", "es_ratio": "ES/判定线"},
            note=labels.KUPIEC_NOTE,
        )

    elif _sel == 1:
        _a = _attr[_attr["caliber"] == caliber].copy()
        _a["口径"] = cal_name
        # 段序按表里的出现次序（利率、利率、信用残差，次口径末尾才是汇率）
        _seg_order = list(_a["factor"].unique())
        st.altair_chart(
            charts.stacked_share(
                _a, cat="口径", seg="factor", val="share_pct",
                seg_order=_seg_order, height=110,
                x_title="波动贡献份额（%）",
            ),
            use_container_width=True,
        )
        charts.data_drawer(
            _a[["factor", "share_pct"]],
            key="ov_drawer_attr",
            title=f"数据抽屉 · {cal_name}的因子波动贡献份额",
            height=220,
            fmt={"share_pct": "%.4f"},
            labels={"factor": "因子", "share_pct": "份额 %"},
            note="利差代理（残差）是回归残差代理量，不是真实信用风险贡献。",
        )

    elif _sel == 2:
        _sh = _shift[_shift["caliber"] == caliber].copy()
        _long = pd.concat([
            _sh[["horizon_days", "sqrt_t_es99_ref_pct"]].rename(
                columns={"sqrt_t_es99_ref_pct": "值"}).assign(算法="√t 缩放值"),
            _sh[["horizon_days", "hist_worst_pct"]].rename(
                columns={"hist_worst_pct": "值"}).assign(算法="同期限实测最差"),
        ])
        _algos = ["√t 缩放值", "同期限实测最差"]
        st.altair_chart(
            alt.Chart(_long).mark_line(
                strokeWidth=charts.LINE_PX,
                point=alt.OverlayMarkDef(size=charts.MARKER_PX, filled=True,
                                         stroke=_tok["surface"],
                                         strokeWidth=charts.SURFACE_RING),
            ).encode(
                x=alt.X("horizon_days:Q", title="持有期（交易日）",
                        axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("值:Q", title="损失（%）"),
                color=alt.Color("算法:N", title="", sort=_algos,
                                scale=alt.Scale(domain=_algos,
                                                range=list(_tok["series"][:2]))),
                tooltip=[alt.Tooltip("horizon_days:Q", title="期限"),
                         alt.Tooltip("算法:N", title="算法"),
                         alt.Tooltip("值:Q", title="损失 %", format=".4f")],
            ).properties(height=340),
            use_container_width=True,
        )
        st.caption(
            "多期读数不参与判定：√t 未做独立的回测验证，"
            "它只能读作「若强行缩放会得到什么量级」。"
        )
        charts.data_drawer(
            _sh[["horizon_days", "sqrt_t_var95_ref_pct", "sqrt_t_var99_ref_pct",
                 "sqrt_t_es99_ref_pct", "hist_worst_pct", "residual_gap_pct"]],
            key="ov_drawer_shift",
            title=f"数据抽屉 · {cal_name}各期限的 √t 缩放对照与实测最差",
            height=260,
            fmt={"horizon_days": "%.0f", "sqrt_t_var95_ref_pct": "%.4f",
                 "sqrt_t_var99_ref_pct": "%.4f", "sqrt_t_es99_ref_pct": "%.4f",
                 "hist_worst_pct": "%.4f", "residual_gap_pct": "%.4f"},
            labels={"horizon_days": "期限", "sqrt_t_var95_ref_pct": "√t 95% VaR %",
                    "sqrt_t_var99_ref_pct": "√t 99% VaR %",
                    "sqrt_t_es99_ref_pct": "√t 99% ES %",
                    "hist_worst_pct": "同期限实测最差 %",
                    "residual_gap_pct": "覆盖缺口 pp"},
        )

    else:
        _s = _impact[(_impact["caliber"] == caliber)
                     & _impact["scenario_id"].astype(str).str.match(r"^S\d+$")].copy()
        _s = _s.sort_values("x_es99", ascending=False)
        _s["情景"] = _s["scenario_id"] + " " + _s["scenario_name"]
        st.altair_chart(
            charts.hbar_with_threshold(
                _s, cat="情景", val="x_es99", height=26 * len(_s) + 50,
                x_title="相当于 1 日 99% 期望损失判定线的倍数",
                tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                         alt.Tooltip("scenario_name:N", title="情景"),
                         alt.Tooltip("direction:N", title="方向"),
                         alt.Tooltip("severity:N", title="严重度"),
                         alt.Tooltip("x_es99:Q", title="判定线倍数", format=".3f"),
                         alt.Tooltip("worst_day_loss_pct:Q", title="最差单日损失 %",
                                     format=".4f")],
                # 情景名是判断句，默认的刻度宽度会把它们截成「S10 利率+信用同…」
            ).configure_axis(labelLimit=260),
            use_container_width=True,
        )
        charts.data_drawer(
            _s[["scenario_id", "scenario_name", "direction", "severity",
                "x_var95", "x_var99", "x_es99", "worst_day_loss_pct"]],
            key="ov_drawer_sev",
            title=f"数据抽屉 · {cal_name}情景库的判定线倍数",
            height=300,
            fmt={"x_var95": "%.3f", "x_var99": "%.3f", "x_es99": "%.3f",
                 "worst_day_loss_pct": "%.4f"},
            labels={"scenario_id": "编号", "scenario_name": "情景", "direction": "方向",
                    "severity": "严重度", "x_var95": "95% VaR 倍数",
                    "x_var99": "99% VaR 倍数", "x_es99": "99% ES 倍数",
                    "worst_day_loss_pct": "最差单日损失 %"},
            note="倍数 = 情景损失 ÷ 该口径 1 日 99% 期望损失判定线。"
                 "同一条情景在各缓冲档位下这个倍数不变。",
        )
    st.caption(CAPTIONS[_sel])
    jump(JUMPS[_sel])

page.footer("docs/final_report.md §1 业务视角核心结论、results/ 下各阶段产物",
            "backtest_results · stress_impact · stress_shift_coverage · "
            "var_attribution · var_historical")
