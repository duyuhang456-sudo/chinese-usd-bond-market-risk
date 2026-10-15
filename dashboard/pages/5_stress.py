"""压力测试交互分析。

一页回答一个问题：极端情形下亏多少、越没越过判定线。

主导统计量是报告推荐的**倍数**（含缓冲净损失 ÷ 1 日 99% 期望损失），不是
「超限 / 未超限」的二值名单——二值判断对缓冲档位敏感，同一个情景换个档位就翻面，
倍数不会整块翻面，更适合跨情景比较。二值名单保留在最后一张卡里作辅助披露。

顶部一条控制带驱动全页：口径与缓冲档位。情景类别、计量口径、情景族、锚点窗长这些
此前各节自己的选择器全部撤掉，一律钉在报告的登记取值上，这样每一节的读数都能与
终期报告逐字对上，读者也不用先猜设定再看数。缓冲档位是滑块而不是连续输入：五个
登记档位之间不做插值。

数据全部查表，页面不重算任何计量结果：`results/stress_impact.csv`（30 行，
15 个情景 × 2 个口径）、`stress_buffer_sens.csv`（150 行，5 个缓冲档位）、
`stress_scenarios.csv`（112 行，情景库映射）、`stress_factor_contrib.csv`。
唯一一处现场算术是「净损失 ÷ 判定线」——这就是 `x_es99` 的定义本身。

版式：三张结论图（情景库映射、损失倍数、因子贡献）与收尾的二值名单各占满宽，
其余两件并排；每张图只挂一句读法，长解释与附录表都收进数据抽屉。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("压力测试交互分析")

_si = data.stress_impact()
_bs = data.stress_buffer_sens()
_sc = data.stress_scenarios()
_fc = data.stress_factor_contrib()
_tok = theme.tokens()

FAMILIES = ["利率上行", "信用利差走阔", "离岸汇率贬值"]
CLASSES = ["历史", "补充", "假设"]
REGISTERED = "0.3365 本报告取值"
SID_ORDER = list(dict.fromkeys(_si["scenario_id"]))
BUF_LABELS = list(dict.fromkeys(_bs["buffer_label"]))

# 满宽卡片里分面图的定宽基准（按窄视口取，宽屏留白而不横向溢出）
FULL_W = 1000

# 控制带要排在页头下面，页头又要读到它们当前的值：先取上一轮落在 session_state
# 里的值喂给页头，控件本身在页头之后再落位。
caliber = st.session_state.get("st_cal") or "主"
buffer_label = st.session_state.get("st_buf") or REGISTERED

es_line = labels.ES_1D_99[caliber]

page.header(
    "压力测试交互分析",
    "十五个登记情景横跨历史、补充、假设三类，全部是 1 日持有期，正值为损失。"
    "主轴是「相当于几倍 1 日 99% 期望损失」，倍数不随缓冲档位整块翻面。",
    crumb="阶段三 · 压力测试",
    caliber=labels.CALIBER_SHORT[caliber],
    span=f"缓冲档位 {buffer_label} · 判定线 {es_line:.4f}% · "
         f"锚点日 2022-03-14 → 2025-04-07",
)

# ---------------------------------------------------------------- 控制带
with st.container(border=True):
    _k1, _k2 = st.columns([2, 5])
    with _k1:
        caliber = st.segmented_control(
            "口径", ["主", "次"], default="主", key="st_cal",
            format_func=lambda c: labels.CALIBER_SHORT[c],
            help="决定判定线、因子贡献与全部损失读数的计价基准。"
                 "纯汇率情景在美元口径下恒为 0，换成人民币口径才显出损失。",
        ) or "主"
        es_line = labels.ES_1D_99[caliber]
    with _k2:
        buffer_label = st.select_slider(
            "缓冲档位（登记的 5 个离散值，不做插值）", BUF_LABELS,
            value=REGISTERED, key="st_buf",
            help="缓冲只在「历史」与「补充」情景上生效，再加到损失上；纯汇率情景按定义"
                 "不加缓冲，所以低档位那一段的曲线是平的。判定线本身不随档位变。",
        )

buf_value = float(_bs.loc[_bs["buffer_label"] == buffer_label, "buffer_value"].iloc[0])

# 当前口径 + 当前档位下的净损失与判定线，全页共用；x_es99 即此定义
_cur = _bs[(_bs["caliber"] == caliber) & (_bs["buffer_label"] == buffer_label)].copy()
_cur["x_es99"] = _cur["loss_net_pct"] / _cur["es_1d_99_pct"]
_base = _si[_si["caliber"] == caliber].drop(
    columns=["loss_after_buffer_pct", "x_es99", "x_var99", "tail_flag"])
M = _base.merge(_cur[["scenario_id", "loss_net_pct", "x_es99", "is_tail"]],
                on="scenario_id", how="left")
M["buffer_now_pct"] = M["loss_net_pct"] - M["benchmark_loss_pct"]
M["类别"] = M["scenario_class"].map(labels.CLASS_LABEL)

_n_hit = int(_cur["is_tail"].sum())
_top = _cur.loc[_cur["x_es99"].idxmax()]
_top_name = str(_si.loc[_si["scenario_id"] == _top["scenario_id"], "scenario_name"].iloc[0])
# 历史与补充情景里未被因子解释的那一份（假设情景的残差恒为 0，不参与取最大）
_hist = _fc[_fc["factor"] == "residual"]
_hist = _hist[(_hist["caliber"] == caliber) & (_hist["scenario_class"] != "假设")]
_res = _hist.loc[_hist["share_pct"].idxmax()]

# 读数带的值只放数字与单位：带了「（主口径）」这个后缀会被读数卡截断成
# 「4.31 倍（主口…」。口径由页头的 meta 行与上面的控制带统一承载。
page.kpi_row([
    {"label": "当前档位下最极端情景仍在判定线之上",
     "value": f"{_top['x_es99']:.2f} 倍",
     "help": f"{labels.CALIBER_SHORT[caliber]}下 {_top['scenario_id']} {_top_name}，"
             f"含缓冲净损失 {_top['loss_net_pct']:.4f}%。"
             "换成次口径且不加缓冲时，极端汇率情景 S09 是 4.51 倍。"},
    {"label": "含缓冲净损失越过判定线的情景占多数",
     "value": f"{_n_hit} / {len(SID_ORDER)}",
     "help": f"本档位下这两个口径分别是 "
             f"{int(_bs[(_bs['caliber'] == '主') & (_bs['buffer_label'] == buffer_label)]['is_tail'].sum())} 条"
             f"与 {int(_bs[(_bs['caliber'] == '次') & (_bs['buffer_label'] == buffer_label)]['is_tail'].sum())} 条。"
             f"缓冲取 0 时是 "
             f"{int(_bs[(_bs['caliber'] == caliber) & (_bs['buffer_label'] == '0.0000 不缓冲')]['is_tail'].sum())} 条，"
             f"档位换到 +1pp 压力档时是 "
             f"{int(_bs[(_bs['caliber'] == caliber) & (_bs['buffer_label'] == '1.0000 +1pp 压力档')]['is_tail'].sum())} 条。"
             "名单见「超限名单与实现值对照」那张卡。"},
    {"label": "1 日口径下缓冲已占到判定线的一半",
     "value": f"{buf_value / es_line * 100:.1f} %",
     "help": f"{labels.CALIBER_SHORT[caliber]}下 {buf_value:.4f} pp ÷ {es_line:.4f}%"
             "（判定线随口径变）。缓冲占的比重越大，「超限 / 未超限」这条二值判断对档位越敏感。"},
    {"label": "历史情景里未被因子解释的份额接近一半",
     "value": f"{float(_res['share_pct']):.1f} %",
     "help": f"{labels.CALIBER_SHORT[caliber]}下最高的一处是 {_res['scenario_id']}。"
             "残差是 δ 回归的拟合误差，不是真实信用风险贡献。"},
])

# ============================================ 1｜情景库映射（满宽）
with page.card("情景库与冲击映射",
               "同一个情景在不同因子上的冲击量级，按 bp 与 % 分面画，不共用一根"
               "坐标轴；汇率冲击记为负值，负号表示人民币贬值方向。"):
    _p1 = _sc[_sc["measure"] == "peak_1d"].copy()
    _p1["类别"] = _p1["scenario_class"].map(labels.CLASS_LABEL)
    _class_order = [labels.CLASS_LABEL[c] for c in CLASSES]
    _fac = alt.Chart(_p1).mark_bar(size=charts.MAX_BAR_PX, cornerRadiusEnd=4).encode(
        x=alt.X("scenario_id:N", title="", sort=SID_ORDER),
        y=alt.Y("shock:Q", title="冲击（带符号）"),
        color=alt.Color("类别:N", title="", sort=_class_order,
                        scale=alt.Scale(domain=_class_order,
                                        range=list(_tok["series"][:3]))),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("factor_cn:N", title="因子"),
                 alt.Tooltip("shock:Q", title="冲击", format=".4f"),
                 alt.Tooltip("unit:N", title="单位"),
                 alt.Tooltip("anchor_date:N", title="锚点日")],
    ).properties(height=250, width=charts.facet_width(2, card_w=FULL_W)).facet(
        column=alt.Column("unit:N", title="")
    ).resolve_scale(y="independent")
    st.altair_chart(_fac, use_container_width=False)
    charts.data_drawer(
        _sc, key="st_mapping", height=340,
        title="数据抽屉 · 情景库映射：112 条「情景 × 因子 × 计量口径」的冲击登记",
        columns=["scenario_id", "scenario_name", "scenario_class", "factor_cn", "unit",
                 "measure", "shock", "multiple", "available", "anchor_date", "note"],
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "factor_cn": "因子", "unit": "单位", "measure": "计量口径",
                "shock": "冲击", "multiple": "倍数", "available": "数据可得",
                "anchor_date": "锚点日", "note": "备注"},
        fmt={"shock": "%.4f", "multiple": "%.2f"},
        note="判定只认单日最差（peak_1d）这一档，其余计量口径是附录对照，"
             "不参与判定。计量口径 peak_1d 共 15 条；single_extreme（16 条）、"
             "peak_3d（31 条）、cum_window（26 条）、worst_day（10 条）、"
             "max_drawdown（10 条）、窗内实现（4 条）都是附录对照。"
             "假设情景的轻、中、极端三档严格成 1× / 1.5× / 2×，是登记情景的性质，"
             "不是拟合出来的斜率。",
    )

# ============================================ 2｜损失与倍数（满宽，本页的锚）
with page.card("损失、判定线与压力倍数",
               "两根条分别是未加缓冲的基准损失与当前档位下的净损失；红菱形是 "
               "1 日 99% 期望损失判定线，条端数字是「含缓冲净损失 ÷ 判定线」。"
               "历史与纯汇率情景的两根条等长，因为这两类按定义不加缓冲。"):
    K_BASE = "基准损失（未加缓冲）"
    K_NET = f"加缓冲后（{buffer_label}）"
    _order_sid = list(M.sort_values("loss_net_pct", ascending=False)["scenario_id"])
    _long = M.melt(id_vars=["scenario_id", "scenario_name", "es_1d_99_pct", "x_es99"],
                   value_vars=["benchmark_loss_pct", "loss_net_pct"],
                   var_name="kind", value_name="loss")
    _long["kind"] = _long["kind"].map({"benchmark_loss_pct": K_BASE,
                                       "loss_net_pct": K_NET})

    _y = alt.Y("scenario_id:N", title="", sort=_order_sid)
    _lossbars = alt.Chart(_long).mark_bar(size=10, cornerRadiusEnd=4).encode(
        y=_y,
        yOffset=alt.YOffset("kind:N", sort=[K_BASE, K_NET]),
        x=alt.X("loss:Q", title=f"损失（%，正值 = 损失）· {labels.CALIBER_SHORT[caliber]}",
                scale=alt.Scale(zero=True)),
        color=alt.Color("kind:N", title="", sort=[K_BASE, K_NET],
                        scale=alt.Scale(domain=[K_BASE, K_NET],
                                        range=list(_tok["series"][:2]))),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("kind:N", title="口径"),
                 alt.Tooltip("loss:Q", title="损失 %", format=".4f")],
    )
    _dia = alt.Chart(M).mark_point(size=130, shape="diamond", filled=True,
                                   color=_tok["status"]["critical"],
                                   stroke=_tok["surface"],
                                   strokeWidth=charts.SURFACE_RING).encode(
        y=_y, x=alt.X("es_1d_99_pct:Q"),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("es_1d_99_pct:Q", title="判定线 %", format=".4f")],
    )
    # 条端倍数用自画的文字层：charts.annotate 走的是数值 y 轴，这里 y 是分类轴
    _lab = alt.Chart(M).mark_text(align="left", dx=6, fontSize=10.5,
                                  color=_tok["ink2"], font=_tok["font"]).encode(
        y=_y, x=alt.X("loss_net_pct:Q"), text=alt.Text("x_es99:Q", format=".2f"),
    )
    st.altair_chart((_lossbars + _dia + _lab).properties(height=330),
                    use_container_width=True)
    st.caption(
        "纯汇率情景在美元口径下整体恒为 0，汇率被计价基准搬运掉了，"
        "同一个情景到了人民币口径才显出 1.59 / 2.38 / 3.17 的损失。"
    )
    charts.data_drawer(
        M.sort_values("loss_net_pct", ascending=False), key="st_detail", height=380,
        title="数据抽屉 · 压力情景明细：15 个情景的基准、缓冲、倍数与实现值",
        columns=["scenario_id", "scenario_name", "scenario_class", "direction", "severity",
                 "anchor_date", "benchmark_loss_pct", "benchmark_source", "buffer_now_pct",
                 "loss_net_pct", "es_1d_99_pct", "x_es99", "loss_realized_pct",
                 "residual_pct", "worst_day_date", "credit_range_lo_pct",
                 "credit_range_hi_pct", "credit_range_note"],
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "direction": "方向", "severity": "档位", "anchor_date": "锚点日",
                "benchmark_loss_pct": "基准损失 %", "benchmark_source": "基准来源",
                "buffer_now_pct": "当前档位缓冲 pp", "loss_net_pct": "含缓冲净损失 %",
                "es_1d_99_pct": "判定线 %", "x_es99": "判定线倍数",
                "loss_realized_pct": "锚点窗累计实现 %", "residual_pct": "未解释残差 pp",
                "worst_day_date": "最差单日", "credit_range_lo_pct": "信用路径下沿 %",
                "credit_range_hi_pct": "信用路径上沿 %", "credit_range_note": "信用区间说明"},
        fmt={"benchmark_loss_pct": "%.4f", "buffer_now_pct": "%.4f",
             "loss_net_pct": "%.4f", "es_1d_99_pct": "%.4f", "x_es99": "%.3f",
             "loss_realized_pct": "%.4f", "residual_pct": "%.4f",
             "credit_range_lo_pct": "%.4f", "credit_range_hi_pct": "%.4f"},
        note="组合极端情景 S10 的基准损失 2.5363% 就是同族的 S03（2.1193%）与 "
             "S06（0.4171%）线性累加，这是登记时的构造，不是测算结果。"
             "「当前档位缓冲 pp」是含缓冲净损失减基准损失，历史与纯汇率情景在任何档位下都为 0。"
             "信用路径上下沿只有信用类情景有值，其说明原文照录：线性外推有偏差，"
             "不作信用风险可控的判断。",
    )

# ============================================ 3｜缓冲档位敏感性 ｜ 族内梯度
row3 = st.columns(2, gap="medium")
with row3[0]:
    with page.card("缓冲档位与超限条数",
                   "五个登记档位之间不插值，连起来只看走向；换档位只推「加缓冲后」的"
                   "损失，判定线本身不动。", stretch=True):
        _tc = (_bs.groupby(["buffer_value", "buffer_label", "caliber"], as_index=False)["is_tail"]
               .sum().rename(columns={"is_tail": "n_tail"}))
        _tot = (_tc.groupby(["buffer_value", "buffer_label"], as_index=False)["n_tail"]
                .sum().assign(caliber="合计"))
        CURVE = pd.concat([_tc, _tot], ignore_index=True)
        CURVE["口径"] = CURVE["caliber"].map(lambda c: labels.CALIBER_SHORT.get(c, c))
        _order_cn = [labels.CALIBER_SHORT["主"], labels.CALIBER_SHORT["次"], "合计"]

        _lines = alt.Chart(CURVE[CURVE["口径"] != "合计"]).mark_line(
            strokeWidth=charts.LINE_PX, point=alt.OverlayMarkDef(
                size=charts.MARKER_PX, filled=True, stroke=_tok["surface"],
                strokeWidth=charts.SURFACE_RING),
        ).encode(
            x=alt.X("buffer_value:Q", title="缓冲（pp）"),
            y=alt.Y("n_tail:Q", title="超限情景条数"),
            color=alt.Color("口径:N", title="", sort=_order_cn,
                            scale=alt.Scale(domain=_order_cn,
                                            range=list(_tok["series"][:2]))),
            tooltip=[alt.Tooltip("buffer_label:N", title="档位"),
                     alt.Tooltip("口径:N"), alt.Tooltip("n_tail:Q", title="超限条数")],
        )
        # 合计是两条线的加总，不是第三个平行序列，所以用中性墨而不是第三个分类色
        _agg = alt.Chart(CURVE[CURVE["口径"] == "合计"]).mark_line(
            strokeWidth=3, color=_tok["ink2"], opacity=0.5).encode(
            x=alt.X("buffer_value:Q"), y=alt.Y("n_tail:Q"),
            tooltip=[alt.Tooltip("buffer_label:N", title="档位"),
                     alt.Tooltip("n_tail:Q", title="两口径合计条数")],
        )
        st.altair_chart(
            charts.add_rule((_lines + _agg).properties(height=250), buf_value,
                            label=f"当前档位 {buf_value:.4f} pp", orient="vertical"),
            use_container_width=True,
        )
        _piv = _tc.pivot_table(index=["buffer_value", "buffer_label"], columns="caliber",
                               values="n_tail")
        _piv["合计"] = _piv.sum(axis=1)
        _piv = _piv.rename(columns=labels.CALIBER_SHORT).reset_index()
        _piv.columns = ["缓冲（pp）", "档位", "主口径", "次口径", "合计"]
        for _c in ("主口径", "次口径", "合计"):
            _piv[_c] = _piv[_c].astype(int)
        charts.data_drawer(
            _piv, key="st_buffer", height=250,
            title="数据抽屉 · 缓冲档位超限条数：5 个档位 × 主 / 次口径与合计",
            note=f"本报告取值是「{REGISTERED}」这一档，两口径合计 19 条，是全报告引用最多的"
                 "那组读数。5 个档位依次是 17 / 17 / 19 / 23 / 23 条，全部落在 17 到 23 之间；"
                 "主口径不缓冲时 7 条，缓冲只会把损失推得更远，条数只增不减。"
                 "次口径的起点就高于主口径，因为汇率一进入计价就带来额外损失。"
                 "低档位那一段是平的：纯汇率情景恒不加缓冲、历史情景取实际路径。",
        )

with row3[1]:
    with page.card("族内梯度",
                   "三族的轻 / 中 / 极端三档在当前档位下各落在哪里。"
                   + labels.GRADIENT_NOTE, stretch=True):
        _grp = M[M["scenario_id"].str.match(r"^S\d+$") & M["direction"].isin(FAMILIES)].copy()
        _grp["倍数"] = _grp["severity"].map(
            {labels.SEVERITY_ORDER[0]: 1.0, labels.SEVERITY_ORDER[1]: 1.5,
             labels.SEVERITY_ORDER[2]: 2.0})
        _grp = _grp.sort_values(["direction", "倍数"])
        _ymax = float(_grp["loss_net_pct"].max()) * 1.24

        _grad = alt.Chart(_grp).mark_line(strokeWidth=charts.LINE_PX, point=alt.OverlayMarkDef(
            size=charts.MARKER_PX, filled=True, stroke=_tok["surface"],
            strokeWidth=charts.SURFACE_RING)).encode(
            x=alt.X("倍数:Q", title="相对本族轻度档的倍数", scale=alt.Scale(domain=[0.9, 2.15])),
            y=alt.Y("loss_net_pct:Q",
                    title=f"含缓冲净损失（%）· {labels.CALIBER_SHORT[caliber]}",
                    scale=alt.Scale(domain=[0, _ymax], zero=True)),
            color=alt.Color("direction:N", title="", sort=FAMILIES,
                            legend=alt.Legend(orient="bottom", direction="horizontal",
                                              columns=3, title=None, symbolType="circle"),
                            scale=alt.Scale(domain=FAMILIES, range=list(_tok["series"][:3]))),
            tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                     alt.Tooltip("scenario_name:N", title="情景"),
                     alt.Tooltip("倍数:Q", title="相对轻度档", format=".2f"),
                     alt.Tooltip("loss_net_pct:Q", title="含缓冲净损失 %", format=".4f"),
                     alt.Tooltip("x_es99:Q", title="判定线倍数", format=".3f")],
        )
        _peak = _grp.loc[_grp["loss_net_pct"].idxmax()]
        # 判定线的标签改挂在线的**下方**：信用利差这一族在左端就贴在判定线附近，
        # 默认的「线上方」写法会和它撞在一起。annotate 的 dot=False 就是纯文字层。
        _rule = charts.annotate(
            charts.add_rule(_grad.properties(height=265), es_line),
            x=1.02, y=es_line, text=f"判定线 {es_line:.4f}%",
            dot=False, align="left", dx=0, dy=14,
        )
        st.altair_chart(
            charts.annotate(_rule,
                            x=float(_peak["倍数"]), y=float(_peak["loss_net_pct"]),
                            text=f"{_peak['scenario_id']} {_peak['loss_net_pct']:.4f}%，"
                                 f"判定线的 {float(_peak['x_es99']):.2f} 倍",
                            align="right", dx=-8, dy=-14),
            use_container_width=True,
        )
        st.caption(
            "缓冲对同一族的三档是同一个常数，所以加缓冲后的三点仍在一条直线上，只是整条线平移。"
        )
        charts.data_drawer(
            _grp[["scenario_id", "scenario_name", "direction", "severity", "倍数",
                  "benchmark_loss_pct", "loss_net_pct", "x_es99"]],
            key="st_gradient",
            title="数据抽屉 · 族内梯度：三个情景族 × 轻 / 中 / 极端三档的损失与倍数",
            labels={"scenario_id": "编号", "scenario_name": "情景", "direction": "族",
                    "severity": "档位", "倍数": "相对轻度档", "benchmark_loss_pct": "基准损失 %",
                    "loss_net_pct": "含缓冲净损失 %", "x_es99": "判定线倍数"},
            fmt={"倍数": "%.2f", "benchmark_loss_pct": "%.4f", "loss_net_pct": "%.4f",
                 "x_es99": "%.3f"},
            note="三族各三档共 9 条，另加一条组合极端 S10 不在族内。"
                 "倍数一列是登记时的相对档位。离岸汇率那一族在美元口径下整条压在 0 上，"
                 "换到人民币口径才抬起来。",
        )

# ============================================ 4｜因子贡献度（满宽）
with page.card("因子贡献度",
               f"每条情景的净损失在五个因子间的份额，100% 堆叠，按当前口径与当前档位下的"
               f"净损失从大到小排列（{labels.CALIBER_SHORT[caliber]}）。"
               f"「{labels.FACTOR_CN['residual']}」是 δ 回归的拟合误差，不是真实信用风险的"
               "贡献，它在历史情景里占到四到五成。"):
    _fcc = _fc[_fc["caliber"] == caliber].copy()
    _fcc["情景"] = _fcc["scenario_id"] + " " + _fcc["scenario_name"]
    _fcc["情景"] = pd.Categorical(
        _fcc["情景"],
        categories=[f"{sid} {nm}" for sid, nm in
                    M.sort_values("loss_net_pct", ascending=False)[
                        ["scenario_id", "scenario_name"]].itertuples(index=False)],
        ordered=True,
    )
    st.altair_chart(
        charts.stacked_share(_fcc, cat="情景", seg="factor_cn", val="share_pct",
                             seg_order=[labels.FACTOR_CN[f] for f in labels.FACTOR_ORDER],
                             height=300),
        use_container_width=True,
    )
    charts.data_drawer(
        _fcc[["scenario_id", "scenario_name", "scenario_class", "factor_cn",
              "contribution_pct", "share_pct"]],
        key="st_contrib", height=340,
        title="数据抽屉 · 因子贡献度：15 个情景 × 5 个因子的份额与贡献",
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "factor_cn": "因子", "contribution_pct": "贡献（pp）",
                "share_pct": "份额（%）"},
        fmt={"contribution_pct": "%.4f", "share_pct": "%.2f"},
        note="纯汇率情景在美元口径下五个因子的份额全为 0——损失本身就是 0，"
             "没有可分解的对象。假设情景的份额是登记时的构造值：利率族按久期比例分，"
             "信用族整条落在利差上，纯汇率情景在次口径下 100% 落在汇率上。"
             "单日损失里有相当一块现有因子解释不了。",
    )

# ============================================ 5｜超限名单（满宽，收尾的二值披露）
with page.card("超限名单与实现值对照",
               "二值名单作辅助披露，换一次档位名单就会变，这正是它不适合当严重度"
               "排序的原因。"):
    _vis = _cur[_cur["is_tail"]].sort_values("x_es99", ascending=False).copy()
    st.markdown(f"**当前档位下的超限名单**（{labels.CALIBER_SHORT[caliber]} · {buffer_label}，"
                f"{len(_vis)} 条）")
    st.dataframe(
        _vis[["scenario_id", "scenario_name", "loss_net_pct", "es_1d_99_pct", "x_es99"]],
        width="stretch", hide_index=True, height=min(320, 36 * len(_vis) + 44),
        key="st_tail_list",
        column_config={
            "scenario_id": st.column_config.Column("编号", width="small"),
            "scenario_name": st.column_config.Column("情景"),
            "loss_net_pct": st.column_config.NumberColumn("含缓冲净损失 %", format="%.4f"),
            "es_1d_99_pct": st.column_config.NumberColumn("判定线 %", format="%.4f"),
            "x_es99": st.column_config.NumberColumn("判定线倍数", format="%.3f"),
        },
    )
    _hist_rows = M[M["scenario_class"].isin(["历史", "补充"])].copy()
    _hist_rows = _hist_rows.sort_values("loss_net_pct", ascending=False)
    charts.data_drawer(
        _hist_rows[["scenario_id", "scenario_name", "direction",
                    "benchmark_loss_pct", "loss_realized_pct", "residual_pct",
                    "worst_day_date", "loss_net_pct", "x_es99"]],
        key="st_hist", height=240,
        title="数据抽屉 · 历史与补充情景：基准损失与窗内实际实现的差",
        labels={"scenario_id": "编号", "scenario_name": "情景", "direction": "方向",
                "benchmark_loss_pct": "基准损失 %", "loss_realized_pct": "锚点窗累计实现 %",
                "residual_pct": "未解释残差 pp", "worst_day_date": "最差单日",
                "loss_net_pct": "含缓冲净损失 %", "x_es99": "判定线倍数"},
        fmt={"benchmark_loss_pct": "%.4f", "loss_realized_pct": "%.4f",
             "residual_pct": "%.4f", "loss_net_pct": "%.4f", "x_es99": "%.3f"},
        note="历史与补充情景的基准损失是「窗内最差单日」，锚点窗累计实现是整段窗口的"
             "实际结果，两者不是同一个量。H1 与 X2 的实现值比单日更差"
             "（+1.5368% 对 1.3449%、+3.0008% 对 1.0349%）；H2 反过来，那段时间整体是赚的，"
             "锚点窗累计实现记为 −1.9981%，方向与情景设定相反而没有做任何符号翻转，"
             "因为那就是窗口里的真实结果。假设情景没有实现值可对，它们的损失本来由 δ 映射算出，"
             "残差恒为 0。",
    )

page.footer("results/stress_scenarios.csv、stress_buffer_sens.csv、"
            "stress_impact.csv、stress_factor_contrib.csv",
            "shock / multiple / loss_net_pct / es_1d_99_pct / share_pct")
