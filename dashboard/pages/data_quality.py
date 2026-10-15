"""数据质量（阶段一·数据体系的一部分）。

这一页回答一份读数建立在什么数上：喂进去的数本身靠不靠得住。四个来源文件各自拉到，
完整率、填补明细、报价口径选型、事件库、存疑日与异常判定。全部只读 `clean_data/`、
`factors/`、`events/` 下的鉴定类产物，与口径无关，因此本页不像其他页那样带口径切换。

这一页由原「稳健性检验与数据质量」页的数据质量块独立而来，按阶段重组时划入阶段一。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, page, theme  # noqa: E402

page.init("数据质量")

_vh = data.var_historical()
SPAN = f"{_vh['date'].min().date()} → {_vh['date'].max().date()}"
_tok = theme.tokens()

_cmp = data.completeness_report()
_worst_file = _cmp.loc[_cmp["completeness_pct"].idxmin()]

page.kpi_row([
    {"label": "最低的一份数据完整率",
     "value": f"{_worst_file['completeness_pct']:.2f} %",
     "help": f"{_worst_file['file']} 填补 {int(_worst_file['ffill_cells'])} 格、"
             f"分布于 {int(_worst_file['fill_runs'])} 段——它只作对照，不参与任何计量。"
             "口径：填补前实测。"},
])

page.header(
    "数据质量",
    "四个来源文件各自回答：完整率、填补、报价口径选型，以及哪些日子被判定为异常与存疑。"
    "内容与计量口径无关，故本页不带口径切换。",
    crumb="阶段一 · 数据质量",
    span=f"样本区间 {SPAN}",
)

# ------------------------------------------------------------ 1 完整率 ｜ 2 报价口径
_q1 = st.columns(2, gap="medium")

with _q1[0]:
    with page.card("数据完整率与填补",
                   "原始序列缺了多少格、补的是哪几段。完整率 = 填补前实际存在的格数 ÷ 应有格数。",
                   stretch=True):
        cmp_df = data.completeness_report().sort_values("completeness_pct").reset_index(drop=True)
        cmp_df["状态"] = ["完整率低于 100%" if v < 100 else "无缺口"
                        for v in cmp_df["completeness_pct"]]
        _fo = list(cmp_df["file"])
        # 八个文件的完整率都挤在 94 ~ 100 之间，柱从 0 起会把差别压平；
        # 改用点图（点不隐含基线）并在轴上标明截断起点，比截断柱诚实。
        _cb = alt.Chart(cmp_df).mark_point(
            size=130, filled=True, stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
        ).encode(
            y=alt.Y("file:N", title="", sort=_fo),
            x=alt.X("completeness_pct:Q", title="完整率（%）",
                    scale=alt.Scale(domain=[94.0, 100.6], clamp=True)),
            color=alt.Color("状态:N", title="",
                            scale=alt.Scale(domain=["无缺口", "完整率低于 100%"],
                                            range=[_tok["status"]["good"],
                                                   _tok["status"]["serious"]])),
            tooltip=[alt.Tooltip("file:N", title="文件"),
                     alt.Tooltip("frequency:N", title="频率"),
                     alt.Tooltip("completeness_pct:Q", title="完整率 %", format=".2f"),
                     alt.Tooltip("expected_days:Q", title="应有天数"),
                     alt.Tooltip("present_before_fill:Q", title="填补前存在"),
                     alt.Tooltip("ffill_cells:Q", title="前值填充格数"),
                     alt.Tooltip("fill_runs:Q", title="填补段数"),
                     alt.Tooltip("span_start:N", title="起始"),
                     alt.Tooltip("span_end:N", title="截止")],
        )
        _ct = alt.Chart(cmp_df).mark_text(
            align="left", dx=7, fontSize=10, color=_tok["ink2"], font=_tok["font"],
        ).encode(y=alt.Y("file:N", sort=_fo), x=alt.X("completeness_pct:Q"),
                 text=alt.Text("completeness_pct:Q", format=".2f"))
        st.altair_chart((_cb + _ct).properties(height=250), use_container_width=True)
        st.caption("横轴从 94% 起截断——点图不隐含 0 基线，截断在此不产生视觉误导。"
                   "绿点可放心引用，橙点补过。")

        fl = data.fill_log()
        charts.data_drawer(
            cmp_df, key="dq_cmp", title="数据抽屉 · 数据完整率：八个来源文件的缺口与填补",
            height=280,
            note=f"填补明细共 **{len(fl)}** 段、合计 **{int(fl['length'].sum())}** 格，"
                 "全部是前值填充，没有线性插值。展开下面的抽屉可以逐段查。"
                 "两份基准对照的完整率最低——"
                 "**这正是它们只作对照、不参与任何计量**的原因之一。",
            fmt={"expected_days": "%.0f", "present_before_fill": "%.0f",
                 "completeness_pct": "%.2f", "ffill_cells": "%.0f",
                 "lin_cells": "%.0f", "fill_runs": "%.0f"},
            labels={"file": "文件", "frequency": "频率", "span_start": "起始",
                    "span_end": "截止", "expected_days": "应有天数",
                    "present_before_fill": "填补前存在", "completeness_pct": "完整率 %",
                    "ffill_cells": "前值填充格数", "lin_cells": "线性插值格数",
                    "fill_runs": "填补段数", "状态": "状态"},
        )
        charts.data_drawer(
            fl, key="dq_fill", title="数据抽屉 · 填补明细：逐段记录（文件 / 列 / 起止 / 长度 / 方法）",
            height=320,
            fmt={"length": "%.0f"},
            labels={"file": "文件", "column": "列", "start": "起", "end": "止",
                    "length": "长度", "method": "方法"},
        )

with _q1[1]:
    with page.card("报价口径为什么选官方 NAV",
                   "同一个组合有四套可算的日收益。选型判据是零收益日占比最低，不是最容易算。",
                   stretch=True):
        sr = data.staleness_remedy_comparison()
        sr = sr.copy()
        sr["选中"] = ["选定方案" if s.startswith("D2") else "备选" for s in sr["方案"]]
        _sr_long = sr.melt(id_vars=["方案", "选中"], value_vars=["零收益占比%", "利率剥离R²"],
                           var_name="metric", value_name="value").dropna(subset=["value"])
        _sr_long["指标"] = _sr_long["metric"].map({"零收益占比%": "零收益日占比（%）",
                                                  "利率剥离R²": "利率剥离 R²"})
        _sb = alt.Chart(_sr_long).mark_bar(size=18, cornerRadiusEnd=4).encode(
            y=alt.Y("方案:N", title="", sort=list(sr["方案"])),
            x=alt.X("value:Q", title=""),
            color=alt.Color("选中:N", title="",
                            scale=alt.Scale(domain=["选定方案", "备选"],
                                            range=[_tok["status"]["good"], _tok["muted"]])),
            tooltip=[alt.Tooltip("方案:N", title="方案"),
                     alt.Tooltip("指标:N", title="指标"),
                     alt.Tooltip("value:Q", title="取值", format=".3f")],
        )
        _st = alt.Chart(_sr_long).mark_text(
            align="left", dx=6, fontSize=10, color=_tok["ink2"], font=_tok["font"],
        ).encode(y=alt.Y("方案:N", sort=list(sr["方案"])), x=alt.X("value:Q"),
                 text=alt.Text("value:Q", format=".2f"))
        st.altair_chart(
            (_sb + _st).properties(
                height=210, width=charts.facet_width(2)).facet(
                column=alt.Column("指标:N", title="", sort=["零收益日占比（%）", "利率剥离 R²"])
            ).resolve_scale(x="independent"),
            use_container_width=False,
        )
        _d1 = sr[sr["方案"].str.startswith("D1")].iloc[0]
        _d2 = sr[sr["方案"].str.startswith("D2")].iloc[0]
        st.caption(
            f"两个指标量纲不同（百分比与 R²），所以小倍数分面、各用各的横轴。"
            f"D1 价格口径有 {_d1['零收益占比%']:.1f}% 的交易日收益为 0——"
            "在这个序列上做回测，会把「无交易日」误当作「无风险日」。"
        )
        charts.data_drawer(
            sr, key="dq_stale", title="数据抽屉 · 报价口径：四套方案并排与选型判据",
            height=220,
            note=f"官方 NAV 的零收益日只有 {_d2['零收益占比%']:.1f}%。"
                 f"选官方 NAV 的代价要一并说清：它的年化波动从 {_d1['年化波动%']:.2f}% 升到 "
                 f"{_d2['年化波动%']:.2f}%，1 日 99% 正态 VaR 从 {_d1['1日99%VaR(正态)%']:.3f}% 升到 "
                 f"{_d2['1日99%VaR(正态)%']:.3f}%，等效久期从 {_d1['等效久期(年)']:.2f} 年升到 "
                 f"{_d2['等效久期(年)']:.2f} 年。读数变高不是变差，是原先被零收益日压低了。",
            fmt={"n": "%.0f", "零收益占比%": "%.1f", "年化波动%": "%.2f", "1日σ%": "%.3f",
                 "1日99%VaR(正态)%": "%.3f", "利率剥离R²": "%.3f", "等效久期(年)": "%.2f",
                 "OAS-ρ": "%.3f", "OAS命中%": "%.1f"},
        )

# ------------------------------------------------------------ 3 存疑日与事件
_q2 = st.columns(2, gap="medium")

with _q2[0]:
    with page.card("事件库：日期与市场维度",
                   "哪些交易日不可信、凭什么判定。「存疑」只做标注，不改数、不剔除主口径。",
                   stretch=True):
        ev = data.risk_events()
        od = data.outlier_judgment()
        dd = data.doubtful_days()
        _order_mkt = ev["market"].value_counts().index.tolist()
        _ed = alt.Chart(ev).mark_point(
            size=70, filled=True, color=_tok["series"][0],
            stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("market:N", title="", sort=_order_mkt),
            tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                     alt.Tooltip("event_cn:N", title="事件"),
                     alt.Tooltip("market:N", title="市场维度"),
                     alt.Tooltip("expected_direction:N", title="预期方向"),
                     alt.Tooltip("note:N", title="备注"),
                     alt.Tooltip("source:N", title="来源")],
        )
        st.altair_chart(_ed.properties(height=250), use_container_width=True)
        st.caption("单色即可——信息在「哪个市场维度、哪一天」，不在事件之间比大小。")
        charts.data_drawer(
            ev, key="dq_ev", title="数据抽屉 · 事件库：全部事件按日期与市场维度",
            height=300,
            note=f"事件库共 **{len(ev)}** 条，自 {ev['date'].min():%Y-%m-%d} 起，"
                 f"覆盖 {ev['market'].nunique()} 个市场维度。"
                 "市场维度有九种组合（如 `rate+credit`），分行原样保留，不做归并。",
            labels={"date": "日期", "event_cn": "事件", "market": "市场维度",
                    "expected_direction": "预期方向", "note": "备注", "source": "来源"},
        )

with _q2[1]:
    with page.card("存疑日与异常判定：只标注、不改数",
                   "哪些日子被判定为真实冲击、哪些只能存疑。判定为「真实冲击」不等于该日正常。",
                   stretch=True):
        o2 = od.copy()
        _oc = alt.Chart(o2).mark_point(
            size=60, filled=True, stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
        ).encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("series:N", title="", sort=sorted(o2["series"].unique())),
            color=alt.Color("verdict:N", title="",
                            scale=alt.Scale(domain=["真实冲击", "存疑(无事件支撑)"],
                                            range=[_tok["series"][0],
                                                   _tok["status"]["critical"]])),
            shape=alt.Shape("verdict:N", title="", legend=None,
                            scale=alt.Scale(domain=["真实冲击", "存疑(无事件支撑)"])),
            tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                     alt.Tooltip("series:N", title="序列"),
                     alt.Tooltip("value:Q", title="取值", format=".4f"),
                     alt.Tooltip("hit:N", title="命中口径"),
                     alt.Tooltip("verdict:N", title="判定"),
                     alt.Tooltip("handling:N", title="处理"),
                     alt.Tooltip("event:N", title="对应事件"),
                     alt.Tooltip("gap_days:Q", title="距事件天数"),
                     alt.Tooltip("note:N", title="说明")],
        )
        st.altair_chart(_oc.properties(height=250), use_container_width=True)
        st.caption("颜色与形状同时编码判定结果：橙色圆点 = 有事件支撑的真实冲击、保留原值；"
                   "红色 = 存疑、仅标注不改数。")
        charts.data_drawer(
            o2, key="dq_od", title="数据抽屉 · 异常候选日的判定结果与处理方式",
            height=320,
            note=f"异常候选日 **{len(od)}** 条（按滚动分位命中），其中判定为真实冲击的 "
                 f"**{int((od['verdict'] == '真实冲击').sum())} / {len(od)} 条**"
                 "（手判，能在事件库里找到对应，保留原值）。"
                 "判定为「真实冲击」不等于该日正常，它只说明这一天的大幅波动"
                 "能在事件库里找到对应。",
            fmt={"value": "%.4f", "gap_days": "%.0f"},
            labels={"date": "日期", "series": "序列", "value": "取值", "hit": "命中口径",
                    "verdict": "判定", "handling": "处理", "event_date": "事件日期",
                    "event": "对应事件", "gap_days": "距事件天数", "note": "说明"},
        )
        charts.data_drawer(
            dd, key="dq_dd", title="数据抽屉 · 存疑日清单（仅标注、不改数）",
            height=260,
            fmt={"value": "%.4f"},
            labels={"series": "序列", "date": "日期", "value": "取值", "hit": "命中口径",
                    "verdict": "判定", "handling": "处理", "note": "说明", "special": "标记"},
            note="这些日子在阶段二回测里作为异常标记保留，"
                 "用来判断某些违规是数据问题还是模型问题造成的。"
                 f"存疑日共 **{len(dd)}** 个（无事件支撑，仅标注、不改数、不剔除主口径），"
                 "全部落在主口径的两个序列上（ETF 收益与 OAS 变化），汇率序列的存疑日只影响次口径。",
        )

page.footer(
    "clean_data/completeness_report.csv、fill_log.csv、outlier_judgment.csv；"
    "factors/staleness_remedy_comparison.csv；events/risk_events_timeline.csv；"
    "results/doubtful_days.csv",
    "completeness_pct / expected_days / present_before_fill / ffill_cells / fill_runs / "
    "零收益占比% / 年化波动% / 1日σ% / 1日99%VaR(正态)% / 利率剥离R² / "
    "等效久期(年) / value / gap_days / verdict / handling",
)
