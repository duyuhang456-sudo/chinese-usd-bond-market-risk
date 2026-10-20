"""主题一 · 数据与因子概览。

回答四个问题：五个报价方案里选了哪一个、收益率怎么算出来的、哪些日子存疑、
风险因子拆成了哪几块。这一段没有「下一步」，读者从哪张卡开始看都行。

左上角试算 chip 里换报价方案或汇率口径，这一段的读数与卡片会跟着变；
换非 D2 方案只改对照卡，不会伪造后续读数。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, pipe, theme  # noqa: E402

page.init("数据与因子概览")
# 两栏版面计划：每张卡 H（半栏）/ W（通栏），按卡片出现顺序写。
# 落在通栏卡之前的那张半栏卡标成 W，自己撑满一整行——否则右边空半屏。
page.grid_plan("H", "H", "H", "H", "H", "H", "W", "W", "H")

page.section_head(
    "数据与因子概览",
    n_cards=9,
    subtitle="五个报价方案里选了哪一个、收益率怎么算出来的、哪些日子存疑、"
             "风险因子拆成了哪几块。",
)

_tok = theme.tokens()
sr = data.staleness_remedy_comparison()
tr_tbl = data.factor_table("factor_table_nav_tr")
nav_tbl = data.factor_table("factor_table_nav")
_n_rows = len(tr_tbl)
_n_doas = int(tr_tbl["doas_bp"].notna().sum())
_n_ex = int(tr_tbl["is_exdate"].sum())
_cmp = data.completeness_report()
_worst_file = _cmp.loc[_cmp["completeness_pct"].idxmin()]

caliber = pipe.pipe_caliber()
quote = pipe.pipe_quote()
_qrow = pipe.quote_row(quote)
_r2_raw = _qrow["利率剥离R²"]
_r2_txt = f"{float(_r2_raw):.3f}" if float(_r2_raw) == float(_r2_raw) else "未登记"

_cols = st.columns(4, gap="small")
for _col, (_label, _value, _sub) in zip(_cols, [
    ("最低数据完整率", f"{_worst_file['completeness_pct']:.2f} %",
     f"{_worst_file['file']}，只作对照、不参与计量。"),
    (f"{quote} 的零收益日占比", f"{_qrow['零收益占比%']:.1f} %",
     "陈旧报价（D1）六成以上日子没有变动；D2 官方 NAV 只有 6.1%。"),
    (f"{quote} 的利率剥离 R²", _r2_txt,
     "D2 为 0.668；δ 传导用的复权口径 R² = 0.757 是另一列。"),
    ("该方案报价样本天数", f"{int(_qrow['n'])} 天",
     "周度方案只有 265 个周度点；计量链用日度 D2 的 1273 天。"),
]):
    with _col:
        page.tile_kpi(_label, _value, sub=_sub)

if quote != pipe.QUOTE_CHOSEN:
    st.info(
        f"{pipe.quote_label(quote)} 只作对照、没有进入计量链"
        f"（零收益占比 {_qrow['零收益占比%']}%、利率剥离 R² {_r2_txt}）。"
        "主题二起的全部读数仍按定案 D2（官方 NAV 日度）计算，"
        "换报价方案不会改动后续结果。"
    )

# -------------------------------------------------- 1.1 五个报价方案对照
with page.card("1.1　五个报价方案：零收益占比与利率剥离 R²"):
    _sr = sr.copy()

    def _qstate(code: str) -> str:
        if code == quote:
            return "当前试算"
        if code == pipe.QUOTE_CHOSEN:
            return "定案 D2"
        return "备选"

    _sr["状态"] = [_qstate(str(s).split(" ", 1)[0]) for s in _sr["方案"]]
    _sr["代号"] = [str(s).split(" ", 1)[0] for s in _sr["方案"]]
    _sr_long = _sr.melt(id_vars=["代号", "状态"], value_vars=["零收益占比%", "利率剥离R²"],
                        var_name="metric", value_name="value").dropna(subset=["value"])
    _sr_long["指标"] = _sr_long["metric"].map({"零收益占比%": "零收益日占比（%）",
                                               "利率剥离R²": "利率剥离 R²"})
    # 两个指标拆成两张独立的图、上下排。原先是一张两格分面——分面 spec 的顶层
    # width 是「每格」宽度，得定死才不出框，而定死的宽度一旦碰上窄视口就压出卡片。
    # 拆成单图之后各自 use_container_width，宽度永远跟着卡片走。
    _Q_ORDER = list(_sr["代号"])
    _Q_DOM = ["当前试算", "定案 D2", "备选"]
    _Q_COL = [_tok["series"][0], _tok["status"]["good"], _tok["muted"]]

    def _qpanel(metric: str, *, legend: bool, height: int):
        part = _sr_long[_sr_long["指标"] == metric]
        bars = alt.Chart(part).mark_bar(size=16, cornerRadiusEnd=4).encode(
            y=alt.Y("代号:N", title="", sort=_Q_ORDER),
            x=alt.X("value:Q", title=metric, scale=alt.Scale(padding=6)),
            color=alt.Color(
                "状态:N", title="",
                scale=alt.Scale(domain=_Q_DOM, range=_Q_COL),
                legend=(alt.Legend(orient="top", direction="horizontal")
                        if legend else None)),
            tooltip=[alt.Tooltip("代号:N", title="方案"),
                     alt.Tooltip("value:Q", title=metric, format=".3f"),
                     alt.Tooltip("状态:N")],
        )
        lab = alt.Chart(part).mark_text(
            align="left", dx=6, fontSize=10, color=_tok["ink2"], font=_tok["font"],
        ).encode(y=alt.Y("代号:N", sort=_Q_ORDER), x=alt.X("value:Q"),
                 text=alt.Text("value:Q", format=".2f"))
        return (bars + lab).properties(height=height)

    st.altair_chart(_qpanel("零收益日占比（%）", legend=True, height=190),
                    use_container_width=True)
    st.altair_chart(_qpanel("利率剥离 R²", legend=False, height=170),
                    use_container_width=True)
    charts.data_drawer(
        _sr, key="dq_stale", title="1.1 四套报价口径对照",
        height=220,
        fmt={"n": "%.0f", "零收益占比%": "%.1f", "年化波动%": "%.2f", "1日σ%": "%.3f",
             "1日99%VaR(正态)%": "%.3f", "利率剥离R²": "%.3f", "等效久期(年)": "%.2f",
             "OAS-ρ": "%.3f", "OAS命中%": "%.1f"},
    )

# -------------------------------------------------- 1.2 三条收益口径
with page.card("1.2　三条收益口径：除权、复权、人民币"):
    _trd = tr_tbl.copy()
    for c in ["etf_ret_pct", "etf_ret_tr_pct", "etf_ret_rmb_pct"]:
        _trd["cum_" + c] = _trd[c].cumsum()
    _tlong = _trd.melt(id_vars=["date"],
                       value_vars=["cum_etf_ret_pct", "cum_etf_ret_tr_pct",
                                   "cum_etf_ret_rmb_pct"],
                       var_name="series", value_name="cum_pct")
    _tmap = {"cum_etf_ret_pct": "除权（不含分红）",
             "cum_etf_ret_tr_pct": "复权（主口径 USD）",
             "cum_etf_ret_rmb_pct": "人民币（次口径）"}
    _tlong["series"] = _tlong["series"].map(_tmap)
    _active = {"主": "复权（主口径 USD）", "次": "人民币（次口径）"}[caliber]
    _inactive = [s for s in _tmap.values() if s != _active]

    _grey = alt.Chart(_tlong[_tlong["series"].isin(_inactive)]).mark_line(
        strokeWidth=1.4, opacity=0.45).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y("cum_pct:Q", title="累计（%）", scale=alt.Scale(zero=True)),
        color=alt.Color("series:N", title="",
                        scale=alt.Scale(domain=_inactive,
                                        range=[_tok["muted"]] * len(_inactive))),
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("series:N", title="序列"),
                 alt.Tooltip("cum_pct:Q", title="累计 %", format=".2f")],
    )
    _bold = alt.Chart(_tlong[_tlong["series"] == _active]).mark_line(
        strokeWidth=charts.LINE_PX + 0.6, color=_tok["series"][0]).encode(
        x=alt.X("date:T"), y=alt.Y("cum_pct:Q"),
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("cum_pct:Q", title="累计 %", format=".2f")],
    )
    _exd = _trd[_trd["is_exdate"]]
    _exch = alt.Chart(_exd).mark_point(
        shape="triangle-down", size=90, filled=True,
        color=_tok["series"][3]).encode(
        x=alt.X("date:T"), y=alt.Y("cum_etf_ret_tr_pct:Q"),
        tooltip=[alt.Tooltip("date:T", title="除息日"),
                 alt.Tooltip("div_yield_pct:Q", title="分红率 %", format=".4f")])
    st.altair_chart(
        (_grey + _bold + _exch).add_params(charts.zoom_x()).properties(height=300),
        use_container_width=True)
    st.caption(
        f"蓝线＝当前试算口径（{_active}），灰线＝另外两条；倒三角是 {_n_ex} 个除息日。"
        "除权与复权之间的缺口是没计入的分红，复权与人民币之间的差是汇率。"
    )
    charts.data_drawer(
        _trd[["date", "etf_ret_pct", "etf_ret_tr_pct", "etf_ret_rmb_pct",
              "is_exdate", "div_yield_pct"]],
        key="df_tr", title="1.2 三条收益口径与除息日标记", height=280,
        labels={"date": "日期", "etf_ret_pct": "除权 %",
                "etf_ret_tr_pct": "复权主口径 %", "etf_ret_rmb_pct": "次口径 %",
                "is_exdate": "是否除息日", "div_yield_pct": "分红率 %"},
        fmt={"etf_ret_pct": "%.4f", "etf_ret_tr_pct": "%.4f",
             "etf_ret_rmb_pct": "%.4f", "div_yield_pct": "%.4f"},
        note=("逐日收益取原值（不累加），对应上面累计线的斜率。"
              f"全表共 {_n_ex} 个除息日，复权表按分红率还原。"),
    )
    charts.data_drawer(
        data.benchmark_comparison(), key="df_bench",
        title="1.2 基准对照：9141.HK 与 3141.HK 的登记值",
        height=260,
        note="美元柜台与港币柜台的同标的两个上市份额。3141.HK 全程只作旁证，"
             "不进入因子回归、回测或压力测试的任何一步。",
    )

# -------------------------------------------------- 1.3 累计收益分解
with page.card("1.3　NAV 除权口径的累计收益分解"):
    _decomp = nav_tbl.copy()
    for c in ["etf_ret_pct", "etf_rate_attrib_pct", "etf_spread_proxy_pct"]:
        _decomp["cum_" + c] = _decomp[c].cumsum()
    _long = _decomp.melt(
        id_vars=["date"],
        value_vars=["cum_etf_ret_pct", "cum_etf_rate_attrib_pct",
                    "cum_etf_spread_proxy_pct"],
        var_name="series", value_name="cum_pct")
    _n_map = {"cum_etf_ret_pct": "组合收益（合计）",
              "cum_etf_rate_attrib_pct": "利率贡献",
              "cum_etf_spread_proxy_pct": "利差代理贡献"}
    _long["series"] = _long["series"].map(_n_map)
    st.altair_chart(
        charts.line_crosshair(
            _long, y="cum_pct", color="series", height=280,
            color_domain=list(_n_map.values()), color_title="",
            y_title="累计（%）", zero=True, legend_select=True),
        use_container_width=True,
    )
    st.caption("利率贡献与利差代理贡献逐日相加精确等于组合收益，是记账恒等式，"
               "别把重合读成「信用维度解释了这么多」。")
    charts.data_drawer(
        _long, key="df_decomp", title="1.3 NAV 除权口径的累计收益分解",
        labels={"date": "日期", "series": "序列", "cum_pct": "累计（%）"},
        fmt={"cum_pct": "%.4f"},
        note="累计值由页面从逐日收益累加得到（cumsum），不是单独落库的列。",
    )

# -------------------------------------------------- 1.4 信用代理 vs 滞后 ΔOAS
with page.card("1.4　信用维度：利差代理收益与滞后 ΔOAS"):
    _proxy = ("etf_spread_proxy_pct" if "etf_spread_proxy_pct" in nav_tbl.columns
              else "etf_spread_proxy_tr_pct")
    _sp = nav_tbl[["date", _proxy, "doas_bp"]].dropna().copy()
    _sp["doas_lag1_bp"] = _sp["doas_bp"].shift(1)
    _sp = _sp.dropna()
    st.altair_chart(
        charts.scatter(
            _sp, x="doas_lag1_bp", y=_proxy, height=280, zoom=True,
            x_title="ΔOAS（滞后一日，bp）",
            y_title="利差代理收益（%）",
            tooltip=[alt.Tooltip("date:T", title="日期"),
                     alt.Tooltip("doas_lag1_bp:Q", title="ΔOAS(t−1) bp",
                                 format=".1f"),
                     alt.Tooltip(f"{_proxy}:Q", title="代理收益 %",
                                 format=".4f")]),
        use_container_width=True,
    )
    st.caption(f"外部 ΔOAS 缺失的 {_n_rows - _n_doas} 天靠代理量顶替信用维度，"
               "散点里看不到这些日子。")
    charts.data_drawer(
        _sp, key="df_sp",
        title="1.4 利差代理收益与滞后一日 ΔOAS 的逐日对照",
        labels={"date": "日期", "doas_bp": "ΔOAS bp",
                "doas_lag1_bp": "ΔOAS 滞后一日 bp", _proxy: "利差代理收益 %"},
        fmt={"doas_bp": "%.1f", "doas_lag1_bp": "%.1f", _proxy: "%.4f"},
        note="滞后一日列由页面用 shift(1) 生成；两列都非空的行才进入散点。",
    )

# -------------------------------------------------- 1.5 数据完整率
with page.card("1.5　八个来源文件的完整率与填补"):
    cmp_df = _cmp.sort_values("completeness_pct").reset_index(drop=True)
    cmp_df["状态"] = ["完整率低于 100%" if v < 100 else "无缺口"
                      for v in cmp_df["completeness_pct"]]
    _fo = list(cmp_df["file"])
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
    charts.data_drawer(
        cmp_df, key="dq_cmp", title="1.5 八个来源文件的缺口与填补",
        height=280,
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
        data.fill_log(), key="dq_fill", title="1.5 填补明细（逐段）",
        height=320,
        fmt={"length": "%.0f"},
        labels={"file": "文件", "column": "列", "start": "起", "end": "止",
                "length": "长度", "method": "方法"},
    )

# -------------------------------------------------- 1.6 事件库
with page.card("1.6　事件库：日期 × 市场维度"):
    ev = data.risk_events()
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
    ).add_params(charts.zoom_x())
    st.altair_chart(_ed.properties(height=250), use_container_width=True)
    charts.data_drawer(
        ev, key="dq_ev", title="1.6 事件库（按日期与市场维度）",
        height=300,
        labels={"date": "日期", "event_cn": "事件", "market": "市场维度",
                "expected_direction": "预期方向", "note": "备注", "source": "来源"},
    )

# -------------------------------------------------- 1.7 存疑日
with page.card("1.7　存疑日与异常判定"):
    od = data.outlier_judgment()
    _oc = alt.Chart(od).mark_point(
        size=60, filled=True, stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
    ).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y("series:N", title="", sort=sorted(od["series"].unique())),
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
    _oc = charts.legend_filter(_oc, "verdict").add_params(charts.zoom_x())
    st.altair_chart(_oc.properties(height=250), use_container_width=True)
    st.caption("颜色与形状同时给判定结果：蓝＝真实冲击，红＝存疑（无事件支撑）。"
               "点图例可以只看其中一类。")
    charts.data_drawer(
        od, key="dq_od", title="1.7 异常候选日的判定结果",
        height=320,
        fmt={"value": "%.4f", "gap_days": "%.0f"},
        labels={"date": "日期", "series": "序列", "value": "取值", "hit": "命中口径",
                "verdict": "判定", "handling": "处理", "event_date": "事件日期",
                "event": "对应事件", "gap_days": "距事件天数", "note": "说明"},
    )
    charts.data_drawer(
        data.doubtful_days(), key="dq_dd", title="1.7 存疑日清单",
        height=260,
        fmt={"value": "%.4f"},
        labels={"series": "序列", "date": "日期", "value": "取值", "hit": "命中口径",
                "verdict": "判定", "handling": "处理", "note": "说明",
                "special": "标记"},
    )

# -------------------------------------------------- 1.8 因子形态
with page.card("1.8　四个因子的分布与逐日序列", wide=True):
    _dist_cols = [c for c in ["d5y_bp", "d10y_bp", "fx_ret_pct", "doas_bp"]
                  if c in tr_tbl.columns and tr_tbl[c].notna().any()]
    inner2 = st.columns(len(_dist_cols))
    for i, col in enumerate(_dist_cols):
        _title = labels.series_title(col)
        if col == "doas_bp":
            _title += f"（仅 {_n_doas} 天）"
        with inner2[i]:
            st.altair_chart(
                charts.hist_with_normal(tr_tbl[col], bins=45, height=220, title=_title),
                use_container_width=True,
            )
    inner = st.columns(len(_dist_cols))
    for i, col in enumerate(_dist_cols):
        sub = tr_tbl[["date", col]].dropna()
        with inner[i]:
            st.altair_chart(
                charts.line_crosshair(
                    sub, y=col, height=200, y_title=labels.series_title(col),
                    zero=False,
                    tooltip=[alt.Tooltip("date:T", title="日期"),
                             alt.Tooltip(f"{col}:Q", title=labels.series_title(col),
                                         format=".4f")],
                ),
                use_container_width=True,
            )
    st.caption("上排是分布与同均值同方差正态的对照，下排是逐日序列；"
               "缺口保留空值，画作断口不补零。")
    charts.data_drawer(
        tr_tbl[["date"] + _dist_cols], key="df_series",
        title="1.8 四个因子的逐日序列",
        labels={"date": "日期", **{c: labels.series_title(c) for c in _dist_cols}},
        note="读 factors/factor_table_nav_tr.csv（复权口径）。",
        height=280,
    )
    charts.data_drawer(
        data.descriptive_stats_nav(), key="df_desc",
        title="1.8 七条序列的描述统计",
        fmt={"均值(日)": "%.6f", "标准差(日)": "%.6f", "年化σ(%)": "%.4f",
             "偏度": "%.4f", "峰度(超额)": "%.4f", "JB-p": "%.3e"},
        height=280,
        note=("描述统计含均值、日标准差、年化 σ、偏度、超额峰度、JB 检验 p 值、"
              "极值与零收益占比。年化 σ 只对收益类序列有定义，bp 类因子为空。"),
    )

# -------------------------------------------------- 1.9 因子相关阵
with page.card("1.9　因子相关阵"):
    _corr = data.factor_corr()
    st.altair_chart(
        charts.heatmap(_corr, x="factor_b", y="factor", v="corr", height=260,
                       x_title="", y_title="", v_title="相关系数"),
        use_container_width=True,
    )
    st.caption("利差代理与利率因子相关恰为 0，那是回归残差按定义正交，不是实证发现。")
    charts.data_drawer(
        _corr, key="df_corr", title="1.9 因子相关阵（3 × 3）",
        labels={"factor": "因子", "factor_b": "对照因子", "corr": "相关系数"},
        fmt={"corr": "%.3f"}, height=220,
        note="读 results/factor_corr.csv；对角线恒为 1。",
    )

page.footer(
    "clean_data/completeness_report.csv、fill_log.csv、outlier_judgment.csv；"
    "factors/staleness_remedy_comparison.csv、factor_table_nav.csv、"
    "factor_table_nav_tr.csv、descriptive_stats_nav.csv、benchmark_comparison.csv；"
    "results/factor_corr.csv、doubtful_days.csv；events/risk_events_timeline.csv"
)