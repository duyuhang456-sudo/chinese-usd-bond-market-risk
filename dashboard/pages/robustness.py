"""稳健性检验（阶段三·压力测试的一部分）。

判据贯穿全页：结论随某个自选参数大幅摆动，它就不能作为判断依据提交。
六项稳健性检验各自换掉一个自选参数，看读数会不会翻面；另附多期 √t 与最差窗两件，
注明不参与判定。

这一页由原「稳健性检验与数据质量」页的稳健性块独立而来，按阶段三维重组时划入阶段三。
横向展开情景的图各占满宽——半宽时 15 个情景挤在 300px 里，参照线标签会互相压住；
只有情景数少的图（窗口平移、信用路径）才并排。锚点窗长按方向固定口径（利率、信用用主，
汇率用次），不随控制带切换，写在该节副标题里。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("稳健性检验")

_si_all = data.stress_impact()
_vh = data.var_historical()
SPAN = f"{_vh['date'].min().date()} → {_vh['date'].max().date()}"
_tok = theme.tokens()

# 满宽卡片里分面图的定宽基准（按窄视口取，宽屏留白而不横向溢出）
FULL_W = 1000


def _derive(cal: str):
    """口径 → （该口径的情景表、判定线、交叉核对线）。"""
    s = _si_all[_si_all["caliber"] == cal]
    return s, labels.ES_1D_99[cal], float(s["es_hd_99_pct"].iloc[0])


caliber = st.session_state.get("rq_cal") or "主"
si, es_line, cross_ref = _derive(caliber)

page.header(
    "稳健性检验",
    "六项稳健性检验各自换掉一个自选参数，看读数会不会翻面；附录给多期读数但不参与判定。",
    crumb="阶段三 · 稳健性检验",
    caliber=labels.CALIBER_FULL[caliber],
    span=f"样本区间 {SPAN} · 主判定线 {es_line:.4f}% · 交叉核对线 {cross_ref:.4f}%",
)

# ---------------------------------------------------------------- 控制带
with st.container(border=True):
    _k1, _k2 = st.columns([2, 5])
    with _k1:
        caliber = st.segmented_control(
            "口径", ["主", "次"], default="主", key="rq_cal",
            format_func=lambda c: labels.CALIBER_SHORT[c],
            help="主口径 = 复权 USD，次口径 = 人民币。切换会改判定线与交叉核对线，"
                 "并改每一节按情景展开的读数；锚点窗长那一节不跟随，它按方向固定口径。",
        ) or "主"
    with _k2:
        st.caption(
            "两条线不是同一个统计量：前者是滚动 250 日的条件期望损失，后者是全样本分位。"
        )

si, es_line, cross_ref = _derive(caliber)

# ---------------------------------------------------------------- 读数带
_buf = data.stress_buffer_sens()
_buf["yes"] = _buf["tail_flag"].astype(str).str.strip() == labels.FLAG_YES
_buf_cnt = _buf.pivot_table(index="buffer_label", columns="caliber",
                            values="yes", aggfunc="sum").astype(int)
_buf_seq = list(_buf_cnt.sum(axis=1))
_n_a_all = int((_si_all["tail_flag"] == labels.FLAG_YES).sum())
_n_b_all = int((_si_all["tail_flag_hd"] == labels.FLAG_YES).sum())
_n_a = int((si["tail_flag"] == labels.FLAG_YES).sum())
_n_b = int((si["tail_flag_hd"] == labels.FLAG_YES).sum())
_n_tot = len(si)

page.kpi_row([
    {"label": "「超限 / 未超限」这条二值判定不稳健",
     "value": f"{labels.BUFFER_PP / es_line * 100:.1f} %",
     "help": f"缓冲 0.3365pp 占{labels.CALIBER_SHORT[caliber]}判定线 {es_line:.4f}% 的一半。"
             f"双口径合计的超限条数随缓冲档位从 {_buf_seq[0]} 条逐档升到 {_buf_seq[-1]} 条。"
             "口径：五档缓冲。"},
    {"label": "名单长度对阈值口径敏感，头部不敏感",
     "value": f"{_n_a} → {_n_b} 条",
     "help": f"{labels.CALIBER_SHORT[caliber]} {_n_tot} 条情景，换线移出 {_n_a - _n_b} 条，"
             f"移出的都贴着阈值。双口径合计是 {_n_a_all} → {_n_b_all} 条；"
             "X1 次口径 1.1382% 在两条线上都超限，未被移出。"},
    {"label": "1 日持有期下情景库的库级覆盖",
     "value": "成立",
     "help": "主口径 S10 2.5363% 高于实测最差单日 1.3449%；次口径 S09 3.1708% 高于 1.4739%。"
             "口径：双口径。"},
])


def _vrefs(chart: alt.Chart, refs: list) -> alt.Chart:
    """若干竖直参照线。**只画线，不在图上写字。**

    refs 是 (值, 标签, 颜色) 的列表；标签由调用方写进卡片下面那一句读法里。
    这一页的图是横向排情景的，三条参照线的数值只差几个百分点，文字挂在图上必然
    互相压住，也会压住条形本身——挂到图外的读法行反而每个值都读得到。

    线的颜色是这三条线唯一的图内标识，所以标签必须按同一顺序点出颜色。
    """
    for v, _lab, col in refs:
        d = pd.DataFrame({"x": [float(v)]})
        chart = chart + alt.Chart(d).mark_rule(
            color=col, strokeWidth=1.5, strokeDash=[4, 3]).encode(x=alt.X("x:Q"))
    return chart

# ------------------------------------------------------------ 1 覆盖度（满宽）
with page.card("库级覆盖度：库内最深的那条够不够深",
               "判据不是「每条情景都过历史最差」，而是「库内最深的一条够深」。",
               ):
    cov = data.stress_coverage()
    cov = cov[cov["caliber"] == caliber].copy()
    order = list(cov.sort_values("loss_pct", ascending=False)["scenario_id"])
    cov["状态"] = cov["below_worst"].astype(str).map(
        {labels.FLAG_YES: "低于实测最差", "否": "高于实测最差"}).fillna("—")
    worst = float(cov["hist_worst_pct"].iloc[0])

    _bars = alt.Chart(cov).mark_bar(size=12, cornerRadiusEnd=4).encode(
        y=alt.Y("scenario_id:N", title="", sort=order,
                axis=alt.Axis(labelOverlap=False, labelFontSize=10)),
        x=alt.X("loss_pct:Q", title=f"1 日基准损失（%，正值 = 损失）· "
                                    f"{labels.CALIBER_SHORT[caliber]}",
                scale=alt.Scale(domain=[0, float(cov["loss_pct"].max()) * 1.12])),
        color=alt.Color("状态:N", title="",
                        scale=alt.Scale(domain=["高于实测最差", "低于实测最差"],
                                        range=[_tok["status"]["good"],
                                               _tok["status"]["serious"]])),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("scenario_class:N", title="类别"),
                 alt.Tooltip("benchmark_source:N", title="基准口径"),
                 alt.Tooltip("loss_pct:Q", title="基准损失 %", format=".4f"),
                 alt.Tooltip("hist_percentile:Q", title="历史分位 %", format=".2f"),
                 alt.Tooltip("hist_worst_pct:Q", title="实测最差 %", format=".4f"),
                 alt.Tooltip("hist_worst_end:N", title="实测最差发生日"),
                 alt.Tooltip("exceeded_by_pct:Q", title="被超越幅度 %", format=".4f"),
                 alt.Tooltip("状态:N")],
    )
    st.altair_chart(
        _vrefs(_bars, [
            (es_line, f"判定线 {es_line:.4f}%", _tok["status"]["critical"]),
            (cross_ref, f"交叉核对线 {cross_ref:.4f}%", _tok["muted"]),
            (worst, f"实测最差单日 {worst:.4f}%", _tok["ink2"]),
        ]).properties(height=240),
        use_container_width=True,
    )

    deep = data.stress_shift_coverage()
    deep = deep[(deep["is_main_horizon"] == True) & (deep["caliber"] == caliber)]  # noqa: E712
    _cov_note = ""
    if not deep.empty:
        d = deep.iloc[0]
        _cov_note = (
            f"库内最深情景为 {d['deepest_scenario']} 号情景（1 日基准损失 "
            f"{d['deepest_loss_shifted']:.4f}%），对实测最差单日 "
            f"{d['hist_worst_pct']:.4f}%（{cov['hist_worst_end'].iloc[0]}）的超出幅度 "
            f"{abs(d['residual_gap_pct']):.4f}pp——库级覆盖的判据就是这一条。"
        )
    st.caption(
        f"橙 = 低于样本内实测最差单日（被超越），绿 = 高于；单条情景低于实测最差是正常的。"
        f"三条竖线自左至右——红线 = 1 日 99% 期望损失判定线 {es_line:.4f}%、"
        f"浅灰线 = 交叉核对线 {cross_ref:.4f}%、深灰线 = 实测最差单日 {worst:.4f}%。"
    )
    charts.data_drawer(
        cov, key="rq_cov", title="数据抽屉 · 覆盖度：损失、历史分位与实测最差对照",
        height=320,
        note="文字与颜色同时给出，不靠颜色单独表意。"
             "情景侧与实测最差单日两侧同为单日量，是直接可比的同类比较；"
             "多期读数在附录里，不参与判定。" + _cov_note,
        fmt={"loss_pct": "%.4f", "hist_percentile": "%.2f", "hist_n_obs": "%.0f",
             "hist_worst_pct": "%.4f", "exceeded_by_pct": "%.4f",
             "loss_cum_pct": "%.4f", "mdd_pct": "%.4f"},
        labels={"scenario_id": "编号", "scenario_name": "情景",
                "scenario_class": "类别", "benchmark_source": "基准口径",
                "loss_pct": "基准损失 %", "loss_cum_pct": "窗内累计 %",
                "mdd_pct": "窗内最大回撤 %", "hist_percentile": "历史分位 %",
                "hist_n_obs": "历史观测数", "hist_worst_pct": "实测最差 %",
                "hist_worst_end": "实测最差发生日", "exceeded_by_pct": "被超越幅度 %",
                "below_worst": "低于实测最差"},
    )

# ------------------------------------------------------------ 2 窗口平移 ｜ 3 信用路径
_r2 = st.columns(2, gap="medium")

with _r2[0]:
    with page.card("窗口起止平移：最差单日会随窗口移动改数",
                   "窗口起止同时平移 ±2 个交易日、窗长不变。", stretch=True):
        ws = data.stress_window_sens()
        ws = ws[ws["caliber"] == caliber].copy()
        longw = ws.melt(id_vars=["scenario_id", "shift_days"],
                        value_vars=["worst_day_pct", "mdd_pct"],
                        var_name="kind", value_name="loss")
        longw["kind"] = longw["kind"].map({"worst_day_pct": "窗内最差单日",
                                           "mdd_pct": "窗内最大回撤"})
        _lines = alt.Chart(longw).mark_line(strokeWidth=charts.LINE_PX).encode(
            x=alt.X("shift_days:Q", title="窗口平移（交易日）",
                    axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("loss:Q", title="损失（%）"),
            color=alt.Color("scenario_id:N", title="",
                            scale=alt.Scale(domain=list(dict.fromkeys(ws["scenario_id"])))),
            tooltip=[alt.Tooltip("scenario_id:N", title="情景"),
                     alt.Tooltip("shift_days:Q", title="平移"),
                     alt.Tooltip("loss:Q", title="损失 %", format=".4f")],
        )
        _dots = _lines.mark_point(size=charts.MARKER_PX, filled=True,
                                  stroke=_tok["surface"],
                                  strokeWidth=charts.SURFACE_RING)
        _fac = (_lines + _dots).properties(
            height=112, width=charts.facet_width(1)).facet(
            row=alt.Row("kind:N", title="", sort=["窗内最差单日", "窗内最大回撤"])
        ).resolve_scale(y="independent")
        st.altair_chart(_fac, use_container_width=False)

        base_wd = ws[ws["shift_days"] == 0].set_index("scenario_id")["worst_day_pct"]
        tab = ws.groupby("scenario_id").agg(
            期限=("n_days", "first"),
            最差单日区间下沿=("worst_day_pct", "min"),
            最差单日区间上沿=("worst_day_pct", "max"),
            回撤区间下沿=("mdd_pct", "min"),
            回撤区间上沿=("mdd_pct", "max"),
        )
        tab.insert(1, "最差单日基准", base_wd)
        tab["最差单日极差"] = tab["最差单日区间上沿"] - tab["最差单日区间下沿"]
        tab["回撤极差"] = tab["回撤区间上沿"] - tab["回撤区间下沿"]
        tab = tab.round(4)
        tab.index.name = "情景"
        fixed = tab.index[tab["最差单日极差"] == 0].tolist()
        nz = tab.loc[tab["最差单日极差"] > 0, "最差单日极差"]
        _win_note = (
            f"{labels.CALIBER_SHORT[caliber]}下 {len(tab)} 个历史与补充情景里，"
            f"有 {len(fixed)} 个（{'、'.join(fixed)}）的最差单日读数在全部五个平移位置上完全不动，"
            "因为最差的那一天始终落在窗内。"
            + (f"其余情景的极差 {nz.min():.4f} ~ {nz.max():.4f}pp，"
               f"最大者 {tab['最差单日极差'].idxmax()}——平移到 ±1 / ±2 后最差日被移出窗口。"
               if not nz.empty else "")
        )
        st.caption("最差单日极差都不超过各自的回撤极差：1 日口径比窗内最大回撤更稳健，"
                   "但仍应给区间而不是单点值。")
        charts.data_drawer(
            tab.reset_index(), key="rq_win",
            title="数据抽屉 · 窗口平移：最差单日与回撤的五档区间",
            height=300,
            note="报告按「情景 × 口径」十个组合统计，其中 3 个（H1 主、H1 次、H2 主）"
                 "在全部五个平移位置上完全不动。" + _win_note,
            fmt={"期限": "%.0f", "最差单日基准": "%.4f", "最差单日区间下沿": "%.4f",
                 "最差单日区间上沿": "%.4f", "最差单日极差": "%.4f",
                 "回撤区间下沿": "%.4f", "回撤区间上沿": "%.4f", "回撤极差": "%.4f"},
        )

with _r2[1]:
    with page.card("信用路径：给的是区间不是点",
                   "只画有信用分量的情景，纯汇率与纯利率情景没有这一项。", stretch=True):
        cr = si[si["credit_range_lo_pct"].notna()].copy()
        if cr.empty:
            st.info("当前口径下没有带信用路径区间的情景。")
        else:
            _order = list(cr.sort_values("credit_range_hi_pct", ascending=False)["scenario_id"])
            _y = alt.Y("scenario_id:N", title="", sort=_order)
            _rule = alt.Chart(cr).mark_rule(strokeWidth=6, color=_tok["muted"],
                                            opacity=0.40).encode(
                y=_y,
                x=alt.X("credit_range_lo_pct:Q", title="损失（%，正值 = 损失）"),
                x2=alt.X2("credit_range_hi_pct:Q"),
                tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                         alt.Tooltip("scenario_name:N", title="情景"),
                         alt.Tooltip("credit_range_lo_pct:Q", title="区间下端 %",
                                     format=".4f"),
                         alt.Tooltip("credit_range_hi_pct:Q", title="区间上端 %",
                                     format=".4f")],
            )
            _dm = alt.Chart(cr).mark_point(
                size=110, filled=True, color=_tok["series"][0],
                stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
                y=_y, x=alt.X("worst_day_modeled_pct:Q"),
                tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                         alt.Tooltip("worst_day_modeled_pct:Q", title="δ 点估计 %",
                                     format=".4f")])
            _db = alt.Chart(cr).mark_point(
                size=110, filled=True, color=_tok["series"][1],
                stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
                y=_y, x=alt.X("loss_after_buffer_pct:Q"),
                tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                         alt.Tooltip("loss_after_buffer_pct:Q", title="点估计 + 缓冲 %",
                                     format=".4f")])
            st.altair_chart(
                _vrefs(_rule + _dm + _db,
                       [(es_line, f"判定线 {es_line:.4f}%", _tok["status"]["critical"])]
                       ).properties(height=215),
                use_container_width=True,
            )
            _n_lo = int((cr["credit_range_lo_pct"] > es_line).sum())
            _n_hi = int((cr["credit_range_hi_pct"] > es_line).sum())
            st.caption(
                f"灰条 = 信用路径区间，蓝点 = δ 点估计，橙点 = 点估计加缓冲，"
                f"红虚线 = 判定线 {es_line:.4f}%。"
                f"{len(cr)} 条里下端越过判定线的 {_n_lo} 条、上端 {_n_hi} 条。"
            )
            charts.data_drawer(
                cr, key="rq_credit",
                title="数据抽屉 · 信用路径：区间两端、点估计与缓冲",
                height=280,
                note="区间**下端**是平静样本的弱传导端（OLS 95% 置信区间的弱端，再加缓冲），"
                     "**上端**是弱端再加 2022 年未解释残差极值 1.1147pp（1 日窗）——"
                     "该加项本身是对偏差的修正，故上端不再叠缓冲。"
                     "上端越线不能读成「信用维度已经超过判定线」——它主要由那个加项抬上去，"
                     "而且信用维度作为参考项的定性没有改变：δ 估计自 2023-09 之后的平静子样本，"
                     "不反映外推到危机的主要风险。报告主张两个口径并列披露、不取其一。",
                fmt={"worst_day_modeled_pct": "%.4f", "loss_after_buffer_pct": "%.4f",
                     "credit_range_lo_pct": "%.4f", "credit_range_hi_pct": "%.4f",
                     "es_1d_99_pct": "%.4f"},
                labels={"scenario_id": "编号", "scenario_name": "情景",
                        "worst_day_modeled_pct": "δ 点估计 %",
                        "loss_after_buffer_pct": "点估计 + 缓冲 %",
                        "credit_range_lo_pct": "区间下端 %",
                        "credit_range_hi_pct": "区间上端 %",
                        "es_1d_99_pct": "判定线 %", "credit_range_note": "区间说明"},
            )

# ------------------------------------------------------------ 4 缓冲五档（满宽）
with page.card("缓冲五档：名单长度随档位逐档变长，二值判定不稳健",
               "把偏差缓冲从 0 调到 +1pp，看超限名单长多少。缓冲按定义要覆盖亏损侧偏差，"
               "档位越高越保守。"):
    bs = data.stress_buffer_sens()
    bs = bs[bs["caliber"] == caliber].copy()
    _hm = bs.copy()
    _hm["情景"] = _hm["scenario_id"] + " " + _hm["scenario_name"]
    _hm_order = [f"{s} {n}" for s, n in
                 bs[["scenario_id", "scenario_name"]].drop_duplicates()
                 .set_index("scenario_id").sort_index().itertuples()]
    # 热图的格数一多，Vega 的两条默认行为都会碍事：五条档位标签的宽度加起来超过
    # 轴长时，它把整条 x 轴转 90°；15 行挤在一起时，它按「看不看得下」丢行标，只剩
    # 4 行有名字。这里不靠标签本身够短去躲，而是把这两条行为直接钉死（见下面两个
    # alt.Axis）。x 轴保持水平、y 轴 15 行全出。
    _hm["档位"] = _hm["buffer_label"]
    _hm["情景"] = pd.Categorical(_hm["情景"], categories=_hm_order, ordered=True)
    _hm = _hm.sort_values(["情景", "buffer_label"])
    st.altair_chart(
        charts.heatmap(
            _hm, x="档位", y="情景", v="loss_net_pct", height=285,
            x_title="", y_title="", v_title="净损失 %",
            x_axis=alt.Axis(labelAngle=0, labelLimit=180),
            y_axis=alt.Axis(labelOverlap=False, labelLimit=215),
        ),
        use_container_width=True,
    )
    _seq_now = " / ".join(str(v) for v in _buf_cnt[caliber].tolist())
    _seq_all = " / ".join(str(v) for v in _buf_seq)
    st.caption(
        f"单色阶：深浅只表示净损失大小。本口径五档的超限条数为 **{_seq_now}**"
        f"（双口径合计 {_seq_all}）。"
    )
    charts.data_drawer(
        bs, key="rq_buf", title="数据抽屉 · 缓冲五档：各情景净损失与尾部标记",
        height=340,
        note="历史情景与纯汇率情景的行整行不变色"
             "（前者按定义不加缓冲，后者主口径恒为 0、次口径不加缓冲）。"
             "结论是**不稳健**：名单长度随档位逐档变长（不降），从完全不缓冲到 +1pp 压力档，"
             "双口径合计从 17 条涨到 23 条。原因不在缓冲取值有误，而在 1 日口径下"
             f"缓冲已占{labels.CALIBER_SHORT[caliber]}判定线的 "
             f"{labels.BUFFER_PP / es_line * 100:.1f}%（0.3365 / {es_line:.4f}）——"
             "「超限 / 未超限」这个二值判定比多日口径脆弱得多。"
             "这正是改用「几倍 99% 期望损失」这一连续读数作主导统计量的理由。",
        fmt={"buffer_value": "%.4f", "horizon_days": "%.0f",
             "loss_net_pct": "%.4f", "es_1d_99_pct": "%.4f"},
        labels={"buffer_label": "缓冲档位", "buffer_value": "缓冲取值 pp",
                "scenario_id": "编号", "scenario_name": "情景",
                "horizon_days": "期限（交易日）", "loss_net_pct": "净损失 %",
                "es_1d_99_pct": "判定线 %", "tail_flag": "是否超限"},
    )

# ------------------------------------------------------------ 5 锚点窗长（满宽）
with page.card("锚点窗长：绝对水平随窗长摆，档间比例不动",
               "本节三个方向各自钉死口径（利率、信用用主口径，汇率用次口径），"
               "不随上面的口径切换。"):
    anc = data.stress_anchor_sens().copy()
    anc["锚点档"] = anc["is_main_horizon"].map({True: "登记档（1 日）", False: "备查档"})
    long_a = anc.melt(id_vars=["direction", "window_days", "is_main_horizon", "caliber",
                               "anchor_date", "shock_sum"],
                      value_vars=["mild_1x_loss_pct", "extreme_2x_loss_pct"],
                      var_name="kind", value_name="loss")
    long_a["kind"] = long_a["kind"].map({"mild_1x_loss_pct": "轻度档 1×",
                                         "extreme_2x_loss_pct": "极端档 2×"})
    _la = alt.Chart(long_a).mark_line(strokeWidth=charts.LINE_PX).encode(
        x=alt.X("window_days:Q", title="锚点窗长（交易日）", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("loss:Q", title="损失（%）"),
        color=alt.Color("kind:N", title="", sort=["轻度档 1×", "极端档 2×"]),
        tooltip=[alt.Tooltip("direction:N", title="方向"),
                 alt.Tooltip("caliber:N", title="口径"),
                 alt.Tooltip("window_days:Q", title="窗长"),
                 alt.Tooltip("kind:N", title="档位"),
                 alt.Tooltip("loss:Q", title="损失 %", format=".4f"),
                 alt.Tooltip("shock_sum:Q", title="锚点冲击合计", format=".2f"),
                 alt.Tooltip("anchor_date:T", title="锚点日")],
    )
    _pa = _la.mark_point(size=charts.MARKER_PX, filled=True,
                         stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING)
    # 登记档参照线取同一份数据、用 transform_filter 切出来——分面要求各图层共用一份数据
    _ma = alt.Chart(long_a).mark_rule(
        color=_tok["status"]["critical"], strokeWidth=1.5, opacity=0.5,
    ).transform_filter(alt.datum.is_main_horizon).encode(x=alt.X("window_days:Q"))
    st.altair_chart(
        (_la + _pa + _ma).properties(
            height=130, width=charts.facet_width(3, card_w=FULL_W)).facet(
            column=alt.Column("direction:N", title="")
        ).resolve_scale(y="independent"),
        use_container_width=False,
    )
    _rng = anc.groupby(["direction", "caliber"])["extreme_2x_loss_pct"].agg(["min", "max"])
    _rng["极差"] = (_rng["max"] - _rng["min"]).round(4)
    st.caption("红线标出登记档（1 日锚点）。三档之间 1 : 1.5 : 2 的倍率是需求文档直接规定的，"
               "线性施加在锚点上，所以变的是绝对水平、不是档间比例。")
    charts.data_drawer(
        anc, key="rq_anchor",
        title="数据抽屉 · 锚点窗长：三方向 × 三档的损失与锚点冲击",
        height=300,
        note="极端档（2×）在三类方向上的极差："
             + "、".join(f"{d}（{c}口径）{v:.4f}pp" for (d, c), v in _rng["极差"].items())
             + "。汇率方向摆动最大，所以假设情景的损失只能作量级参考，不能当点估计精确引用。"
             "登记档的登记值与表里的 `registered_extreme_pct` 逐值一致。",
        fmt={"window_days": "%.0f", "shock_sum": "%.2f", "mild_1x_loss_pct": "%.4f",
             "extreme_2x_loss_pct": "%.4f", "registered_mild_pct": "%.4f",
             "registered_extreme_pct": "%.4f"},
        labels={"direction": "方向", "caliber": "口径", "window_days": "窗长",
                "anchor_date": "锚点日", "shock_sum": "锚点冲击合计",
                "mild_1x_loss_pct": "轻度 1× %", "extreme_2x_loss_pct": "极端 2× %",
                "registered_mild_pct": "登记轻度 %",
                "registered_extreme_pct": "登记极端 %", "锚点档": "档位"},
    )

# ------------------------------------------------------------ 6 阈值口径（满宽）
with page.card("阈值口径核对：名单长度变，头部不变",
               "换一条判据线，超限名单会不会变。两条线的定义见页头上方的控制带。"):
    si2 = si.copy()
    si2["挪动"] = si2["tail_flag"] != si2["tail_flag_hd"]
    _order2 = list(si2.sort_values("loss_after_buffer_pct", ascending=False)["scenario_id"])
    _y2 = alt.Y("scenario_id:N", title="", sort=_order2,
                axis=alt.Axis(labelOverlap=False, labelFontSize=10))
    _dd = alt.Chart(si2).mark_point(
        size=110, filled=True, color=_tok["series"][0],
        stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
        y=_y2,
        x=alt.X("loss_after_buffer_pct:Q", title="含缓冲净损失（%，正值 = 损失）"),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("loss_after_buffer_pct:Q", title="净损失 %", format=".4f"),
                 alt.Tooltip("es_1d_99_pct:Q", title="主判定线 %", format=".4f"),
                 alt.Tooltip("es_hd_99_pct:Q", title="交叉核对线 %", format=".4f"),
                 alt.Tooltip("tail_flag:N", title="主判定线超限"),
                 alt.Tooltip("tail_flag_hd:N", title="交叉核对线超限")],
    )
    _mv = alt.Chart(si2[si2["挪动"]]).mark_text(
        align="left", dx=8, fontSize=10, color=_tok["status"]["serious"],
        font=_tok["font"]).encode(
        y=_y2, x=alt.X("loss_after_buffer_pct:Q"), text=alt.Text("scenario_id:N"))
    st.altair_chart(
        _vrefs(_dd + _mv, [
            (es_line, f"主判定线 {es_line:.4f}%", _tok["status"]["critical"]),
            (cross_ref, f"交叉核对线 {cross_ref:.4f}%", _tok["muted"]),
        ]).properties(height=240),
        use_container_width=True,
    )
    _mv_ids = "、".join(si2.loc[si2["挪动"], "scenario_id"].tolist()) or "无"
    st.caption(
        f"红虚线 = 主判定线 {es_line:.4f}%，浅灰虚线 = 交叉核对线 {cross_ref:.4f}%。"
        f"{labels.CALIBER_SHORT[caliber]}：主判定线超限 **{_n_a} / {_n_tot} 条**，"
        f"交叉核对线超限 **{_n_b} / {_n_tot} 条**，换线移出 {_n_a - _n_b} 条；"
        f"两条线相差 **{(cross_ref / es_line - 1) * 100:.1f}%**。"
    )
    charts.data_drawer(
        si2, key="rq_thresh",
        title="数据抽屉 · 阈值口径：两条判定线下的净损失与超限标记",
        height=320,
        note=f"橙字编号标出换线后被移出名单的情景（本口径为 {_mv_ids}）——都贴着阈值，"
             "换线后落回线内。**X1 次口径并未被移出**：它 1.1382% 高于交叉核对线 1.0646%，"
             "在两条线上都超限。报告按两个口径合计给出 19 条 → 15 条，本节只按当前口径计。"
             "名单长度对阈值口径敏感，名单头部不敏感"
             "（汇率情景 S07–S09 次口径与 S10 在两条线下都超限）。"
             "主判定线仅由 8（主）/ 7（次）个违规日支撑，样本极小，读数本身带有很大不确定性。",
        fmt={"loss_after_buffer_pct": "%.4f", "es_1d_99_pct": "%.4f",
             "es_hd_99_pct": "%.4f", "buffer_pct": "%.4f", "residual_pct": "%.4f",
             "x_es99": "%.3f"},
        labels={"scenario_id": "编号", "scenario_name": "情景",
                "scenario_class": "类别", "loss_after_buffer_pct": "净损失 %",
                "buffer_pct": "缓冲 pp", "residual_pct": "残差偏差 pp",
                "x_es99": "倍数（÷ 99% 期望损失）",
                "es_1d_99_pct": "主判定线 %", "tail_flag": "主判定线超限",
                "es_hd_99_pct": "交叉核对线 %", "tail_flag_hd": "交叉核对线超限",
                "挪动": "换线后改变", "worst_day_date": "最差单日"},
    )

# ------------------------------------------------------------ 7 附录（两件并排）
_r4 = st.columns(2, gap="medium")

with _r4[0]:
    with page.card("附录：√t 缩放与多期最差窗",
                   "多期读数不参与判定。√t 未做独立回测验证，只能读作「若强行用 √t "
                   "会得到什么量级」。", stretch=True):
        sh = data.stress_shift_coverage()
        _sl = alt.Chart(sh).mark_line(strokeWidth=charts.LINE_PX).encode(
            x=alt.X("horizon_days:Q", title="持有期（交易日）", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("sqrt_t_es99_ref_pct:Q", title="√t × 1 日 99% 期望损失（%）"),
            color=alt.Color("caliber:N", title="", scale=alt.Scale(domain=["主", "次"])),
            tooltip=[alt.Tooltip("caliber:N", title="口径"),
                     alt.Tooltip("horizon_days:Q", title="期限"),
                     alt.Tooltip("sqrt_t_var95_ref_pct:Q", title="√t 95% VaR %",
                                 format=".4f"),
                     alt.Tooltip("sqrt_t_var99_ref_pct:Q", title="√t 99% VaR %",
                                 format=".4f"),
                     alt.Tooltip("sqrt_t_es99_ref_pct:Q", title="√t 99% ES %",
                                 format=".4f"),
                     alt.Tooltip("hist_worst_pct:Q", title="同期限实测最差 %",
                                 format=".4f"),
                     alt.Tooltip("is_main_horizon:N", title="是否登记档")],
        )
        _sp = _sl.mark_point(size=charts.MARKER_PX, filled=True,
                             stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING)
        st.altair_chart((_sl + _sp).properties(height=205), use_container_width=True)
        _n_below = int((sh["sqrt_t_es99_ref_pct"] < sh["hist_worst_pct"]).sum())
        st.caption(
            f"√t 参考值在全部 {len(sh)} 个（期限 × 口径）组合上都低于同期限实测最差"
            f"（{_n_below} / {len(sh)}）：波动聚集的直接后果，不是随机误差。"
        )
        charts.data_drawer(
            sh, key="rq_sqrt",
            title="数据抽屉 · √t 参考值：各期限的缩放对照与实测最差",
            height=260,
            note="本节的 1 日以外的实测最差取自滚动窗内最差单日，聚集段里的连续下跌被 √t 拉平了。"
                 "本节与上面各节的 1 日基准**不同量纲**，不参与任何判定。",
            fmt={"horizon_days": "%.0f", "sqrt_t_var95_ref_pct": "%.4f",
                 "sqrt_t_var99_ref_pct": "%.4f", "sqrt_t_es99_ref_pct": "%.4f",
                 "hist_worst_pct": "%.4f", "residual_gap_pct": "%.4f",
                 "deepest_loss_shifted": "%.4f"},
            labels={"horizon_days": "期限", "caliber": "口径", "is_main_horizon": "登记档",
                    "sqrt_t_var95_ref_pct": "√t 95% VaR %",
                    "sqrt_t_var99_ref_pct": "√t 99% VaR %",
                    "sqrt_t_es99_ref_pct": "√t 99% ES %",
                    "hist_worst_pct": "同期限实测最差 %",
                    "residual_gap_pct": "覆盖缺口 pp",
                    "deepest_scenario": "库内最深情景",
                    "deepest_loss_shifted": "最深情景损失 %", "ref_note": "说明"},
        )

with _r4[1]:
    with page.card("附录：各期限前十最差窗",
                   "同样是多期读数、不参与判定；1 日档那一列就是覆盖度里实测最差单日的出处。",
                   stretch=True):
        ww = data.stress_worst_windows()
        _hz = sorted(ww["horizon_days"].unique())
        _wb = alt.Chart(ww).mark_bar(size=9, cornerRadiusEnd=4).encode(
            y=alt.Y("rank:Q", title="排名（1 = 最差）", axis=alt.Axis(tickMinStep=1),
                    scale=alt.Scale(reverse=True)),
            x=alt.X("loss_pct:Q", title="窗内损失（%，正值 = 损失）"),
            color=alt.Color("caliber:N", title="", scale=alt.Scale(domain=["主", "次"])),
            tooltip=[alt.Tooltip("caliber:N", title="口径"),
                     alt.Tooltip("horizon_days:Q", title="期限"),
                     alt.Tooltip("rank:Q", title="排名"),
                     alt.Tooltip("window_start:T", title="窗起"),
                     alt.Tooltip("window_end:T", title="窗止"),
                     alt.Tooltip("loss_pct:Q", title="损失 %", format=".4f")],
        )
        # 四个持有期：并排四格在半个卡片里每格只剩 70 多 px，折成两行两列。
        st.altair_chart(
            _wb.properties(height=170, width=charts.facet_width(2)).facet(
                alt.Facet("horizon_days:N", title="持有期（交易日）",
                          sort=[str(h) for h in _hz]),
                columns=2,
            ).resolve_scale(y="independent"),
            use_container_width=False,
        )
        st.caption(
            "各期限前十最差窗，两口径并排。1 日档（左起第一列）就是覆盖度那条实测最差单日的出处"
            "——主 2022-06-13 的 1.3449%、次 2022-09-29 的 1.4739%。"
        )
        charts.data_drawer(
            ww, key="rq_ww", title="数据抽屉 · 最差窗明细：各期限前十窗口",
            height=300,
            fmt={"horizon_days": "%.0f", "rank": "%.0f", "loss_pct": "%.4f"},
            labels={"horizon_days": "期限", "is_main_horizon": "登记档", "caliber": "口径",
                    "rank": "排名", "window_start": "窗起", "window_end": "窗止",
                    "loss_pct": "损失 %", "caliber_note": "说明"},
        )

page.footer(
    "results/stress_coverage.csv、stress_shift_coverage.csv、stress_window_sens.csv、"
    "stress_buffer_sens.csv、stress_impact.csv、stress_anchor_sens.csv、"
    "stress_worst_windows.csv；docs/tool_usage.md",
    "loss_pct / hist_worst_pct / worst_day_pct / mdd_pct / shift_days / n_days / "
    "loss_net_pct / buffer_value / tail_flag / credit_range_lo_pct / hi_pct / "
    "extreme_2x_loss_pct / is_main_horizon / shock_sum / x_es99",
)
