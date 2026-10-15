"""VaR 模型对比与回测。

看板的第三页，回答「哪个模型该用、它准不准」。迁移自静态图：var_series_hs250、
var_hs_vs_parametric、phase2_rollvol_regimes、var_backtest（失败率）、
model_scorecard / baseline_decision、baseline_model_tradeoff、delta_transmission。

本页有三个最容易读错的地方，页内都单独写了提示：
  1. `conf` 列的 95%/99% 是 **Kupiec 检验水平**，不是 VaR 置信度；
  2. 失败率点图的横杆是 Kupiec 接受区间，不是置信区间意义上的误差棒；
  3. 平均 VaR 的横向对比只在**同一个基准范围**内成立，换范围换样本就不可比。

页级筛选（口径、检验水平、模型、回测范围、区间）统一收在页首的控制带；正文是几组图卡，
每张图卡下方一只数据抽屉。所有数字都是引用，不在本页重算。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("VaR 模型对比与回测")

_tok = theme.tokens()
ORDER10 = labels.MODEL_ORDER + [m for m in labels.NON_EVIDENCE_MODELS
                                if m not in labels.MODEL_ORDER]


def _runs(reg: pd.DataFrame, flag: str) -> pd.DataFrame:
    """布尔列 → 连续区间表（start/end），供背景色带用。

    色带按整段画，而不是逐日画矩形——逐日画会在相邻矩形之间露出底色细缝。
    """
    out = []
    start, prev = None, None
    for d, f in zip(reg["date"], reg[flag].fillna(False).astype(bool)):
        if f and start is None:
            start = d
        elif not f and start is not None:
            out.append({"start": start, "end": prev})
            start = None
        prev = d
    if start is not None:
        out.append({"start": start, "end": prev})
    return pd.DataFrame(out, columns=["start", "end"])


# ---------------------------------------------------------------- 侧栏筛选
_v = data.var_historical()
_lo, _hi = _v["date"].min().date(), _v["date"].max().date()
_bt_all = data.backtest_results()
_scopes = list(dict.fromkeys(
    labels.CORE_SCOPES + [s for s in _bt_all["scope"].unique()
                          if s not in labels.CORE_SCOPES]))

def _parse_range(v, hi):
    """date_input 在选取过程中可能只回半个区间，退回到端点值而不是报错。"""
    if isinstance(v, tuple):
        return (v[0], v[1]) if len(v) == 2 else (v[0], hi)
    return (v, hi)


# 页头在控制带之前渲染，跨度与口径得先知道当前选的是哪一组，从会话态里取上一轮的值
caliber = st.session_state.get("vb_caliber") or "主"
conf = st.session_state.get("vb_conf") or "95"
start, end = _parse_range(st.session_state.get("vb_range", (_lo, _hi)), _hi)

# ------------------------------------------------------------ 页头与读数
_pick = _bt_all[(_bt_all["caliber"] == caliber)
                & (_bt_all["conf"].astype(int) == int(conf))
                & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_mean_m1 = float(_pick.set_index("model")["mean_var"].get("M1", float("nan")))
_mean_hs = float(_pick.set_index("model")["mean_var"].get("HS250", float("nan")))
_gap_now = (_mean_m1 / _mean_hs - 1) * 100
_rate_hs = float(_pick.set_index("model")["rate_pct"].get("HS250", float("nan")))
_rej = _bt_all[_bt_all["lr_uc"] > 3.8415]
_rej95 = int((_rej["conf"].astype(int) == 95).sum())
_rej99 = int((_rej["conf"].astype(int) == 99).sum())

page.header(
    "VaR 模型对比与回测",
    "两个方法逐日样本外比，再用 Kupiec 与 Christoffersen 回头验，最后单独验 δ 传导链。",
    crumb="阶段二 · 对比与回测",
    span=f"{start} 至 {end}",
    caliber=labels.CALIBER_FULL[caliber],
)

# ------------------------------------------------------------ 控制带（页内驱动）
# 口径与检验水平统辖全页；模型多选只作用于逐日 −VaR 那一节；区间只作用于前两节；
# 回测范围只作用于失败率与平均 VaR 两节。
with st.container(border=True):
    _k1, _k2, _k3 = st.columns([2, 3, 2])
    with _k1:
        caliber = st.segmented_control(
            "口径", ["主", "次"], default="主", key="vb_caliber",
            format_func=lambda c: labels.CALIBER_SHORT[c],
            help="统辖全页。两个口径的差别是计价基准的差别，不是精度的差别。",
        ) or "主"
    with _k2:
        conf = st.segmented_control(
            "检验水平", ["95", "99"], default="95", key="vb_conf",
            format_func=lambda c: f"{c}%（名义 {labels.NOMINAL_RATE[c]:g}%）",
            help="统辖全页。这里选的是 Kupiec 检验水平，不是 VaR 置信度。",
        ) or "95"
    with _k3:
        scope = st.selectbox("回测范围", _scopes, index=_scopes.index("full"),
                             format_func=lambda s: labels.SCOPE_LABEL.get(s, s),
                             key="vb_scope",
                             help="只作用于失败率与平均 VaR 两节。换范围必须换标签。")
    _k4, _k5 = st.columns([3, 2])
    with _k4:
        picked = st.multiselect(
            "模型（上限 5 条）", ORDER10, default=labels.DEFAULT_MODELS,
            format_func=lambda m: labels.MODEL_LABEL.get(m, m), key="vb_models",
            help="只作用于逐日 −VaR 那一节；取色按固定次序，筛选后幸存序列不换色。")
    with _k5:
        _rng_in = st.date_input("区间", value=(_lo, _hi), min_value=_lo,
                                max_value=_hi, key="vb_range",
                                help="只作用于前两节。")
    if len(picked) > labels.MODEL_PICK_CAP:
        st.warning(
            f"一次最多画 {labels.MODEL_PICK_CAP} 条线（配色槽位与可读性所限）；"
            f"已按固定次序保留 {labels.MODEL_PICK_CAP} 条。"
        )
        picked = [m for m in ORDER10 if m in picked][:labels.MODEL_PICK_CAP]

# 固定次序取色：筛选后幸存序列不换色
picked = [m for m in ORDER10 if m in picked]
start, end = _parse_range(_rng_in, _hi)

page.kpi_row([
    {"label": "日度基准 HS250 在本档几乎无偏差", "value": f"{_rate_hs:.2f} %",
     "help": f"历史模拟法 250 日窗、{labels.CALIBER_FULL[caliber]}、检验水平 {conf}%、"
             f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}上的真实违规率，"
             f"名义是 {labels.NOMINAL_RATE[conf]:g}%。阶段二定案：95% / 99% 两档实测 "
             "4.92% / 1.03%。"},
    {"label": "Kupiec 拒绝全部是过度覆盖、没有一处低估风险", "value": f"{len(_rej)} 处",
     "help": f"全表 {len(_bt_all)} 处检验里共 {len(_rej)} 处被拒绝，"
             f"95% 档 {_rej95} 处、99% 档 {_rej99} 处。方向全部是实际违规次数低于期望，"
             "等于 VaR 算得偏高、资本占用过多。"},
    {"label": f"参数法平均 VaR 相对 HS250（{labels.CALIBER_SHORT[caliber]}）",
     "value": f"{_gap_now:+.1f} %",
     "help": f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}、检验水平 {conf}%。"
             "正值表示参数法更保守、占用更多资本，负值表示它反而算得更低，"
             "所以「参数法系统性高估风险」不能一般化。"},
    {"label": "δ 偏差的八成八来自信用残差而非参数本身", "value": "88.1 %",
     "help": "偏差构成里 |23.30| 来自信用残差，参数差异只有 |3.13|；"
             "δ 本身估得稳（放大倍数 1.00），问题出在信用维度。"},
])


# ------------------------------------------------ 1 波动状态与回测区间
_reg = data.regimes()
# 阈值必须按生成规则复算：它是**未填头**的 60 日滚动年化波动的全样本 2/3 分位
# （code/prep_phase2.py §3a）。序列前 59 天窗口未满、用 expanding σ 顶替过，
# 把这段算进去阈值会偏低约 0.03pp，高波动段就少几天。
_rv_th = float(_reg["rv60_ann_%"].iloc[59:].quantile(2 / 3))
_n_high = int((_reg["regime"] == "高波动").sum())
_n_crisis = int(_reg["crisis_window"].sum())

with page.card("波动状态与回测区间",
               "先给全画布定一个状态标签，平稳与高波动是后面所有分段回测的分组依据。"):
    _vol = charts.line_crosshair(
        _reg, y="rv60_ann_%", height=232, zero=True,
        y_title="60 日滚动年化波动（%）",
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("rv60_ann_%:Q", title="年化 σ %", format=".4f"),
                 alt.Tooltip("regime:N", title="状态")],
    )
    _vol = charts.shade_bands(_vol, _runs(_reg, "chronic_high"),
                              color=_tok["muted"], opacity=0.10)
    _vol = charts.shade_bands(_vol, _runs(_reg, "crisis_window"), opacity=0.18)
    _vol = charts.add_rule(_vol, _rv_th, label=f"慢性高波动阈值 {_rv_th:.2f}%")
    st.altair_chart(_vol, use_container_width=True)
    st.caption(
        "浅灰底=慢性高波动段，橙底=急性危机窗；曲线读的是 NAV 除权口径。"
        "该标志含当日滚动量，不是事前信号，危机窗样本外命中 0 天。"
    )
    charts.data_drawer(
        _reg, key="vb_regime", title="数据抽屉 · 逐日滚动年化波动与状态标签",
        labels={"date": "日期", "rv60_ann_%": "60 日滚动年化 σ %",
                "chronic_high": "慢性高波动", "crisis_window": "急性危机窗",
                "regime": "状态"},
        fmt={"rv60_ann_%": "%.4f"}, height=280,
        note=f"读 results/regimes.csv；阈值 {_rv_th:.4f}% 由页面按生成规则复算、"
             "与 CSV 的状态列一致，它是全样本三分之二分位，取到它就算慢性高波动；"
             f"急性危机窗 {_n_crisis} 天（2022-06-06 至 06-22）由 |日收益| ≥ 1.2% "
             "自动触发，没有人工挑日子，两段合起来是高波动段，"
             f"共 {_n_high} 天，占样本 {_n_high / len(_reg) * 100:.0f}%。",
    )

# ------------------------------------ 2 逐日 −VaR 与违规日
with page.card("历史模拟法与参数法：逐日 −VaR 与违规日",
               "同一根时间轴上叠多个模型，看谁在最坏的那些天把线放得够低。"):
    if not picked:
        st.info("先在左侧栏里选择至少一个模型。")
    else:
        long = data.var_series_long(caliber, conf, tuple(picked))
        long = data.slice_dates(long, start, end)
        long["neg_var"] = -pd.to_numeric(long["var_pct"], errors="coerce")

        order = [m for m in ORDER10 if m in picked]
        fig = charts.line_crosshair(
            long, y="neg_var", color="model", height=232,
            color_domain=order, color_title="",
            y_title="日收益 / −VaR（%）", zero=True,
            tooltip=[alt.Tooltip("date:T", title="日期"),
                     alt.Tooltip("model:N", title="模型"),
                     alt.Tooltip("neg_var:Q", title="−VaR %", format=".4f"),
                     alt.Tooltip("ret_pct:Q", title="日收益 %（正值=上涨）",
                                 format=".4f")],
        )

        # 实际收益线：中性墨色，不进图例（图例留给模型），靠题注点名
        ret = long.drop_duplicates("date")[["date", "ret_pct"]].dropna()
        if not ret.empty:
            fig = fig + alt.Chart(ret).mark_line(
                strokeWidth=charts.LINE_PX, color=_tok["ink"], opacity=0.55
            ).encode(x=alt.X("date:T"), y=alt.Y("ret_pct:Q"))

        # 波动段背景：高波动段宽而浅垫底，危机窗窄而深压在上层
        reg = data.slice_dates(_reg, start, end)
        if not reg.empty:
            fig = charts.shade_bands(
                fig, _runs(reg, "chronic_high"), color=_tok["muted"], opacity=0.10)
            fig = charts.shade_bands(fig, _runs(reg, "crisis_window"), opacity=0.16)

        # 次口径 GARCH 的边界解段：σ_t 退化成常数，这一段 M1g 的线没有时变含义。
        # 只在读者确实画了 M1g 且口径为次时才标，免得主口径也被误读成同样退化。
        _bnd = None
        if caliber == "次" and "M1g" in picked:
            _b0, _b1 = [pd.Timestamp(x.strip())
                        for x in labels.GARCH_BOUNDARY_SPAN.split("~")]
            _bnd = pd.DataFrame({"start": [_b0], "end": [_b1]})
            fig = charts.shade_bands(fig, _bnd, color=_tok["status"]["serious"],
                                     opacity=0.16)

        # 违规日：每个已选模型各一个点，同日多模型违规时点会重叠
        be = data.backtest_exceptions()
        be = be[(be["caliber"] == caliber) & (be["conf"].astype(int) == int(conf))
                & (be["model"].isin(picked))]
        be = data.slice_dates(be, start, end)
        if not be.empty:
            fig = fig + alt.Chart(be).mark_point(
                size=charts.MARKER_PX, filled=True, color=_tok["status"]["critical"],
                stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
            ).encode(
                x=alt.X("date:T"), y=alt.Y("ret:Q"),
                tooltip=[alt.Tooltip("date:T", title="违规日"),
                         alt.Tooltip("model:N", title="模型"),
                         alt.Tooltip("ret:Q", title="日收益 %", format=".4f"),
                         alt.Tooltip("var:Q", title="VaR %", format=".4f"),
                         alt.Tooltip("exceed_ratio:Q", title="超出倍数", format=".2f"),
                         alt.Tooltip("is_doubtful:N", title="存疑日"),
                         alt.Tooltip("near_ghost:N", title="近空窗"),
                         alt.Tooltip("near_event:N", title="近事件"),
                         alt.Tooltip("co_3141_pct:Q", title="3141 同期 %",
                                     format=".4f")],
            )

        st.altair_chart(fig, use_container_width=True)
        st.caption(
            "黑线=日收益（正值=上涨），彩色线=各模型的 −VaR（越靠下越保守），"
            "红点=违规日；同日多模型违规时点重叠，悬停可看具体模型。"
        )
        if _bnd is not None:
            st.caption(
                f"另有一段橙红底：次口径 GARCH 的 α 落到 0 的区间"
                f"（{labels.GARCH_BOUNDARY_SPAN}），那段 σ_t 退化为常数，"
                "M1g 的线在那里没有时变含义。主口径没有出现这个退化。"
            )
        if be.empty:
            st.info("该口径 / 水平 / 区间下，所选模型没有违规日。")
        _exc_note = (
            f"区间内违规合计 {len(be)} 条（跨 {be['date'].nunique()} 个交易日）。"
            if not be.empty
            else "该口径 / 水平 / 区间下，所选模型没有违规日。"
        )
        charts.data_drawer(
            long[["date", "model", "var_pct", "ret_pct", "neg_var"]],
            key="vb_varseries", title="数据抽屉 · 逐日 VaR 与实际收益（长表）",
            height=280,
            labels={"date": "日期", "model": "模型", "var_pct": "VaR %（正值）",
                    "ret_pct": "日收益 %", "neg_var": "−VaR %"},
            fmt={"var_pct": "%.4f", "ret_pct": "%.4f", "neg_var": "%.4f"},
            note="读 results/var_historical.csv 与 results/var_parametric.csv 合并后的长表；"
                 "图上线是同一天同一模型一行，违规点另见 backtest_exceptions.csv。"
                 + _exc_note,
        )


# ------------------------------------------------ 3 失败率与接受区间
sub = _bt_all[(_bt_all["caliber"] == caliber)
              & (_bt_all["conf"].astype(int) == int(conf))
              & (_bt_all["scope"] == scope)].copy()
order = [m for m in ORDER10 if m in set(sub["model"])]
sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
sub = sub.sort_values("model")
sub["lo_pct"] = sub["x_lo"] / sub["T"] * 100
sub["hi_pct"] = sub["x_hi"] / sub["T"] * 100
# 轴标签要短：半幅卡片里放得下 10 个类别，全角括号会把相邻标签挤在一起
sub["model_lab"] = sub["model"].astype(str).map(
    lambda m: f"{m} 对照" if m in labels.NON_EVIDENCE_MODELS else m)
nominal = labels.NOMINAL_RATE[conf]

with page.card("失败率与 Kupiec 接受区间",
               f"{labels.SCOPE_LABEL.get(scope, scope)} · {conf}% 检验水平。"
               "点=真实违规率，灰杆=接受区间，虚线=名义水平。"):
    st.warning(labels.KUPIEC_NOTE)
    dot = charts.point_interval(
        sub, cat="model_lab", val="rate_pct", lo="lo_pct", hi="hi_pct",
        nominal=nominal, height=232,
        y_title="真实违规率（%）",
        tooltip=[alt.Tooltip("model_lab:N", title="模型"),
                 alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
                 alt.Tooltip("lo_pct:Q", title="接受区间下沿 %", format=".4f"),
                 alt.Tooltip("hi_pct:Q", title="接受区间上沿 %", format=".4f"),
                 alt.Tooltip("x:Q", title="实际违规次数"),
                 alt.Tooltip("T:Q", title="样本天数"),
                 alt.Tooltip("exp_x_eff:Q", title="期望次数", format=".1f"),
                 alt.Tooltip("p_uc:Q", title="Kupiec p", format=".4f"),
                 alt.Tooltip("p_ind_mc:Q", title="独立性 p（MC）", format=".4f"),
                 alt.Tooltip("role:N", title="角色")],
    )
    st.altair_chart(dot, use_container_width=True)
    st.caption(
        "标「对照」的两个模型是恒等式与分布分位对照，不参与评选。"
    )
    charts.data_drawer(
        sub[["model", "role", "T", "x", "rate_pct", "exp_x_eff", "lo_pct", "hi_pct",
             "p_uc", "p_ind_mc", "es", "es_ratio"]],
        key="vb_rate",
        title=f"数据抽屉 · {labels.SCOPE_LABEL.get(scope, scope)}的失败率明细",
        height=280,
        labels={"model": "模型", "role": "角色", "T": "样本天数", "x": "违规次数",
                "rate_pct": "真实违规率 %", "exp_x_eff": "期望次数",
                "lo_pct": "区间下沿 %", "hi_pct": "区间上沿 %",
                "p_uc": "Kupiec p", "p_ind_mc": "独立性 p", "es": "ES %",
                "es_ratio": "ES/VaR"},
        fmt={"rate_pct": "%.4f", "exp_x_eff": "%.1f", "lo_pct": "%.4f",
             "hi_pct": "%.4f", "p_uc": "%.4f", "p_ind_mc": "%.4f", "es": "%.4f",
             "es_ratio": "%.4f"},
        note="读 results/backtest_results.csv；接受区间由 x_lo / x_hi 除以 T 得到。"
             f"全表 {len(_bt_all)} 行里被 Kupiec 拒绝的有 {len(_rej)} 处，"
             "方向全部是过度覆盖，实际违规次数低于期望，等于 VaR 算得偏高、资本占用过多。"
             "没有一处是把风险看小了，这与「新兴市场工具在尾部容易低估」的先验恰好相反；"
             f"分档看：95% 档 {_rej95} 处、99% 档 {_rej99} 处。",
    )

# ------------------------------------------------ 4 平均 VaR 对照
_b = _bt_all[(_bt_all["caliber"] == caliber)
             & (_bt_all["conf"].astype(int) == int(conf))
             & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_b["T"] = _b["T"].astype(int)
_t_ref = int(_b["T"].max())
_b_in = _b[_b["T"] == _t_ref].copy()
_b_out = _b[_b["T"] != _t_ref]
_b_in["model_lab"] = _b_in["model"].map(lambda m: labels.MODEL_LABEL.get(m, m))
_b_in = _b_in.sort_values("mean_var")
_mv_m1 = float(_b_in.set_index("model")["mean_var"].get("M1", float("nan")))
_mv_hs = float(_b_in.set_index("model")["mean_var"].get("HS250", float("nan")))

_sort = list(_b_in["model_lab"])
_bars = alt.Chart(_b_in).mark_bar(size=18, cornerRadiusEnd=4).encode(
    y=alt.Y("model_lab:N", title="", sort=_sort),
    x=alt.X("mean_var:Q", title="平均 VaR（%，正值 = 损失）",
            scale=alt.Scale(domain=[0, float(_b_in["mean_var"].max()) * 1.55])),
    color=alt.value(_tok["series"][0]),
    tooltip=[alt.Tooltip("model:N", title="模型"),
             alt.Tooltip("mean_var:Q", title="平均 VaR %", format=".4f"),
             alt.Tooltip("es:Q", title="ES %", format=".4f"),
             alt.Tooltip("es_ratio:Q", title="ES/VaR", format=".4f"),
             alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
             alt.Tooltip("T:Q", title="样本天数")],
)
_txt = alt.Chart(_b_in).mark_text(
    align="left", dx=6, fontSize=10, color=_tok["ink2"], font=_tok["font"],
).encode(y=alt.Y("model_lab:N", sort=_sort),
         x=alt.X("mean_var:Q"), text=alt.Text("mean_var:Q", format=".4f"))
_mv_chart = _bars + _txt
# 结论直接标在 HS250 那一行上：标注用文字色，不穿序列色。
if _mv_hs == _mv_hs and _mv_m1 == _mv_m1:
    _ann = alt.Chart(pd.DataFrame({
        "model_lab": [labels.MODEL_LABEL.get("HS250", "HS250")],
        "mean_var": [_mv_hs],
        "t": [f"低于 M1 的 {_mv_m1:.4f}%，历史模拟法并没有更保守"],
    })).mark_text(
        align="left", dx=62, fontSize=10.5, fontWeight="bold",
        color=_tok["ink2"], font=_tok["font"],
    ).encode(y=alt.Y("model_lab:N", sort=_sort),
             x=alt.X("mean_var:Q",
                     scale=alt.Scale(domain=[0, float(_b_in["mean_var"].max()) * 1.55])),
             text="t:N")
    _mv_chart = _mv_chart + _ann

with page.card("参数法 vs 历史模拟法：平均 VaR 对照",
               "准不准之外还要看贵不贵。平均 VaR 就是资本占用，同一范围下才可比。"):
    st.altair_chart(
        _mv_chart.properties(height=max(220, 24 * len(_b_in) + 46)),
        use_container_width=True)
    st.caption(
        f"同一条横轴上共 {_t_ref} 天样本，模型之间可比；"
        "「参数法更贵」不是普遍规律，换档换口径会变号。"
    )
    _out_note = ""
    if not _b_out.empty:
        _out_note = (
            "未画进来的："
            + "、".join(f"{r['model']}（只有 {int(r['T'])} 天样本）"
                        for _, r in _b_out.iterrows())
            + "，它的窗宽更长，公共样本自然更短，放在同一条轴上比平均值会误导。"
        )
    charts.data_drawer(
        _b[["model", "role", "T", "mean_var", "es", "es_ratio", "rate_pct",
            "x", "exp_x_eff", "p_uc"]],
        key="vb_meanvar",
        title=f"数据抽屉 · {labels.SCOPE_LABEL[labels.BASE_SCOPE]}的平均 VaR、ES 与资本指数",
        height=280,
        labels={"model": "模型", "role": "角色", "T": "样本天数",
                "mean_var": "平均 VaR %", "es": "ES %", "es_ratio": "ES/VaR",
                "rate_pct": "真实违规率 %", "x": "违规次数",
                "exp_x_eff": "期望次数", "p_uc": "Kupiec p"},
        fmt={"mean_var": "%.4f", "es": "%.4f", "es_ratio": "%.4f",
             "rate_pct": "%.4f", "exp_x_eff": "%.1f", "p_uc": "%.4f"},
        note="含上面未画进来的短样本模型，它们的 T 列小于其余行；"
             "资本指数与校准偏差见下一张评分卡。" + _out_note,
    )


# ------------------------------------------------ 5 评分卡与定案
ms = data.model_scorecard()
ms = ms[(ms["caliber"] == caliber) & (ms["conf"].astype(int) == int(conf))]
ms = ms.set_index("model").reindex([m for m in ORDER10 if m in set(ms["model"])])
ms = ms.reset_index()
n_rec = int((ms["verdict"] == "推荐").sum())

bd = data.baseline_decision()
bd = bd[bd["caliber"] == caliber]
bd = bd.set_index("model").reindex([m for m in ORDER10 if m in set(bd["model"])])
bd = bd.reset_index()

with page.card("模型选择：评分卡、定案与取舍",
               "把准不准与贵不贵合成一次评选，两道闸都过才进推荐。"):
    st.caption(
        f"本档推荐模型 {n_rec} / {len(ms)} 个。资本指数以 M1 为 1，小于 1 表示更省资本；"
        "过 Kupiec 的范围数按 χ²(1) 5% 临界值 3.8415 判定。"
    )
    st.dataframe(
        ms[["model", "verdict", "rate_base_pct", "nominal_pct", "rate_over_nominal",
            "cal_dev", "cap_idx_vs_M1", "mean_var_base_pct", "es_base_pct",
            "n_pass_uc_core", "n_core_scopes", "stab_scope_pp", "gap_regime_pp",
            "n_ind_suspect"]],
        width="stretch", height=205, hide_index=True,
        column_config={
            "model": st.column_config.Column("模型"),
            "verdict": st.column_config.Column("结论"),
            "rate_base_pct": st.column_config.NumberColumn("基准范围违规率 %",
                                                           format="%.4f"),
            "nominal_pct": st.column_config.NumberColumn("名义 %", format="%.4f"),
            "rate_over_nominal": st.column_config.NumberColumn("违规率/名义",
                                                               format="%.4f"),
            "cal_dev": st.column_config.NumberColumn("校准偏差", format="%.4f"),
            "cap_idx_vs_M1": st.column_config.NumberColumn("资本指数（M1=1）",
                                                           format="%.4f"),
            "mean_var_base_pct": st.column_config.NumberColumn("平均 VaR %",
                                                               format="%.4f"),
            "es_base_pct": st.column_config.NumberColumn("ES %", format="%.4f"),
            "n_pass_uc_core": st.column_config.Column("过 Kupiec 的核心范围数"),
            "n_core_scopes": st.column_config.Column("核心范围总数"),
            "stab_scope_pp": st.column_config.NumberColumn("跨范围摆幅 pp",
                                                           format="%.4f"),
            "gap_regime_pp": st.column_config.NumberColumn("分段缺口 pp",
                                                           format="%.4f"),
            "n_ind_suspect": st.column_config.Column("独立性存疑次数"),
        },
    )
    st.dataframe(
        bd[["model", "tier", "reason", "g1_pass_rate", "g2_prudent", "cal_dev_95",
            "cal_dev_99", "cap_idx_95", "cap_idx_99", "rate_95", "rate_99",
            "n_ind_max"]],
        width="stretch", height=180, hide_index=True,
        column_config={
            "model": st.column_config.Column("模型"),
            "tier": st.column_config.Column("结论"),
            "reason": st.column_config.Column("理由"),
            "g1_pass_rate": st.column_config.Column("过失败率闸"),
            "g2_prudent": st.column_config.Column("过审慎闸"),
            "cal_dev_95": st.column_config.NumberColumn("校准偏差 95%", format="%.4f"),
            "cal_dev_99": st.column_config.NumberColumn("校准偏差 99%", format="%.4f"),
            "cap_idx_95": st.column_config.NumberColumn("资本指数 95%", format="%.4f"),
            "cap_idx_99": st.column_config.NumberColumn("资本指数 99%", format="%.4f"),
            "rate_95": st.column_config.NumberColumn("违规率 95%", format="%.4f"),
            "rate_99": st.column_config.NumberColumn("违规率 99%", format="%.4f"),
            "n_ind_max": st.column_config.Column("独立性存疑（最大）"),
        },
    )

with page.card("失败率随回测范围的变化",
               "同一模型换回测范围后的真实违规率。框架基准是公共样本 773 日。"):
    st.caption(
        f"框架基准范围是 {labels.SCOPE_LABEL[labels.BASE_SCOPE]}；「剔除存疑日」两列"
        "是把存疑交易日整段剔除后的复算，用来看结论是否因此翻转。"
    )
    piv = _bt_all[(_bt_all["caliber"] == caliber)
                  & (_bt_all["conf"].astype(int) == int(conf))]
    piv = piv.pivot_table(index="model", columns="scope", values="rate_pct")
    piv = piv.reindex([m for m in ORDER10 if m in piv.index])
    piv.columns = [labels.SCOPE_LABEL.get(c, c) for c in piv.columns]
    st.dataframe(piv.round(3), width="stretch", height=225)
    st.caption("数值是真实违规率（%），按模型 × 回测范围展开；"
               "转置后的同名列见上一张失败率点图的数据抽屉。")

# ------------------------------------------------ 6 取舍坐标（满宽）
ms2 = data.model_scorecard()
ms2 = ms2[(ms2["caliber"] == caliber) & (ms2["conf"].astype(int) == int(conf))].copy()
ms2 = ms2[ms2["model"].isin(ORDER10)]
nom = float(ms2["nominal_pct"].iloc[0]) if not ms2.empty else 0.0

with page.card("模型选择：取舍坐标",
               "横轴资本占用（以 M1 为 1），纵轴基准范围违规率，越靠左上越划算。"):
    sc = charts.scatter(
        ms2, x="cap_idx_vs_M1", y="rate_base_pct", color="verdict",
        color_domain=["推荐", "淘汰"], color_title="",
        x_title="资本指数（M1 = 1，越小越省资本）",
        y_title="基准范围违规率（%）", height=270,
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("verdict:N", title="结论"),
                 alt.Tooltip("cap_idx_vs_M1:Q", title="资本指数", format=".4f"),
                 alt.Tooltip("rate_base_pct:Q", title="违规率 %", format=".4f"),
                 alt.Tooltip("cal_dev:Q", title="校准偏差", format=".4f"),
                 alt.Tooltip("es_base_pct:Q", title="ES %", format=".4f")],
    )
    _txt2 = alt.Chart(ms2).mark_text(
        align="left", dx=8, dy=-6, fontSize=10, color=_tok["ink2"],
        font=_tok["font"],
    ).encode(x=alt.X("cap_idx_vs_M1:Q"), y=alt.Y("rate_base_pct:Q"),
             text=alt.Text("model:N"))
    sc = sc + _txt2
    sc = charts.add_rule(sc, nom, label=f"名义 {nom:g}%", dashed=True)
    sc = charts.add_rule(sc, nom * labels.PRUDENCE,
                         label=f"审慎带（名义 ×{labels.PRUDENCE:g}）",
                         color=_tok["muted"], dashed=True)
    st.altair_chart(sc, use_container_width=True)
    st.caption(
        "两条虚线分别是名义水平与审慎带（名义 ×1.10）；点已直接标注模型名，"
        "颜色只作辅助、不单独表意。"
    )
    charts.data_drawer(
        ms2[["model", "verdict", "cap_idx_vs_M1", "rate_base_pct", "cal_dev",
             "mean_var_base_pct", "es_base_pct", "stab_scope_pp"]],
        key="vb_tradeoff", title="数据抽屉 · 资本占用与违规率的取舍坐标", height=260,
        labels={"model": "模型", "verdict": "结论", "cap_idx_vs_M1": "资本指数",
                "rate_base_pct": "违规率 %", "cal_dev": "校准偏差",
                "mean_var_base_pct": "平均 VaR %", "es_base_pct": "ES %",
                "stab_scope_pp": "跨范围摆幅 pp"},
        fmt={"cap_idx_vs_M1": "%.4f", "rate_base_pct": "%.4f", "cal_dev": "%.4f",
             "mean_var_base_pct": "%.4f", "es_base_pct": "%.4f",
             "stab_scope_pp": "%.4f"},
        note="读 results/model_scorecard.csv；verdict 列即结论。",
    )

# ------------------------------------------------ 7 δ 传导：四个读数（满宽）
de_all = data.delta_transmission_events()
ds_all = data.delta_transmission_summary()
# 这三个下拉原本是页内第二层筛选器，现在钉死成报告的主设定，页内不再另设筛选。
dmodel = sorted(de_all["model"].unique())[0]
dsrc = "B 事件前δ"
dwin = "3日(主)"

de = de_all[(de_all["model"] == dmodel) & (de_all["caliber"] == caliber)
            & (de_all["delta_src"] == dsrc) & (de_all["window"] == dwin)].copy()
de["understate_cn"] = de["understate"].map({True: "是（低估损失）", False: "否"})
# 原始的 market 列是「credit+rate+fx」这样的组合标签，类别太多不能直接上色；
# 散点的着色上限是 3 个序列，所以只保留报告真正要区分的那一维：有没有信用成分
de["credit_cn"] = de["market"].astype(str).map(
    lambda m: "含信用维度" if "credit" in m else "不含信用维度")
ds = ds_all[(ds_all["model"] == dmodel) & (ds_all["caliber"] == caliber)
            & (ds_all["delta_src"] == dsrc) & (ds_all["window"] == dwin)]

if not ds.empty:
    r = ds.iloc[0]
    # 满宽一排四张：挤在半幅卡片里时 st.metric 会把长读数截成「+0....」。
    with page.card("δ 传导与偏差构成：四个读数",
                   f"固定设定：{dmodel} · {labels.CALIBER_FULL[caliber]} · {dsrc} · "
                   f"{dwin} 窗，共 {len(de)} 个事件；这是报告采用的设定。"):
        page.kpi_row([
            {"label": "偏差均值（实际 − 预测）", "value": f"{r['dev_mean']:+.4f} %",
             "help": f"为负即 δ 系统性低估损失。{dwin} 窗、{dsrc}、"
                     f"{labels.CALIBER_FULL[caliber]}。"},
            {"label": "偏差中位数", "value": f"{r['dev_median']:+.4f} %",
             "help": f"5% / 95% 分位 {r['dev_p05']:+.3f} / {r['dev_p95']:+.3f}，"
                     "看的是分布形状。"},
            {"label": "低估事件占比", "value": f"{r['under_share'] * 100:.1f} %",
             "help": "全部事件口径；亏损侧单独统计时更高。"},
            {"label": "信用类事件占比", "value": f"{r['credit_share'] * 100:.1f} %",
             "help": f"绝对偏差均值 {r['absdev_mean']:.4f}%；"
                     "偏差的 88.1% 来自信用残差。"},
        ], per_row=4)


# ------------------------------------------------ 8 δ 传导：两张散点
with page.card("δ 传导与偏差构成：预测 vs 实际",
               "两张散点分别看 δ 的预测误差有多大、偏差与利率冲击有没有关系。"):
    st.markdown("**预测 vs 实际**（45° 线为完全命中）")
    lo = float(min(de["r_pred_pct"].min(), de["r_actual_pct"].min()))
    hi = float(max(de["r_pred_pct"].max(), de["r_actual_pct"].max()))
    s = charts.scatter(
        de, x="r_pred_pct", y="r_actual_pct", color="credit_cn",
        color_domain=["含信用维度", "不含信用维度"], color_title="", height=240,
        x_title="δ 预测收益（%）", y_title="实际收益（%）",
        tooltip=[alt.Tooltip("event_date:T", title="事件日"),
                 alt.Tooltip("event_cn:N", title="事件"),
                 alt.Tooltip("market:N", title="冲击类别"),
                 alt.Tooltip("r_pred_pct:Q", title="预测 %", format=".4f"),
                 alt.Tooltip("r_actual_pct:Q", title="实际 %", format=".4f"),
                 alt.Tooltip("dev_pct:Q", title="偏差 %", format=".4f"),
                 alt.Tooltip("understate_cn:N", title="低估损失")],
    )
    st.altair_chart(charts.diagonal(s, lo, hi), use_container_width=True)
    st.markdown("**偏差 vs 利率冲击幅度**")
    s2 = charts.scatter(
        de, x="rate_shock_bp", y="dev_pct", color="credit_cn",
        color_domain=["含信用维度", "不含信用维度"], color_title="", height=240,
        x_title="事件窗利率冲击（bp）", y_title="偏差 %（负值=低估损失）",
        tooltip=[alt.Tooltip("event_date:T", title="事件日"),
                 alt.Tooltip("event_cn:N", title="事件"),
                 alt.Tooltip("market:N", title="冲击类别"),
                 alt.Tooltip("rate_shock_bp:Q", title="冲击 bp", format=".1f"),
                 alt.Tooltip("dev_pct:Q", title="偏差 %", format=".4f")],
    )
    st.altair_chart(charts.add_rule(s2, 0.0, label="无偏差", dashed=True),
                    use_container_width=True)
    st.caption(
        "点落在 45° 线下方即实际比预测更差，也就是 δ 低估了这一步的损失。"
    )
    charts.data_drawer(
        de[["event_date", "event_cn", "market", "r_actual_pct", "r_pred_pct",
            "dev_pct", "understate_cn", "rate_shock_bp", "d5y_sum_bp",
            "d10y_sum_bp", "r2_delta_src"]],
        key="vb_delta", title="数据抽屉 · δ 传导的逐事件预测与偏差", height=280,
        labels={"event_date": "事件日", "event_cn": "事件", "market": "冲击类别",
                "r_actual_pct": "实际 %", "r_pred_pct": "预测 %", "dev_pct": "偏差 %",
                "understate_cn": "低估损失", "rate_shock_bp": "利率冲击 bp",
                "d5y_sum_bp": "Δ5Y 合计 bp", "d10y_sum_bp": "Δ10Y 合计 bp",
                "r2_delta_src": "δ 来源 R²"},
        fmt={"r_actual_pct": "%.4f", "r_pred_pct": "%.4f", "dev_pct": "%.4f",
             "rate_shock_bp": "%.1f", "d5y_sum_bp": "%.1f", "d10y_sum_bp": "%.1f",
             "r2_delta_src": "%.4f"},
        note="读 results/delta_transmission_events.csv，已按上面钉死的设定筛过。"
             "偏差与利率冲击幅度的相关系数只有 +0.151，偏差不随冲击变大而系统性恶化。"
             "着色只区分事件是否含信用成分；`market` 原始列是组合标签，类别过多不做多色。",
    )

# ------------------------------------------------ 9 δ 传导：偏差分布与因子暴露
with page.card("δ 传导与偏差构成：偏差分布与因子暴露",
               "偏差的构成是这条链上最该记住的一件事，其次是 δ 的因子暴露。"):
    st.caption(
        "偏差的 88.1% 来自信用残差项（|23.30| 对参数差异的 |3.13|），"
        "δ 本身估得稳，放大倍数 1.00。"
    )
    st.altair_chart(
        charts.hist_with_normal(de["dev_pct"], bins=30, height=240,
                                title=f"偏差（%）· n={len(de)}"),
        use_container_width=True,
    )
    st.caption("δ 的因子暴露（主口径每 1 单位因子的收益敏感度）；"
               "汇率一行是口径定义而非估计，主口径恒为 0、人民币次口径为 +1。")
    st.dataframe(data.factor_exposure(), width="stretch", height=160,
                 hide_index=True)
    charts.data_drawer(
        de[["event_cn", "market", "dev_pct", "dev_param_pct", "dev_credit_pct",
            "rate_shock_bp", "r2_delta_src"]],
        key="vb_delta_split", title="数据抽屉 · δ 偏差的参数项与信用项分解",
        height=260,
        labels={"event_cn": "事件", "market": "冲击类别", "dev_pct": "总偏差 %",
                "dev_param_pct": "参数差异项 %", "dev_credit_pct": "信用残差项 %",
                "rate_shock_bp": "利率冲击 bp", "r2_delta_src": "δ 来源 R²"},
        fmt={"dev_pct": "%.4f", "dev_param_pct": "%.4f", "dev_credit_pct": "%.4f",
             "rate_shock_bp": "%.1f", "r2_delta_src": "%.4f"},
        note="分布图与上面两张散点读的是同一批事件；这里把总偏差拆成参数项与信用项。",
    )

page.caliber_note(
    f"当前 {labels.CALIBER_FULL[caliber]} · 检验水平 {conf}% · 区间 {start} 至 {end}；"
    "页内所有读数都建立在这套口径与区间上。"
)
page.footer(
    "results/regimes.csv、var_historical.csv、var_parametric.csv、"
    "backtest_exceptions.csv、backtest_results.csv、model_scorecard.csv、"
    "baseline_decision.csv、delta_transmission_events.csv、"
    "delta_transmission_summary.csv、factor_exposure.csv",
    "var_pct / ret_pct / exceed_ratio / rate_pct / mean_var / cap_idx_vs_M1 / "
    "verdict / dev_pct",
)
