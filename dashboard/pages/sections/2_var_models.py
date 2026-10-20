"""主题二 · VaR 模型对比看板。

八个模型逐日比一遍：谁给得小、谁经得起 Kupiec 与 Christoffersen、波动来自哪一块。
试算模型与置信度在右上角 chip 里改；这张表是全页唯一一处「勾行就切换」的地方，
勾行首方框会把整张卡的读数、锚图高亮与后面所有主题一起带动。

全部引用既有 CSV，不在本段重算。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, pipe, theme  # noqa: E402

page.init("VaR 模型对比看板")
# 两栏版面计划：每张卡 H（半栏）/ W（通栏），按卡片出现顺序写。
# 落在通栏卡之前的那张半栏卡标成 W，自己撑满一整行——否则右边空半屏。
page.grid_plan("H", "H", "H", "H", "W", "H", "H", "H", "H", "H")

page.section_head(
    "VaR 模型对比看板",
    n_cards=10,
    subtitle="八个模型逐日比一遍：谁给得小、谁经得起 Kupiec 与 Christoffersen、"
             "波动到底来自哪一块。",
)

_tok = theme.tokens()

ORDER10 = labels.MODEL_ORDER + [m for m in labels.NON_EVIDENCE_MODELS
                                if m not in labels.MODEL_ORDER]

# 评分卡勾行 → 切换试算模型；必须在本段任何 pipe_model() 之前
_ms0 = data.model_scorecard()
_ms0 = _ms0[(_ms0["caliber"] == pipe.pipe_caliber())
            & (_ms0["conf"].astype(str) == pipe.pipe_conf())]
_order_present = [m for m in ORDER10 if m in set(_ms0["model"])]
if pipe.consume_table_model("var_scorecard_pick", _order_present):
    st.rerun()

caliber = pipe.pipe_caliber()
model = pipe.pipe_model()
conf = pipe.pipe_conf()
quote = pipe.pipe_quote()
_row = pipe.scorecard_row()
_mlab = labels.MODEL_LABEL.get(model, model)
_es_txt = (f"{float(_row['es_base_pct']):.4f} %"
           if float(_row["es_base_pct"]) == float(_row["es_base_pct"]) else "未登记")

_cols = st.columns(4, gap="small")
for _col, (_label, _value, _sub) in zip(_cols, [
    (f"{_mlab} 的平均 VaR", f"{float(_row['mean_var_base_pct']):.4f} %",
     f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}上的平均资本占用（正值＝损失）。"),
    (f"{_mlab} 的 {conf}% ES", _es_txt,
     "超过 VaR 之后的条件平均损失；99% 档的 1 日 ES 是压力测试的判定线分母。"),
    ("资本指数（M1 = 1）", f"{float(_row['cap_idx_vs_M1']):.4f}",
     "小于 1 表示比 M1 省资本；省不省由主题四的检验说了算。"),
    ("校准偏差（正＝过度覆盖）", f"{float(_row['cal_dev']):+.4f}",
     "1 − 真实违规率/名义违规率。负值才是低估风险。"),
]):
    with _col:
        page.tile_kpi(_label, _value, sub=_sub)

st.caption(f"当前试算：{pipe.trial_meta()} · common773 公共样本。"
           "正负号约定：损失以正值表示。")

# -------------------------------------------------- 2.1 八模型逐日 −VaR
long = data.var_series_long(caliber, conf, tuple(labels.MODEL_ORDER))
long["neg_var"] = -pd.to_numeric(long["var_pct"], errors="coerce")
_sel = long[long["model"] == model]
_other = long[long["model"] != model]

_grey = alt.Chart(_other).mark_line(
    strokeWidth=1.3, color=_tok["muted"], opacity=0.42,
).encode(
    x=alt.X("date:T", title=""),
    y=alt.Y("neg_var:Q", title="日收益 / −VaR（%）"),
    detail="model:N",
    tooltip=[alt.Tooltip("date:T", title="日期"),
             alt.Tooltip("model:N", title="模型"),
             alt.Tooltip("neg_var:Q", title="−VaR %", format=".4f")],
)
_bold = alt.Chart(_sel).mark_line(
    strokeWidth=2.6, color=_tok["series"][0],
).encode(
    x=alt.X("date:T"), y=alt.Y("neg_var:Q"),
    tooltip=[alt.Tooltip("date:T", title="日期"),
             alt.Tooltip("neg_var:Q", title="−VaR %", format=".4f")],
)
_ret = long.drop_duplicates("date")[["date", "ret_pct"]].dropna()
_reline = alt.Chart(_ret).mark_line(
    strokeWidth=charts.LINE_PX, color=_tok["ink"], opacity=0.55,
).encode(x=alt.X("date:T"), y=alt.Y("ret_pct:Q"))

_reg = data.regimes()
_rv_th = float(_reg["rv60_ann_%"].iloc[59:].quantile(2 / 3))


def _runs(reg: pd.DataFrame, flag: str) -> pd.DataFrame:
    """布尔列 → 连续区间表（start/end），供背景色带整段画。"""
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


_fig = _grey + _bold + _reline
_fig = charts.shade_bands(_fig, _runs(_reg, "chronic_high"),
                          color=_tok["muted"], opacity=0.10)
_fig = charts.shade_bands(_fig, _runs(_reg, "crisis_window"), opacity=0.16)

if caliber == "次" and model == "M1g":
    _b0, _b1 = [pd.Timestamp(x.strip())
                for x in labels.GARCH_BOUNDARY_SPAN.split("~")]
    _fig = charts.shade_bands(
        _fig, pd.DataFrame({"start": [_b0], "end": [_b1]}),
        color=_tok["status"]["serious"], opacity=0.16)

_be = data.backtest_exceptions()
_be = _be[(_be["caliber"] == caliber) & (_be["conf"].astype(str) == conf)
          & (_be["model"] == model)]
if not _be.empty:
    _fig = _fig + alt.Chart(_be).mark_point(
        size=charts.MARKER_PX, filled=True, color=_tok["status"]["critical"],
        stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
    ).encode(
        x=alt.X("date:T"), y=alt.Y("ret:Q"),
        tooltip=[alt.Tooltip("date:T", title="违规日"),
                 alt.Tooltip("ret:Q", title="日收益 %", format=".4f"),
                 alt.Tooltip("var:Q", title="VaR %", format=".4f"),
                 alt.Tooltip("exceed_ratio:Q", title="超出倍数", format=".2f"),
                 alt.Tooltip("is_doubtful:N", title="存疑日")],
    )

with page.card("2.1　八个模型的逐日 −VaR：蓝线是当前试算模型，红点是它的违规日"):
    st.altair_chart(
        _fig.add_params(charts.zoom_x()).properties(height=300),
        use_container_width=True)
    st.caption(
        f"蓝线 {_mlab}，灰线其余七模型，黑线是实际日收益。越靠下越保守；"
        f"浅灰底＝慢性高波动段，橙底＝急性危机窗；"
        f"红点＝{_mlab} 在 {conf}% 档的违规日，共 {len(_be)} 天。"
    )
    if caliber == "次" and model == "M1g":
        st.caption(
            f"另有一段橙红底：次口径 GARCH 的 α 落到 0 的区间"
            f"（{labels.GARCH_BOUNDARY_SPAN}），那段 σ_t 退化为常数，"
            "M1g 的线在那里没有时变含义。主口径没有这个退化。"
        )
    charts.data_drawer(
        long[["date", "model", "var_pct", "ret_pct", "neg_var"]],
        key="var_varseries", title="2.1 逐日 VaR 与实际收益（长表）",
        height=280,
        labels={"date": "日期", "model": "模型", "var_pct": "VaR %（正值）",
                "ret_pct": "日收益 %", "neg_var": "−VaR %"},
        fmt={"var_pct": "%.4f", "ret_pct": "%.4f", "neg_var": "%.4f"},
        note="读 results/var_historical.csv 与 var_parametric.csv 合并后的长表；"
             "违规点另见 backtest_exceptions.csv。"
             f"体制色带阈值 {_rv_th:.4f}% 按生成规则页面现算。",
    )

# -------------------------------------------------- 2.2 平均 VaR 对照
_bt_all = data.backtest_results()
_b = _bt_all[(_bt_all["caliber"] == caliber)
             & (_bt_all["conf"].astype(str) == conf)
             & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_t_ref = int(_b["T"].astype(int).max())
_b_in = _b[_b["T"].astype(int) == _t_ref].copy()
_b_in["model_lab"] = _b_in["model"].map(lambda m: labels.MODEL_LABEL.get(m, m))
_b_in["选中"] = _b_in["model"].map(lambda m: m == model)
_b_in = _b_in.sort_values("mean_var")
_sort = list(_b_in["model_lab"])

with page.card("2.2　平均 VaR 对照：同一根横轴上的资本占用"):
    _bars = alt.Chart(_b_in).mark_bar(size=18, cornerRadiusEnd=4).encode(
        # 半栏只有 700px，默认 labelLimit 会把「FHS-E 滤波历史模拟」砍成
        # 「FHS-E 滤波历史…」，八根条读不出是哪八个模型。0 = 不截断。
        y=alt.Y("model_lab:N", title="", sort=_sort,
                axis=alt.Axis(labelLimit=0)),
        x=alt.X("mean_var:Q", title="平均 VaR（%，正值＝损失）",
                scale=alt.Scale(
                    domain=[0, float(_b_in["mean_var"].max()) * 1.55])),
        color=alt.condition(alt.datum.选中,
                            alt.value(_tok["series"][0]),
                            alt.value(_tok["muted"])),
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("mean_var:Q", title="平均 VaR %", format=".4f"),
                 alt.Tooltip("es:Q", title="ES %", format=".4f"),
                 alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f")],
    )
    st.altair_chart(
        _bars.properties(height=max(200, 24 * len(_b_in) + 46)),
        use_container_width=True)
    _b_out = _b[_b["T"].astype(int) != _t_ref]
    st.caption(f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}、{_t_ref} 天公共样本；"
               "蓝条是当前试算模型。")
    st.caption(
        "「参数法更贵」不是普遍规律，换档换口径会变号。"
        + ("" if _b_out.empty else
           "未画进来的："
           + "、".join(f"{r['model']}（只有 {int(r['T'])} 天样本）"
                       for _, r in _b_out.iterrows())
           + "，窗更长、公共样本更短，同轴比平均值会误导。")
    )
    charts.data_drawer(
        _b[["model", "role", "T", "mean_var", "es", "es_ratio", "rate_pct",
            "x", "exp_x_eff", "p_uc"]],
        key="var_meanvar",
        title=f"2.2 {labels.SCOPE_LABEL[labels.BASE_SCOPE]}的平均 VaR、ES 与资本指数",
        height=280,
        labels={"model": "模型", "role": "角色", "T": "样本天数",
                "mean_var": "平均 VaR %", "es": "ES %", "es_ratio": "ES/VaR",
                "rate_pct": "真实违规率 %", "x": "违规次数",
                "exp_x_eff": "期望次数", "p_uc": "Kupiec p"},
        fmt={"mean_var": "%.4f", "es": "%.4f", "es_ratio": "%.4f",
             "rate_pct": "%.4f", "exp_x_eff": "%.1f", "p_uc": "%.4f"},
    )

# -------------------------------------------------- 2.3 取舍坐标
ms2 = data.model_scorecard()
ms2 = ms2[(ms2["caliber"] == caliber)
          & (ms2["conf"].astype(str) == conf)].copy()
ms2 = ms2[ms2["model"].isin(labels.MODEL_ORDER)]
_nom = float(ms2["nominal_pct"].iloc[0])

with page.card("2.3　取舍坐标：横轴资本占用，纵轴真实违规率"):
    sc = charts.scatter(
        ms2, x="cap_idx_vs_M1", y="rate_base_pct", color="verdict",
        color_domain=["推荐", "淘汰"], color_title="",
        x_title="资本指数（M1 = 1，越小越省资本）",
        y_title="基准范围违规率（%）", height=280,
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("verdict:N", title="结论"),
                 alt.Tooltip("cap_idx_vs_M1:Q", title="资本指数", format=".4f"),
                 alt.Tooltip("rate_base_pct:Q", title="违规率 %", format=".4f"),
                 alt.Tooltip("cal_dev:Q", title="校准偏差", format=".4f"),
                 alt.Tooltip("es_base_pct:Q", title="ES %", format=".4f")],
    )
    _ring = alt.Chart(ms2[ms2["model"] == model]).mark_point(
        size=260, filled=False, stroke=_tok["series"][0], strokeWidth=2.4,
    ).encode(x="cap_idx_vs_M1:Q", y="rate_base_pct:Q")
    _txt2 = alt.Chart(ms2).mark_text(
        align="left", dx=8, dy=-6, fontSize=10, color=_tok["ink2"],
        font=_tok["font"],
    ).encode(x=alt.X("cap_idx_vs_M1:Q"), y=alt.Y("rate_base_pct:Q"),
             text=alt.Text("model:N"))
    sc = sc + _ring + _txt2
    sc = charts.add_rule(sc, _nom, label=f"名义 {_nom:g}%", dashed=True, dy=13)
    sc = charts.add_rule(sc, _nom * labels.PRUDENCE,
                         label=f"审慎带（名义 ×{labels.PRUDENCE:g}）",
                         color=_tok["muted"], dashed=True, dy=-6)
    st.altair_chart(sc, use_container_width=True)
    st.caption("越靠左上越划算；蓝圈是当前试算模型，两条虚线是名义水平与审慎带"
               "（名义 ×1.10）。颜色只作辅助、不单独表意。")
    charts.data_drawer(
        ms2[["model", "verdict", "cap_idx_vs_M1", "rate_base_pct", "cal_dev",
             "mean_var_base_pct", "es_base_pct", "stab_scope_pp"]],
        key="var_tradeoff", title="2.3 资本占用与违规率的取舍坐标",
        height=260,
        labels={"model": "模型", "verdict": "结论",
                "cap_idx_vs_M1": "资本指数", "rate_base_pct": "违规率 %",
                "cal_dev": "校准偏差", "mean_var_base_pct": "平均 VaR %",
                "es_base_pct": "ES %", "stab_scope_pp": "跨范围摆幅 pp"},
        fmt={"cap_idx_vs_M1": "%.4f", "rate_base_pct": "%.4f",
             "cal_dev": "%.4f", "mean_var_base_pct": "%.4f",
             "es_base_pct": "%.4f", "stab_scope_pp": "%.4f"},
        note="读 results/model_scorecard.csv；verdict 列即结论。",
    )

# -------------------------------------------------- 2.4 评分卡（全页唯一勾选处）
ms = data.model_scorecard()
ms = ms[(ms["caliber"] == caliber) & (ms["conf"].astype(str) == conf)]
ms = ms.set_index("model").reindex(
    [m for m in ORDER10 if m in set(ms["model"])]).reset_index()
n_rec = int((ms["verdict"] == "推荐").sum())

with page.card("2.4　评分卡：两道闸全过才进推荐"):
    st.caption(f"本档推荐 {n_rec}/{len(ms)} 个。资本指数以 M1 为 1，小于 1 表示更省资本；"
               "过 Kupiec 的范围数按 χ²(1) 5% 临界值 3.8415 判定。")
    st.dataframe(
        ms[["model", "verdict", "rate_base_pct", "nominal_pct",
            "rate_over_nominal", "cal_dev", "cap_idx_vs_M1",
            "mean_var_base_pct", "es_base_pct", "n_pass_uc_core",
            "n_core_scopes", "stab_scope_pp", "gap_regime_pp",
            "n_ind_suspect"]],
        width="stretch", height=230, hide_index=True,
        column_config={
            "model": st.column_config.Column("模型"),
            "verdict": st.column_config.Column("结论"),
            "rate_base_pct": st.column_config.NumberColumn(
                "基准范围违规率 %", format="%.4f"),
            "nominal_pct": st.column_config.NumberColumn("名义 %", format="%.4f"),
            "rate_over_nominal": st.column_config.NumberColumn(
                "违规率/名义", format="%.4f"),
            "cal_dev": st.column_config.NumberColumn("校准偏差", format="%.4f"),
            "cap_idx_vs_M1": st.column_config.NumberColumn(
                "资本指数（M1=1）", format="%.4f"),
            "mean_var_base_pct": st.column_config.NumberColumn(
                "平均 VaR %", format="%.4f"),
            "es_base_pct": st.column_config.NumberColumn("ES %", format="%.4f"),
            "n_pass_uc_core": st.column_config.Column("过 Kupiec 核心范围数"),
            "n_core_scopes": st.column_config.Column("核心范围总数"),
            "stab_scope_pp": st.column_config.NumberColumn(
                "跨范围摆幅 pp", format="%.4f"),
            "gap_regime_pp": st.column_config.NumberColumn(
                "分段缺口 pp", format="%.4f"),
            "n_ind_suspect": st.column_config.Column("独立性存疑次数"),
        },
        on_select="rerun", selection_mode="single-row",
        key="var_scorecard_pick",
    )
    st.caption("勾行首方框切换的是整个试算单：顶部读数、2.1 的蓝线与红点、"
               "以及后面所有主题都跟着换。M1δ 与 M1g-t 是对照，勾了也不切换。")
    charts.data_drawer(
        ms[["model", "verdict", "rate_base_pct", "nominal_pct", "rate_over_nominal",
            "cal_dev", "cap_idx_vs_M1", "mean_var_base_pct", "es_base_pct",
            "n_pass_uc_core", "n_core_scopes", "stab_scope_pp", "gap_regime_pp",
            "n_ind_suspect"]],
        key="var_scorecard_drawer", title="2.4 评分卡全表",
        height=260,
        labels={"model": "模型", "verdict": "结论", "nominal_pct": "名义 %",
                "rate_base_pct": "基准范围违规率 %", "rate_over_nominal": "违规率/名义",
                "cal_dev": "校准偏差", "cap_idx_vs_M1": "资本指数",
                "mean_var_base_pct": "平均 VaR %", "es_base_pct": "ES %",
                "n_pass_uc_core": "过 Kupiec 核心范围数",
                "n_core_scopes": "核心范围总数", "stab_scope_pp": "跨范围摆幅 pp",
                "gap_regime_pp": "分段缺口 pp", "n_ind_suspect": "独立性存疑次数"},
        fmt={"rate_base_pct": "%.4f", "nominal_pct": "%.4f",
             "rate_over_nominal": "%.4f", "cal_dev": "%.4f",
             "cap_idx_vs_M1": "%.4f", "mean_var_base_pct": "%.4f",
             "es_base_pct": "%.4f", "stab_scope_pp": "%.4f",
             "gap_regime_pp": "%.4f"},
    )

# -------------------------------------------------- 2.5 五条 σ 路径
_pre = "" if caliber == "主" else "rmb_"
_vp = data.var_parametric()
_sig_cols = [c for c in ["sig_uncond", "sig_garch", "sig_garch_t",
                         "sig_ewma", "sig_dnorm"]
             if f"{_pre}{c}" in _vp.columns]
_sig = _vp[["date"] + [f"{_pre}{c}" for c in _sig_cols]].melt(
    id_vars="date", var_name="s", value_name="sigma")
_sig["sigma"] = pd.to_numeric(_sig["sigma"], errors="coerce")
_sig = _sig.dropna(subset=["sigma"])
_sig["路径"] = _sig["s"].str.replace(r"^rmb_", "", regex=True).map(labels.SIG_LABEL)

with page.card("2.5　五条波动率路径：参数法的机理层", wide=True):
    # 五条 σ 各画一张单图，三列一行地铺。分面 spec 的宽度只能定死，窄视口会压出
    # 卡片；单图各自跟卡片伸缩，任何视口都不会出框。各格纵轴仍独立（图上本来
    # 也看刻度），共轴会把量级小的那几条压成直线。
    _sig_order = [labels.SIG_LABEL[c] for c in _sig_cols]
    for _k5 in range(0, len(_sig_order), 3):
        _row5 = st.columns(min(3, len(_sig_order) - _k5), gap="small")
        for _col5, _pth in zip(_row5, _sig_order[_k5:_k5 + 3]):
            with _col5:
                _fig5 = alt.Chart(_sig[_sig["路径"] == _pth]).mark_line(
                    strokeWidth=1.4, color=_tok["series"][0]).encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("sigma:Q", title="日波动率 σ（%）"),
                    tooltip=[alt.Tooltip("date:T", title="日期"),
                             alt.Tooltip("路径:N", title="σ 路径"),
                             alt.Tooltip("sigma:Q", title="σ %", format=".4f")],
                ).add_params(charts.zoom_x())
                st.altair_chart(
                    _fig5.properties(height=140, title=_pth),
                    use_container_width=True)
    st.caption(f"{labels.CALIBER_FULL[caliber]}。五条 σ 同为日波动率（%），"
               "分面且各格纵轴独立——共轴会把量级小的那几条压成直线。")
    charts.data_drawer(
        _sig.assign(sigma=lambda d: d["sigma"].round(6)),
        key="var_sig", title="2.5 五条 σ 路径逐日值",
        columns=["date", "路径", "sigma"],
        labels={"date": "日期", "路径": "σ 路径", "sigma": "日波动率 %"},
        note="sig_garch_t 是 t 似然估出来的 σ，本身是波动率，不是分位数。",
        fmt={"sigma": "%.6f"}, height=300,
    )

# -------------------------------------------------- 2.6 HS 分位路径
_vh = data.var_historical()
_hs = _vh[["date"] + [f"{_pre}hs{w}_var_{c}" for w in (250, 500, 750)
                      for c in ("95", "99")]].melt(
    id_vars="date", var_name="col", value_name="var_pct")
_hs["var_pct"] = pd.to_numeric(_hs["var_pct"], errors="coerce")
_hs = _hs.dropna(subset=["var_pct"])
_hs["窗长"] = _hs["col"].str.extract(r"hs(\d+)_")[0].map(lambda w: f"HS{w}")
_hs["水平"] = _hs["col"].str.extract(r"_var_(\d+)$")[0] + "% 检验水平"

with page.card("2.6　历史模拟的分位路径：三个窗长各自的经验分位"):
    # 95% / 99% 两档拆成上下两张单图。原先并排一份分面 spec，顶层 width 是每格
    # 宽度，只能定死，窄视口下就压出卡片；拆开之后各自跟卡片伸缩。
    # 纵轴各管各的（沿用原分面的独立纵轴）：99% 的量级比 95% 高近一倍，
    # 共一根轴会把 95% 那条压成贴底的一条线。时间轴则统一。
    _hs_x0, _hs_x1 = _hs["date"].min(), _hs["date"].max()
    _panels = []
    for _i, _lev in enumerate(["95% 检验水平", "99% 检验水平"]):
        _part = _hs[_hs["水平"] == _lev].reset_index(drop=True).copy()
        _part["v"] = _part["var_pct"].astype(float)
        _y0 = float(_part["v"].min())
        _y1 = float(_part["v"].max())
        _pad = (_y1 - _y0) * 0.06
        _panels.append(alt.Chart(_part).mark_line(
            strokeWidth=charts.LINE_PX).encode(
            x=alt.X("date:T", title="", scale=alt.Scale(domain=[_hs_x0, _hs_x1])),
            y=alt.Y("v:Q", title=f"{_lev} · VaR（%，正值）",
                    scale=alt.Scale(domain=[_y0 - _pad, _y1 + _pad])),
            color=alt.Color("窗长:N", title="", sort=["HS250", "HS500", "HS750"],
                            legend=(alt.Legend(orient="top", direction="horizontal")
                                    if _i == 0 else None)),
            tooltip=[alt.Tooltip("date:T", title="日期"),
                     alt.Tooltip("窗长:N"), alt.Tooltip("水平:N"),
                     alt.Tooltip("v:Q", title="VaR %", format=".4f")],
        ).properties(height=185))
    # 两档放进同一个 vconcat，纵轴各自独立。不用两张独立的 st.altair_chart：
    # 同一张卡里连着的两张图，第一张会在容器宽度还没定下来时先渲染，
    # 结果是三条线被压成贴顶的一条直线（实测 bbox 只有 4px 高）。
    # vconcat 不是分面，顶层宽度就是整张的宽度，use_container_width 照常生效。
    st.altair_chart(
        alt.vconcat(*_panels, spacing=10).resolve_scale(y="independent"),
        use_container_width=True)
    st.caption(f"{labels.CALIBER_FULL[caliber]}。窗越长越平滑、对近期冲击越钝；"
               "点图例可以只看其中一档。")
    charts.data_drawer(
        _hs, key="var_hs", title="2.6 三个窗长的 HS 分位逐日值",
        columns=["date", "窗长", "水平", "var_pct"],
        labels={"date": "日期", "窗长": "窗长", "水平": "检验水平",
                "var_pct": "VaR %"},
        note="VaR 以正值存，与收益同图时画成 −VaR。",
        fmt={"var_pct": "%.6f"}, height=300,
    )

# -------------------------------------------------- 2.7 δ 传导
de_all = data.delta_transmission_events()
ds_all = data.delta_transmission_summary()
dmodel = sorted(de_all["model"].unique())[0]
dsrc = "B 事件前δ"
dwin = "3日(主)"

de = de_all[(de_all["model"] == dmodel) & (de_all["caliber"] == caliber)
            & (de_all["delta_src"] == dsrc)
            & (de_all["window"] == dwin)].copy()
de["understate_cn"] = de["understate"].map({True: "是（低估损失）", False: "否"})
de["credit_cn"] = de["market"].astype(str).map(
    lambda m: "含信用维度" if "credit" in m else "不含信用维度")
ds = ds_all[(ds_all["model"] == dmodel) & (ds_all["caliber"] == caliber)
            & (ds_all["delta_src"] == dsrc) & (ds_all["window"] == dwin)]

with page.card("2.7　δ 传导：预测 vs 实际"):
    if not ds.empty:
        r = ds.iloc[0]
        _c2 = st.columns(4, gap="small")
        for _col, (_label, _value, _sub) in zip(_c2, [
            ("偏差均值（实际 − 预测）", f"{r['dev_mean']:+.4f} %",
             "为负即 δ 系统性低估损失。"),
            ("偏差中位数", f"{r['dev_median']:+.4f} %",
             f"5% / 95% 分位 {r['dev_p05']:+.3f} / {r['dev_p95']:+.3f}。"),
            ("低估事件占比", f"{r['under_share'] * 100:.1f} %", "全部事件口径。"),
            ("信用类事件占比", f"{r['credit_share'] * 100:.1f} %",
             f"绝对偏差均值 {r['absdev_mean']:.4f}%。"),
        ]):
            with _col:
                page.tile_kpi(_label, _value, sub=_sub)
    st.caption(f"固定设定：{dmodel} · {labels.CALIBER_FULL[caliber]} · {dsrc} · "
               f"{dwin} 窗，共 {len(de)} 个事件；这是报告采用的设定。")
    lo = float(min(de["r_pred_pct"].min(), de["r_actual_pct"].min()))
    hi = float(max(de["r_pred_pct"].max(), de["r_actual_pct"].max()))
    s = charts.scatter(
        de, x="r_pred_pct", y="r_actual_pct", color="credit_cn",
        color_domain=["含信用维度", "不含信用维度"], color_title="",
        height=260, x_title="δ 预测收益（%）", y_title="实际收益（%）",
        tooltip=[alt.Tooltip("event_date:T", title="事件日"),
                 alt.Tooltip("event_cn:N", title="事件"),
                 alt.Tooltip("market:N", title="冲击类别"),
                 alt.Tooltip("r_pred_pct:Q", title="预测 %", format=".4f"),
                 alt.Tooltip("r_actual_pct:Q", title="实际 %", format=".4f"),
                 alt.Tooltip("dev_pct:Q", title="偏差 %", format=".4f"),
                 alt.Tooltip("understate_cn:N", title="低估损失")],
    )
    st.altair_chart(
        charts.diagonal(s, lo, hi).add_params(charts.zoom_xy()),
        use_container_width=True)
    st.caption("点落在 45° 线下方即实际比预测更差，也就是 δ 低估了这一步损失。")
    charts.data_drawer(
        de[["event_date", "event_cn", "market", "r_actual_pct", "r_pred_pct",
            "dev_pct", "understate_cn", "rate_shock_bp", "d5y_sum_bp",
            "d10y_sum_bp", "r2_delta_src"]],
        key="var_delta", title="2.7 δ 传导的逐事件预测与偏差",
        height=280,
        labels={"event_date": "事件日", "event_cn": "事件",
                "market": "冲击类别", "r_actual_pct": "实际 %",
                "r_pred_pct": "预测 %", "dev_pct": "偏差 %",
                "understate_cn": "低估损失", "rate_shock_bp": "利率冲击 bp",
                "d5y_sum_bp": "Δ5Y 合计 bp", "d10y_sum_bp": "Δ10Y 合计 bp",
                "r2_delta_src": "δ 来源 R²"},
        fmt={"r_actual_pct": "%.4f", "r_pred_pct": "%.4f", "dev_pct": "%.4f",
             "rate_shock_bp": "%.1f", "d5y_sum_bp": "%.1f",
             "d10y_sum_bp": "%.1f", "r2_delta_src": "%.4f"},
        note="读 results/delta_transmission_events.csv，已按上面钉死的设定筛过。",
    )

# -------------------------------------------------- 2.8 δ 偏差构成
with page.card("2.8　δ 偏差的构成与因子暴露"):
    st.caption("偏差的 88.1% 来自信用残差项（|23.30| 对参数差异的 |3.13|），"
               "δ 本身估得稳，放大倍数 1.00。")
    _c3 = st.columns([1, 1], gap="small")
    with _c3[0]:
        st.altair_chart(
            charts.hist_with_normal(de["dev_pct"], bins=30, height=250,
                                    title=f"偏差（%）· n={len(de)}"),
            use_container_width=True)
    with _c3[1]:
        st.markdown("**偏差 vs 事件窗利率冲击**")
        s2 = charts.scatter(
            de, x="rate_shock_bp", y="dev_pct", color="credit_cn",
            color_domain=["含信用维度", "不含信用维度"], color_title="",
            height=215, x_title="利率冲击（bp）",
            y_title="偏差 %（负＝低估损失）",
            tooltip=[alt.Tooltip("event_date:T", title="事件日"),
                     alt.Tooltip("event_cn:N", title="事件"),
                     alt.Tooltip("market:N", title="冲击类别"),
                     alt.Tooltip("rate_shock_bp:Q", title="冲击 bp", format=".1f"),
                     alt.Tooltip("dev_pct:Q", title="偏差 %", format=".4f")])
        st.altair_chart(
            charts.add_rule(s2, 0.0, label="无偏差", dashed=True),
            use_container_width=True)
        st.caption("偏差与冲击幅度的相关系数只有 +0.151，不随冲击变大而系统性恶化。")
    st.caption("δ 的因子暴露（主口径每 1 单位因子的收益敏感度）；"
               "汇率一行是口径定义而非估计，主口径恒为 0、人民币次口径为 +1。")
    st.dataframe(data.factor_exposure(), width="stretch", height=160,
                 hide_index=True)
    charts.data_drawer(
        de[["event_cn", "market", "dev_pct", "dev_param_pct",
            "dev_credit_pct", "rate_shock_bp", "r2_delta_src"]],
        key="var_delta_split",
        title="2.8 δ 偏差的参数项与信用项分解",
        height=260,
        labels={"event_cn": "事件", "market": "冲击类别",
                "dev_pct": "总偏差 %", "dev_param_pct": "参数差异项 %",
                "dev_credit_pct": "信用残差项 %",
                "rate_shock_bp": "利率冲击 bp", "r2_delta_src": "δ 来源 R²"},
        fmt={"dev_pct": "%.4f", "dev_param_pct": "%.4f",
             "dev_credit_pct": "%.4f", "rate_shock_bp": "%.1f",
             "r2_delta_src": "%.4f"},
        note="把总偏差拆成参数项与信用项。",
    )

# -------------------------------------------------- 2.9 波动来自哪
_attr = data.var_attribution().copy()
_ATTR_NAME = {
    "Δ5Y": labels.FACTOR_CN["d5y_bp"],
    "Δ10Y": labels.FACTOR_CN["d10y_bp"],
    "利差代理（残差）": labels.FACTOR_CN["residual"],
    "汇率 CNY/USD": labels.FACTOR_CN["fx_ret_pct"],
}
_ATTR_ORDER = [_ATTR_NAME[k] for k in
               ["Δ5Y", "Δ10Y", "利差代理（残差）", "汇率 CNY/USD"]]
_attr["因子"] = _attr["factor"].map(_ATTR_NAME)
_attr["口径"] = _attr["caliber"].map(labels.CALIBER_SHORT)
_attr = _attr.sort_values("口径", key=lambda s: s.map(
    {labels.CALIBER_SHORT["主"]: 0, labels.CALIBER_SHORT["次"]: 1}))

with page.card("2.9　波动来自哪一维（因子方差贡献份额）"):
    st.altair_chart(
        charts.stacked_share(_attr, cat="口径", seg="因子", val="share_pct",
                             seg_order=_ATTR_ORDER, height=260,
                             x_title="方差贡献份额（%）"),
        use_container_width=True,
    )
    st.caption("主口径没有汇率一行，次口径多出的那块汇率占 44.2%。")
    charts.data_drawer(
        _attr[["caliber_raw", "factor", "share_pct"]], key="var_attr",
        height=200,
        title="2.9 因子波动贡献份额（两口径 × 三到四个因子）",
        labels={"caliber_raw": "口径", "factor": "因子",
                "share_pct": "份额（%）"},
        fmt={"share_pct": "%.4f"},
        note=("读 results/var_attribution.csv。主口径没有汇率一行"
              "（该口径下汇率不进分解）；美元口径下汇率变动被计价基准搬运掉了，"
              "人民币口径因此多出一块汇率、占 44.2%。利差代理那一块是同一次回归的"
              "残差，是代理量、不是真实信用风险贡献；主口径整行为 100% 是把三块"
              "重新归一的结果，不能用它与次口径比绝对水平。"),
    )

# -------------------------------------------------- 2.10 试算 vs 基准
with page.card("2.10　当前试算与报告基准的差"):
    _base = pipe.BASELINE
    _rows = [
        ("模型", labels.MODEL_LABEL.get(model, model), labels.MODEL_LABEL.get(_base["model"], _base["model"])),
        ("置信度", f"{conf}%", f"{_base['conf']}%"),
        ("汇率口径", labels.CALIBER_SHORT[caliber], labels.CALIBER_SHORT[_base["caliber"]]),
        ("平均 VaR", f"{float(_row['mean_var_base_pct']):.4f} %", "0.4527 %"),
        ("1 日 ES（判定线）", _es_txt, "0.6659 %"),
        ("真实违规率", f"{float(_row['rate_base_pct']):.4f} %", "1.0349 %"),
        ("资本指数（M1=1）", f"{float(_row['cap_idx_vs_M1']):.4f}", "0.8720"),
    ]
    st.dataframe(
        pd.DataFrame(_rows, columns=["读数", "当前试算", "报告基准"]),
        width="stretch", height=250, hide_index=True)
    st.caption("换口径或模型后这里一眼能看出偏到哪；报告全文与结论按基准那一列写。")

page.footer(
    "results/var_historical.csv、var_parametric.csv、regimes.csv、"
    "backtest_exceptions.csv、backtest_results.csv、model_scorecard.csv、"
    "delta_transmission_events.csv、delta_transmission_summary.csv、"
    "factor_exposure.csv、var_attribution.csv",
    "var_pct / ret_pct / neg_var / rate_pct / mean_var / es / cap_idx_vs_M1 / "
    "cal_dev / dev_pct / share_pct",
)