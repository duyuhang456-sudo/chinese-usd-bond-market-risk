"""模型诊断与验证。

这一页回答一个问题：检验到底做了没有、做出来的结论站不站得住。
全部读既有 CSV，不重算——连边界解的天数都是现算的而不是抄文档，好让文档与产物
一旦分家就在页面上显形。

顶部一条控制带驱动全页：口径、检验水平、回测范围。往下是一列满宽的大图，
顺序按「先看检验、再看机理」：转移四格与三个检验（做了没有）→ 检验力与两个非证据
对照（能读出什么）→ 五条 σ 路径与 GARCH 边界解（参数法的机理层）→ 历史模拟族
（分位路径与滤波对照）。每张图只挂一句读法，长解释与逐行数据都收进数据抽屉。

分面图的每格宽度必须定死，不能配 use_container_width——顶层 width 对分面 spec 是
「每格」的宽度，Streamlit 会把它撑成容器宽、画出 k 倍宽压住右边那一栏。
满宽卡片的内容宽在 1280 视口下约 874，这里取 1000 作上限，宽屏自然留白。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("模型诊断与验证")

ORDER10 = labels.MODEL_ORDER + [m for m in labels.NON_EVIDENCE_MODELS
                                if m not in labels.MODEL_ORDER]

# 满宽卡片里分面图的定宽基准（不是 1600 视口的 1194：按窄视口取，宽屏留白不溢出）
FULL_W = 1000

_bt = data.backtest_results()
_vh = data.var_historical()
_span_text = f"{_vh['date'].min().date()} → {_vh['date'].max().date()}"

_scopes = list(dict.fromkeys(
    labels.CORE_SCOPES + [s for s in _bt["scope"].unique()
                          if s not in labels.CORE_SCOPES]))

# 三相控件要排在页头下面，但页头又要读到它们当前的值。先取上一轮落在 session_state
# 里的值喂给页头，控件本身在页头之后再落位，返回值覆盖这三行。
caliber = st.session_state.get("mv_cal") or "主"
conf = st.session_state.get("mv_conf") or "99"
scope = st.session_state.get("mv_scope") or labels.BASE_SCOPE

page.header(
    "模型诊断与验证",
    "检验做了没有、做出来的结论站不站得住。四块诊断，每一格都能追到它在 CSV 里的来源。",
    crumb="阶段二 · 诊断与验证",
    caliber=labels.CALIBER_FULL[caliber],
    span=_span_text,
)

# ---------------------------------------------------------------- 控制带
with st.container(border=True):
    _k1, _k2, _k3 = st.columns([2, 3, 3])
    with _k1:
        caliber = st.segmented_control(
            "口径", ["主", "次"], default="主", key="mv_cal",
            format_func=lambda c: labels.CALIBER_SHORT[c],
            help="决定五条 σ 路径、三个检验与 FHS 对照读哪一列；转移四格与"
                 "GARCH 轨迹两口径并排，不随它变。",
        ) or "主"
    with _k2:
        conf = st.segmented_control(
            "检验水平", ["95", "99"], default="99", key="mv_conf",
            format_func=lambda c: f"{c}%（名义 {labels.NOMINAL_RATE[c]:g}%）",
            help="决定转移四格、三个检验与两个对照读哪一档。99% 档的 n11 恒为 0，"
                 "这是本页最要紧的一条读法。",
        ) or "99"
    with _k3:
        scope = st.selectbox(
            "回测范围", _scopes, index=_scopes.index(labels.BASE_SCOPE),
            format_func=lambda s: labels.SCOPE_LABEL.get(s, s), key="mv_scope",
            help="转移四格、p 值与 FHS 对照读哪个区间。σ 路径与 GARCH 轨迹是全样本，"
                 "不随它变。",
        )

sel = _bt[(_bt["caliber"] == caliber) & (_bt["conf"].astype(str) == conf)
          & (_bt["scope"] == scope)].copy()
_models = [m for m in ORDER10 if m in set(sel["model"])]
_cal_note = (f"{labels.CALIBER_FULL[caliber]} · {conf}% 检验水平 · "
             f"{labels.SCOPE_LABEL.get(scope, scope)}")

# ---------------------------------------------------------------- 读数带
_n_rows = len(_bt)
_n_closed = int((_bt[["n00", "n01", "n10", "n11"]].sum(axis=1) == _bt["n_pairs"]).sum())
_p_min = float(pd.to_numeric(_bt["p_ind_mc_cond"], errors="coerce").min())
_bonf = 0.05 / _n_rows
_g_main = data.garch_boundary_summary("主")
_g_rmb = data.garch_boundary_summary("次")

page.kpi_row([
    {"label": "转移四格逐行闭合，全表无例外" if _n_closed == _n_rows
              else "转移四格有行不闭合，须回查产物",
     "value": f"{_n_closed}/{_n_rows} 行",
     "help": "n00 + n01 + n10 + n11 ≡ n_pairs 在 backtest_results 的 262 行上逐行成立，"
             "本页现算而非引用文档。"},
    {"label": "独立性检验未发现稳健的聚集证据", "value": f"{_p_min:.5f}",
     "help": f"全表条件 MC p 的最小值（95% 档 · HS500 · 高波动段）。"
             f"Bonferroni 门槛 0.05/{_n_rows} = {_bonf:.5f}，够不着。"},
    {"label": "GARCH 边界解只出现在次口径",
     "value": f"{_g_main['n_boundary']} / {_g_rmb['n_boundary']} 天",
     "help": f"判据 α ≤ {labels.GARCH_BOUNDARY_ALPHA:g}，天数由 CSV 现算。"
             "主口径 α 最小 0.0271，从不贴近 0；"
             f"次口径 {_g_rmb['n_boundary']} 天全部落在 {labels.GARCH_BOUNDARY_SPAN}。"},
    {"label": "滤波并未改善历史模拟", "value": "未改善",
     "help": "95% 档公共样本（common773）下 FHS-E 的真实违规率 5.692%，"
             "高于 HS250 的 4.916%。滤波处理掉的是 σ 的时变，分位数本身没变。"},
])

page.caliber_note(labels.KUPIEC_NOTE)

_tok = theme.tokens()

# ================================================= 1｜转移四格（满宽，本页的锚）
with page.card("Christoffersen 转移四格：每一格是计数不是比例",
               f"{_cal_note}。n_ij = 前一日状态 i、当日状态 j 的配对天数，"
               "0 = 未违规，1 = 违规。看 n11 那一格：99% 档它在全表 131 行里"
               "恒为 0，从来没有出现过「连续两天违规」。"):
    _rows = []
    for _, r in sel.iterrows():
        for i, j, key in ((0, 0, "n00"), (0, 1, "n01"), (1, 0, "n10"), (1, 1, "n11")):
            _rows.append({"model": r["model"],
                          "前一日": "0 未违规" if i == 0 else "1 违规",
                          "当日": "0 未违规" if j == 0 else "1 违规",
                          "n": float(pd.to_numeric(r[key], errors="coerce"))})
    _t = pd.DataFrame(_rows)
    _thr = float(_t["n"].max()) / 2.0

    _rect = alt.Chart(_t).mark_rect(
        stroke=_tok["surface"], strokeWidth=charts.GAP_PX).encode(
        x=alt.X("前一日:N", title="", sort=["0 未违规", "1 违规"],
                axis=alt.Axis(labelAngle=0, labelFontSize=9)),
        y=alt.Y("当日:N", title="", sort=["1 违规", "0 未违规"],
                axis=alt.Axis(labelFontSize=9)),
        color=alt.Color("n:Q", title="配对天数",
                        scale=alt.Scale(range=list(_tok["blue_ramp"]))),
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("前一日:N"), alt.Tooltip("当日:N"),
                 alt.Tooltip("n:Q", title="配对天数", format=".0f")],
    )
    _label = _rect.mark_text(fontSize=10, fontWeight="bold").encode(
        text=alt.Text("n:Q", format=".0f"),
        color=alt.condition(alt.datum.n > _thr, alt.value("#FFFFFF"),
                            alt.value(_tok["ink"])),
    )
    _fig = (_rect + _label).properties(
        height=128, width=charts.facet_width(5, card_w=FULL_W)).facet(
        alt.Column("model:N", title="", sort=_models,
                   header=alt.Header(labelFontSize=10, labelColor=_tok["ink"],
                                     titleColor=_tok["muted"])),
        columns=5,
    )
    st.altair_chart(_fig, use_container_width=False)
    _probs = sel[["model", "n00", "n01", "n10", "n11", "n_pairs",
                  "n_pair_dropped"]].copy()
    _probs["p01_次日违规率%"] = pd.to_numeric(_probs["n01"], errors="coerce") / (
        pd.to_numeric(_probs["n00"], errors="coerce")
        + pd.to_numeric(_probs["n01"], errors="coerce")) * 100
    _probs["p10_次日回落率%"] = pd.to_numeric(_probs["n10"], errors="coerce") / (
        pd.to_numeric(_probs["n10"], errors="coerce")
        + pd.to_numeric(_probs["n11"], errors="coerce")) * 100
    charts.data_drawer(
        _probs, key="mv_n",
        title="数据抽屉 · Christoffersen 转移四格计数与转移概率",
        columns=["model", "n00", "n01", "n10", "n11", "n_pairs",
                 "n_pair_dropped", "p01_次日违规率%", "p10_次日回落率%"],
        note=f"{labels.TRANSITION_NOTE}{labels.CHRISTOFFERSEN_NOTE}"
             "99% 档的转移表有一格是空的，独立性检验在这种情况下几乎无从发力。",
        fmt={"n00": "%.0f", "n01": "%.0f", "n10": "%.0f", "n11": "%.0f",
             "n_pairs": "%.0f", "n_pair_dropped": "%.0f",
             "p01_次日违规率%": "%.4f", "p10_次日回落率%": "%.4f"},
        labels={"model": "模型", "n00": "n00（0→0）", "n01": "n01（0→1）",
                "n10": "n10（1→0）", "n11": "n11（1→1）",
                "n_pairs": "配对数 n_pairs", "n_pair_dropped": "丢弃的配对"},
        height=340,
    )

# ================================================= 1｜三个检验（满宽）
TESTS = [("p_uc", "Kupiec 无条件覆盖"),
         ("p_ind", "独立性 · χ² 近似"),
         ("p_ind_mc_cond", "独立性 · 条件 MC"),
         ("p_cc", "条件覆盖（联合）")]

with page.card("三个检验的 p 值并列：无条件覆盖、独立性、条件覆盖",
               f"{_cal_note}。p 越小越不支持原假设，横轴取对数。"):
    _p = sel[["model"] + [c for c, _ in TESTS]].melt(
        id_vars="model", var_name="test", value_name="p")
    _p["p"] = pd.to_numeric(_p["p"], errors="coerce")
    _p = _p.dropna(subset=["p"])
    _p["检验"] = _p["test"].map(dict(TESTS))

    _rules_meta = [(0.05, "名义 0.05"), (_bonf, f"Bonferroni {_bonf:.5f}")]
    # 分面要求图层共用同一份数据，所以阈值线并进同一张长表、用 kind 拆层，
    # 而不是各自 alt.Chart(...) 再相加——那样 Altair 会因为数据不同而拒绝分面。
    _p["kind"] = "点"
    _p["门槛"] = None
    _rp = pd.DataFrame([{"model": None, "test": t, "p": v, "检验": cn, "kind": "线",
                         "门槛": lab}
                        for t, cn in TESTS for v, lab in _rules_meta])
    _p_all = pd.concat([_p, _rp], ignore_index=True)

    _base = alt.Chart(_p_all)
    _pts = _base.transform_filter(alt.datum.kind == "点").mark_point(
        size=110, filled=True, color=_tok["series"][0],
        stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
    ).encode(
        x=alt.X("p:Q", title="p 值（对数轴）",
                scale=alt.Scale(type="log", domain=[5e-5, 1.1], clamp=True)),
        y=alt.Y("model:N", title="", sort=_models),
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("检验:N", title="检验"),
                 alt.Tooltip("p:Q", title="p 值", format=".6f")],
    )
    _rules = _base.transform_filter(alt.datum.kind == "线").mark_rule(
        strokeWidth=1.5, strokeDash=[4, 3],
    ).encode(
        x="p:Q",
        color=alt.Color("门槛:N", title="门槛",
                        scale=alt.Scale(domain=[lab for _, lab in _rules_meta],
                                        range=[_tok["muted"],
                                               _tok["status"]["critical"]])),
        tooltip=[alt.Tooltip("门槛:N", title="门槛"),
                 alt.Tooltip("p:Q", format=".5f")],
    )
    _fig = (_pts + _rules).properties(
        height=150, width=charts.facet_width(4, card_w=FULL_W)).facet(
        alt.Column("检验:N", title="", sort=[cn for _, cn in TESTS],
                   header=alt.Header(labelFontSize=10, labelColor=_tok["ink"])),
        columns=4,
    )
    st.altair_chart(_fig, use_container_width=False)
    _det = sel[["model", "role", "T", "x", "exp_x", "exp_x_eff", "n_zero",
                "lr_uc", "p_uc", "lr_ind", "p_ind", "p_ind_mc", "p_ind_mc_cond",
                "lr_cc", "p_cc", "size_uc", "size_ind"]]
    charts.data_drawer(
        _det, key="mv_tests",
        title="数据抽屉 · 三个检验的统计量与 p 值",
        columns=["model", "role", "T", "x", "exp_x", "exp_x_eff", "n_zero",
                 "lr_uc", "p_uc", "lr_ind", "p_ind", "p_ind_mc", "p_ind_mc_cond",
                 "lr_cc", "p_cc", "size_uc", "size_ind"],
        note=f"χ²(1) 5% 临界值 3.8415（Kupiec、独立性），"
             f"χ²(2) 5% 临界值 5.9915（条件覆盖）；size_uc / size_ind 是检验的"
             f"实际水平。{labels.P_TWO_KINDS_NOTE}{labels.SIGNIFICANCE_NOTE}"
             "条件 MC 与 χ² 近似是同一件事的两把尺子，小样本时 χ² 不可靠，"
             "文档明文以条件 MC p 为准；红虚线是全表 262 次的 Bonferroni 门槛。",
        fmt={"T": "%.0f", "x": "%.0f", "exp_x": "%.2f", "exp_x_eff": "%.2f",
             "n_zero": "%.0f", "lr_uc": "%.4f", "lr_ind": "%.4f", "lr_cc": "%.4f",
             "p_uc": "%.6f", "p_ind": "%.6f", "p_ind_mc": "%.6f",
             "p_ind_mc_cond": "%.6f", "p_cc": "%.6f",
             "size_uc": "%.4f", "size_ind": "%.4f"},
        labels={"model": "模型", "role": "角色", "T": "样本天数", "x": "违规次数",
                "exp_x": "期望次数", "exp_x_eff": "期望次数（有效）",
                "n_zero": "零收益天数", "lr_uc": "LR_uc", "p_uc": "p_uc",
                "lr_ind": "LR_ind", "p_ind": "p_ind（χ²）",
                "p_ind_mc": "p_ind（MC 无条件下）",
                "p_ind_mc_cond": "p_ind（条件 MC，以此为准）",
                "lr_cc": "LR_cc", "p_cc": "p_cc", "size_uc": "实际水平 无条件",
                "size_ind": "实际水平 独立性"},
        height=340,
    )

# ================================================= 2｜检验力 ｜ 两个非证据对照

with page.card(labels.TEST_POWER_TITLE,
               "全表 262 行的独立性条件 MC p 分布，按检验水平分开；"
               "红虚线是 Bonferroni 门槛。"):
    _hist_all = _bt[["conf", "p_ind_mc_cond", "n11"]].copy()
    _hist_all["p_ind_mc_cond"] = pd.to_numeric(_hist_all["p_ind_mc_cond"],
                                               errors="coerce")
    _hist_all = _hist_all.dropna(subset=["p_ind_mc_cond"])
    _hist_all["置信度"] = _hist_all["conf"].astype(str) + "%"
    _hist_all["b"] = (_hist_all["p_ind_mc_cond"] * 20).clip(0, 19.99).astype(int) / 20.0
    _hist = _hist_all.groupby(["置信度", "b"]).size().reset_index(name="行数")
    _hist["kind"] = "柱"
    _hr = pd.DataFrame([{"置信度": c, "b": _bonf, "行数": 0.0, "kind": "线",
                         "门槛": f"Bonferroni {_bonf:.5f}"} for c in ("95%", "99%")])
    _h_all = pd.concat([_hist, _hr], ignore_index=True)

    _hbase = alt.Chart(_h_all)
    _hbar = _hbase.transform_filter(alt.datum.kind == "柱").mark_bar(
        size=charts.MAX_BAR_PX, cornerRadiusEnd=4, color=_tok["series"][0],
    ).encode(
        x=alt.X("b:Q", title="独立性条件 MC p（分箱 0.05）",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("行数:Q", title="行数"),
        tooltip=[alt.Tooltip("置信度:N"),
                 alt.Tooltip("b:Q", title="p 分箱下沿", format=".2f"),
                 alt.Tooltip("行数:Q", title="行数", format=".0f")],
    )
    _hrules = _hbase.transform_filter(alt.datum.kind == "线").mark_rule(
        strokeWidth=1.5, strokeDash=[4, 3], color=_tok["status"]["critical"],
    ).encode(x="b:Q",
             tooltip=[alt.Tooltip("门槛:N", title="门槛"),
                      alt.Tooltip("b:Q", format=".5f")])
    _fig = (_hbar + _hrules).properties(
        height=240, width=charts.facet_width(2)).facet(
        alt.Column("置信度:N", title="", sort=["95%", "99%"],
                   header=alt.Header(labelFontSize=10, labelColor=_tok["ink"])),
        columns=2,
    )
    st.altair_chart(_fig, use_container_width=False)
    charts.data_drawer(
        _hist_all[["置信度", "p_ind_mc_cond", "n11"]].assign(
            n11_为零=lambda d: d["n11"] == 0),
        key="mv_power",
        title="数据抽屉 · 检验力：独立性条件 MC p 与 n11 逐行",
        columns=["置信度", "p_ind_mc_cond", "n11", "n11_为零"],
        note=f"n11 是「连续两天违规」的配对数，99% 档里它全为 0。"
             f"{labels.TEST_POWER_NOTE}{labels.MULTIPLE_COMPARISON_NOTE}"
             "99% 档的转移四格里 n11 恒为 0，独立性检验因此几乎没有检验力——"
             "「未拒绝」是检验力问题，不是模型优劣的证据。",
        fmt={"p_ind_mc_cond": "%.6f", "n11": "%.0f"},
        labels={"p_ind_mc_cond": "独立性条件 MC p", "n11": "n11（1→1）",
                "n11_为零": "n11 是否为 0"},
        height=300,
    )


with page.card("两个不作独立证据的对照",
               f"{_cal_note}。两者进 CSV 也进检验表，但不参与评选、"
               "不画进模型对比图。"):
    _be = data.backtest_exceptions()
    _be = _be[(_be["caliber"] == caliber) & (_be["conf"].astype(str) == conf)]
    _m1 = set(_be[_be["model"] == "M1"]["date"].astype(str))
    _m1d = set(_be[_be["model"] == "M1d"]["date"].astype(str))
    _gap = len(_m1 ^ _m1d)

    _vp = data.var_parametric()
    _ratio95 = (_vp["garch_t_var_95"] / _vp["sig_garch_t"]).dropna()
    _ratio99 = (_vp["garch_t_var_99"] / _vp["sig_garch_t"]).dropna()

    _chk = pd.DataFrame([
        {"对照": "M1δ", "本页现算的核对": f"与 M1 违规日集合的对称差 {_gap} 天"},
        {"对照": "M1g-t（95%）",
         "本页现算的核对": f"VaR ÷ σ 恒为 {_ratio95.mean():.4f}，即正态分位"},
        {"对照": "M1g-t（99%）",
         "本页现算的核对": f"VaR ÷ σ 恒为 {_ratio99.mean():.4f}，即正态分位"},
    ])
    st.dataframe(_chk, width="stretch", hide_index=True,
                 height=38 * len(_chk) + 40)
    charts.data_drawer(
        pd.DataFrame([
            {"对照": "M1δ", "口径": labels.CALIBER_SHORT[caliber],
             "检验水平": f"{conf}%", "本页现算的核对": "与 M1 的违规日集合对称差",
             "读数": f"{_gap} 天"},
            {"对照": "M1g-t", "口径": labels.CALIBER_SHORT[caliber],
             "检验水平": "95%", "本页现算的核对": "garch_t_var_95 ÷ sig_garch_t",
             "读数": f"[{_ratio95.min():.6f}, {_ratio95.max():.6f}]"},
            {"对照": "M1g-t", "口径": labels.CALIBER_SHORT[caliber],
             "检验水平": "99%", "本页现算的核对": "garch_t_var_99 ÷ sig_garch_t",
             "读数": f"[{_ratio99.min():.6f}, {_ratio99.max():.6f}]"},
        ]),
        key="mv_nonevidence",
        title="数据抽屉 · 两个非证据对照的逐日核对读数",
        columns=["对照", "口径", "检验水平", "本页现算的核对", "读数"],
        note="两个对照的处理方式与报告一致：留在产物与检验表里备查，"
             "不参与评选。M1δ 与 M1 的违规日逐日相同；"
             "M1g-t 的 VaR ÷ σ 恒等于正态分位 1.6449 / 2.3263。"
             "M1δ：代数恒等式，不是第二个模型。"
             + labels.NON_EVIDENCE_NOTE["M1d"]
             + f"本页现算的核对：{labels.CALIBER_SHORT[caliber]} · {conf}% 档下，"
               f"M1 违规 {len(_m1)} 天、M1δ 违规 {len(_m1d)} 天，"
               f"两者日期集合的对称差 = {_gap} 天。"
             + "M1g-t：t 估 σ，但乘正态分位。"
             + labels.NON_EVIDENCE_NOTE["M1g-t"]
             + f"本页现算的核对：garch_t_var_95 ÷ sig_garch_t 在 "
               f"{len(_ratio95)} 个有效值上落在 "
               f"[{_ratio95.min():.6f}, {_ratio95.max():.6f}]，99% 档则是 "
               f"[{_ratio99.min():.6f}, {_ratio99.max():.6f}]，即正态分位的 "
               f"1.6449 与 2.3263，不是 t 分位。"
               "两个对照留在产物与检验表里备查，既不进模型对比图，"
               "也不进评分卡与定案表。",
        height=180,
    )

# ================================================= 3｜五条 σ 路径（满宽）
_pre = "" if caliber == "主" else "rmb_"
_vp = data.var_parametric()
_sig_cols = [c for c in ["sig_uncond", "sig_garch", "sig_garch_t", "sig_ewma", "sig_dnorm"]
             if f"{_pre}{c}" in _vp.columns]
_sig = _vp[["date"] + [f"{_pre}{c}" for c in _sig_cols]].melt(
    id_vars="date", var_name="s", value_name="sigma")
_sig["sigma"] = pd.to_numeric(_sig["sigma"], errors="coerce")
_sig = _sig.dropna(subset=["sigma"])
_sig["路径"] = _sig["s"].str.replace(r"^rmb_", "", regex=True).map(labels.SIG_LABEL)

with page.card("五条波动率路径：参数法的机理层",
               f"{labels.CALIBER_FULL[caliber]}。五条 σ 同为日波动率（%），"
               "量纲一致，分面比较；不共轴、不用双 Y 轴。"):
    _fig = alt.Chart(_sig).mark_line(
        strokeWidth=1.4, color=_tok["series"][0]).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y("sigma:Q", title="日波动率 σ（%）"),
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("路径:N", title="σ 路径"),
                 alt.Tooltip("sigma:Q", title="σ %", format=".4f")],
    ).properties(height=112, width=charts.facet_width(3, card_w=FULL_W)).facet(
        alt.Column("路径:N", title="",
                   sort=[labels.SIG_LABEL[c] for c in _sig_cols],
                   header=alt.Header(labelFontSize=10, labelColor=_tok["ink"])),
        columns=3,
    ).resolve_scale(y="independent")
    st.altair_chart(_fig, use_container_width=False)
    charts.data_drawer(
        _sig.assign(sigma=lambda d: d["sigma"].round(6)),
        key="mv_sig",
        title="数据抽屉 · 五条 σ 路径逐日值",
        columns=["date", "路径", "sigma"],
        labels={"date": "日期", "路径": "σ 路径", "sigma": "日波动率 %"},
        note="各格 y 轴独立：五条 σ 的量级与摆幅不同，共用一根轴会把变化小的"
             "那几条压成直线。sig_garch_t 是 t 似然估出来的 σ，本身是波动率，"
             "不是分位数。",
        fmt={"sigma": "%.6f"}, height=300,
    )

# ================================================= 3｜GARCH 边界解（满宽）
_traces = []
for _cal in ("主", "次"):
    _tr = data.garch_refit(_cal)[["date", "alpha", "beta", "persist", "boundary"]].copy()
    _tr["口径"] = labels.CALIBER_SHORT[_cal]
    _traces.append(_tr)
_trace = pd.concat(_traces, ignore_index=True).dropna(subset=["alpha"])
_trace = _trace.melt(id_vars=["date", "口径", "boundary"],
                     value_vars=["alpha", "beta", "persist"],
                     var_name="量", value_name="值")
_trace["量"] = _trace["量"].map({"alpha": "α", "beta": "β", "persist": "α + β"})

_bound_span = {"start": pd.Timestamp("2022-08-03"), "end": pd.Timestamp("2023-04-10")}
# 与检验那一行同一个道理：分面要求图层共用数据。这里把色带并进轨迹表、
# 用 kind 拆层。色带只画在次口径那一行——主口径 0 次边界解，划段反而是误导。
_trace["kind"] = "线"
_band_rows = pd.DataFrame([
    {"date": _bound_span["start"], "end": _bound_span["end"], "口径": "次", "量": m,
     "kind": "带"}
    for m in ("α", "β", "α + β")])
_g_all = pd.concat([_trace, _band_rows], ignore_index=True)
_gbase = alt.Chart(_g_all)
_band = _gbase.transform_filter(alt.datum.kind == "带").mark_rect(
    color=_tok["series"][1], opacity=0.14,
).encode(x=alt.X("date:T"), x2=alt.X2("end:T"),
         tooltip=[alt.Tooltip("date:T", title="边界解段起"),
                  alt.Tooltip("end:T", title="边界解段止")])
_ln = _gbase.transform_filter(alt.datum.kind == "线").mark_line(
    strokeWidth=1.4, color=_tok["series"][0],
).encode(
    x=alt.X("date:T", title=""),
    y=alt.Y("值:Q", title=""),
    tooltip=[alt.Tooltip("date:T", title="日期"),
             alt.Tooltip("口径:N"), alt.Tooltip("量:N"),
             alt.Tooltip("值:Q", format=".6f")],
)

with page.card("GARCH 边界解逐日轨迹：这张表是「次口径边界解」的唯一锚点",
               f"主口径与次口径并排；橙色带是文档 §3.1 记的边界解段 "
               f"{labels.GARCH_BOUNDARY_SPAN}。α、β、persist 三者量级不同，"
               "分面且各格独立纵轴。主口径 "
               f"{_g_main['n_boundary']} 次边界解、次口径 {_g_rmb['n_boundary']} 次，"
               "σ_t 在边界解段退化为常数，那一整段的条件化优势并不成立。"):
    _fig = (_band + _ln).properties(
        height=120, width=charts.facet_width(3, card_w=FULL_W)).facet(
        row=alt.Row("口径:N", title="", sort=["主", "次"],
                    header=alt.Header(labelFontSize=10, labelColor=_tok["ink"])),
        column=alt.Column("量:N", title="", sort=["α", "β", "α + β"],
                          header=alt.Header(labelFontSize=10,
                                            labelColor=_tok["ink"])),
    ).resolve_scale(y="independent")
    st.altair_chart(_fig, use_container_width=False)
    _gs = pd.DataFrame([
        {"口径": labels.CALIBER_SHORT[c],
         "边界解天数": k["n_boundary"], "重估总数": k["n_total"],
         "首次": k["start"].date() if k["start"] is not None else None,
         "末次": k["end"].date() if k["end"] is not None else None,
         "段内 β 均值": k["beta_mean"], "persist 峰值": k["persist_max"],
         "persist ≥ 1 天数": k["persist_ge1"],
         "β ≥ 0.999 天数": k["beta_ge_0999"]}
        for c, k in (("主", _g_main), ("次", _g_rmb))
    ])
    charts.data_drawer(
        _gs, key="mv_garch_sum",
        title="数据抽屉 · GARCH 边界解汇总（天数、跨度与段内极值）",
        columns=["口径", "边界解天数", "重估总数", "首次", "末次",
                 "段内 β 均值", "persist 峰值", "persist ≥ 1 天数",
                 "β ≥ 0.999 天数"],
        note=f"{labels.GARCH_BOUNDARY_SOURCE}。"
             f"{labels.GARCH_BOUNDARY_NOTE['主']}"
             f"{labels.GARCH_BOUNDARY_NOTE['次']}",
        fmt={"边界解天数": "%.0f", "重估总数": "%.0f", "段内 β 均值": "%.6f",
             "persist 峰值": "%.6f", "persist ≥ 1 天数": "%.0f",
             "β ≥ 0.999 天数": "%.0f"},
        height=140,
    )
    charts.data_drawer(
        _trace[["date", "口径", "量", "值", "boundary"]],
        key="mv_garch_tr",
        title="数据抽屉 · GARCH 逐日重估轨迹（α / β / persist）",
        columns=["date", "口径", "量", "值", "boundary"],
        fmt={"值": "%.6f"},
        labels={"date": "日期", "口径": "口径", "量": "参数", "值": "取值",
                "boundary": "是否边界解（α ≤ 0.001）"},
        height=340,
    )

# ================================================= 4｜HS 分位 ｜ FHS 对照
_hs = _vh[["date"] + [f"{_pre}hs{w}_var_{c}" for w in (250, 500, 750)
                      for c in ("95", "99")]].melt(
    id_vars="date", var_name="col", value_name="var_pct")
_hs["var_pct"] = pd.to_numeric(_hs["var_pct"], errors="coerce")
_hs = _hs.dropna(subset=["var_pct"])
_hs["窗长"] = _hs["col"].str.extract(r"hs(\d+)_")[0].map(lambda w: f"HS{w}")
_hs["水平"] = _hs["col"].str.extract(r"_var_(\d+)$")[0] + "% 检验水平"

_fhs_scope = labels.BASE_SCOPE
_fhs = _bt[(_bt["model"].isin(["HS250", "FHS-E", "FHS-G"]))
           & (_bt["caliber"] == caliber) & (_bt["scope"] == _fhs_scope)].copy()
_fhs["水平"] = _fhs["conf"].astype(str) + "%"

with page.card("历史模拟的分位路径：三个窗长各自的经验分位",
               f"{labels.CALIBER_FULL[caliber]}。窗越长越平滑、对近期冲击越钝，"
               "此图画正值以看量级。"):
    _fig = alt.Chart(_hs).mark_line(strokeWidth=charts.LINE_PX).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y("var_pct:Q", title="VaR（%，正值）"),
        color=alt.Color("窗长:N", title="", sort=["HS250", "HS500", "HS750"]),
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("窗长:N"), alt.Tooltip("水平:N"),
                 alt.Tooltip("var_pct:Q", title="VaR %", format=".4f")],
    ).properties(height=205, width=charts.facet_width(2)).facet(
        alt.Column("水平:N", title="", sort=["95% 检验水平", "99% 检验水平"],
                   header=alt.Header(labelFontSize=10, labelColor=_tok["ink"])),
        columns=2,
    ).resolve_scale(y="independent")
    st.altair_chart(_fig, use_container_width=False)
    charts.data_drawer(
        _hs, key="mv_hs",
        title="数据抽屉 · 三个窗长的 HS 分位逐日值",
        columns=["date", "窗长", "水平", "var_pct"],
        labels={"date": "日期", "窗长": "窗长", "水平": "检验水平",
                "var_pct": "VaR %"},
        note="VaR 以正值存，与收益同图时画成 −VaR。三个窗长是相邻形态，"
             "配色按固定次序取，不因为某个窗长被筛掉而给幸存序列换色。",
        fmt={"var_pct": "%.6f"}, height=300,
    )


with page.card("滤波历史模拟对照：滤波把 σ 的时变处理掉了，分位数没变",
               f"{labels.CALIBER_FULL[caliber]} · "
               f"{labels.SCOPE_LABEL[_fhs_scope]}。柱是真实违规率，"
               "红虚线是名义水平。"):
    _fig = alt.Chart(_fhs).mark_bar(
        size=charts.MAX_BAR_PX, cornerRadiusEnd=4).encode(
        x=alt.X("model:N", title="", sort=["HS250", "FHS-E", "FHS-G"],
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate_pct:Q", title="真实违规率（%）", scale=alt.Scale(zero=True)),
        color=alt.Color("水平:N", title="", sort=["95%", "99%"]),
        xOffset=alt.XOffset("水平:N", sort=["95%", "99%"]),
        tooltip=[alt.Tooltip("model:N", title="模型"), alt.Tooltip("水平:N"),
                 alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
                 alt.Tooltip("x:Q", title="违规次数"),
                 alt.Tooltip("T:Q", title="样本天数"),
                 alt.Tooltip("mean_var:Q", title="平均 VaR %", format=".4f")],
    )
    _nom = (
        alt.Chart(pd.DataFrame({"y": [5.0, 1.0], "t": ["名义 5%", "名义 1%"]}))
        .mark_rule(strokeWidth=1.5, strokeDash=[4, 3],
                   color=_tok["status"]["critical"])
        .encode(y="y:Q", tooltip=[alt.Tooltip("t:N"),
                                  alt.Tooltip("y:Q", format=".1f")])
    )
    st.altair_chart((_fig + _nom).properties(height=205), use_container_width=True)
    st.caption(labels.FHS_NOTE)
    charts.data_drawer(
        _fhs[["model", "conf", "T", "x", "rate_pct", "mean_var", "es", "es_ratio",
              "p_uc", "p_ind_mc_cond"]],
        key="mv_fhs",
        title="数据抽屉 · HS250 / FHS-E / FHS-G 覆盖率与检验对照",
        columns=["model", "conf", "T", "x", "rate_pct", "mean_var", "es",
                 "es_ratio", "p_uc", "p_ind_mc_cond"],
        fmt={"T": "%.0f", "x": "%.0f", "rate_pct": "%.6f", "mean_var": "%.6f",
             "es": "%.6f", "es_ratio": "%.4f", "p_uc": "%.6f",
             "p_ind_mc_cond": "%.6f"},
        labels={"model": "模型", "conf": "检验水平", "T": "样本天数",
                "x": "违规次数", "rate_pct": "真实违规率 %",
                "mean_var": "平均 VaR %", "es": "ES %", "es_ratio": "ES/VaR",
                "p_uc": "Kupiec p", "p_ind_mc_cond": "独立性条件 MC p"},
        height=220,
    )

page.footer(
    "results/backtest_results.csv、results/var_parametric.csv、"
    "results/var_historical.csv、results/garch_refit_trace_{main,rmb}.csv、"
    "results/backtest_exceptions.csv",
    "n00 / n01 / n10 / n11 / n_pairs、p_uc / p_ind / p_ind_mc / p_ind_mc_cond / p_cc、"
    "rate_pct / mean_var / es / es_ratio、sig_*、hs*_var_95/99、"
    "alpha / beta / persist；次口径列加 rmb_ 前缀",
)
