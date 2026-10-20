"""主题四 · 稳健性检验探索。

把推荐模型能站住的理由摊开：失败率落点、Christoffersen 四格、四个 p 值、
GARCH 重估轨迹、滤波与体制。判定词全部取自 model_scorecard.csv 的 verdict 列，
页面不另造规则。

本页不引入新选择，模型、置信度、汇率口径一律沿用右上角 chip 里的试算单。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, pipe, theme  # noqa: E402

page.init("稳健性检验探索")
# 两栏版面计划：每张卡 H（半栏）/ W（通栏），按卡片出现顺序写。
# 落在通栏卡之前的那张半栏卡标成 W，自己撑满一整行——否则右边空半屏。
page.grid_plan("W", "W", "W", "W", "W", "H", "H")

page.section_head(
    "稳健性检验探索",
    n_cards=7,
    subtitle="把失败率、Kupiec、Christoffersen、GARCH 重估、FHS-G 滤波全拿出来"
             "看一遍，确认推荐不是偶然。",
)

_tok = theme.tokens()

caliber = pipe.pipe_caliber()
conf = pipe.pipe_conf()
model = pipe.pipe_model()
ORDER10 = labels.MODEL_ORDER + [m for m in labels.NON_EVIDENCE_MODELS
                                if m not in labels.MODEL_ORDER]

_sc = pipe.scorecard_row()
_bt = pipe.backtest_row(labels.BASE_SCOPE)
_mlab = labels.MODEL_LABEL.get(model, model)
_verdict = str(_sc["verdict"])
_ok = _verdict == "推荐"


def _num(v, spec: str = ".4f") -> str:
    """NaN/None → 「未登记」，否则按 spec 格式化。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "未登记"
    return format(f, spec) if f == f else "未登记"


_cols = st.columns(4, gap="small")
for _col, (_label, _value, _sub) in zip(_cols, [
    (f"检验结论 · {_mlab}", _verdict,
     f"{int(_sc['n_pass_uc_core'])}/{int(_sc['n_core_scopes'])} 个核心范围"
     "通过 Kupiec，两道闸全过才进推荐。"),
    ("common773 真实违规率", f"{float(_bt['rate_pct']):.4f} %",
     f"实际违规 {int(_bt['x'])} 天 / {int(_bt['T'])} 天，期望 "
     f"{float(_bt['exp_x_eff']):.1f} 天。报告 p 列读的是它，不是名义水平。"),
    (f"名义水平（{conf}% 档）", f"{labels.NOMINAL_RATE[conf]:g} %",
     "Kupiec 检验的 H0：真实违规率等于名义水平。"),
    ("Kupiec p / 条件覆盖 p", f"{_num(_bt['p_uc'])} / {_num(_bt['p_cc'])}",
     "χ²(1) 临界值 3.8415、χ²(2) 临界值 5.9915。"),
]):
    with _col:
        page.tile_kpi(_label, _value, sub=_sub)

_bt_all = data.backtest_results()
_bonf = 0.05 / len(_bt_all)

# -------------------------------------------------- 4.1 失败率与接受区间
_sub = _bt_all[(_bt_all["caliber"] == caliber)
               & (_bt_all["conf"].astype(str) == conf)
               & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_order = [m for m in ORDER10 if m in set(_sub["model"])]
_sub["model"] = pd.Categorical(_sub["model"], categories=_order, ordered=True)
_sub = _sub.sort_values("model")
_sub["lo_pct"] = _sub["x_lo"] / _sub["T"] * 100
_sub["hi_pct"] = _sub["x_hi"] / _sub["T"] * 100
_sub["model_lab"] = _sub["model"].astype(str).map(
    lambda m: f"{m} 对照" if m in labels.NON_EVIDENCE_MODELS else m)
_nominal = labels.NOMINAL_RATE[conf]
_sel_row = _sub[_sub["model"].astype(str) == model]
_rest = _sub[_sub["model"].astype(str) != model]
_labs = list(_sub["model_lab"])

_rule = alt.Chart(_sub).mark_rule(
    strokeWidth=6, color=_tok["muted"], opacity=0.30,
).encode(
    x=alt.X("model_lab:N", title="", sort=_labs, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("lo_pct:Q", title="真实违规率（%）"), y2=alt.Y2("hi_pct:Q"),
)
_dot = alt.Chart(_rest).mark_point(
    size=95, filled=True, color=_tok["series"][0],
    stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING,
).encode(
    x=alt.X("model_lab:N", sort=_labs), y=alt.Y("rate_pct:Q"),
    tooltip=[alt.Tooltip("model:N", title="模型"),
             alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
             alt.Tooltip("p_uc:Q", title="Kupiec p", format=".4f")],
)
_sel_color = _tok["status"]["good"] if _ok else _tok["status"]["critical"]
_sel_dot = alt.Chart(_sel_row).mark_point(
    size=300, filled=True, color=_sel_color,
    stroke=_tok["surface"], strokeWidth=2.6,
).encode(
    x=alt.X("model_lab:N", sort=_labs), y=alt.Y("rate_pct:Q"),
    tooltip=[alt.Tooltip("model:N", title="模型"),
             alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
             alt.Tooltip("lo_pct:Q", title="区间下沿 %", format=".4f"),
             alt.Tooltip("hi_pct:Q", title="区间上沿 %", format=".4f"),
             alt.Tooltip("p_uc:Q", title="Kupiec p", format=".4f"),
             alt.Tooltip("p_cc:Q", title="Christoffersen p", format=".4f")],
)
# 名义线与审慎带只差 10%，两行字默认都锚在线上方 5px，会叠成一坨；
# 一条往上放、一条往下放。
_anchor = charts.add_rule(_rule + _dot + _sel_dot, _nominal,
                          label=f"名义 {_nominal:g}%", dashed=True, dy=13)
_anchor = charts.add_rule(_anchor, _nominal * labels.PRUDENCE,
                          label=f"审慎带 ×{labels.PRUDENCE:g}",
                          color=_tok["muted"], dashed=True, dy=-7)

with page.card("4.1　八个模型同一条检验线：大圈是当前试算模型"):
    st.altair_chart(_anchor.properties(height=300), use_container_width=True)
    st.caption(
        f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]} · {conf}% 档。"
        f"大圈＝{_mlab}，{'绿＝推荐' if _ok else '红＝淘汰'}；"
        "灰杆是 Kupiec 接受区间（不是误差棒），虚线上沿是审慎带。"
        "标「对照」的 M1δ 与 M1g-t 不参选。" + labels.KUPIEC_NOTE
    )
    charts.data_drawer(
        _sub[["model", "role", "T", "x", "rate_pct", "exp_x_eff", "lo_pct",
              "hi_pct", "p_uc", "p_ind_mc_cond", "p_cc", "es", "es_ratio"]],
        key="4_rate",
        title=f"4.1 {labels.SCOPE_LABEL[labels.BASE_SCOPE]}的失败率明细",
        height=280,
        labels={"model": "模型", "role": "角色", "T": "样本天数",
                "x": "违规次数", "rate_pct": "真实违规率 %",
                "exp_x_eff": "期望次数", "lo_pct": "区间下沿 %",
                "hi_pct": "区间上沿 %", "p_uc": "Kupiec p",
                "p_ind_mc_cond": "独立性 p（条件 MC）", "p_cc": "条件覆盖 p",
                "es": "ES %", "es_ratio": "ES/VaR"},
        fmt={"rate_pct": "%.4f", "exp_x_eff": "%.1f", "lo_pct": "%.4f",
             "hi_pct": "%.4f", "p_uc": "%.4f", "p_ind_mc_cond": "%.4f",
             "p_cc": "%.4f", "es": "%.4f", "es_ratio": "%.4f"},
        note="读 results/backtest_results.csv；接受区间由 x_lo / x_hi 除以 T 得到。",
    )

# -------------------------------------------------- 4.2 Christoffersen 四格
sel = _bt_all[(_bt_all["caliber"] == caliber)
              & (_bt_all["conf"].astype(str) == conf)
              & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_models = [m for m in ORDER10 if m in set(sel["model"])]

with page.card("4.2　Christoffersen 转移四格：每一格是计数不是比例", wide=True):
    _rows = []
    for _, rr in sel.iterrows():
        for i, j, key in ((0, 0, "n00"), (0, 1, "n01"),
                          (1, 0, "n10"), (1, 1, "n11")):
            _rows.append({
                "model": rr["model"],
                "前一日": "0 未违规" if i == 0 else "1 违规",
                "当日": "0 未违规" if j == 0 else "1 违规",
                "n": float(pd.to_numeric(rr[key], errors="coerce"))})
    _tt = pd.DataFrame(_rows)
    _thr = float(_tt["n"].max()) / 2.0
    _rect = alt.Chart(_tt).mark_rect(
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
    _lab = _rect.mark_text(fontSize=10, fontWeight="bold").encode(
        text=alt.Text("n:Q", format=".0f"),
        color=alt.condition(alt.datum.n > _thr, alt.value("#FFFFFF"),
                            alt.value(_tok["ink"])),
    )
    # 十个模型各画一张小四格，五行两排地铺。原来是按 model 分面的单张 spec，
    # 每格宽度得定死，视口一窄就压出卡片；拆成单图后跟着卡片伸缩。
    for _ri, _r0 in enumerate(range(0, len(_models), 5)):
        _row_models = _models[_r0:_r0 + 5]
        _lastrow = _r0 + 5 >= len(_models)
        _mrow = st.columns(len(_row_models), gap="small")
        for _ci, (_mc, _m) in enumerate(zip(_mrow, _row_models)):
            with _mc:
                _cm = _tt[_tt["model"] == _m]
                # 小倍数惯例：y 轴标签只在每行第一格、x 轴标签只在最后一行。
                # 十格各印一遍「0 未违规 / 1 违规」，字比格子还占地方。
                _r1 = alt.Chart(_cm).mark_rect(
                    stroke=_tok["surface"], strokeWidth=charts.GAP_PX).encode(
                    x=alt.X("前一日:N", title="", sort=["0 未违规", "1 违规"],
                            axis=alt.Axis(labelAngle=0, labelFontSize=9,
                                          labels=_lastrow)),
                    y=alt.Y("当日:N", title="", sort=["1 违规", "0 未违规"],
                            axis=alt.Axis(labelFontSize=9, labels=_ci == 0)),
                    color=alt.Color("n:Q", title="配对天数", legend=None,
                                    scale=alt.Scale(range=list(_tok["blue_ramp"]))),
                    tooltip=[alt.Tooltip("前一日:N"), alt.Tooltip("当日:N"),
                             alt.Tooltip("n:Q", title="配对天数", format=".0f")],
                )
                _l1 = _r1.mark_text(fontSize=10, fontWeight="bold").encode(
                    text=alt.Text("n:Q", format=".0f"),
                    color=alt.condition(alt.datum.n > _thr, alt.value("#FFFFFF"),
                                        alt.value(_tok["ink"])),
                )
                st.altair_chart(
                    (_r1 + _l1).properties(height=118, title=_m),
                    use_container_width=True)
    st.caption(f"{labels.CALIBER_FULL[caliber]} · {conf}% · "
               f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}。"
               "n_ij＝前一日状态 i、当日状态 j 的配对天数。")
    _probs = sel[["model", "n00", "n01", "n10", "n11", "n_pairs",
                  "n_pair_dropped"]].copy()
    _probs["p01_次日违规率%"] = pd.to_numeric(
        _probs["n01"], errors="coerce") / (
        pd.to_numeric(_probs["n00"], errors="coerce")
        + pd.to_numeric(_probs["n01"], errors="coerce")) * 100
    _probs["p10_次日回落率%"] = pd.to_numeric(
        _probs["n10"], errors="coerce") / (
        pd.to_numeric(_probs["n10"], errors="coerce")
        + pd.to_numeric(_probs["n11"], errors="coerce")) * 100
    charts.data_drawer(
        _probs, key="4_n",
        title="4.2 Christoffersen 转移四格计数与转移概率",
        columns=["model", "n00", "n01", "n10", "n11", "n_pairs",
                 "n_pair_dropped", "p01_次日违规率%", "p10_次日回落率%"],
        note=f"{labels.TRANSITION_NOTE}{labels.CHRISTOFFERSEN_NOTE}"
             "99% 档 n11 恒为 0，独立性检验在这种情况下几乎无从发力。",
        fmt={"n00": "%.0f", "n01": "%.0f", "n10": "%.0f", "n11": "%.0f",
             "n_pairs": "%.0f", "n_pair_dropped": "%.0f",
             "p01_次日违规率%": "%.4f", "p10_次日回落率%": "%.4f"},
        labels={"model": "模型", "n00": "n00（0→0）",
                "n01": "n01（0→1）", "n10": "n10（1→0）",
                "n11": "n11（1→1）", "n_pairs": "配对数 n_pairs",
                "n_pair_dropped": "丢弃的配对"},
        height=340,
    )

# -------------------------------------------------- 4.3 四个 p 值并列
TESTS = [("p_uc", "Kupiec 无条件覆盖"),
         ("p_ind", "独立性 · χ² 近似"),
         ("p_ind_mc_cond", "独立性 · 条件 MC"),
         ("p_cc", "条件覆盖（联合）")]

with page.card("4.3　四个 p 值并列：覆盖率、独立性两把尺、联合", wide=True):
    _p = sel[["model"] + [c for c, _ in TESTS]].melt(
        id_vars="model", var_name="test", value_name="p")
    _p["p"] = pd.to_numeric(_p["p"], errors="coerce")
    _p = _p.dropna(subset=["p"])
    _p["检验"] = _p["test"].map(dict(TESTS))
    _rules_meta = [(0.05, "名义 0.05"), (_bonf, f"Bonferroni {_bonf:.5f}")]
    _p["kind"] = "点"
    _rp = pd.DataFrame([
        {"model": None, "test": t, "p": v, "检验": cn, "kind": "线", "门槛": lab}
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
        color=alt.Color("门槛:N", title="",
                        scale=alt.Scale(
                            domain=[lab for _, lab in _rules_meta],
                            range=[_tok["muted"],
                                   _tok["status"]["critical"]])),
    )
    # 四个检验各一张单图，四栏并排。分面定宽在窄视口下会压出卡片。
    _t4 = st.columns(len(TESTS), gap="small")
    for _tc, (_tkey, _tname) in zip(_t4, TESTS):
        with _tc:
            _tp = _pts.transform_filter(alt.datum.test == _tkey)
            _tr = _rules.transform_filter(alt.datum.test == _tkey)
            st.altair_chart(
                (_tp + _tr).properties(height=150, title=_tname),
                use_container_width=True)
    st.caption(f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]} · {conf}% 档，横轴取对数；"
               "红虚线是全表 Bonferroni 门槛。")

    _hist_all = _bt_all[["conf", "p_ind_mc_cond", "n11"]].copy()
    _hist_all["p_ind_mc_cond"] = pd.to_numeric(
        _hist_all["p_ind_mc_cond"], errors="coerce")
    _hist_all = _hist_all.dropna(subset=["p_ind_mc_cond"])
    _hist_all["置信度"] = _hist_all["conf"].astype(str) + "%"
    _hist_all["b"] = (
        _hist_all["p_ind_mc_cond"] * 20).clip(0, 19.99).astype(int) / 20.0
    _hist = _hist_all.groupby(["置信度", "b"]).size().reset_index(name="行数")
    _hist["kind"] = "柱"
    _hr = pd.DataFrame([{"置信度": c, "b": _bonf, "行数": 0.0, "kind": "线"}
                        for c in ("95%", "99%")])
    _hbase = alt.Chart(pd.concat([_hist, _hr], ignore_index=True))
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
    ).encode(x="b:Q")
    # 95% / 99% 两档拆成两栏单图，不再是一张定宽分面。
    _conf2 = st.columns(2, gap="small")
    for _cc, _cf in zip(_conf2, ["95%", "99%"]):
        with _cc:
            _hb = alt.Chart(_hist[_hist["置信度"] == _cf]).mark_bar(
                size=charts.MAX_BAR_PX, cornerRadiusEnd=4,
                color=_tok["series"][0],
            ).encode(
                x=alt.X("b:Q", title="独立性条件 MC p（分箱 0.05）",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("行数:Q", title="行数"),
                tooltip=[alt.Tooltip("置信度:N"),
                         alt.Tooltip("b:Q", title="p 分箱下沿", format=".2f"),
                         alt.Tooltip("行数:Q", title="行数", format=".0f")],
            )
            _hl = alt.Chart(pd.DataFrame({"b": [_bonf]})).mark_rule(
                strokeWidth=1.5, strokeDash=[4, 3],
                color=_tok["status"]["critical"]).encode(x="b:Q")
            st.altair_chart(
                (_hb + _hl).properties(height=240, title=f"{_cf} 档"),
                use_container_width=True)
    st.caption(labels.TEST_POWER_NOTE)
    _det = sel[["model", "role", "T", "x", "exp_x", "exp_x_eff",
                "lr_uc", "p_uc", "lr_ind", "p_ind", "p_ind_mc",
                "p_ind_mc_cond", "lr_cc", "p_cc", "size_uc", "size_ind"]]
    charts.data_drawer(
        _det, key="4_tests",
        title="4.3 三个检验的统计量与 p 值",
        columns=["model", "role", "T", "x", "exp_x", "exp_x_eff",
                 "lr_uc", "p_uc", "lr_ind", "p_ind", "p_ind_mc",
                 "p_ind_mc_cond", "lr_cc", "p_cc", "size_uc", "size_ind"],
        note="χ²(1) 5% 临界值 3.8415，χ²(2) 5% 临界值 5.9915；"
             "文档明文以条件 MC p 为准。" + labels.P_TWO_KINDS_NOTE
             + labels.SIGNIFICANCE_NOTE,
        fmt={"T": "%.0f", "x": "%.0f", "exp_x": "%.2f",
             "exp_x_eff": "%.2f", "lr_uc": "%.4f", "lr_ind": "%.4f",
             "lr_cc": "%.4f", "p_uc": "%.6f", "p_ind": "%.6f",
             "p_ind_mc": "%.6f", "p_ind_mc_cond": "%.6f",
             "p_cc": "%.6f", "size_uc": "%.4f", "size_ind": "%.4f"},
        labels={"model": "模型", "role": "角色", "T": "样本天数",
                "x": "违规次数", "exp_x": "期望次数",
                "exp_x_eff": "期望次数（有效）", "lr_uc": "LR_uc",
                "p_uc": "p_uc", "lr_ind": "LR_ind",
                "p_ind": "p_ind（χ²）",
                "p_ind_mc": "p_ind（MC 无条件下）",
                "p_ind_mc_cond": "p_ind（条件 MC，以此为准）",
                "lr_cc": "LR_cc", "p_cc": "p_cc",
                "size_uc": "实际水平 无条件", "size_ind": "实际水平 独立性"},
        height=340,
    )
    charts.data_drawer(
        _hist_all[["置信度", "p_ind_mc_cond", "n11"]].assign(
            n11_为零=lambda d: d["n11"] == 0),
        key="4_power",
        title="4.3 检验力：独立性条件 MC p 与 n11 逐行",
        columns=["置信度", "p_ind_mc_cond", "n11", "n11_为零"],
        note="n11 是「连续两天违规」的配对数，99% 档里它全为 0。"
             + labels.TEST_POWER_NOTE + labels.MULTIPLE_COMPARISON_NOTE
             + "99% 档「未拒绝」是检验力问题，不是模型优劣的证据。",
        fmt={"p_ind_mc_cond": "%.6f", "n11": "%.0f"},
        labels={"p_ind_mc_cond": "独立性条件 MC p",
                "n11": "n11（1→1）", "n11_为零": "n11 是否为 0"},
        height=300,
    )

# -------------------------------------------------- 4.4 24 组通过 / 淘汰
_all_scope = _bt_all[(_bt_all["caliber"] == caliber)
                     & (_bt_all["conf"].astype(str) == conf)].copy()
_ms_all = data.model_scorecard()
_ms_all = _ms_all[(_ms_all["caliber"] == caliber)
                  & (_ms_all["conf"].astype(str) == conf)]
_vs = _all_scope[["model", "scope", "rate_pct", "T", "x"]].merge(
    _ms_all[["model", "verdict"]], on="model", how="left").copy()
_vs["范围"] = _vs["scope"].map(labels.SCOPE_LABEL)
_vs["pass"] = _vs["verdict"].map({"推荐": "推荐", "淘汰": "淘汰"}).fillna("对照")
_n_groups = len(_vs)

with page.card("4.4　全部回测范围上的通过与淘汰"):
    _fig = charts.scatter(
        _vs[_vs["pass"] != "对照"], x="rate_pct", y="范围", color="pass",
        color_domain=["推荐", "淘汰"], color_title="",
        x_title="真实违规率（%）", y_title="", height=280,
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("范围:N", title="回测范围"),
                 alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
                 alt.Tooltip("x:Q", title="违规次数"),
                 alt.Tooltip("T:Q", title="样本天数", format=".0f")],
    )
    _fig = charts.add_rule(_fig, _nominal,
                           label=f"名义 {_nominal:g}%", dashed=True, dy=13)
    _fig = charts.add_rule(_fig, _nominal * labels.PRUDENCE,
                           label=f"审慎带 ×{labels.PRUDENCE:g}",
                           color=_tok["muted"], dashed=True, dy=-6)
    st.altair_chart(_fig, use_container_width=True)
    st.caption(f"{conf}% 档 · {_n_groups} 个「模型 × 回测范围」组合。"
               "M1δ 与 M1g-t 是恒等式与分位对照，不画在这里。")

    bd = data.baseline_decision()
    bd = bd[bd["caliber"] == caliber].set_index("model").reindex(
        [m for m in ORDER10 if m in set(bd["model"])]).reset_index()
    st.caption("分档结论与理由直接读 baseline_decision.csv。")
    st.dataframe(
        bd[["model", "tier", "reason", "g1_pass_rate", "g2_prudent",
            "cal_dev_95", "cal_dev_99", "cap_idx_95", "cap_idx_99",
            "rate_95", "rate_99", "n_ind_max"]],
        width="stretch", height=200, hide_index=True,
        column_config={
            "model": st.column_config.Column("模型"),
            "tier": st.column_config.Column("结论"),
            "reason": st.column_config.Column("理由"),
            "g1_pass_rate": st.column_config.Column("过失败率闸"),
            "g2_prudent": st.column_config.Column("过审慎闸"),
            "cal_dev_95": st.column_config.NumberColumn(
                "校准偏差 95%", format="%.4f"),
            "cal_dev_99": st.column_config.NumberColumn(
                "校准偏差 99%", format="%.4f"),
            "cap_idx_95": st.column_config.NumberColumn(
                "资本指数 95%", format="%.4f"),
            "cap_idx_99": st.column_config.NumberColumn(
                "资本指数 99%", format="%.4f"),
            "rate_95": st.column_config.NumberColumn("违规率 95%", format="%.4f"),
            "rate_99": st.column_config.NumberColumn("违规率 99%", format="%.4f"),
            "n_ind_max": st.column_config.Column("独立性存疑（最大）"),
        },
    )
    _be_all = data.backtest_exceptions()
    _be_all = _be_all[(_be_all["caliber"] == caliber)
                      & (_be_all["conf"].astype(str) == conf)]
    _m1 = set(_be_all[_be_all["model"] == "M1"]["date"].astype(str))
    _m1d = set(_be_all[_be_all["model"] == "M1d"]["date"].astype(str))
    _gap = len(_m1 ^ _m1d)
    _vp = data.var_parametric()
    _r95 = (_vp["garch_t_var_95"] / _vp["sig_garch_t"]).dropna()
    _r99 = (_vp["garch_t_var_99"] / _vp["sig_garch_t"]).dropna()
    st.caption(f"M1δ 与 M1 的违规日集合对称差 {_gap} 天，是同一条线的两种写法；"
               f"M1g-t 的 VaR ÷ σ 恒为 {_r95.mean():.4f}（95%）与 "
               f"{_r99.mean():.4f}（99%），即正态分位。两者都不作独立证据。")
    charts.data_drawer(
        _vs[["model", "范围", "rate_pct", "T", "x", "verdict"]], key="4_groups",
        title=f"4.4 全部 {_n_groups} 组「模型 × 回测范围」的违规率与结论",
        fmt={"rate_pct": "%.4f", "T": "%.0f", "x": "%.0f"},
        labels={"model": "模型", "范围": "回测范围", "rate_pct": "真实违规率 %",
                "T": "样本天数", "x": "违规次数", "verdict": "结论"},
        note="verdict 取自 model_scorecard.csv，是两道闸全过的结果，不是本页判定。",
        height=300,
    )
    charts.data_drawer(
        bd[["model", "tier", "reason", "g1_pass_rate", "g2_prudent",
            "cal_dev_95", "cal_dev_99", "cap_idx_95", "cap_idx_99",
            "rate_95", "rate_99", "n_ind_max"]],
        key="4_baseline", title="4.4 定案表：分档结论与理由", height=260,
        fmt={"cal_dev_95": "%.4f", "cal_dev_99": "%.4f", "cap_idx_95": "%.4f",
             "cap_idx_99": "%.4f", "rate_95": "%.4f", "rate_99": "%.4f"},
        labels={"model": "模型", "tier": "结论", "reason": "理由",
                "g1_pass_rate": "过失败率闸", "g2_prudent": "过审慎闸",
                "cal_dev_95": "校准偏差 95%", "cal_dev_99": "校准偏差 99%",
                "cap_idx_95": "资本指数 95%", "cap_idx_99": "资本指数 99%",
                "rate_95": "违规率 95%", "rate_99": "违规率 99%",
                "n_ind_max": "独立性存疑（最大）"},
        note="读 results/baseline_decision.csv。",
    )

# -------------------------------------------------- 4.5 GARCH 边界解
_g_main = data.garch_boundary_summary("主")
_g_rmb = data.garch_boundary_summary("次")
_traces = []
for _cal in ("主", "次"):
    _tr = data.garch_refit(_cal)[["date", "alpha", "beta", "persist",
                                 "boundary"]].copy()
    _tr["口径"] = labels.CALIBER_SHORT[_cal]
    _traces.append(_tr)
_trace = pd.concat(_traces, ignore_index=True).dropna(subset=["alpha"])
_trace = _trace.melt(id_vars=["date", "口径", "boundary"],
                     value_vars=["alpha", "beta", "persist"],
                     var_name="量", value_name="值")
_trace["量"] = _trace["量"].map(
    {"alpha": "α", "beta": "β", "persist": "α + β"})
_band_rows = pd.DataFrame([
    {"date": pd.Timestamp("2022-08-03"),
     "end": pd.Timestamp("2023-04-10"), "口径": "次", "量": m}
    for m in ("α", "β", "α + β")])
_gdf = pd.concat([_trace.assign(kind="线"), _band_rows.assign(kind="带")],
                 ignore_index=True)
_gbase = alt.Chart(_gdf)
_band = _gbase.transform_filter(alt.datum.kind == "带").mark_rect(
    color=_tok["series"][1], opacity=0.14,
).encode(x=alt.X("date:T"), x2=alt.X2("end:T"))
_ln = _gbase.transform_filter(alt.datum.kind == "线").mark_line(
    strokeWidth=1.4, color=_tok["series"][0],
).encode(
    x=alt.X("date:T", title=""),
    y=alt.Y("值:Q", title=""),
    tooltip=[alt.Tooltip("date:T", title="日期"),
             alt.Tooltip("口径:N"), alt.Tooltip("量:N"),
             alt.Tooltip("值:Q", format=".6f")],
).add_params(charts.zoom_x())

with page.card("4.5　GARCH 边界解逐日轨迹", wide=True):
    # 2 口径 × 3 条量，共 6 格。原来是一张分面 spec，每格宽定死；现在铺成
    # 两排、每排三栏的单图，宽度跟着卡片走。
    for _cai in ["主", "次"]:
        _g3 = st.columns(3, gap="small")
        for _gc, _q in zip(_g3, ["α", "β", "α + β"]):
            with _gc:
                _gl = _gdf[(_gdf["kind"] == "线") & (_gdf["口径"] == _cai)
                           & (_gdf["量"] == _q)]
                _gb = _gdf[(_gdf["kind"] == "带") & (_gdf["口径"] == _cai)
                           & (_gdf["量"] == _q)]
                _band1 = alt.Chart(_gb).mark_rect(
                    color=_tok["series"][1], opacity=0.14,
                ).encode(x=alt.X("date:T"), x2=alt.X2("end:T"))
                _ln1 = alt.Chart(_gl).mark_line(
                    strokeWidth=1.4, color=_tok["series"][0],
                ).encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("值:Q", title=""),
                    tooltip=[alt.Tooltip("date:T", title="日期"),
                             alt.Tooltip("量:N"),
                             alt.Tooltip("值:Q", format=".6f")],
                ).add_params(charts.zoom_x())
                st.altair_chart(
                    (_band1 + _ln1).properties(
                        height=120,
                        title=f"{labels.CALIBER_SHORT[_cai]}口径 · {_q}"),
                    use_container_width=True)
    st.caption(f"主口径 {_g_main['n_boundary']} 次、次口径 {_g_rmb['n_boundary']} 次；"
               f"橙带是边界解段 {labels.GARCH_BOUNDARY_SPAN}，"
               "那段 α 落到 0，σ_t 退化成常数。")
    _gs = pd.DataFrame([
        {"口径": labels.CALIBER_SHORT[c], "边界解天数": k["n_boundary"],
         "重估总数": k["n_total"],
         "首次": k["start"].date() if k["start"] is not None else None,
         "末次": k["end"].date() if k["end"] is not None else None,
         "段内 β 均值": k["beta_mean"], "persist 峰值": k["persist_max"],
         "persist ≥ 1 天数": k["persist_ge1"], "β ≥ 0.999 天数": k["beta_ge_0999"]}
        for c, k in (("主", _g_main), ("次", _g_rmb))])
    charts.data_drawer(
        _gs, key="4_garch_sum",
        title="4.5 GARCH 边界解汇总（天数、跨度与段内极值）",
        columns=["口径", "边界解天数", "重估总数", "首次", "末次",
                 "段内 β 均值", "persist 峰值", "persist ≥ 1 天数",
                 "β ≥ 0.999 天数"],
        note=f"{labels.GARCH_BOUNDARY_SOURCE}。"
             f"{labels.GARCH_BOUNDARY_NOTE['主']}{labels.GARCH_BOUNDARY_NOTE['次']}",
        fmt={"边界解天数": "%.0f", "重估总数": "%.0f", "段内 β 均值": "%.6f",
             "persist 峰值": "%.6f", "persist ≥ 1 天数": "%.0f",
             "β ≥ 0.999 天数": "%.0f"},
        height=140,
    )
    charts.data_drawer(
        _trace[["date", "口径", "量", "值", "boundary"]],
        key="4_garch_tr",
        title="4.5 GARCH 逐日重估轨迹（α / β / persist）",
        columns=["date", "口径", "量", "值", "boundary"],
        fmt={"值": "%.6f"},
        labels={"date": "日期", "口径": "口径", "量": "参数", "值": "取值",
                "boundary": "是否边界解（α ≤ 0.001）"},
        height=340,
    )

# -------------------------------------------------- 4.6 滤波历史模拟
_fhs = _bt_all[(_bt_all["model"].isin(["HS250", "FHS-E", "FHS-G"]))
               & (_bt_all["caliber"] == caliber)
               & (_bt_all["scope"] == labels.BASE_SCOPE)].copy()
_fhs["水平"] = _fhs["conf"].astype(str) + "%"

with page.card("4.6　滤波历史模拟对照：滤波处理掉 σ 的时变，分位数没变"):
    _fig = alt.Chart(_fhs).mark_bar(
        size=charts.MAX_BAR_PX, cornerRadiusEnd=4).encode(
        x=alt.X("model:N", title="", sort=["HS250", "FHS-E", "FHS-G"],
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate_pct:Q", title="真实违规率（%）",
                scale=alt.Scale(zero=True)),
        color=alt.Color("水平:N", title="", sort=["95%", "99%"]),
        xOffset=alt.XOffset("水平:N", sort=["95%", "99%"]),
        tooltip=[alt.Tooltip("model:N", title="模型"),
                 alt.Tooltip("水平:N"),
                 alt.Tooltip("rate_pct:Q", title="真实违规率 %", format=".4f"),
                 alt.Tooltip("x:Q", title="违规次数"),
                 alt.Tooltip("mean_var:Q", title="平均 VaR %", format=".4f")],
    )
    _nom = alt.Chart(pd.DataFrame(
        {"y": [5.0, 1.0], "t": ["名义 5%", "名义 1%"]})).mark_rule(
        strokeWidth=1.5, strokeDash=[4, 3], color=_tok["status"]["critical"]
    ).encode(y="y:Q", tooltip=[alt.Tooltip("t:N"),
                               alt.Tooltip("y:Q", format=".1f")])
    _fig = charts.legend_filter(_fig, "水平")
    st.altair_chart((_fig + _nom).properties(height=220),
                    use_container_width=True)
    st.caption(f"{labels.CALIBER_FULL[caliber]} · "
               f"{labels.SCOPE_LABEL[labels.BASE_SCOPE]}，红虚线是名义水平。"
               + labels.FHS_NOTE)
    charts.data_drawer(
        _fhs[["model", "conf", "T", "x", "rate_pct", "mean_var", "es",
              "es_ratio", "p_uc", "p_ind_mc_cond"]],
        key="4_fhs",
        title="4.6 HS250 / FHS-E / FHS-G 覆盖率与检验对照",
        columns=["model", "conf", "T", "x", "rate_pct", "mean_var",
                 "es", "es_ratio", "p_uc", "p_ind_mc_cond"],
        fmt={"T": "%.0f", "x": "%.0f", "rate_pct": "%.6f",
             "mean_var": "%.6f", "es": "%.6f", "es_ratio": "%.4f",
             "p_uc": "%.6f", "p_ind_mc_cond": "%.6f"},
        labels={"model": "模型", "conf": "检验水平", "T": "样本天数",
                "x": "违规次数", "rate_pct": "真实违规率 %",
                "mean_var": "平均 VaR %", "es": "ES %", "es_ratio": "ES/VaR",
                "p_uc": "Kupiec p", "p_ind_mc_cond": "独立性条件 MC p"},
        height=220,
    )


# -------------------------------------------------- 4.7 体制期
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


_reg = data.regimes()
_rv_th = float(_reg["rv60_ann_%"].iloc[59:].quantile(2 / 3))
_n_high = int((_reg["regime"] == "高波动").sum())
_n_crisis = int(_reg["crisis_window"].sum())

with page.card("4.7　波动状态与回测区间：分段回测的分组依据"):
    _vol = charts.line_crosshair(
        _reg, y="rv60_ann_%", height=260, zero=True,
        y_title="60 日滚动年化波动（%）",
        tooltip=[alt.Tooltip("date:T", title="日期"),
                 alt.Tooltip("rv60_ann_%:Q", title="年化 σ %", format=".4f"),
                 alt.Tooltip("regime:N", title="状态")],
    )
    _vol = charts.shade_bands(_vol, _runs(_reg, "chronic_high"),
                              color=_tok["muted"], opacity=0.10)
    _vol = charts.shade_bands(_vol, _runs(_reg, "crisis_window"), opacity=0.18)
    _vol = charts.add_rule(_vol, _rv_th,
                           label=f"慢性高波动阈值 {_rv_th:.2f}%")
    st.altair_chart(_vol, use_container_width=True)
    st.caption("浅灰底＝慢性高波动段，橙底＝急性危机窗；该标志含当日滚动量，"
               "不是事前信号，危机窗样本外命中 0 天。")
    charts.data_drawer(
        _reg, key="4_regime",
        title="4.7 逐日滚动年化波动与状态标签",
        labels={"date": "日期", "rv60_ann_%": "60 日滚动年化 σ %",
                "chronic_high": "慢性高波动", "crisis_window": "急性危机窗",
                "regime": "状态"},
        fmt={"rv60_ann_%": "%.4f"}, height=280,
        note=f"阈值 {_rv_th:.4f}% 由页面按生成规则复算；急性危机窗 {_n_crisis} 天"
             "（2022-06-06 至 06-22）由 |日收益| ≥ 1.2% 自动触发；两段合起来"
             f"高波动共 {_n_high} 天，占样本 {_n_high / len(_reg) * 100:.0f}%。",
    )

page.footer(
    "results/model_scorecard.csv、baseline_decision.csv、backtest_results.csv、"
    "backtest_exceptions.csv、garch_refit_trace_{main,rmb}.csv、regimes.csv",
    "verdict / rate_base_pct / nominal_pct / cap_idx_vs_M1 / rate_pct / "
    "p_uc / p_ind_mc_cond / p_cc / n00-n11 / alpha / beta / persist",
)