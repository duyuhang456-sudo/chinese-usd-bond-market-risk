"""主题三 · 压力测试交互分析。

30 个历史与假设情景重演一遍：哪个最狠、缓冲吃掉多少、库级覆盖够不够、因子贡献
排第几。缓冲档位在右上角 chip 里改，本段的倍数、超限条数与热图一起跟着变。

判定线是双轨的：报告登记线＝HS250 的 1 日 99% ES（主 0.6659% / 次 0.7024%），
超限名单只认这一条；试算线＝chip 里所选模型的 es_base_pct，99% 档时倍数改按这条
线现场除——情景损失不变，换的是分母。全部读既有 CSV，不重算任何情景损失。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, pipe, theme  # noqa: E402

page.init("压力测试交互分析")
# 两栏版面计划：每张卡 H（半栏）/ W（通栏），按卡片出现顺序写。
# 落在通栏卡之前的那张半栏卡标成 W，自己撑满一整行——否则右边空半屏。
page.grid_plan("H", "H", "H", "H", "H", "H", "W", "W", "W", "H", "H", "H", "H", "W")

page.section_head(
    "压力测试交互分析",
    n_cards=8,
    subtitle="30 个情景重演一遍：哪个最狠、缓冲吃掉多少、库级覆盖够不够、"
             "因子贡献排第几。",
)

_si = data.stress_impact()
_bs = data.stress_buffer_sens()
_sc = data.stress_scenarios()
_fc = data.stress_factor_contrib()
_tok = theme.tokens()

FAMILIES = ["利率上行", "信用利差走阔", "离岸汇率贬值"]
CLASSES = ["历史", "补充", "假设"]
SID_ORDER = list(dict.fromkeys(_si["scenario_id"]))

caliber = pipe.pipe_caliber()
model = pipe.pipe_model()
conf = pipe.pipe_conf()
buffer_label = pipe.pipe_buffer()
buf_value = pipe.buffer_value(buffer_label)

es_reg = labels.ES_1D_99[caliber]      # 报告登记线（tail_flag 以此为准）
es_sel = pipe.selected_es()            # 试算线（95% 档为 None）

_cur = _bs[(_bs["caliber"] == caliber) & (_bs["buffer_label"] == buffer_label)].copy()
_cur["x_reg"] = _cur["loss_net_pct"] / _cur["es_1d_99_pct"]
if es_sel is not None:
    _cur["x_sel"] = _cur["loss_net_pct"] / es_sel
    _x_field, _x_name = "x_sel", f"÷ {model} ES"
else:
    _cur["x_sel"] = pd.NA
    _x_field, _x_name = "x_reg", "判定线倍数"

_base = _si[_si["caliber"] == caliber].drop(
    columns=["loss_after_buffer_pct", "x_es99", "x_var99", "tail_flag"])
M = _base.merge(_cur[["scenario_id", "loss_net_pct", "x_reg", "x_sel", "is_tail"]],
                on="scenario_id", how="left")
M["buffer_now_pct"] = M["loss_net_pct"] - M["benchmark_loss_pct"]
M["类别"] = M["scenario_class"].map(labels.CLASS_LABEL)
if es_sel is not None:
    M["es_sel_pct"] = es_sel

_n_hit = int(_cur["is_tail"].sum())
_top = _cur.loc[_cur["loss_net_pct"].idxmax()]
_top_name = str(_si.loc[_si["scenario_id"] == _top["scenario_id"],
                         "scenario_name"].iloc[0])
_top_x = float(_top[_x_field])
_hist = _fc[_fc["factor"] == "residual"]
_hist = _hist[(_hist["caliber"] == caliber) & (_hist["scenario_class"] != "假设")]
_res = _hist.loc[_hist["share_pct"].idxmax()]
_es_denom = es_sel if es_sel is not None else es_reg

_cols = st.columns(4, gap="small")
for _col, (_label, _value, _sub) in zip(_cols, [
    ("当前档位下最极端情景是判定线的几倍", f"{_top_x:.2f} 倍",
     f"{_top['scenario_id']} {_top_name}，含缓冲净损失 {_top['loss_net_pct']:.4f}%。"),
    ("越过报告登记线的情景", f"{_n_hit} / {len(SID_ORDER)}",
     "tail_flag 名单只认报告登记线（HS250 ES），是表内登记字段，不随模型切换重算。"),
    ("缓冲已占到判定线的多少", f"{buf_value / _es_denom * 100:.1f} %",
     f"{buf_value:.4f} pp ÷ {_es_denom:.4f}%"
     + (f"（{model} ES，试算线）。" if es_sel is not None
        else "（报告登记线 HS250 ES；95% 档不换分母）。")),
    ("历史情景里未被因子解释的份额", f"{float(_res['share_pct']):.1f} %",
     f"{labels.CALIBER_SHORT[caliber]}下最高的一处是 {_res['scenario_id']}；"
     "残差是 δ 回归的拟合误差，不是真实信用风险贡献。"),
]):
    with _col:
        page.tile_kpi(_label, _value, sub=_sub)

st.caption(f"当前试算：{pipe.trial_meta(with_buffer=True)} · "
           "锚点日 2022-03-14 → 2025-04-07 · 损失以正值表示。")

K_BASE = "基准损失（未加缓冲）"
K_NET = f"加缓冲后（{buffer_label}）"
_order_sid = list(M.sort_values("loss_net_pct", ascending=False)["scenario_id"])
_y = alt.Y("scenario_id:N", title="", sort=_order_sid)


def _anchor_layers() -> list:
    """情景净损失双条 + 两条判定线锚点 + 条端倍数标签。"""
    _long = M.melt(id_vars=["scenario_id", "scenario_name"],
                   value_vars=["benchmark_loss_pct", "loss_net_pct"],
                   var_name="kind", value_name="loss")
    _long["kind"] = _long["kind"].map({"benchmark_loss_pct": K_BASE,
                                       "loss_net_pct": K_NET})
    bars = alt.Chart(_long).mark_bar(size=10, cornerRadiusEnd=4).encode(
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
    dia_reg = alt.Chart(M).mark_point(
        size=130, shape="diamond", filled=True, color=_tok["status"]["critical"],
        stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
        y=_y, x=alt.X("es_1d_99_pct:Q"),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("es_1d_99_pct:Q", title="登记线 HS250 ES %", format=".4f")],
    )
    layers = [bars, dia_reg]
    if es_sel is not None and abs(es_sel - es_reg) > 1e-9:
        layers.append(alt.Chart(M).mark_point(
            size=120, shape="triangle", filled=True, color=_tok["series"][0],
            stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
            y=_y, x=alt.X("es_sel_pct:Q"),
            tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                     alt.Tooltip("es_sel_pct:Q", title=f"{model} ES %", format=".4f")],
        ))
    layers.append(alt.Chart(M).mark_text(
        align="left", dx=6, fontSize=10.5, color=_tok["ink2"], font=_tok["font"]
    ).encode(y=_y, x=alt.X("loss_net_pct:Q"),
             text=alt.Text(f"{_x_field}:Q", format=".2f")))
    return layers


_DRAW_COLS = ["scenario_id", "scenario_name", "scenario_class", "direction", "severity",
              "anchor_date", "benchmark_loss_pct", "benchmark_source", "buffer_now_pct",
              "loss_net_pct", "es_1d_99_pct", "x_reg"]
_DRAW_LAB = {"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
             "direction": "方向", "severity": "档位", "anchor_date": "锚点日",
             "benchmark_loss_pct": "基准损失 %", "benchmark_source": "基准来源",
             "buffer_now_pct": "当前档位缓冲 pp", "loss_net_pct": "含缓冲净损失 %",
             "es_1d_99_pct": "登记线 %", "x_reg": "登记线倍数"}
_DRAW_FMT = {"benchmark_loss_pct": "%.4f", "buffer_now_pct": "%.4f",
             "loss_net_pct": "%.4f", "es_1d_99_pct": "%.4f", "x_reg": "%.3f"}
if es_sel is not None:
    _DRAW_COLS.insert(12, "x_sel")
    _DRAW_LAB["x_sel"] = f"÷{model} ES 倍数"
    _DRAW_FMT["x_sel"] = "%.3f"

# -------------------------------------------------- 3.1 旗舰下钻（锚点图 + 因子贡献）
# 情景条图与下钻面板合成一张 vconcat，而不是先印一张静态锚点图再印一遍带联动的。
# 早先拆成 3.1 / 3.2 两张卡时，同一张锚点图连着印了两遍，白占一屏。
_fac_order = [labels.FACTOR_CN[f] for f in labels.FACTOR_ORDER]
_det = _fc[_fc["caliber"] == caliber].copy()
# 选择态在同一张 vconcat spec 内共享，不经过 Streamlit rerun
_pick = alt.selection_point(
    name="pick_scenario", fields=["scenario_id"], on="click",
    toggle=False, clear=False, empty=False,
    value=[{"scenario_id": str(_top["scenario_id"])}])
_dy = alt.Y("factor_cn:N", title="", sort=_fac_order,
            axis=alt.Axis(labelLimit=0))   # 0 = 不截断，五条因子名全写出来
_dbars = alt.Chart(_det).mark_bar(size=12, cornerRadiusEnd=4).transform_filter(
    _pick
).encode(
    y=_dy,
    x=alt.X("share_pct:Q", title="该情景内因子贡献份额（%）",
            scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("factor_cn:N", title="", sort=_fac_order,
                    scale=alt.Scale(domain=_fac_order), legend=None),
    tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
             alt.Tooltip("scenario_name:N", title="情景"),
             alt.Tooltip("factor_cn:N", title="因子"),
             alt.Tooltip("contribution_pct:Q", title="贡献 pp", format=".4f"),
             alt.Tooltip("share_pct:Q", title="份额 %", format=".2f")],
)
_dlab = alt.Chart(_det).mark_text(
    align="left", dx=4, fontSize=10, color=_tok["ink2"], font=_tok["font"]
).transform_filter(_pick).encode(
    # 标签层不能写 axis=None：layer 合并轴定义时 null 会连 _dbars 那层的 y 轴
    # 一起抹掉，结果五根条全都没有名字，读者分不出哪根是哪个因子。
    y=_dy,
    x=alt.X("share_pct:Q"),
    text=alt.Text("share_pct:Q", format=".1f"),
)
_detail = (_dbars + _dlab).properties(height=28 * len(_fac_order) + 34)
_anchor = alt.layer(*_anchor_layers()).properties(height=330)

with page.card("3.1　旗舰下钻：点情景条看它的因子贡献"):
    st.altair_chart(
        alt.vconcat(_anchor, _detail, spacing=18).add_params(_pick).resolve_scale(
            x="independent", y="independent", color="independent"),
        use_container_width=True)
    if es_sel is not None and abs(es_sel - es_reg) > 1e-9:
        st.caption(f"点上方任一情景条，下方面板即时切换为该情景的因子贡献份额；"
                   f"默认是当前档位下最极端的情景。红菱形＝报告登记线"
                   f"（HS250 ES {es_reg:.4f}%，超限名单以此为准），蓝三角＝所选 {model} ES "
                   f"{es_sel:.4f}%，条端倍数按蓝三角算——情景损失不变，换的是分母。"
                   "这是全页唯一保留的点击联动。")
    elif es_sel is not None:
        st.caption(f"点上方任一情景条，下方面板即时切换为该情景的因子贡献份额；"
                   f"默认是当前档位下最极端的情景。菱形＝报告登记线"
                   f"（HS250 ES {es_reg:.4f}%），所选 {model} 与登记模型同为 "
                   f"{es_sel:.4f}%，倍数就是报告口径。这是全页唯一保留的点击联动。")
    else:
        st.caption(f"点上方任一情景条，下方面板即时切换为该情景的因子贡献份额；"
                   f"默认是当前档位下最极端的情景。95% 档没有登记 ES 判定线，"
                   f"菱形与倍数都只用报告登记线（HS250 ES {es_reg:.4f}%）。"
                   "这是全页唯一保留的点击联动。")
    charts.data_drawer(
        M.sort_values("loss_net_pct", ascending=False), key="3_impact", height=380,
        title="3.1 压力情景明细：30 个情景的基准、缓冲、倍数与实现值",
        columns=_DRAW_COLS + ["loss_realized_pct", "residual_pct", "worst_day_date",
                              "credit_range_lo_pct", "credit_range_hi_pct",
                              "credit_range_note"],
        labels={**_DRAW_LAB,
                "loss_realized_pct": "锚点窗累计实现 %", "residual_pct": "未解释残差 pp",
                "worst_day_date": "最差单日", "credit_range_lo_pct": "信用路径下沿 %",
                "credit_range_hi_pct": "信用路径上沿 %", "credit_range_note": "信用区间说明"},
        fmt={**_DRAW_FMT, "loss_realized_pct": "%.4f", "residual_pct": "%.4f",
             "credit_range_lo_pct": "%.4f", "credit_range_hi_pct": "%.4f"},
    )
    charts.data_drawer(
        _det[["scenario_id", "scenario_name", "scenario_class", "factor_cn",
              "contribution_pct", "share_pct"]],
        key="3_drill",
        title="3.1 下钻面板的底表：全部情景 × 因子贡献",
        height=300,
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "factor_cn": "因子", "contribution_pct": "贡献（pp）",
                "share_pct": "份额（%）"},
        fmt={"contribution_pct": "%.4f", "share_pct": "%.2f"},
        note="选择态不经过 Streamlit rerun，所以下方面板的 CSV 是全量底表——"
             "按上面选中的 scenario_id 过滤即可得到当前面板那 5 行。",
    )

# -------------------------------------------------- 3.3 缓冲五档热图
_hm = _bs[_bs["caliber"] == caliber].copy()
_hm["情景"] = _hm["scenario_id"] + " " + _hm["scenario_name"]
_hm_order = [f"{s} {n}" for s, n in
             _hm[["scenario_id", "scenario_name"]].drop_duplicates()
             .set_index("scenario_id").sort_index().itertuples()]
_hm["档位"] = _hm["buffer_label"]
_hm["情景"] = pd.Categorical(_hm["情景"], categories=_hm_order, ordered=True)
_hm = _hm.sort_values(["情景", "buffer_label"])

with page.card("3.2　缓冲五档热图：各情景净损失随档位变化"):
    st.altair_chart(
        charts.heatmap(
            _hm, x="档位", y="情景", v="loss_net_pct", height=285,
            x_title="", y_title="", v_title="净损失 %",
            x_axis=alt.Axis(labelAngle=0, labelLimit=180),
            y_axis=alt.Axis(labelOverlap=False, labelLimit=215),
        ),
        use_container_width=True,
    )
    charts.data_drawer(
        _hm, key="3_buf", title="3.2 缓冲五档：各情景净损失与尾部标记",
        height=340,
        note="历史情景与纯汇率情景的行整行不变色（前者按定义不加缓冲，后者主口径"
             "恒为 0、次口径不加缓冲）。双口径合计超限条数从完全不缓冲的 17 条逐档"
             "涨到 +1pp 压力档的 23 条——这正是改用「几倍 ES」连续读数作主导统计量的理由。",
        fmt={"buffer_value": "%.4f", "horizon_days": "%.0f",
             "loss_net_pct": "%.4f", "es_1d_99_pct": "%.4f"},
        labels={"buffer_label": "缓冲档位", "buffer_value": "缓冲取值 pp",
                "scenario_id": "编号", "scenario_name": "情景",
                "horizon_days": "期限（交易日）", "loss_net_pct": "净损失 %",
                "es_1d_99_pct": "判定线 %", "tail_flag": "是否超限"},
    )

# -------------------------------------------------- 3.4 超限名单
_vis = _cur[_cur["is_tail"]].sort_values("x_reg", ascending=False).copy()

with page.card("3.3　超限名单与实现值对照"):
    st.caption(f"当前档位（{buffer_label}）下越过报告登记线的 {len(_vis)} 条。")
    st.dataframe(
        _vis[["scenario_id", "scenario_name", "loss_net_pct",
              "es_1d_99_pct", "x_reg"]],
        width="stretch", hide_index=True, height=min(320, 36 * len(_vis) + 44),
        key="3_tail_list",
        column_config={
            "scenario_id": st.column_config.Column("编号", width="small"),
            "scenario_name": st.column_config.Column("情景"),
            "loss_net_pct": st.column_config.NumberColumn("含缓冲净损失 %", format="%.4f"),
            "es_1d_99_pct": st.column_config.NumberColumn("登记线 %", format="%.4f"),
            "x_reg": st.column_config.NumberColumn("登记线倍数", format="%.3f"),
        },
    )
    _hist_rows = M[M["scenario_class"].isin(["历史", "补充"])].copy()
    _hist_rows = _hist_rows.sort_values("loss_net_pct", ascending=False)
    charts.data_drawer(
        _hist_rows[["scenario_id", "scenario_name", "direction",
                    "benchmark_loss_pct", "loss_realized_pct", "residual_pct",
                    "worst_day_date", "loss_net_pct", "x_reg"]],
        key="3_hist", height=240,
        title="3.3 历史与补充情景：基准损失与窗内实际实现的差",
        labels={"scenario_id": "编号", "scenario_name": "情景", "direction": "方向",
                "benchmark_loss_pct": "基准损失 %", "loss_realized_pct": "锚点窗累计实现 %",
                "residual_pct": "未解释残差 pp", "worst_day_date": "最差单日",
                "loss_net_pct": "含缓冲净损失 %", "x_reg": "登记线倍数"},
        fmt={"benchmark_loss_pct": "%.4f", "loss_realized_pct": "%.4f",
             "residual_pct": "%.4f", "loss_net_pct": "%.4f", "x_reg": "%.3f"},
    )

# -------------------------------------------------- 3.5 库级覆盖
si_cal = _si[_si["caliber"] == caliber]
cross_ref = float(si_cal["es_hd_99_pct"].iloc[0])


def _vrefs(chart: alt.Chart, refs: list) -> alt.Chart:
    """若干竖直参照线，只画线不写字——标签写在图下读法行里。"""
    for v, _lab, col in refs:
        d = pd.DataFrame({"x": [float(v)]})
        chart = chart + alt.Chart(d).mark_rule(
            color=col, strokeWidth=1.5, strokeDash=[4, 3]).encode(x=alt.X("x:Q"))
    return chart


cov = data.stress_coverage()
cov = cov[cov["caliber"] == caliber].copy()
order = list(cov.sort_values("loss_pct", ascending=False)["scenario_id"])
cov["状态"] = cov["below_worst"].astype(str).map(
    {labels.FLAG_YES: "低于实测最差", "否": "高于实测最差"}).fillna("—")
worst = float(cov["hist_worst_pct"].iloc[0])

with page.card("3.4　库级覆盖：库内最深的那条够不够深"):
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
            (es_reg, f"判定线 {es_reg:.4f}%", _tok["status"]["critical"]),
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
        f"橙＝低于样本内实测最差单日（被超越），绿＝高于；单条情景低于实测最差是正常的。"
        f"三条竖线自左至右——判定线 {es_reg:.4f}%、交叉核对线 {cross_ref:.4f}%、"
        f"实测最差单日 {worst:.4f}%。"
    )
    charts.data_drawer(
        cov, key="3_cov", title="3.4 覆盖度：损失、历史分位与实测最差对照",
        height=320,
        note="情景侧与实测最差单日两侧同为单日量，是直接可比的同类比较；"
             "多期读数在 3.13、3.14 两张卡里，不参与判定。" + _cov_note,
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

# -------------------------------------------------- 3.6 因子贡献
_fcc = _fc[_fc["caliber"] == caliber].copy()
_fcc["情景"] = _fcc["scenario_id"] + " " + _fcc["scenario_name"]
_fcc["情景"] = pd.Categorical(
    _fcc["情景"],
    categories=[f"{sid} {nm}" for sid, nm in
                M.sort_values("loss_net_pct", ascending=False)[
                    ["scenario_id", "scenario_name"]].itertuples(index=False)],
    ordered=True,
)
_top_fc = _fcc[_fcc["scenario_id"] == _top["scenario_id"]].copy()

with page.card("3.5　因子贡献：逐情景份额与最狠情景的横向分解"):
    _c2 = st.columns([3, 2], gap="medium")
    with _c2[0]:
        st.altair_chart(
            charts.stacked_share(_fcc, cat="情景", seg="factor_cn", val="share_pct",
                                 seg_order=_fac_order, height=330),
            use_container_width=True,
        )
    with _c2[1]:
        _hb = alt.Chart(_top_fc).mark_bar(size=12, cornerRadiusEnd=4).encode(
            y=alt.Y("factor_cn:N", title="", sort=_fac_order),
            x=alt.X("share_pct:Q", title=f"{_top['scenario_id']} 份额（%）",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("factor_cn:N", title="", sort=_fac_order,
                            scale=alt.Scale(domain=_fac_order), legend=None),
            tooltip=[alt.Tooltip("factor_cn:N", title="因子"),
                     alt.Tooltip("contribution_pct:Q", title="贡献 pp", format=".4f"),
                     alt.Tooltip("share_pct:Q", title="份额 %", format=".2f")],
        )
        st.altair_chart(_hb.properties(
            height=28 * len(_fac_order) + 34), use_container_width=True)
        st.caption(f"左：30 个情景的份额堆叠；右：当前最狠的 {_top['scenario_id']} "
                   f"单情景横向分解。")
    charts.data_drawer(
        _fcc[["scenario_id", "scenario_name", "scenario_class", "factor_cn",
              "contribution_pct", "share_pct"]],
        key="3_contrib", height=340,
        title="3.5 因子贡献度：30 个情景 × 5 个因子的份额与贡献",
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "factor_cn": "因子", "contribution_pct": "贡献（pp）",
                "share_pct": "份额（%）"},
        fmt={"contribution_pct": "%.4f", "share_pct": "%.2f"},
    )

# -------------------------------------------------- 3.7 族内梯度
_grp = M[M["scenario_id"].str.match(r"^S\d+$") & M["direction"].isin(FAMILIES)].copy()
_grp["倍数"] = _grp["severity"].map(
    {labels.SEVERITY_ORDER[0]: 1.0, labels.SEVERITY_ORDER[1]: 1.5,
     labels.SEVERITY_ORDER[2]: 2.0})
_grp = _grp.sort_values(["direction", "倍数"])
_ymax = float(_grp["loss_net_pct"].max()) * 1.24
_grad_sel = alt.selection_point(fields=["direction"], bind="legend")

with page.card("3.6　族内梯度：轻 / 中 / 极端三档"):
    _grad = alt.Chart(_grp).mark_line(
        strokeWidth=charts.LINE_PX,
        point=alt.OverlayMarkDef(size=charts.MARKER_PX, filled=True,
                                 stroke=_tok["surface"],
                                 strokeWidth=charts.SURFACE_RING)).encode(
        x=alt.X("倍数:Q", title="相对本族轻度档的倍数",
                scale=alt.Scale(domain=[0.9, 2.15])),
        y=alt.Y("loss_net_pct:Q",
                title=f"含缓冲净损失（%）· {labels.CALIBER_SHORT[caliber]}",
                scale=alt.Scale(domain=[0, _ymax], zero=True)),
        color=alt.Color("direction:N", title="", sort=FAMILIES,
                        legend=alt.Legend(orient="bottom", direction="horizontal",
                                          columns=3, title=None, symbolType="circle"),
                        scale=alt.Scale(domain=FAMILIES, range=list(_tok["series"][:3]))),
        opacity=alt.condition(_grad_sel, alt.value(1), alt.value(0.2)),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("倍数:Q", title="相对轻度档", format=".2f"),
                 alt.Tooltip("loss_net_pct:Q", title="含缓冲净损失 %", format=".4f"),
                 alt.Tooltip("x_reg:Q", title="登记线倍数", format=".3f")],
    )
    _peak = _grp.loc[_grp["loss_net_pct"].idxmax()]
    _rule = charts.annotate(
        charts.add_rule(_grad.properties(height=300), es_reg),
        x=1.02, y=es_reg, text=f"登记线 {es_reg:.4f}%",
        dot=False, align="left", dx=0, dy=14,
    )
    st.altair_chart(
        charts.annotate(_rule,
                        x=float(_peak["倍数"]), y=float(_peak["loss_net_pct"]),
                        text=f"{_peak['scenario_id']} {_peak['loss_net_pct']:.4f}%，"
                             f"登记线的 {float(_peak['x_reg']):.2f} 倍",
                        align="right", dx=-8, dy=-14).add_params(_grad_sel),
        use_container_width=True,
    )
    st.caption("三个情景族 × 1× / 1.5× / 2× 三档；点图例只看其中一族。")
    charts.data_drawer(
        _grp[["scenario_id", "scenario_name", "direction", "severity", "倍数",
              "benchmark_loss_pct", "loss_net_pct", "x_reg"]],
        key="3_gradient",
        title="3.6 族内梯度：三个情景族 × 轻 / 中 / 极端三档的损失与倍数",
        labels={"scenario_id": "编号", "scenario_name": "情景", "direction": "族",
                "severity": "档位", "倍数": "相对轻度档",
                "benchmark_loss_pct": "基准损失 %",
                "loss_net_pct": "含缓冲净损失 %", "x_reg": "登记线倍数"},
        fmt={"倍数": "%.2f", "benchmark_loss_pct": "%.4f", "loss_net_pct": "%.4f",
             "x_reg": "%.3f"},
    )

# -------------------------------------------------- 3.8 锚点窗长
anc = data.stress_anchor_sens().copy()
anc = anc[anc["caliber"] == caliber].copy()
anc["锚点档"] = anc["is_main_horizon"].map({True: "登记档（1 日）", False: "备查档"})
long_a = anc.melt(id_vars=["direction", "window_days", "is_main_horizon", "caliber",
                           "anchor_date", "shock_sum"],
                  value_vars=["mild_1x_loss_pct", "extreme_2x_loss_pct"],
                  var_name="kind", value_name="loss")
long_a["kind"] = long_a["kind"].map({"mild_1x_loss_pct": "轻度档 1×",
                                     "extreme_2x_loss_pct": "极端档 2×"})
_anc_sel = alt.selection_point(fields=["kind"], bind="legend")

with page.card("3.7　锚点窗长：三方向 × 三档", wide=True):
    _la = alt.Chart(long_a).mark_line(strokeWidth=charts.LINE_PX).encode(
        x=alt.X("window_days:Q", title="锚点窗长（交易日）",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("loss:Q", title="损失（%）"),
        color=alt.Color("kind:N", title="", sort=["轻度档 1×", "极端档 2×"]),
        opacity=alt.condition(_anc_sel, alt.value(1), alt.value(0.2)),
        tooltip=[alt.Tooltip("direction:N", title="方向"),
                 alt.Tooltip("window_days:Q", title="窗长"),
                 alt.Tooltip("kind:N", title="档位"),
                 alt.Tooltip("loss:Q", title="损失 %", format=".4f"),
                 alt.Tooltip("shock_sum:Q", title="锚点冲击合计", format=".2f"),
                 alt.Tooltip("anchor_date:T", title="锚点日")],
    )
    _pa = _la.mark_point(size=charts.MARKER_PX, filled=True,
                         stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING)
    _ma = alt.Chart(long_a).mark_rule(
        color=_tok["status"]["critical"], strokeWidth=1.5, opacity=0.5,
    ).transform_filter(alt.datum.is_main_horizon).encode(x=alt.X("window_days:Q"))
    # 三个方向各画一张单图，并排三栏。分面只能定宽，窄视口下会压出卡片。
    _dirs = list(dict.fromkeys(long_a["direction"]))
    _d3 = st.columns(len(_dirs), gap="small")
    for _dc, _dname in zip(_d3, _dirs):
        with _dc:
            _dl = _la.transform_filter(alt.datum.direction == _dname)
            _dp = _pa.transform_filter(alt.datum.direction == _dname)
            _dm = _ma.transform_filter(alt.datum.direction == _dname)
            st.altair_chart(
                (_dl + _dp + _dm).add_params(_anc_sel).properties(
                    height=150, title=_dname),
                use_container_width=True)
    _rng = anc.groupby(["direction", "caliber"])["extreme_2x_loss_pct"].agg(["min", "max"])
    _rng["极差"] = (_rng["max"] - _rng["min"]).round(4)
    st.caption("红线标出登记档（1 日锚点）。三档之间 1 : 1.5 : 2 的倍率是需求文档"
               "直接规定的，线性施加在锚点上，所以变的是绝对水平、不是档间比例。")
    charts.data_drawer(
        anc, key="3_anchor",
        title="3.7 锚点窗长：三方向 × 三档的损失与锚点冲击",
        height=300,
        note="极端档（2×）在三类方向上的极差："
             + "、".join(f"{d}（{c}口径）{v:.4f}pp" for (d, c), v in _rng["极差"].items())
             + "。汇率方向摆动最大，假设情景的损失只能作量级参考。",
        fmt={"window_days": "%.0f", "shock_sum": "%.2f", "mild_1x_loss_pct": "%.4f",
             "extreme_2x_loss_pct": "%.4f", "registered_mild_pct": "%.4f",
             "registered_extreme_pct": "%.4f"},
        labels={"direction": "方向", "caliber": "口径", "window_days": "窗长",
                "anchor_date": "锚点日", "shock_sum": "锚点冲击合计",
                "mild_1x_loss_pct": "轻度 1× %", "extreme_2x_loss_pct": "极端 2× %",
                "registered_mild_pct": "登记轻度 %",
                "registered_extreme_pct": "登记极端 %", "锚点档": "档位"},
    )

# -------------------------------------------------- 3.9 双口径倍数
_tc = (_bs.groupby(["buffer_value", "buffer_label", "caliber"], as_index=False)["is_tail"]
       .sum().rename(columns={"is_tail": "n_tail"}))
_tot = (_tc.groupby(["buffer_value", "buffer_label"], as_index=False)["n_tail"]
        .sum().assign(caliber="合计"))
CURVE = pd.concat([_tc, _tot], ignore_index=True)
CURVE["口径"] = CURVE["caliber"].map(lambda c: labels.CALIBER_SHORT.get(c, c))
_order_cn = [labels.CALIBER_SHORT["主"], labels.CALIBER_SHORT["次"], "合计"]

with page.card("3.8　缓冲档位与超限条数（双口径）"):
    _lines = alt.Chart(CURVE[CURVE["口径"] != "合计"]).mark_line(
        strokeWidth=charts.LINE_PX,
        point=alt.OverlayMarkDef(size=charts.MARKER_PX, filled=True,
                                 stroke=_tok["surface"],
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
    _agg = alt.Chart(CURVE[CURVE["口径"] == "合计"]).mark_line(
        strokeWidth=3, color=_tok["ink2"], opacity=0.5).encode(
        x=alt.X("buffer_value:Q"), y=alt.Y("n_tail:Q"),
        tooltip=[alt.Tooltip("buffer_label:N", title="档位"),
                 alt.Tooltip("n_tail:Q", title="两口径合计条数")],
    )
    st.altair_chart(
        charts.add_rule((_lines + _agg).properties(height=290), buf_value,
                        label=f"当前档位 {buf_value:.4f} pp", orient="vertical"),
        use_container_width=True,
    )
    st.caption("竖线是 chip 里选中的缓冲档；灰线是两口径合计。")
    _piv = _tc.pivot_table(index=["buffer_value", "buffer_label"], columns="caliber",
                           values="n_tail")
    _piv["合计"] = _piv.sum(axis=1)
    _piv = _piv.rename(columns=labels.CALIBER_SHORT).reset_index()
    _piv.columns = ["缓冲（pp）", "档位", "主口径", "次口径", "合计"]
    for _c in ("主口径", "次口径", "合计"):
        _piv[_c] = _piv[_c].astype(int)
    charts.data_drawer(
        _piv, key="3_buffer", height=250,
        title="3.8 缓冲档位超限条数：5 个档位 × 主 / 次口径与合计",
    )

# ============================== 附录块：3.10 起，另起一个 st.title 分块
page.section_head(
    "情景库附录",
    n_cards=6,
    subtitle="前面八张卡回答「这轮压力测算读多少、算多少」；这一块回答"
             "「情景库本身是怎么建的、换个窗口会怎样」。",
)

# -------------------------------------------------- 3.10 情景库映射
_p1 = _sc[_sc["measure"] == "peak_1d"].copy()
_p1["类别"] = _p1["scenario_class"].map(labels.CLASS_LABEL)
_class_order = [labels.CLASS_LABEL[c] for c in CLASSES]

with page.card("3.9　情景库与冲击映射", wide=True):
    # 原来是按 unit 分成一格一格的分面，每格宽度得定死，窄视口就压出卡片。
    # 改成两栏各画一张单图，宽度交给卡片。
    _units = list(dict.fromkeys(_p1["unit"]))
    _ucols = st.columns(len(_units), gap="small")
    for _ui, (_ucol, _u) in enumerate(zip(_ucols, _units)):
        with _ucol:
            _ub = alt.Chart(_p1[_p1["unit"] == _u]).mark_bar(
                size=charts.MAX_BAR_PX, cornerRadiusEnd=4).encode(
                x=alt.X("scenario_id:N", title=f"单位：{_u}", sort=SID_ORDER),
                y=alt.Y("shock:Q", title="冲击（带符号）"),
                color=alt.Color("类别:N", title="", sort=_class_order,
                                scale=alt.Scale(domain=_class_order,
                                                range=list(_tok["series"][:3])),
                                legend=(alt.Legend(orient="top", direction="horizontal")
                                        if _ui == 0 else None)),
                tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                         alt.Tooltip("scenario_name:N", title="情景"),
                         alt.Tooltip("factor_cn:N", title="因子"),
                         alt.Tooltip("shock:Q", title="冲击", format=".4f"),
                         alt.Tooltip("unit:N", title="单位"),
                         alt.Tooltip("anchor_date:N", title="锚点日")],
            )
            st.altair_chart(_ub.properties(height=300), use_container_width=True)
    charts.data_drawer(
        _sc, key="3_mapping", height=340,
        title="3.9 情景库映射：情景 × 因子 × 计量口径的冲击登记",
        columns=["scenario_id", "scenario_name", "scenario_class", "factor_cn", "unit",
                 "measure", "shock", "multiple", "available", "anchor_date", "note"],
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "factor_cn": "因子", "unit": "单位", "measure": "计量口径",
                "shock": "冲击", "multiple": "倍数", "available": "数据可得",
                "anchor_date": "锚点日", "note": "备注"},
        fmt={"shock": "%.4f", "multiple": "%.2f"},
    )

# -------------------------------------------------- 3.11 信用路径区间
cr = si_cal[si_cal["credit_range_lo_pct"].notna()].copy()

with page.card("3.10　信用路径：给的是区间不是点"):
    if cr.empty:
        st.info("当前口径下没有带信用路径区间的情景。")
    else:
        _order_cr = list(cr.sort_values("credit_range_hi_pct",
                                        ascending=False)["scenario_id"])
        _yc = alt.Y("scenario_id:N", title="", sort=_order_cr)
        _rule_c = alt.Chart(cr).mark_rule(
            strokeWidth=6, color=_tok["muted"], opacity=0.40).encode(
            y=_yc,
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
            y=_yc, x=alt.X("worst_day_modeled_pct:Q"),
            tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                     alt.Tooltip("worst_day_modeled_pct:Q", title="δ 点估计 %",
                                 format=".4f")])
        _db = alt.Chart(cr).mark_point(
            size=110, filled=True, color=_tok["series"][1],
            stroke=_tok["surface"], strokeWidth=charts.SURFACE_RING).encode(
            y=_yc, x=alt.X("loss_after_buffer_pct:Q"),
            tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                     alt.Tooltip("loss_after_buffer_pct:Q", title="点估计 + 缓冲 %",
                                 format=".4f")])
        st.altair_chart(
            _vrefs(_rule_c + _dm + _db,
                   [(es_reg, f"判定线 {es_reg:.4f}%", _tok["status"]["critical"])]
                   ).properties(height=215),
            use_container_width=True,
        )
        _n_lo = int((cr["credit_range_lo_pct"] > es_reg).sum())
        _n_hi = int((cr["credit_range_hi_pct"] > es_reg).sum())
        st.caption(f"灰条＝信用路径区间，蓝点＝δ 点估计，橙点＝点估计加缓冲，"
                   f"红虚线＝判定线 {es_reg:.4f}%。{len(cr)} 条里下端越过判定线的 "
                   f"{_n_lo} 条、上端 {_n_hi} 条。上端越线不能读成「信用维度已经超过判定线」。")
        charts.data_drawer(
            cr, key="3_credit",
            title="3.10 信用路径：区间两端、点估计与缓冲",
            height=280,
            note="区间下端是平静样本的弱传导端（OLS 95% 置信区间的弱端，再加缓冲），"
                 "上端是弱端再加 2022 年未解释残差极值 1.1147pp（1 日窗）。"
                 "上端越线不能读成「信用维度已经超过判定线」；报告主张两个口径并列披露。",
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

# -------------------------------------------------- 3.12 阈值口径核对
si2 = si_cal.copy()
si2["挪动"] = si2["tail_flag"] != si2["tail_flag_hd"]
_order2 = list(si2.sort_values("loss_after_buffer_pct",
                                ascending=False)["scenario_id"])
_y2 = alt.Y("scenario_id:N", title="", sort=_order2,
            axis=alt.Axis(labelOverlap=False, labelFontSize=10))
_n_a = int((si_cal["tail_flag"] == labels.FLAG_YES).sum())
_n_b = int((si_cal["tail_flag_hd"] == labels.FLAG_YES).sum())

with page.card("3.11　阈值口径核对：名单长度变，头部不变"):
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
            (es_reg, f"主判定线 {es_reg:.4f}%", _tok["status"]["critical"]),
            (cross_ref, f"交叉核对线 {cross_ref:.4f}%", _tok["muted"]),
        ]).properties(height=240),
        use_container_width=True,
    )
    st.caption(
        f"{labels.CALIBER_SHORT[caliber]}：主判定线 {_n_a} / {len(si2)} 条，"
        f"交叉核对线 {_n_b} / {len(si2)} 条，换线移出 {_n_a - _n_b} 条；"
        f"两条线相差 {(cross_ref / es_reg - 1) * 100:.1f}%。"
    )
    charts.data_drawer(
        si2, key="3_thresh",
        title="3.11 阈值口径：两条判定线下的净损失与超限标记",
        height=320,
        note="名单长度对阈值口径敏感，名单头部不敏感（汇率情景 S07–S09 次口径与 S10 "
             "在两条线下都超限）。主判定线仅由 8（主）/ 7（次）个违规日支撑，"
             "样本极小，读数本身带有很大不确定性。",
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

# -------------------------------------------------- 3.13 窗口平移
ws = data.stress_window_sens()
ws = ws[ws["caliber"] == caliber].copy()
longw = ws.melt(id_vars=["scenario_id", "shift_days"],
                value_vars=["worst_day_pct", "mdd_pct"],
                var_name="kind", value_name="loss")
longw["kind"] = longw["kind"].map({"worst_day_pct": "窗内最差单日",
                                   "mdd_pct": "窗内最大回撤"})
_win_sel = alt.selection_point(fields=["scenario_id"], bind="legend")

with page.card("3.12　窗口起止平移：最差单日会随窗口移动改数"):
    _lines_w = alt.Chart(longw).mark_line(strokeWidth=charts.LINE_PX).encode(
        x=alt.X("shift_days:Q", title="窗口平移（交易日）",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("loss:Q", title="损失（%）"),
        color=alt.Color("scenario_id:N", title="",
                        scale=alt.Scale(domain=list(dict.fromkeys(ws["scenario_id"])))),
        opacity=alt.condition(_win_sel, alt.value(1), alt.value(0.2)),
        tooltip=[alt.Tooltip("scenario_id:N", title="情景"),
                 alt.Tooltip("shift_days:Q", title="平移"),
                 alt.Tooltip("loss:Q", title="损失 %", format=".4f")],
    )
    _dots_w = _lines_w.mark_point(size=charts.MARKER_PX, filled=True,
                                  stroke=_tok["surface"],
                                  strokeWidth=charts.SURFACE_RING)
    # 两行（最差单日 / 最大回撤）拆成上下两张单图。原先是一份按行分面的
    # 宽图——通栏 1400 宽、一屏高的十几分之一，两条线细得像发丝。
    for _ki, _kind in enumerate(["窗内最差单日", "窗内最大回撤"]):
        _kw = (_lines_w + _dots_w).transform_filter(
            alt.datum.kind == _kind).add_params(_win_sel, charts.zoom_x())
        st.altair_chart(
            _kw.properties(height=170, title=_kind), use_container_width=True)
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
    st.caption("最差单日极差都不超过各自的回撤极差：1 日口径比窗内最大回撤更稳健，"
               "但仍应给区间而不是单点值。")
    charts.data_drawer(
        tab.reset_index(), key="3_win",
        title="3.12 窗口平移：最差单日与回撤的五档区间",
        height=300,
        fmt={"期限": "%.0f", "最差单日基准": "%.4f", "最差单日区间下沿": "%.4f",
             "最差单日区间上沿": "%.4f", "最差单日极差": "%.4f",
             "回撤区间下沿": "%.4f", "回撤区间上沿": "%.4f", "回撤极差": "%.4f"},
    )

# -------------------------------------------------- 3.14 √t 缩放
sh = data.stress_shift_coverage()
_sqrt_sel = alt.selection_point(fields=["caliber"], bind="legend")

with page.card("3.13　附录：√t 缩放与多期最差窗"):
    _sl = alt.Chart(sh).mark_line(strokeWidth=charts.LINE_PX).encode(
        x=alt.X("horizon_days:Q", title="持有期（交易日）",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("sqrt_t_es99_ref_pct:Q", title="√t × 1 日 99% 期望损失（%）"),
        color=alt.Color("caliber:N", title="", scale=alt.Scale(domain=["主", "次"])),
        opacity=alt.condition(_sqrt_sel, alt.value(1), alt.value(0.2)),
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
    st.altair_chart((_sl + _sp).add_params(_sqrt_sel).properties(height=205),
                    use_container_width=True)
    _n_below = int((sh["sqrt_t_es99_ref_pct"] < sh["hist_worst_pct"]).sum())
    st.caption(
        f"√t 参考值在全部 {len(sh)} 个（期限 × 口径）组合上都低于同期限实测最差"
        f"（{_n_below} / {len(sh)}）：波动聚集的直接后果，不是随机误差。"
    )
    charts.data_drawer(
        sh, key="3_sqrt",
        title="3.13 √t 参考值：各期限的缩放对照与实测最差",
        height=260,
        note="本节的 1 日以外实测最差取自滚动窗内最差单日，聚集段里的连续下跌被 √t "
             "拉平了。本节与上面各节的 1 日基准不同量纲，不参与任何判定。",
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

# -------------------------------------------------- 3.15 各期限前十最差窗
ww = data.stress_worst_windows()
_hz = sorted(ww["horizon_days"].unique())

with page.card("3.14　附录：各期限前十最差窗", wide=True):
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
    # 每个持有期一张单图，两列一行地排。分面 spec 的顶层 width 是「每格」宽，
    # 只能定死；改成两栏单图之后宽度跟着卡片走，不会压出边框。
    for _k in range(0, len(_hz), 2):
        _rowc = st.columns(min(2, len(_hz) - _k), gap="small")
        for _ci, (_colh, _h) in enumerate(zip(_rowc, _hz[_k:_k + 2])):
            with _colh:
                _wbh = alt.Chart(ww[ww["horizon_days"] == _h]).mark_bar(
                    size=9, cornerRadiusEnd=4).encode(
                    y=alt.Y("rank:Q", title="排名（1 = 最差）",
                            axis=alt.Axis(tickMinStep=1),
                            scale=alt.Scale(reverse=True)),
                    x=alt.X("loss_pct:Q", title="窗内损失（%，正值 = 损失）"),
                    color=alt.Color("caliber:N", title="",
                                    scale=alt.Scale(domain=["主", "次"]),
                                    legend=(None if _k or _ci else alt.Legend(
                                        orient="top", direction="horizontal"))),
                    tooltip=[alt.Tooltip("caliber:N", title="口径"),
                             alt.Tooltip("rank:Q", title="排名"),
                             alt.Tooltip("window_start:T", title="窗起"),
                             alt.Tooltip("window_end:T", title="窗止"),
                             alt.Tooltip("loss_pct:Q", title="损失 %", format=".4f")],
                )
                st.altair_chart(
                    _wbh.properties(height=160, title=f"持有期 {_h} 个交易日"),
                    use_container_width=True)
    st.caption("各期限前十最差窗，两口径并排。1 日档（左起第一列）就是 3.5 那条"
               "实测最差单日的出处——主 2022-06-13 的 1.3449%、次 2022-09-29 的 1.4739%。")
    charts.data_drawer(
        ww, key="3_ww", title="3.14 最差窗明细：各期限前十窗口",
        height=300,
        fmt={"horizon_days": "%.0f", "rank": "%.0f", "loss_pct": "%.4f"},
        labels={"horizon_days": "期限", "is_main_horizon": "登记档", "caliber": "口径",
                "rank": "排名", "window_start": "窗起", "window_end": "窗止",
                "loss_pct": "损失 %", "caliber_note": "说明"},
    )

page.footer("results/stress_scenarios.csv、stress_buffer_sens.csv、stress_impact.csv、"
            "stress_factor_contrib.csv、stress_coverage.csv、stress_shift_coverage.csv、"
            "stress_window_sens.csv、stress_anchor_sens.csv、stress_worst_windows.csv、"
            "model_scorecard.csv")