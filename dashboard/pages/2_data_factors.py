"""数据与因子概览。

看板的第二页，回答「后面所有读数建立在什么数据上」。迁移自静态图的
descriptive_hist_qq_nav / portfolio_return_decomp_nav / tr_return_series /
spread_factor_vs_oas(9141) / spread_factor_vs_oas_3141。

本页最大的坑是口径：四个因子表列名大量重名而含义不同，所以页首显式选表，
不做隐式默认；量纲不同的序列一律拆成小倍数，不共用一根轴。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("数据与因子概览")

_tok = theme.tokens()
TABLES = list(labels.FACTOR_TABLES.keys())

# 页头在控制带之前渲染，跨度与口径得先知道当前选的是哪张表，从会话态里取上一轮的值
_cur_key = st.session_state.get("df_table") or TABLES[0]
_peek = data.factor_table(_cur_key)
_lo, _hi = _peek["date"].min().date(), _peek["date"].max().date()

page.header(
    "数据与因子概览",
    "先选因子表，再看因子形态、相关结构、三条收益口径与信用代理量。",
    crumb="阶段一 · 数据与因子",
    span=f"{_lo} → {_hi}",
    caliber=labels.FACTOR_TABLES[_cur_key].split("（")[0],
)

# ------------------------------------------------------------ 控制带（页内驱动）
# 选表驱动本页全部图；因子多选只作用于「因子序列」；日期区间作用于后三组。
with st.container(border=True):
    _tab = st.segmented_control(
        "因子表（口径）", TABLES, default=TABLES[0],
        format_func=lambda k: labels.FACTOR_TABLES[k].split("（")[0],
        key="df_table",
        help="四张表的 etf_ret_pct 含义不同：复权表含分红，除权表不含，"
             "市价表用二级市场报价。选表决定「因子序列」「因子分布」「信用维度」"
             "三卡读的是哪张表；相关阵、波动分解、收益口径与存疑日各有固定的列。",
    )
    table_key = _tab or TABLES[0]
    base = data.factor_table(table_key)

    factor_cols = [c for c in base.columns if c != "date"]
    pickable = ([c for c in ["d5y_bp", "d10y_bp", "oas_bp", "doas_bp", "fx_ret_pct"]
                 if c in factor_cols]
                + [c for c in factor_cols if c.startswith("etf_")])
    lo, hi = base["date"].min().date(), base["date"].max().date()

    _c1, _c2 = st.columns([3, 2])
    with _c1:
        picked = st.multiselect(
            "因子（小倍数，各自独立纵轴）", pickable,
            default=[c for c in ["d5y_bp", "d10y_bp", "fx_ret_pct"] if c in pickable],
            format_func=labels.series_title, key="df_pick",
            help="只作用于下面的「因子序列」一卡；其余各图有各自固定的列。",
        )
    with _c2:
        _rng_in = st.date_input("日期区间", value=(lo, hi), min_value=lo,
                                max_value=hi, key="df_range",
                                help="作用于因子序列、收益分解与信用维度三组图。")

# date_input 在选取过程中可能只回半个区间，退回到端点值而不是报错
if isinstance(_rng_in, tuple) and len(_rng_in) == 2:
    start, end = _rng_in
elif isinstance(_rng_in, tuple):
    start, end = _rng_in[0], hi
else:
    start, end = _rng_in, hi

tr_tbl = data.factor_table("factor_table_nav_tr")
nav_tbl = data.factor_table("factor_table_nav")

_n_rows = len(tr_tbl)
_n_doas = int(tr_tbl["doas_bp"].notna().sum())
_n_dd = len(data.doubtful_days())
_n_ex = int(tr_tbl["is_exdate"].sum())

page.kpi_row([
    {"label": "四个因子表共用一条日期轴", "value": f"{_n_rows} 天",
     "help": f"{lo} 起至 {hi}；后面各页的窗口都从这条轴切。"},
    {"label": "利率双因子高度同向，共线性由此而来", "value": "0.943",
     "help": "5 年期与 10 年期美债收益率变动高度同向，回归里靠正交化处理。"},
    {"label": "信用维度只有一部分样本有外部序列", "value": f"{_n_doas / _n_rows * 100:.1f} %",
     "help": f"OAS 自 2023-09-05 起才有值；ΔOAS 是它的一阶差分、首日无差分可算，"
             f"故只有 {_n_doas} 天有值。其余 {_n_rows - _n_doas} 天靠回归残差代理。"},
    {"label": "存疑日与除息日一律只标注不改数", "value": f"{_n_dd} / {_n_ex} 天",
     "help": "存疑交易日 / 除息日。稳健性检验会把存疑日整段剔除后回看结论是否翻转；"
             f"复权表按分红率还原这 {_n_ex} 次除息。"},
])

# ============================================================ 因子序列与分布
with page.card("因子序列（小倍数，各自独立纵轴）",
               "量纲不同，不共用一根轴。"):
    if not picked:
        st.info("先在页首的因子多选里选择至少一个因子。")
    else:
        window = data.slice_dates(base, start, end)
        inner = st.columns(min(len(picked), 3))
        _series_note = []
        for i, col in enumerate(picked):
            sub = window[["date", col]].dropna()
            with inner[i % len(inner)]:
                st.altair_chart(
                    charts.line_crosshair(
                        sub, y=col, height=240, y_title=labels.series_title(col),
                        zero=False,
                        tooltip=[alt.Tooltip("date:T", title="日期"),
                                 alt.Tooltip(f"{col}:Q", title=labels.series_title(col),
                                             format=".4f")],
                    ),
                    use_container_width=True,
                )
            _series_note.append(
                f"{labels.series_title(col)}；非空 {len(sub)} 天"
                + (f"（缺口 {len(window) - len(sub)} 天，画作断口不补零）"
                   if len(sub) < len(window) else ""))
        charts.data_drawer(
            window[["date"] + picked], key="df_series",
            title="数据抽屉 · 所选因子的逐日序列",
            labels={"date": "日期", **{c: labels.series_title(c) for c in picked}},
            note=(f"读 factors/{table_key}.csv，区间 {start} → {end}；缺口保留空值。"
                  + "".join(f"{t}。" for t in _series_note)),
            height=280,
        )

with page.card("因子分布（直方图 + 同均值方差的正态参照线）",
               "是否偏离正态由超额峰度与 JB 检验 p 值说话，不靠看图。"):
    _dist_cols = [c for c in ["d5y_bp", "d10y_bp", "fx_ret_pct"]
                  if c in base.columns]
    if "doas_bp" in base.columns and base["doas_bp"].notna().any():
        _dist_cols = _dist_cols + ["doas_bp"]
    if _dist_cols:
        inner2 = st.columns(min(len(_dist_cols), 4))
        for i, col in enumerate(_dist_cols):
            _title = labels.series_title(col)
            if col == "doas_bp":
                _title += f"（仅 {int(base['doas_bp'].notna().sum())} 天）"
            with inner2[i % len(inner2)]:
                st.altair_chart(
                    charts.hist_with_normal(base[col], bins=45, height=220,
                                            title=_title),
                    use_container_width=True,
                )
    else:
        st.caption("当前表没有可画的分布列。")
    charts.data_drawer(
        data.descriptive_stats_nav(), key="df_desc",
        title="数据抽屉 · 七条序列的描述统计",
        fmt={"均值(日)": "%.6f", "标准差(日)": "%.6f", "年化σ(%)": "%.4f",
             "偏度": "%.4f", "峰度(超额)": "%.4f", "JB-p": "%.3e"},
        height=280,
        note=("读 factors/descriptive_stats_nav.csv；窗口标注为序列的起始月。"
              "描述统计含均值、日标准差、年化 σ、偏度、超额峰度、JB 检验 p 值、"
              "极值与零收益占比，可点列头排序。注意年化 σ 那一列只对收益类序列"
              "有定义，bp 类因子为空是正常的。"),
    )

# ============================================== 相关结构与波动贡献

with page.card("因子相关阵（单色阶，深色 = 相关更强）",
               "利差代理与利率因子相关恰为 0，那是回归残差按定义正交，不是实证发现。"):
    _corr = data.factor_corr()
    st.altair_chart(
        charts.heatmap(_corr, x="factor_b", y="factor", v="corr", height=260,
                       x_title="", y_title="", v_title="相关系数"),
        use_container_width=True,
    )
    charts.data_drawer(
        _corr, key="df_corr", title="数据抽屉 · 因子相关阵（3 × 3）",
        labels={"factor": "因子", "factor_b": "对照因子", "corr": "相关系数"},
        fmt={"corr": "%.3f"}, height=220,
        note="读 results/factor_corr.csv；对角线恒为 1。",
    )


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
with page.card("波动来自哪一维（因子方差贡献份额）",
               "主口径没有汇率一行；次口径多出的那块汇率占 44.2%。"):
    st.altair_chart(
        charts.stacked_share(_attr, cat="口径", seg="因子", val="share_pct",
                             seg_order=_ATTR_ORDER, height=260,
                             x_title="方差贡献份额（%）"),
        use_container_width=True,
    )
    charts.data_drawer(
        _attr[["caliber_raw", "factor", "share_pct"]], key="df_attr", height=200,
        title="数据抽屉 · 因子波动贡献份额（两口径 × 三到四个因子）",
        labels={"caliber_raw": "口径", "factor": "因子", "share_pct": "份额（%）"},
        fmt={"share_pct": "%.4f"},
        note=("读 results/var_attribution.csv；主口径没有汇率一行"
              "（该口径下汇率不进分解）。美元口径下汇率变动被计价基准搬运掉了；"
              "人民币口径因此多出一块汇率、占 44.2%，「人民币视角下汇率是常态"
              "波动的主要来源」这个读数就出自这里。利差代理那一块是同一次回归的"
              "残差，是代理量、不是真实信用风险贡献；主口径整行为 100% 是把三块"
              "重新归一的结果，不能用它与次口径比绝对水平。"),
    )

# ============================================================ 三条收益口径
with page.card("NAV 除权口径的累计收益分解",
               "两条分项本该与合计重合：记账恒等式，不是「信用解释了这么多」。"):
    _decomp = nav_tbl.copy()
    for c in ["etf_ret_pct", "etf_rate_attrib_pct", "etf_spread_proxy_pct"]:
        _decomp["cum_" + c] = _decomp[c].cumsum()
    # 这个分解是记账恒等式：两条分项逐日精确等于合计，末端差只剩舍入。先把那个数
    # 算出来，免得读者把缺口读成「没解释掉的那一块」。
    _win = data.slice_dates(_decomp, start, end)
    _gap = abs(float((_win["etf_ret_pct"] - _win["etf_rate_attrib_pct"]
                      - _win["etf_spread_proxy_pct"]).sum()))
    _long = _decomp.melt(
        id_vars="date",
        value_vars=["cum_etf_ret_pct", "cum_etf_rate_attrib_pct",
                    "cum_etf_spread_proxy_pct"],
        var_name="series", value_name="cum_pct")
    _n_map = {"cum_etf_ret_pct": "组合收益（合计）",
              "cum_etf_rate_attrib_pct": "利率贡献",
              "cum_etf_spread_proxy_pct": "利差代理贡献"}
    _long["series"] = _long["series"].map(_n_map)
    _long = data.slice_dates(_long, start, end)
    st.altair_chart(
        charts.line_crosshair(
            _long, y="cum_pct", color="series", height=280,
            color_domain=list(_n_map.values()), color_title="",
            y_title="累计（%）", zero=True),
        use_container_width=True,
    )
    charts.data_drawer(
        _long, key="df_decomp", title="数据抽屉 · NAV 除权口径的累计收益分解",
        labels={"date": "日期", "series": "序列", "cum_pct": "累计（%）"},
        fmt={"cum_pct": "%.4f"},
        note=("累计值由页面从逐日收益累加得到（cumsum），不是单独落库的列。"
              "利率贡献与利差代理贡献逐日精确相加等于组合收益，当前区间的末端"
              f"两条分项之和与合计相差 {_gap:.4f} pp，那是四位小数的舍入而不是"
              "残差。别把重合读成「信用维度解释了这么多」。"),
    )

with page.card("三条收益口径：除权、复权、人民币",
               "除权不含分红，复权是主口径，人民币是次口径；▼ 是除息日。"):
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
    _tlong = data.slice_dates(_tlong, start, end)
    _base_ch = charts.line_crosshair(
        _tlong, y="cum_pct", color="series", height=280,
        color_domain=list(_tmap.values()), color_title="", y_title="累计（%）",
        zero=True)
    _exd = data.slice_dates(_trd[_trd["is_exdate"]], start, end)
    if not _exd.empty:
        _base_ch = _base_ch + alt.Chart(_exd).mark_point(
            shape="triangle-down", size=90, filled=True,
            color=_tok["series"][3]).encode(
            x=alt.X("date:T"), y=alt.Y("cum_etf_ret_tr_pct:Q"),
            tooltip=[alt.Tooltip("date:T", title="除息日"),
                     alt.Tooltip("div_yield_pct:Q", title="分红率 %", format=".4f")])
    st.altair_chart(_base_ch, use_container_width=True)
    charts.data_drawer(
        _trd[["date", "etf_ret_pct", "etf_ret_tr_pct", "etf_ret_rmb_pct",
              "is_exdate", "div_yield_pct"]],
        key="df_tr", title="数据抽屉 · 三条收益口径与除息日标记", height=280,
        labels={"date": "日期", "etf_ret_pct": "除权 %",
                "etf_ret_tr_pct": "复权主口径 %", "etf_ret_rmb_pct": "次口径 %",
                "is_exdate": "是否除息日", "div_yield_pct": "分红率 %"},
        fmt={"etf_ret_pct": "%.4f", "etf_ret_tr_pct": "%.4f",
             "etf_ret_rmb_pct": "%.4f", "div_yield_pct": "%.4f"},
        note=("逐日收益取原值（不累加），对应上面两条累计线的斜率。"
              "复权主口径与人民币次口径之间的差就是汇率那条线。除权与复权两条累计线"
              f"之间的缺口就是没计入的分红。全表共 {_n_ex} 个除息日"
              f"（当前区间内 {0 if _exd.empty else len(_exd)} 个）。"),
    )

# ============================================================== 信用维度
# 利差代理列随口径改名：复权表叫 _tr_pct，其余三张叫 _pct；含义相同但不能串
_proxy = "etf_spread_proxy_pct" if "etf_spread_proxy_pct" in base.columns \
    else "etf_spread_proxy_tr_pct"

# 这两张是同一件事的两面（代理量与 ΔOAS 的散点、两条累计），并排看更省版面

with page.card("信用维度：利差代理与滞后 ΔOAS",
               "信用因子自 2023-09 起才有外部序列，此前用回归残差代理。"):
    _sp = base[["date", _proxy, "doas_bp"]].dropna().copy()
    _sp["doas_lag1_bp"] = _sp["doas_bp"].shift(1)
    _sp = _sp.dropna()
    st.altair_chart(
        charts.scatter(
            _sp, x="doas_lag1_bp", y=_proxy, height=280,
            x_title="ΔOAS（滞后一日，bp）",
            y_title="利差代理收益（%）",
            tooltip=[alt.Tooltip("date:T", title="日期"),
                     alt.Tooltip("doas_lag1_bp:Q", title="ΔOAS(t−1) bp",
                                 format=".1f"),
                     alt.Tooltip(f"{_proxy}:Q", title="代理收益 %",
                                 format=".4f")]),
        use_container_width=True,
    )
    charts.data_drawer(
        _sp, key="df_sp",
        title="数据抽屉 · 利差代理收益与滞后一日 ΔOAS 的逐日对照",
        labels={"date": "日期", "doas_bp": "ΔOAS bp",
                "doas_lag1_bp": "ΔOAS 滞后一日 bp", _proxy: "利差代理收益 %"},
        fmt={"doas_bp": "%.1f", "doas_lag1_bp": "%.1f", _proxy: "%.4f"},
        note=("口径提示：NAV 口径取当日（lag 0），市价口径取滞后一日（lag 1），"
              "两者不可混用。滞后一日列由页面用 shift(1) 生成；两列都非空的行才"
              "进入散点。代理量的用途只有一个：在外部 ΔOAS 缺失的 "
              f"{_n_rows - _n_doas} 天里顶替信用维度，它和 ΔOAS 的等级相关程度"
              "决定了这段顶替有多可信。"),
    )


with page.card("利差代理与 ΔOAS 的累计对照",
               "两个序列量纲不同，纵向分面、不共轴。"):
    _cum = base[["date", _proxy, "doas_bp"]].copy()
    _cum["cum_proxy"] = _cum[_proxy].cumsum()
    _cum["cum_doas"] = _cum["doas_bp"].cumsum()
    _cl = _cum.melt(id_vars="date", value_vars=["cum_proxy", "cum_doas"],
                    var_name="series", value_name="cum")
    _cl["series"] = _cl["series"].map({"cum_proxy": "利差代理收益累计（%）",
                                       "cum_doas": "ΔOAS 累计（bp）"})
    _cl = data.slice_dates(_cl, start, end)
    _reg = data.benchmark_comparison()
    _rho_row = _reg[_reg["metric"].astype(str).str.contains("ρ", na=False)]
    st.altair_chart(
        alt.Chart(_cl).mark_line(strokeWidth=charts.LINE_PX).encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("cum:Q", title=""),
            color=alt.Color("series:N", title="",
                            scale=alt.Scale(domain=["利差代理收益累计（%）",
                                                    "ΔOAS 累计（bp）"])),
            tooltip=[alt.Tooltip("date:T", title="日期"),
                     alt.Tooltip("cum:Q", format=".2f")],
        ).properties(height=126, width=charts.facet_width(1)).facet(
            row=alt.Row("series:N", title="")
        ).resolve_scale(y="independent"),
        use_container_width=False,
    )
    if not _rho_row.empty:
        _r = _rho_row.iloc[0]
        st.caption(
            "登记值：9141.HK 利差代理 vs 滞后 ΔOAS 的 ρ = "
            f"{pd.to_numeric(_r['9141.HK'], errors='coerce'):.3f}。"
        )

# ====================================================== 存疑日与基准对照
_dd = data.doubtful_days()
with page.card("存疑交易日（仅标注，不改数）",
               f"共 {_n_dd} 个，报告里的稳健性检验会把它们整段剔除，回看结论是否翻转。"):
    st.dataframe(
        _dd[["series", "date", "value", "hit", "verdict", "handling",
             "event", "note", "special"]],
        width="stretch", height=240, hide_index=True,
        column_config={
            "series": st.column_config.Column("序列"),
            "date": st.column_config.Column("日期"),
            "value": st.column_config.NumberColumn("当日读数", format="%.4f"),
            "hit": st.column_config.Column("是否命中阈值"),
            "verdict": st.column_config.Column("判定"),
            "handling": st.column_config.Column("处理方式"),
            "event": st.column_config.Column("邻近事件"),
            "note": st.column_config.Column("说明"),
            "special": st.column_config.Column("备注"),
        },
    )
    charts.data_drawer(
        data.benchmark_comparison(), key="df_bench",
        title="数据抽屉 · 基准对照：9141.HK 与 3141.HK 的登记值",
        height=260,
        note="美元柜台与港币柜台的同标的两个上市份额。3141.HK 全程只作旁证，"
             "它的读数只用来印证 9141.HK 的序列没有走样，不进入因子回归、回测"
             "或压力测试的任何一步。",
    )

page.footer(f"factors/{table_key}.csv、results/factor_corr.csv、"
            "factors/descriptive_stats_nav.csv、results/var_attribution.csv、"
            "factors/factor_table_nav.csv、factors/factor_table_nav_tr.csv、"
            "results/doubtful_days.csv、factors/benchmark_comparison.csv",
            "因子列 / 相关系数 / 描述统计 / 波动分解 / 三条收益口径 / "
            "存疑日清单 / 双柜台对照")
