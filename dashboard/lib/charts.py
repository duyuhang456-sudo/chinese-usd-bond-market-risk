"""可复用图元。

形状规范（dataviz skill，全看板统一）：
  线 2px；标记 ≥8px 且带 2px 底色描边；柱 ≤24px、数据端 4px 圆角；
  堆叠段之间留 2px 底色缝；面积填充 10%；网格实线发丝，虚线只留给真阈值线。
"""

from __future__ import annotations

import datetime
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from . import theme

MAX_BAR_PX = 24
LINE_PX = 2
MARKER_PX = 70          # 面积约 8–9px 直径
GAP_PX = 2
SURFACE_RING = 2

# 两栏布局里一栏的内容宽。按 1280 视口取（此时一栏约 470），宽屏自然留余量。
CARD_W = 470


def facet_width(columns: int, *, card_w: int = CARD_W, chrome: int = 40) -> int:
    """分面图每格的宽度——**分面图不能配 `use_container_width`**。

    对分面的 Vega-Lite spec，顶层的 `width` 是「每一格」的宽度，而 Streamlit 会把
    它撑成整个容器的宽度，于是 k 格的图画出 k 倍宽，从卡片里溢出去盖住右边那一栏。
    实测溢出 86～1579px，而且 Python 侧一声不响，只有渲染出来才看得见。

    所以每格宽度必须定死，并且 `st.altair_chart(..., use_container_width=False)`，
    交给内容定宽。chrome 是每格 y 轴刻度与留白的经验值。
    """
    return max(48, card_w // columns - chrome)


def _tok() -> dict:
    return theme.tokens()


def _color_scale(field: str, title: str, domain: list | None) -> alt.Color:
    """分类色阶。域固定成传入次序，子集筛选后幸存序列不换色。"""
    kw = {"domain": domain} if domain else {}
    return alt.Color(f"{field}:N", title=title, scale=alt.Scale(**kw))


# ------------------------------------------------------------------ 折线


def line_crosshair(
    df: pd.DataFrame,
    *,
    y: str,
    color: str | None = None,
    x: str = "date",
    tooltip: list | None = None,
    y_title: str = "",
    x_title: str = "",
    color_title: str = "",
    height: int = 320,
    color_domain: list | None = None,
    zero: bool = False,
) -> alt.Chart:
    """折线 + 竖直准星。一个 tooltip 列出该 X 上的所有序列，不必对准某条线。"""
    tok = _tok()
    x_enc = alt.X(f"{x}:T", title=x_title)
    y_enc = alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=zero))
    base = alt.Chart(df).encode(x=x_enc, y=y_enc)
    cenc = {"color": _color_scale(color, color_title, color_domain)} if color else {}

    line = base.mark_line(strokeWidth=LINE_PX).encode(**cenc)

    # 不可见的拾取层：读者瞄日期，不必瞄 2px 线
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)

    ring = dict(size=MARKER_PX, filled=True, stroke=tok["surface"], strokeWidth=SURFACE_RING)
    if color:
        pts = base.mark_point(**ring).encode(
            **cenc, opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
    else:
        pts = base.mark_point(**ring, color=tok["series"][0]).encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

    rule = (
        alt.Chart(df)
        .mark_rule(color=tok["axis"], strokeWidth=1)
        .encode(x=alt.X(f"{x}:T"),
                opacity=alt.condition(nearest, alt.value(0.55), alt.value(0)))
    )

    tips = tooltip or [alt.Tooltip(f"{x}:T", title="日期"),
                       alt.Tooltip(f"{y}:Q", title=y_title)]
    tt = base.mark_point(opacity=0).encode(tooltip=tips).transform_filter(nearest)

    # nearest 必须挂在**返回出去的那个图**上。挂在一个不进返回值的中间图层上，
    # Altair 会把 params 声明丢掉，只留下 condition 里对 param_1 的引用——
    # spec 校验不过，浏览器画一个空盒子，而 Python 侧一声不响（AppTest 查不出来）。
    return (rule + line + pts + tt).add_params(nearest).properties(height=height)


def add_rule(chart: alt.Chart, value: float, *, label: str = "", color: str | None = None,
             dashed: bool = True, orient: str = "horizontal") -> alt.Chart:
    """真阈值线——虚线只用在真正的阈值上，网格线仍是实线。"""
    tok = _tok()
    col = color or tok["status"]["critical"]
    dash = [4, 3] if dashed else []
    if orient == "horizontal":
        rule = alt.Chart(pd.DataFrame({"y": [value]})).mark_rule(
            color=col, strokeWidth=1.5, strokeDash=dash).encode(y="y:Q")
        if not label:
            return chart + rule
        txt = alt.Chart(pd.DataFrame({"y": [value], "t": [label]})).mark_text(
            align="left", dx=4, dy=-5, color=tok["ink2"], fontSize=10,
            font=tok["font"]).encode(y="y:Q", text="t:N")
        return chart + rule + txt
    rule = alt.Chart(pd.DataFrame({"x": [value]})).mark_rule(
        color=col, strokeWidth=1.5, strokeDash=dash).encode(x="x:Q")
    if not label:
        return chart + rule
    txt = alt.Chart(pd.DataFrame({"x": [value], "t": [label]})).mark_text(
        align="left", dx=4, dy=-6, color=tok["ink2"], fontSize=10,
        font=tok["font"]).encode(x="x:Q", text="t:N")
    return chart + rule + txt


def shade_bands(chart: alt.Chart, bands: pd.DataFrame, *, color: str | None = None,
                opacity: float = 0.13) -> alt.Chart:
    """背景色带（危机窗 / 高波动段）。bands 需含 start/end 两列。

    两类区间叠在同图上时给不同的 opacity：危机窗是窄而深的重点，
    长期高波动段是宽而浅的底色，靠深浅分开，不引入第二种色相。
    """
    if bands is None or bands.empty:
        return chart
    tok = _tok()
    enc = {"x": alt.X("start:T"), "x2": alt.X2("end:T")}
    if "label" in bands.columns:
        enc["tooltip"] = alt.Tooltip("label:N", title="区间")
    band = alt.Chart(bands).mark_rect(
        color=color or tok["series"][1], opacity=opacity).encode(**enc)
    return band + chart


def vline_labels(chart: alt.Chart, xs: list, *, color: str | None = None) -> alt.Chart:
    """若干竖直参照线（如历史 99% / 99.9% 分位）。"""
    tok = _tok()
    df = pd.DataFrame({"x": list(xs)})
    return chart + alt.Chart(df).mark_rule(
        color=color or tok["muted"], strokeWidth=1, strokeDash=[3, 3]).encode(x="x:Q")


# --------------------------------------------------- 点 + 区间（失败率）


def point_interval(
    df: pd.DataFrame,
    *,
    cat: str,
    val: str,
    lo: str,
    hi: str,
    nominal: float | None = None,
    height: int = 340,
    y_title: str = "",
    x_title: str = "",
    tooltip: list | None = None,
) -> alt.Chart:
    """点图 + 接受区间横杆。

    静态图用柱，柱从 0 起会把区间吃掉；这里改成点 + 区间，
    名义水平另画一条虚线参照。
    """
    tok = _tok()
    order = list(dict.fromkeys(df[cat]))
    x = alt.X(f"{cat}:N", title=x_title, sort=order, axis=alt.Axis(labelAngle=0))
    y = alt.Y(f"{val}:Q", title=y_title)

    rule = alt.Chart(df).mark_rule(strokeWidth=6, color=tok["muted"], opacity=0.35).encode(
        x=x, y=alt.Y(f"{lo}:Q", title=y_title), y2=alt.Y2(f"{hi}:Q"))
    dot = alt.Chart(df).mark_point(
        size=110, filled=True, color=tok["series"][0],
        stroke=tok["surface"], strokeWidth=SURFACE_RING).encode(
        x=x, y=y, tooltip=tooltip or [alt.Tooltip(f"{cat}:N", title="模型"),
                                      alt.Tooltip(f"{val}:Q", title="真实违规率 %"),
                                      alt.Tooltip(f"{lo}:Q", title="区间下沿 %"),
                                      alt.Tooltip(f"{hi}:Q", title="区间上沿 %")])
    chart = rule + dot
    if nominal is not None:
        chart = add_rule(chart, nominal, label=f"名义 {nominal:g}%（Kupiec 检验水平）")
    return chart.properties(height=height)


# --------------------------------------------------- 横向条（带阈值菱形）


def hbar_with_threshold(
    df: pd.DataFrame,
    *,
    cat: str,
    val: str,
    threshold: str | None = None,
    label: str | None = None,
    height: int | None = None,
    color: str | None = None,
    x_title: str = "",
    tooltip: list | None = None,
    sort_desc: bool = True,
) -> alt.Chart:
    """单序列横向条（≤24px、数据端 4px 圆角），可叠加阈值菱形。"""
    tok = _tok()
    order = "-x" if sort_desc else "x"
    x = alt.X(f"{val}:Q", title=x_title, scale=alt.Scale(zero=True))
    y = alt.Y(f"{cat}:N", title="", sort=order)
    chart = alt.Chart(df).mark_bar(size=MAX_BAR_PX, cornerRadiusEnd=4).encode(
        x=x, y=y, color=alt.value(color or tok["series"][0]),
        tooltip=tooltip or [alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q")])
    if label:
        chart = chart + alt.Chart(df).mark_text(
            align="left", dx=4, color=tok["ink2"], fontSize=10, font=tok["font"]
        ).encode(x=x, y=y, text=alt.Text(f"{label}:N"))
    if threshold:
        chart = chart + alt.Chart(df).mark_point(
            shape="diamond", size=130, filled=True, color=tok["status"]["critical"],
            stroke=tok["surface"], strokeWidth=SURFACE_RING).encode(
            x=alt.X(f"{threshold}:Q"), y=y,
            tooltip=[alt.Tooltip(f"{cat}:N"),
                     alt.Tooltip(f"{threshold}:Q", title="判定线 %")])
    h = height or max(160, 26 * len(df) + 60)
    return chart.properties(height=h)


# --------------------------------------------------- 100% 堆叠份额条


def stacked_share(
    df: pd.DataFrame,
    *,
    cat: str,
    seg: str,
    val: str,
    seg_order: list,
    seg_label: dict | None = None,
    height: int | None = None,
    x_title: str = "占比（%）",
) -> alt.Chart:
    """100% 堆叠横条。

    段间的缝用底色描边实现——描边色即底色，视觉上就是一个 2px 缝，
    不引入任何与底色有对比的墨，也不给每条段加边框。
    """
    tok = _tok()
    labels = seg_label or {}
    # 在图例里要显示中文名，最稳的做法是直接把段名换成显示名再按原次序排
    disp = [labels.get(s, s) for s in seg_order]
    plot = df.assign(**{seg: df[seg].map(lambda v: labels.get(v, v))})
    x = alt.X(f"{val}:Q", title=x_title, stack="zero", scale=alt.Scale(domain=[0, 100]))
    y = alt.Y(f"{cat}:N", title="", sort=list(dict.fromkeys(df[cat])))
    chart = alt.Chart(plot).mark_bar(
        stroke=tok["surface"], strokeWidth=GAP_PX
    ).encode(
        x=x, y=y,
        color=alt.Color(f"{seg}:N", title="", sort=disp, scale=alt.Scale(domain=disp)),
        order=alt.Order(f"{seg}:N", sort="ascending"),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{seg}:N", title="因子"),
                 alt.Tooltip(f"{val}:Q", title="份额 %", format=".2f")],
    )
    h = height or max(150, 30 * df[cat].nunique() + 60)
    return chart.properties(height=h).configure_axis(labelLimit=260)


# --------------------------------------------------- 散点


def scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    shape: str | None = None,
    tooltip: list | None = None,
    x_title: str = "",
    y_title: str = "",
    height: int = 340,
    color_domain: list | None = None,
    color_title: str = "",
    shape_domain: list | None = None,
    size: int = 130,
) -> alt.Chart:
    """散点。全对比形态，着色序列上限 3。"""
    tok = _tok()
    enc = {
        "x": alt.X(f"{x}:Q", title=x_title),
        "y": alt.Y(f"{y}:Q", title=y_title),
        "tooltip": tooltip or [alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q")],
    }
    if color:
        enc["color"] = _color_scale(color, color_title, color_domain)
    if shape:
        enc["shape"] = alt.Shape(
            f"{shape}:N", title="",
            scale=alt.Scale(domain=shape_domain) if shape_domain else alt.Scale())
    chart = alt.Chart(df).mark_point(
        size=size, filled=True, stroke=tok["surface"], strokeWidth=SURFACE_RING
    ).encode(**enc)
    return chart.properties(height=height)


def diagonal(chart: alt.Chart, lo: float, hi: float, *, label: str = "45°") -> alt.Chart:
    """45° 参照线。"""
    tok = _tok()
    ln = alt.Chart(pd.DataFrame({"a": [lo, hi]})).mark_line(
        color=tok["muted"], strokeWidth=1).encode(x="a:Q", y="a:Q")
    return chart + ln


# --------------------------------------------------- 热图


def heatmap(df: pd.DataFrame, *, x: str, y: str, v: str,
            x_title: str = "", y_title: str = "", height: int = 260,
            v_title: str = "", x_axis: alt.Axis | None = None,
            y_axis: alt.Axis | None = None) -> alt.Chart:
    """单色阶（蓝）热图——连续量级不用彩虹。

    色阶取**已注册的 BLUE_RAMP**，不用 Altair 内置的 `scheme="blues"`：
    内置色阶与令牌无关，换了底色它仍是原来的紫蓝，热图会与其余图分裂成两套观感。

    x_axis / y_axis 是给轴留的逃生口。热图的格数一多，Vega 的两条默认行为都会碍事：
    标签宽度加起来超过轴长时把整条 x 轴转 90°，以及标签太密时按「看不看得下」丢标签。
    行列一多就得自己钉 labelAngle 与 labelOverlap。
    """
    tok = _tok()
    return alt.Chart(df).mark_rect(stroke=tok["surface"], strokeWidth=GAP_PX).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=list(dict.fromkeys(df[x])), axis=x_axis),
        y=alt.Y(f"{y}:N", title=y_title, sort=list(dict.fromkeys(df[y])), axis=y_axis),
        color=alt.Color(f"{v}:Q", title=v_title,
                        scale=alt.Scale(range=list(tok["blue_ramp"]))),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:N"),
                 alt.Tooltip(f"{v}:Q", format=".3f")],
    ).properties(height=height)


# --------------------------------------------------- 直方图 + 正态参照


def hist_with_normal(s: pd.Series, *, bins: int = 40, height: int = 200,
                     title: str = "", x_title: str = "") -> alt.Chart:
    """分布小倍数的一格：直方图 + 同均值方差的正态参考线。"""
    tok = _tok()
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty or clean.std(ddof=1) == 0:
        return alt.Chart(pd.DataFrame({"t": ["无数据"]})).mark_text().encode(
            text="t:N").properties(height=height, title=title)
    counts, edges = np.histogram(clean.to_numpy(), bins=bins)
    hist = pd.DataFrame({"x0": edges[:-1], "x1": edges[1:], "n": counts})
    mu, sd = float(clean.mean()), float(clean.std(ddof=1))
    width = float(edges[1] - edges[0])
    xs = np.linspace(edges[0], edges[-1], 201)
    dens = np.exp(-((xs - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    norm = pd.DataFrame({"x": xs, "y": dens * len(clean) * width})
    bars = alt.Chart(hist).mark_bar().encode(
        x=alt.X("x0:Q", title=x_title, bin="binned"), x2="x1:Q",
        y=alt.Y("n:Q", title=""))
    line = alt.Chart(norm).mark_line(strokeWidth=LINE_PX, color=tok["series"][1]).encode(
        x=alt.X("x:Q"), y=alt.Y("y:Q"))
    return (bars + line).properties(height=height, title=title)


# --------------------------------------------------- 图上标注


def annotate(chart: alt.Chart, *, x, y, text: str, dx: float = 6, dy: float = -9,
             align: str = "left", color: str | None = None, dot: bool = True,
             size: int = 110, font_size: int = 10.5) -> alt.Chart:
    """把一个读数直接标在它发生的位置。

    读者不必自己从图里找那个点。x 给 Timestamp/date 就按时间轴对齐，给数字按数量轴；
    标注与点都用**文字色**，不穿序列色——文字一律用文字色是这套规范的硬要求。
    """
    tok = _tok()
    is_time = isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date,
                             np.datetime64))
    df = pd.DataFrame({"x": [x], "y": [y], "t": [text]})
    xe = alt.X("x:T") if is_time else alt.X("x:Q")
    col = color or tok["ink2"]
    out = chart + alt.Chart(df).mark_text(
        align=align, dx=dx, dy=dy, fontSize=font_size, fontWeight="bold",
        color=col, font=tok["font"],
    ).encode(x=xe, y=alt.Y("y:Q"), text="t:N")
    if dot:
        out = out + alt.Chart(df).mark_point(
            size=size, filled=True, color=col,
            stroke=tok["surface"], strokeWidth=SURFACE_RING,
        ).encode(x=xe, y=alt.Y("y:Q"))
    return out


# --------------------------------------------------- 数据抽屉（表格视图孪生）


def data_drawer(df: pd.DataFrame, *, key: str, title: str,
                columns: Any = None, labels: dict | None = None,
                fmt: dict | None = None, note: str = "",
                height: int = 300) -> None:
    """每张图的表格孪生——不悬停也能取到每个值。

    与此前那个「表格视图」的不同在于：**标题要说清这是哪张表**（此前 14 个折叠块
    全叫「表格视图」，读者展开前不知道会看到什么）、数字有格式（百分比 4 位、
    计数整数、p 值 4 位）、列名可改成人话。

    columns 给列名即按此顺序取列；labels 是列名到中文名的映射；fmt 是列名到 printf
    格式的映射（Streamlit 的 NumberColumn 认这种写法）。中文名走 column_config 的
    label，**不改 dataframe 自己的列名**——否则 column_config 的键就与列对不上了。
    """
    import streamlit as st

    show = df[list(columns)] if columns is not None else df
    labs, fmts = labels or {}, fmt or {}
    cfg = {}
    for col in show.columns:
        disp = labs.get(col, col)
        spec = fmts.get(col)
        if spec is not None:
            kw = {"format": spec} if isinstance(spec, str) else dict(spec)
            cfg[col] = st.column_config.NumberColumn(disp, **kw)
        elif disp != col:
            cfg[col] = st.column_config.Column(disp)
    with st.expander(title, expanded=False):
        if note:
            st.caption(note)
        st.dataframe(show, width="stretch", height=height, key=key,
                     column_config=cfg or None)
