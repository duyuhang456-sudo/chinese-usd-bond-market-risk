"""页面骨架：单页长文档·5 主题分组的容器与块状组件。

骨架全部用 Streamlit 原生组件搭，**不注入任何 CSS**——外观只由
`.streamlit/config.toml` 的 `[theme]` 与 `lib/theme.py` 的 Altair 主题决定。
这是 Streamlit 官方设计规范的路子（design.md：「Do not use custom CSS to polish
an app unless the user explicitly asks for it」）：原生主题跨元素一致，且不会因为
Streamlit 换了内部 class 名就整块失效。

整张看板是 `app.py` 装配的一根可滚长页，由以下块状组件拼装：

1. `trial_chip()`     右上 popover，封装 pipe 五控（口径/报价/模型/置信度/缓冲）；
                      默认显示为 chip，关掉后主区零控件
2. `section_head()`   主题色块头——caps 小标题 + 原生 st.title + 一句导读
3. `card()`           图卡——`st.container(border=True)` + `st.subheader()`，卡里
                      放图、一句读法、`charts.data_drawer()` 数据抽屉
4. `tile_kpi()`       参考图风格的 stat tile——一个大数字 + 一句说人话
5. `text_card()`      文本卡——序号 + 结论 + 数据锚点，用于「结论与边界」主题
6. `kpi_row()`        读数带，一排 `st.metric(border=True)`
7. `banner()`         页顶的期刊式 running head，整页只在最顶上打一条
8. `footer()`         数据来源页脚

去掉了上一轮 5 步管线的 `stepper / step_nav / PIPE_STEPS`：单页骨架不需要步骤条，
整张可滚动页本身就是导航。"""

from __future__ import annotations

from contextlib import contextmanager

from . import theme

APP_TITLE = "中资投资级美元债市场风险计量与压力测试"


def init(title: str, *, layout: str = "wide") -> None:
    """每个主题段的第一步。

    注册 Altair 主题（幂等）。`set_page_config` 全进程只允许调一次，由 app.py
    负责；单段单独跑（例如 tools/check_dashboard.py 的 AppTest）时这里补上，
    重复调用被吞掉。
    """
    import streamlit as st

    try:
        st.set_page_config(page_title=f"{title} · {APP_TITLE}",
                           page_icon=":material/query_stats:", layout=layout)
    except Exception:
        pass
    grid_reset()          # 每段从新的一行开始，不吃上一段剩下的半栏
    theme.register()


def banner(crumb: str, meta: str = "", *, height: int = 40) -> None:
    """期刊式 running head：顶端一条发丝横线，线下小号衬线——左步骤名、右运行口径。

    不写 CSS：用一张装饰性 Altair 图定位（横线一条 rect、文字走像素坐标）。
    crumb 用 ink 色加粗，meta 右贴用 muted。文字坐标里 x 走 0–1 数据域随容器伸缩。
    """
    import altair as alt
    import pandas as pd
    import streamlit as st

    from . import theme

    tok = theme.tokens()

    def _xdom() -> alt.X:
        return alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0.0, 1.0]))

    rule = (
        alt.Chart(pd.DataFrame({"x": [0.0], "x2": [1.0]}))
        .mark_rect()
        .encode(x=_xdom(), x2="x2:Q", y=alt.value(0), y2=alt.value(1),
                color=alt.value(tok["axis"]))
    )
    crumb_text = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_text(align="left", baseline="middle", fontSize=12.5, fontWeight=700,
                   font=tok["font"], color=tok["ink"])
        .encode(x=alt.value(2), y=alt.value(24), text=alt.value(crumb))
    )
    layers = [rule, crumb_text]
    if meta:
        meta_text = (
            alt.Chart(pd.DataFrame({"x": [1.0]}))
            .mark_text(align="right", baseline="middle", dx=-2, fontSize=11,
                       font=tok["font"], color=tok["muted"])
            .encode(x=_xdom(), y=alt.value(24), text=alt.value(meta))
        )
        layers.append(meta_text)
    st.altair_chart(
        alt.layer(*layers).properties(height=height, padding=0),
        use_container_width=True,
    )


def header(title: str, subtitle: str = "", *, crumb: str = "",
           caliber: str = "", span: str = "", meta: str = "") -> None:
    """页头：发丝横线 + 原生标题（保留函数名以便旧页签迁移期复用）。

    crumb 上眉带；meta 是眉带右侧的运行元信息裸串，没给时由 caliber / span 拼。
    subtitle 已废弃——一句定位不放在页头。
    """
    import streamlit as st

    if not meta:
        meta = " · ".join(x for x in (
            f"口径 {caliber}" if caliber else "",
            f"数据区间 {span}" if span else "",
        ) if x)
    if crumb:
        banner(crumb, meta)
    st.title(title)
    if subtitle:
        st.markdown(subtitle)


def section_head(title: str, *, subtitle: str = "", n_cards: int | None = None,
                 color: str | None = None, anchor: str | None = None) -> None:
    """主题块头：caps 小标题 + 原生 st.title + 一句导读。

    全部用 `st.markdown` / `st.title` 原生元素。**不画装饰图**——一个主题块只允许
    放真图（card 里那几张），免得读者分不清哪张是结论。也不做每块一色的色块头：
    零 CSS 的前提下能换色的只有 theme 主色一条，画不出八槽分类色，与其塞一条装饰
     rect 假装有主题色，不如让块头就是块头。

    anchor 不传时按标题去 `lib/toc.py` 的目录表查。Streamlit 自动生成的锚点会把
    中文剥空、退化成哈希，目录里的 `#中文标题` 因此跳不过去；显式 ASCII 锚点才跳得动。
    """
    import streamlit as st

    if anchor is None:
        from . import toc

        anchor = toc.anchor_for(title)

    # caps 小标题只在它比大标题多说一句话时才有意义（张数）。没有张数可报时
    # 那一行就是标题的原样重复——封面上会连着出现两个「本页一览」。
    if n_cards:
        st.markdown(f":gray[{title.upper()} · {n_cards} 张卡]")
    st.title(title, anchor=anchor)
    if subtitle:
        st.caption(subtitle)


def tile_kpi(label: str, value: str, *, sub: str = "") -> None:
    """参考图风格的 stat tile：一个判断 + 一个大数字 + 一句说人话。

    直接用原生 `st.metric(border=True)`：白卡、边框、数字字号都由
    `.streamlit/config.toml` 的 `[theme]` 决定，页面源码不注入任何 CSS。
    sub 进 `help`，鼠标悬停才展开，正文不堆字。
    """
    import streamlit as st

    st.metric(label, value, help=sub or None, border=True, height="stretch")


def text_card(title: str, body: str, *, anchor: str = "") -> None:
    """文本卡：序号 + 结论 + 数据锚点，零绘图成本，给主题五「结论与边界」用。

    视觉与 ref 图下方文字 tile 同——白卡 + caps 序号 + 一句话正文 + 行尾锚点。
    """
    import streamlit as st

    if _resolve_width(False):
        with st.container(border=True):
            st.markdown(f":gray[{title}]")
            st.markdown(body)
            if anchor:
                st.caption(anchor)
        return
    with _grid_slot():
        with st.container(border=True):
            st.markdown(f":gray[{title}]")
            st.markdown(body)
            if anchor:
                # 用 st.caption 而不是反引号包等宽：Streamlit 把内联 <code> 的颜色
                # 硬编码成亮绿，与这套深蓝学术色板不相干，零 CSS 也改不动。
                # 锚点要的就是「小一号的附注」这层意思，caption 正好，不必等宽。
                st.caption(anchor)



def trial_chip() -> None:
    """右上 popover：默认是只读 chip，pop 开后是 pipe 五控。

    主区零控件——读者改口径时主动点 chip。这条收紧上一轮「主区零控件，紧凑」的目标。
    """
    import streamlit as st

    from . import pipe

    meta = pipe.trial_meta()
    # meta 自己就以「试算」开头，别再套一个「试算 ·」，会读成「试算 · 试算 …」
    with st.popover(meta, use_container_width=False):
        pipe.controls_row()


# （旧 5 步骨架已下线：stepper / step_nav / PIPE_STEPS 不再保留。
# 单页长文档里没有"下一步"的逻辑位置，整张可滚动页本身就是路径。）


def kpi_row(items: list, *, per_row: int = 4) -> None:
    """读数带：一排读数卡。

    items 里每一项是 dict：`label`（这句判断在说什么，写成一句话而不是一个名词）、
    `value`（读数，含单位）、`delta`（可选的对照，**只在是真数值差时给**——非数值
    字符串会被 Streamlit 当成正数配绿箭头，语义就反了）、`help`（展开说明，鼠标悬停）。

    超过 per_row 项自动折行，不把一行挤到五列以上。
    """
    import streamlit as st

    items = list(items)
    if not items:
        return
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        for col, it in zip(st.columns(len(chunk), gap="small"), chunk):
            col.metric(it["label"], it["value"], delta=it.get("delta"),
                       help=it.get("help"), border=True, height="stretch")


# --- 两栏网格游标 -----------------------------------------------------------
# 整页是单栏通铺时，图会被 use_container_width 撑成 1500×320（4.7:1），
# 又扁又长，页高也一路堆到二十几屏。这里让连续的卡自动两两并排：每张图
# 收在约 700px 的栏里，比例回到 2:1 上下，页高大约减半。
# 需要通栏的卡（分面图每格定宽、hconcat 双图）显式标 wide=True，它自己起一行。
_GRID = {"cols": None, "n": 0}
# 版面计划：每张卡是半栏（H）还是通栏（W）。有它才能在「这一张要不要撑满」
# 的时候知道下一张是什么——否则一张半栏卡后面跟一张通栏卡，右边会空半屏。
_PLAN = {"items": [], "i": 0}


def grid_plan(*widths: str) -> None:
    """声明本段每张卡的宽度，按卡片出现顺序写 'H' / 'W'。

    计划由段文件自己维护；写少了就退回按 `wide=` 参数走，不会崩。
    手写的好处是「落单的半栏卡撑满」这件事在源码里看得见，不用猜。
    """
    _PLAN["items"] = [str(w).upper() for w in widths]
    _PLAN["i"] = 0
    grid_reset()


def grid_reset() -> None:
    """开新的一行。每个主题段开头由 init() 调一次，避免上一段剩下的半行串过来。"""
    _GRID["cols"] = None
    _GRID["n"] = 0


def _resolve_width(wide: bool) -> bool:
    """本张卡实际要不要通栏：计划里有就用计划的，没有就用调用处的 wide。"""
    items, i = _PLAN["items"], _PLAN["i"]
    if i < len(items):
        _PLAN["i"] = i + 1
        return items[i] == "W"
    return wide


def _grid_slot():
    """取本张卡要落的栏；满两栏就新起一行。"""
    import streamlit as st

    if _GRID["cols"] is None or _GRID["n"] >= 2:
        _GRID["cols"] = st.columns(2, gap="medium")
        _GRID["n"] = 0
    col = _GRID["cols"][_GRID["n"]]
    _GRID["n"] += 1
    return col


@contextmanager
def card(title: str, note: str = "", *, wide: bool = False, stretch: bool = False):
    """一张图卡：卡片底 + 发丝边框 + 卡片标题 + 一句读法。

    边框与圆角来自 `[theme]` 的 `borderColor` / `baseRadius`，不写 CSS。
    默认与上一张卡并排占半栏；`wide=True` 通栏独占一行，给分面图与双图并置用
    ——它们的每格宽度是定死的，塞进半栏会溢出。

    高度默认按内容收边，不拉平：同一行的两张卡常常一高一矮（3.1 的旗舰下钻有
    一屏高，旁边的热图只有一半），拉平之后矮的那张框里会空出四百像素，像没画完。
    """
    import streamlit as st

    wide = _resolve_width(wide)
    kw = {"height": "stretch"} if (stretch and not wide) else {}
    if wide:
        grid_reset()
        with st.container(border=True):
            st.subheader(title)
            if note:
                st.caption(note)
            yield
        return
    with _grid_slot():
        with st.container(border=True, **kw):
            st.subheader(title)
            if note:
                st.caption(note)
            yield


def caliber_note(text: str = "") -> None:
    """口径与符号方向的常驻提示。"""
    import streamlit as st

    from . import labels

    st.caption(f"口径与符号：损失以正值表示（正值 = 损失）；{labels.SIGN_NOTE_VAR}。{text}")


def footer(source: str, columns: str = "") -> None:
    """每页脚注：数据来自哪张表、读了哪几列。"""
    import streamlit as st

    body = f"数据来源：{source}"
    if columns:
        body += f"；读取列：{columns}"
    st.caption(body)


# 旧名保留：页脚语义未变，只是换了呈现
origin_note = footer
