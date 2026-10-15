"""页面骨架：页头、读数带、图卡、页脚。

骨架全部用 Streamlit 原生组件搭，**不注入任何 CSS**——外观只由
`.streamlit/config.toml` 的 `[theme]` 与 `lib/theme.py` 的 Altair 主题决定。
这是 Streamlit 官方设计规范的路子（design.md：「Do not use custom CSS to polish
an app unless the user explicitly asks for it」）：原生主题跨元素一致，且不会因为
Streamlit 换了内部 class 名就整块失效。

每页固定三段，不再有页内第二层筛选器：

1. `header()`   页头——面包屑、页标题、一句话定位、口径与数据区间。
2. `kpi_row()`  读数带——一排 `st.metric(border=True)`。标签写的是**判断**而不是
   名词，读者看标题那一行就知道这一页主张什么。
3. `card()`     图卡——`st.container(border=True)` + `st.subheader()`，卡里放图、
   一句读法、以及 `charts.data_drawer()` 数据抽屉。卡片本身就分节，页内不再另起标题。
4. `footer()`   页脚——数据来源与读取列。

页头与卡片标题都不带图标：`icon=` 是 Streamlit 开发版才有的参数，1.50 的
`st.title` / `st.subheader` 收不到它。
"""

from __future__ import annotations

from contextlib import contextmanager

from . import theme

APP_TITLE = "中资投资级美元债市场风险计量与压力测试"


def init(title: str, *, layout: str = "wide") -> None:
    """每个页面的第一步。

    注册 Altair 主题（幂等）。`set_page_config` 只允许调一次，在 `st.navigation`
    装配下由入口页负责；页面单独跑（例如 AppTest）时这里补上，重复调用被吞掉。
    """
    import streamlit as st

    try:
        st.set_page_config(page_title=f"{title} · {APP_TITLE}",
                           page_icon=":material/query_stats:", layout=layout)
    except Exception:
        pass
    theme.register()


def header(title: str, subtitle: str = "", *, crumb: str = "",
           caliber: str = "", span: str = "") -> None:
    """页头。

    subtitle 是一句话定位（这一页回答什么问题）；caliber 与 span 是两枚元信息，
    读者不必翻到页脚就知道本页读数建立在哪个口径、哪段区间上。
    """
    import streamlit as st

    if crumb:
        st.caption(crumb)
    st.title(title)
    if subtitle:
        st.markdown(subtitle)
    meta = [x for x in (
        f"口径 {caliber}" if caliber else "",
        f"数据区间 {span}" if span else "",
    ) if x]
    if meta:
        st.caption(" · ".join(meta))


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


@contextmanager
def card(title: str, note: str = "", *, stretch: bool = False):
    """一张图卡：白底圆角边框 + 卡片标题 + 一句读法。

    边框与圆角来自 `[theme]` 的 `borderColor` / `baseRadius`，不写 CSS。
    stretch=True 让同一行的卡片等高——一行里卡片内容长短不一时用它，
    否则矮的几张会在底部参差。
    """
    import streamlit as st

    kw = {"height": "stretch"} if stretch else {}
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
    body += "。本页只读既有产物，不重算任何数字。"
    st.caption(body)


# 旧名保留：页脚语义未变，只是换了呈现
origin_note = footer
