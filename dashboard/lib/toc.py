"""全页导航：右下角的「目录」胶囊与回到顶部圆钮。

这是整个看板**唯一**一处 CSS，理由是一条硬需求：悬浮件必须 `position: fixed`，
而 Streamlit 没有任何原生组件能钉在视口上。其余外观仍由
`.streamlit/config.toml` 的 `[theme]` 决定，图表外观由 `lib/theme.py` 的 Altair
主题决定，页面里其它地方一行样式都没有。

为什么不开 HTML 注入开关：那条禁令保持原样（见 `tools/check_dashboard.py` 的
`_FORBIDDEN_SRC` 第一条）。这里走的是 `st.html`——它不做 iframe，DOMPurify 的
html 白名单里 `<style>` `<details>` `<summary>` `<nav>` 都在，且不允许执行脚本。
校验工具里另加了一道闸，把 `st.html` 的调用收窄到本文件的唯一一处。

为什么不用侧边栏：侧边栏常驻要占掉约 330px 正文宽度，两栏卡片会被挤小、整页变高；
悬浮面板默认收起，不占正文也不占页高。

锚点为什么手写：Streamlit 1.50 自动生成的锚点只保留 `a-z0-9`，中文标题会被剥空并
退化成 xxhash（`数据与因子概览` → `9608beea`），`#中文标题` 跳不过去。所以正文标题
一律由 `page.section_head()` 按下面的表取 ASCII 锚点，`st.title(body, anchor=...)`
把它落到 `<h1 id="...">`；目录里的链接就是原生 `<a href="#...">`，浏览器自己滚，
不触发 rerun、不新开标签。表只此一份，正文与目录不会各写一套。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """目录里的一行，也是正文那个 h1 的锚点来源。"""

    anchor: str          # 正文 st.title 的 anchor，同时是 href="#…"
    no: str              # 面板里的编号；正文标题不带编号
    title: str           # 正文 h1 的原文，page.section_head 靠它查锚点
    count: str           # 这一段有多少内容，给读者一个长度预期
    hint: str            # 一句定位语，说这一段在回答什么
    sub: bool = False    # 是否是某段下面的附录（面板里缩进一级）


SECTIONS: tuple[Entry, ...] = (
    Entry("cover", "01", "本页一览", "4 数字", "项目主结论与锚点表"),
    Entry("data", "02", "数据与因子概览", "9 张", "报价、复权、因子拆分"),
    Entry("var", "03", "VaR 模型对比看板", "10 张", "八个模型逐日比一遍"),
    Entry("stress", "04", "压力测试交互分析", "8 张", "30 个情景与缓冲占比"),
    Entry("stress-appendix", "05", "情景库附录", "6 张", "情景库怎么建、换窗口", sub=True),
    Entry("robust", "06", "稳健性检验探索", "7 张", "失败率与三项检验"),
    Entry("verdict", "07", "结论与边界", "10 条", "八条结论、两条边界"),
)

_BY_TITLE = {e.title: e.anchor for e in SECTIONS}


def anchor_for(title: str) -> str | None:
    """按正文标题查锚点；不在表里的标题返回 None，交给 Streamlit 自动生成。"""
    return _BY_TITLE.get(title)


# 样式只作用于本文件写出的元素，不碰 Streamlit 的内部 class 与 testid，
# 这样 Streamlit 升级改了内部结构也不会整块失效。
_CSS = """
<style>
  .cnd-toc, .cnd-top {
    font-family: "Source Serif 4", Georgia, "Songti SC", "STSong", "SimSun", serif;
  }
  .cnd-toc { position: fixed; right: 18px; bottom: 78px; z-index: 1000; }
  .cnd-toc > summary {
    list-style: none; cursor: pointer; user-select: none;
    display: inline-flex; align-items: center; gap: 6px;
    padding: 9px 16px; border-radius: 999px;
    background: #101A2C; border: 1px solid #22314C; color: #D6DEEB;
    font-size: 13px; letter-spacing: 0.14em;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
  }
  .cnd-toc > summary::-webkit-details-marker { display: none; }
  .cnd-toc > summary:hover { border-color: #4C8DD6; color: #BFD6F0; }
  .cnd-toc[open] > summary {
    background: #16233A; border-color: #4C8DD6; color: #BFD6F0;
  }
  .cnd-toc nav {
    position: absolute; right: 0; bottom: calc(100% + 10px);
    width: 332px; max-height: 68vh; overflow: auto;
    background: #101A2C; border: 1px solid #22314C; border-radius: 10px;
    padding: 6px; box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
  }
  .cnd-head {
    padding: 8px 10px 7px; margin-bottom: 4px;
    border-bottom: 1px solid #22314C;
    color: #6B7891; font-size: 11.5px; letter-spacing: 0.14em;
  }
  .cnd-row {
    display: grid; grid-template-columns: 2.2em 1fr auto; column-gap: 0.5em;
    padding: 7px 10px; border-radius: 6px;
    color: #D6DEEB; text-decoration: none; line-height: 1.35;
  }
  .cnd-row:hover { background: #16233A; }
  /* 附录缩进一级，靠留白和浅一档的字色区分，不靠制表符——箱线字符在衬线字体里
     会渲染成一个像 "L" 的怪东西。 */
  .cnd-sub .cnd-title { padding-left: 1.1em; color: #BFD6F0; }
  .cnd-sub .cnd-no { opacity: 0.75; }
  .cnd-no { color: #6B7891; font-size: 12px; }
  .cnd-count { color: #6B7891; font-size: 11.5px; white-space: nowrap; }
  .cnd-hint {
    grid-column: 2 / 4; margin-top: 2px;
    color: #94A2BA; font-size: 12px;
  }
  #pagetop {
    /* 锚点在正文容器的上内边距之内，直接滚过来会停在 padding 下沿、把 running head
       顶到固定页头底下。留一个比视口还高的 scroll-margin，落点一律夹到 scrollTop=0。 */
    scroll-margin-top: 100vh;
  }
  .cnd-top {
    position: fixed; right: 18px; bottom: 18px; z-index: 1000;
    width: 44px; height: 44px; border-radius: 50%;
    display: grid; place-items: center;
    background: #101A2C; border: 1px solid #22314C; color: #D6DEEB;
    text-decoration: none; font-size: 17px; line-height: 1;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
  }
  .cnd-top:hover { background: #16233A; border-color: #4C8DD6; color: #BFD6F0; }
  @media (max-width: 640px) { .cnd-toc nav { width: 78vw; } }
</style>
"""


def _rows() -> str:
    out = []
    for e in SECTIONS:
        cls = "cnd-row cnd-sub" if e.sub else "cnd-row"
        out.append(
            f'<a class="{cls}" href="#{e.anchor}">'
            f'<span class="cnd-no">{e.no}</span>'
            f'<span class="cnd-title">{e.title}</span>'
            f'<span class="cnd-count">{e.count}</span>'
            f'<span class="cnd-hint">{e.hint}</span>'
            f'</a>'
        )
    return "".join(out)


def _html() -> str:
    """整段 HTML。

    开头的 `#pagetop` 零尺寸锚点是「回到顶部」的目标：第一个 h1 下面已经压着
    一条 running head 与试算 chip，指向它只能回到「本页一览」，回不到真正的页顶。
    这个块因此必须放在 app.py 的最前面，锚点才落在文档第 0 像素处。
    """
    return (
        _CSS
        + '<span id="pagetop" aria-hidden="true"></span>'
        + '<details class="cnd-toc">'
        + '<summary>目录</summary>'
        + '<nav aria-label="全页目录">'
        + f'<div class="cnd-head">全页目录 · {len(SECTIONS)} 段</div>'
        + _rows()
        + '</nav></details>'
        + '<a class="cnd-top" href="#pagetop" aria-label="回到顶部" title="回到顶部">'
        + '↑</a>'
    )


def render() -> None:
    """在 app.py 的最前面调用一次：`#pagetop` 必须真的落在页顶。"""
    import streamlit as st

    st.html(_html())
