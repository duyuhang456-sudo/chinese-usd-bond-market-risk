"""设计令牌与 Altair 主题。

令牌取自**结项汇报那套视觉语言**（`中资投资级美元债券…结项汇报`），
这样两份交付物是同一个人的东西：内容页米白 `#F7F5F0`、标题深蓝 `#16283C`、
正文 `#465667`、次要 `#667484`、金色 `#C9A66B` 作强调与卡片左竖条。

深色档不是把浅色反转，而是直接用那份汇报稿的**另一种页面**——封面与章节页的深蓝
`#101F30`。两种版式你都已经定过，看板的两套主题就落在它们上面。

金色有两种用法，分开取色有对比度依据：装饰性竖条与衬底用 `#C9A66B`（不承载信息），
需要读的字用金字 `#8C6832`（对米白约 5.3:1）。`#C9A66B` 直接当文字色只有约 2:1，不合格。

分类色仍按 dataviz 校验过的固定顺序取用、不循环。底色从 `#fcfcfb` 换成 `#F7F5F0`、
深色底从 `#1a1a19` 换成 `#101F30` 之后**必须重算**——用 `tools/check_palette.py`，
不靠眼睛。
"""

from __future__ import annotations

import altair as alt

# --- 分类色：固定顺序，永不循环 -------------------------------------------
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# --- 状态色：保留色，只用于状态，且永远配图标或文字 -------------------------
# 分模式给，因为同一组色标对两种底色不可能同时达标：参考档的
# warning #fab219 对米白只有 1.68:1，做色块几乎看不见；而它对自己那套深底有 9.08:1。
# 浅色档的 warning / serious 因此各深一档（3.70:1 / 3.35:1），色相家族不变，
# 严重度仍读成绿→琥珀→橙→红。四个状态色彼此是**有序**的严重度阶梯，
# 不用分类色那套「相邻 ΔE ≥ 15」去要求它——参考档自己的 warning↔serious 也只有 13.6。
STATUS_LIGHT = {"good": "#0ca30c", "warning": "#a87500",
                "serious": "#dd5f2c", "critical": "#d03b3b"}
STATUS_DARK = {"good": "#0ca30c", "warning": "#fab219",
               "serious": "#ec835a", "critical": "#d03b3b"}

# --- 蓝色单色阶（连续量级用：热图、分量深浅） ------------------------------
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
# 离散有序档（分档着色）与连续色阶的要求不一样：连续色阶本来就跨明度、浅端贴着底色，
# 而离散档的每一步都要与相邻步看得出区别、且离底色最近的那一端仍须达到 2:1。
# 13 阶原样用会达不到（相邻 ΔL 约 0.047，低于 0.06），所以隔步取值、并把贴底色的一端裁掉。
BLUE_RAMP_ORDINAL = {
    "light": [BLUE_RAMP[i] for i in (4, 6, 8, 10, 12)],
    "dark": [BLUE_RAMP[i] for i in (0, 2, 4, 6, 8, 10)],
}

# 发散色对：冷暖两极 + 暖灰中点（中点跟着页面底色走，不再用冷灰）
DIVERGING = {"low": "#2a78d6", "mid_light": "#EFEAE0", "mid_dark": "#22334A",
             "high": "#d03b3b"}

# Inter 装了就用，没装回落 Helvetica Neue；中文回落 PingFang SC。
# 不拉 Google Fonts：不引入第三方请求，云端构建也不受影响。
FONT_STACK = ('"Inter", "Helvetica Neue", system-ui, -apple-system, '
              '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif')

_TOKENS = {
    "light": {
        # 底与面
        "page": "#F7F5F0",        # 内容页米白（= config.toml 的 backgroundColor）
        "surface": "#F7F5F0",     # 卡片/图底色＝同页面米白，一片纸色、只靠边框区隔
        # 文字：由强到弱
        "ink": "#16283C",         # 标题
        "ink2": "#465667",        # 正文
        "muted": "#667484",       # 次要（轴标签、说明）
        "faint": "#8A97A5",       # 页脚级（只用在深蓝章节带上，米白底上不够）
        # 线与边
        "grid": "#E7E1D6",
        "axis": "#C9C2B4",
        "border": "#E3DDD1",
        # 强调
        "accent": "#C9A66B",      # 金色，仅装饰（竖条、衬底）
        "accent_ink": "#8C6832",  # 金字，可读
        "navy": "#101F30",        # 章节带底色
        "highlight": "#FFF0B3",   # 荧光笔衬底
        "highlight_ink": "#16283C",
        "series": SERIES_LIGHT,
    },
    "dark": {
        "page": "#101F30",
        "surface": "#16283C",
        "ink": "#F7F5F0",
        "ink2": "#C4CFD9",
        "muted": "#8A97A5",
        "faint": "#8A97A5",
        "grid": "#22334A",
        "axis": "#33455C",
        "border": "#2A3B52",
        "accent": "#C9A66B",
        "accent_ink": "#DCC08A",
        "navy": "#0A1520",
        "highlight": "#4A3F1C",
        "highlight_ink": "#FFF0B3",
        "series": SERIES_DARK,
    },
}


def base_mode() -> str:
    """当前主题基色，默认 light。

    先问 `st.context.theme.type`——这才是浏览器实际解析出来的主题，
    用户手动切到深色时它跟着变；`st.get_option("theme.base")` 读的是
    `.streamlit/config.toml` 里钉的配置值，用户切换它不变，只能作回落。
    脱离 Streamlit 运行时（被别的脚本 import、冒烟测试）两者都取不到，回落 light。
    """
    try:
        import streamlit as st

        th = getattr(st.context, "theme", None)
        t = getattr(th, "type", None)
        if t in ("light", "dark"):
            return t
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "light"


def tokens(mode: str | None = None) -> dict:
    """取当前主题的一套令牌。"""
    mode = mode or base_mode()
    t = dict(_TOKENS[mode])
    t["mode"] = mode
    t["status"] = STATUS_LIGHT if mode == "light" else STATUS_DARK
    t["font"] = FONT_STACK
    t["diverging"] = DIVERGING
    t["blue_ramp"] = BLUE_RAMP
    t["blue_ramp_ordinal"] = BLUE_RAMP_ORDINAL[mode]
    return t


def series_color(i: int, mode: str | None = None) -> str:
    """第 i 个分类序列色（0 基）。超过 8 个不该再有序列——应折进「其他」或分面。"""
    pal = tokens(mode)["series"]
    if i >= len(pal):
        raise ValueError(
            f"分类色只有 {len(pal)} 个槽位，第 {i + 1} 个序列应折进「其他」或改用分面。"
        )
    return pal[i]


def _theme_config(mode: str) -> dict:
    t = _TOKENS[mode]
    return {
        # 图落在卡片里，图底色取 surface：浅色=同页面米白（整页一片纸色），
        # 深色=浅一档深蓝，图与卡片同色边界自然。
        "background": t["surface"],
        "font": FONT_STACK,
        "config": {
            "view": {"stroke": None, "continuousWidth": 640, "continuousHeight": 320},
            "padding": {"top": 8, "bottom": 6, "left": 4, "right": 12},
            "axis": {
                "domainColor": t["axis"],
                "tickColor": t["axis"],
                "gridColor": t["grid"],
                "gridDash": [],           # 网格恒为实线发丝；虚线只留给真阈值线
                "gridWidth": 1,
                "labelColor": t["muted"],
                "titleColor": t["ink2"],
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
            },
            "legend": {
                "labelColor": t["ink2"],
                "titleColor": t["ink2"],
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "symbolSize": 80,
                "orient": "right",
            },
            "title": {
                "color": t["ink"],
                "font": FONT_STACK,
                "fontSize": 13,
                "fontWeight": "normal",
                "anchor": "start",
                "offset": 8,
            },
            # 分面与小倍数此前吃的是 Altair 裸默认，这里补齐
            "header": {
                "labelColor": t["muted"],
                "titleColor": t["ink2"],
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": 11,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "labelPadding": 4,
            },
            "facet": {"spacing": 14},
            "concat": {"spacing": 14},
            "range": {
                "category": list(t["series"]),
                "ordinal": list(BLUE_RAMP_ORDINAL[mode]),
                "ramp": list(BLUE_RAMP),
                "heatmap": list(BLUE_RAMP),
            },
            "line": {"strokeWidth": 2},
            "point": {"size": 70, "filled": True, "stroke": t["surface"], "strokeWidth": 2},
            "bar": {"cornerRadiusEnd": 4},
            "area": {"opacity": 0.10, "line": {"strokeWidth": 2}},
            "text": {"font": FONT_STACK},
        },
    }


_REGISTERED = False


def register() -> None:
    """把 light/dark 两套主题注册进 Altair，并启用当前基色那套。

    幂等：重复调用只重新 enable，不重复 register。
    """
    global _REGISTERED
    if not _REGISTERED:
        for mode in ("light", "dark"):
            cfg = _theme_config(mode)
            name = f"jpm-{mode}"
            fn = (lambda c: (lambda: c))(cfg)
            try:
                alt.theme.register(name, enable=False)(fn)      # altair >= 5.5
            except AttributeError:
                alt.themes.register(name, fn)                   # 旧接口
            except Exception:
                pass
        _REGISTERED = True
    try:
        alt.theme.enable(f"jpm-{base_mode()}")
    except AttributeError:
        alt.themes.enable(f"jpm-{base_mode()}")
        # 旧接口下 enable 会顺带把主题置为默认，这里再确认一次
    except Exception:
        pass
