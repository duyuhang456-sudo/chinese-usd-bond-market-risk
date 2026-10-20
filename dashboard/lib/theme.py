"""设计令牌与 Altair 主题。

视觉语言＝**清爽的学术版式**：纸白底、墨黑字、西文衬线 / 中文宋体、发丝横线、
小圆角、干净透亮的数据色（蓝、红、金黄、青绿、橙、紫、绿、粉）。图也是论文图
的规矩：细实线、浅灰网格、柱端微圆、选中点才用重色。

状态色是保留色，只用于状态（推荐 / 淘汰、告警），且永远配文字标签，不参与
分类序列。分类色板过 dataviz 五项可计算检查（明度带、彩度、色觉分离、常视觉
下限、对比度，见 tools/check_palette.py）；换底色或换色板必须重算，不靠眼睛。
"""

from __future__ import annotations

import altair as alt

# --- 分类色：固定顺序，永不循环 -------------------------------------------
# 槽位有语义：[0] 蓝＝选中/当前试算/基准，[1] 红＝对照损失/加缓冲后；顺序不是
# 随意排的——tools/check_palette.py 对相邻配对做色觉模拟（Machado 2009），
# 蓝-红-金黄-青绿-橙-紫-绿-粉这个次序在浅/深两底色上最差相邻 ΔE 都过线。
SERIES_LIGHT = ["#3B7DBF", "#D94E4E", "#B58E00", "#09A095",
                "#D87718", "#8466C9", "#59A14F", "#D47095"]
SERIES_DARK = ["#4784C2", "#CA5553", "#B18F2C", "#039E93",
               "#C57127", "#8C77C4", "#599C4F", "#CA7392"]

# --- 状态色：保留色，只用于状态，且永远配图标或文字 -------------------------
# 分模式给，因为同一组色标对两种底色不可能同时达标。四个状态色彼此是**有序**
# 的严重度阶梯，不用分类色那套「相邻 ΔE ≥ 15」去要求它。good 取深一档绿，
# 「推荐」两个字直接印在纸白上也要过 4.5:1。
STATUS_LIGHT = {"good": "#177A35", "warning": "#A87500",
                "serious": "#DD5F2C", "critical": "#D03B3B"}
STATUS_DARK = {"good": "#3FAE5C", "warning": "#FAB219",
               "serious": "#EC835A", "critical": "#E06060"}

# --- 蓝色单色阶（连续量级用：热图、分量深浅），hue 与板蓝一致 ---------------
BLUE_RAMP = ["#e9eff5", "#dae3ec", "#c7d4e2", "#a5c3e3", "#8aadd2", "#6e96c0",
             "#507dac", "#36679a", "#1a5388", "#004178", "#00336c", "#002560",
             "#001754"]
# 离散有序档（分档着色）与连续色阶的要求不一样：连续色阶本来就跨明度、浅端贴着底色，
# 而离散档的每一步都要与相邻步看得出区别、且离底色最近的那一端仍须达到 2:1。
# 深色档不能拿浅色阶直接用：热图低端会变成一片近白，压过整页深蓝。
# 深色阶从贴着底色的深靛走到亮蓝，量级越大越亮，与浅色档的方向相反。
# 低端不能贴着卡片底色：热图里 0 损失是一大片格子，压在 #101A2C 的卡片上会
# 整块消失，读者分不出「值为 0」和「没有数据」。最低一档取离卡片 1.6:1 的深靛。
BLUE_RAMP_DARK = ["#233F63", "#28476D", "#2D4F77", "#335881", "#3A628C",
                  "#416C97", "#4A78A3", "#5585B0", "#6393BD", "#74A2CA",
                  "#87B2D6", "#9CC2E3", "#B4D9F8"]

BLUE_RAMP_ORDINAL = {
    "light": [BLUE_RAMP[i] for i in (4, 6, 8, 10, 12)],
    # 最暗一档必须离底色 >= 2:1，否则第一档直接看不见；所以从第 6 步起步。
    "dark": [BLUE_RAMP_DARK[i] for i in (2, 4, 6, 8, 10)],
}

# 发散色对：板蓝 / 板红两极 + 暖灰中点（中点跟着页面底色走）
DIVERGING = {"low": "#3B7DBF", "mid_light": "#EFEDE6", "mid_dark": "#2C2B28",
             "high": "#D94E4E"}

# 衬线栈：西文 Source Serif/Charter/Georgia，中文宋体。本机/云端有哪支用哪支，
# 不拉 Google Fonts：不引入第三方请求，云端构建也不受影响。
FONT_STACK = ('"Source Serif 4", "Source Serif Pro", "Charter", "Iowan Old Style", '
              'Georgia, "Times New Roman", "Songti SC", "STSong", "SimSun", serif')

_TOKENS = {
    "light": {
        # 底与面
        "page": "#FCFBF8",        # 纸白（= config.toml 的 backgroundColor）
        "surface": "#FCFBF8",     # 卡片/图底色＝同纸色，一片纸、只靠发丝线分节
        # 文字：由强到弱
        "ink": "#1C1B17",         # 标题（墨黑）
        "ink2": "#2B2A24",        # 正文与读数数字
        "muted": "#6F6A5E",       # 次要（轴标签、说明）
        "faint": "#A09A8C",       # 页脚级
        # 线与边
        "grid": "#E9E6DE",
        "axis": "#C4BFB3",
        "border": "#DBD7CC",
        # 强调：蓝色家族，不再用金
        "accent": "#2F6DB0",
        "accent_ink": "#2F6DB0",
        "navy": "#16233D",        # 深藏青（备用）
        "highlight": "#E4EDF6",   # 浅蓝荧光笔衬底
        "highlight_ink": "#1C1B17",
        "series": SERIES_LIGHT,
    },
    "dark": {
        # 暗色档＝深夜纸：深海军蓝底、冷白字。底色定在靛蓝一侧而不是中性炭黑，
        # 数据色（蓝红金黄青绿）贴上去才有学院蓝本的观感。
        "page": "#0A1120",
        "surface": "#101A2C",     # 卡片与图底，比页底亮一档
        "ink": "#EDF1F9",         # 标题
        "ink2": "#D6DEEB",        # 正文与读数数字
        "muted": "#94A2BA",       # 轴标签、说明
        "faint": "#6B7891",       # 页脚级
        "grid": "#1B2740",
        "axis": "#2E3D58",
        "border": "#22314C",
        "accent": "#5B9AE0",
        "accent_ink": "#8FB9EA",
        "navy": "#060B15",
        "highlight": "#16233A",
        "highlight_ink": "#BFD6F0",
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

        # 配置里钉死的那一档优先：这套看板是**深色专用**的，读者浏览器的
        # 系统主题不该把图表翻成浅色（页面底是深蓝、图底是浅纸，会分裂成两层）。
        pinned = st.get_option("theme.base")
        if pinned in ("light", "dark"):
            return pinned
        th = getattr(st.context, "theme", None)
        t = getattr(th, "type", None)
        if t in ("light", "dark"):
            return t
        return "light"
    except Exception:
        return "dark"


def tokens(mode: str | None = None) -> dict:
    """取当前主题的一套令牌。"""
    mode = mode or base_mode()
    t = dict(_TOKENS[mode])
    t["mode"] = mode
    t["status"] = STATUS_LIGHT if mode == "light" else STATUS_DARK
    t["font"] = FONT_STACK
    t["diverging"] = DIVERGING
    t["blue_ramp"] = BLUE_RAMP if mode == "light" else BLUE_RAMP_DARK
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
    t = tokens(mode)   # 走 tokens() 才拿得到 blue_ramp 这类按档派生的键
    return {
        # 图落在卡片里，图底色取 surface：浅色=同纸色，深色=浅一档暖炭，
        # 图与卡片同色边界自然。
        "background": t["surface"],
        "font": FONT_STACK,
        "config": {
            "view": {"stroke": None, "continuousWidth": 640, "continuousHeight": 320},
            "padding": {"top": 8, "bottom": 6, "left": 4, "right": 12},
            "axis": {
                "domainColor": t["axis"],
                "domainWidth": 1,
                "tickColor": t["axis"],
                "tickWidth": 1,
                "tickSize": 4,
                "gridColor": t["grid"],
                "gridDash": [],           # 网格恒为细实线；虚线只留给真阈值线
                "gridWidth": 1,
                "labelColor": t["muted"],
                "titleColor": t["ink2"],
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": 11,
                # 不截断轴标签。两栏之后每张图只有约 700px，Vega 默认按 180px
                # 砍字，模型名与因子名会变成「FHS-E 滤波历史…」，读者认不出是哪条。
                "labelLimit": 0,
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
                "fontWeight": 600,
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
                "labelLimit": 0,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "labelPadding": 4,
            },
            "facet": {"spacing": 14},
            "concat": {"spacing": 14},
            "range": {
                "category": list(t["series"]),
                "ordinal": list(BLUE_RAMP_ORDINAL[mode]),
                "ramp": list(t["blue_ramp"]),
                "heatmap": list(t["blue_ramp"]),
            },
            # 论文图：细线条、柱端微圆、浅色晕。
            "line": {"strokeWidth": 1.8},
            "point": {"size": 60, "filled": True, "stroke": t["surface"], "strokeWidth": 1.5},
            "bar": {"cornerRadiusEnd": 3},
            "area": {"opacity": 0.09, "line": {"strokeWidth": 1.8}},
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
