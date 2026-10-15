"""结论分级与已知边界。

看板的收尾页。这一页不引入任何新数字，只做一件事：把终期报告 §5.1 已经分好的三层结论
（稳健 / 参考 / 不可外推）连同它们的**数据锚点**摆在一起，让读者知道
哪一句话可以直接引用、哪一句话必须带上条件、哪一句话根本不能往外推。

版式：页首一排读数带给出每层的条数与最极端的那个锚点；下面一张满宽的倍数图，
同一情景的两个口径画成并排的两根条——行对齐了，「纯汇率情景在美元口径恒为 0」
这句话才是看出来的而不是读出来的；再下面是一张满宽的结论表，顶部控制带按层级筛它。

控制带只有层级一个控件：选「全部」看十四条的完整清单，选某一层就只留那一层。
早先这里用四个页签装同一批结论，页签里又是两列文字卡——一层只看见一半，
还得自己左右对照，所以改成一张表加一个筛选器。

分层的标准不是「结论重不重要」，而是**结论对自选参数的敏感程度**：
换一个窗口、换一条判定线、换一个锚点窗长，读数会不会翻面。
逐条文案来自 `dashboard/lib/labels.py` 的 `CONCLUSION_TIERS` 与 `KNOWN_LIMITS`，
与报告 §5.1、`docs/tool_usage.md` §8 逐字对应；数据锚点里的数都已回到 CSV 复核过。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import charts, data, labels, page, theme  # noqa: E402

page.init("结论分级与已知边界")

_si = data.stress_impact()
_vh = data.var_historical()
SPAN = f"{_vh['date'].min().date()} → {_vh['date'].max().date()}"
_tok = theme.tokens()

_tiers = {t["tier"]: t for t in labels.CONCLUSION_TIERS}
TIERS = ["全部", "稳健", "参考", "不可外推", "已知边界"]

# 表里那张卡片的副标题：整体分层标准，或当前这一层的判据
_NOTE_ALL = "分层标准不是「结论重不重要」，而是结论换一个窗口、一条判定线、一个锚点窗长之后会不会翻面。"
_NOTE_LIM = "数据可得性的边界。在报告与汇报里引用任何数字，这四项都要一并带上。"


def _note(name: str) -> str:
    if name == "全部":
        return _NOTE_ALL
    if name == "已知边界":
        return _NOTE_LIM
    return _tiers[name]["criterion"]


page.header(
    "结论分级与已知边界",
    "同一份计量结果，不同句子的可信度不一样——按「对自选参数的敏感程度」分三级。",
    crumb="阶段四 · 结论分级",
    span=SPAN,
)

# ---------------------------------------------------------------- 控制带
with st.container(border=True):
    _k1, _k2 = st.columns([3, 4])
    with _k1:
        tier = st.segmented_control(
            "结论层级", TIERS, default="全部", key="cc_tier",
            help="「稳健」= 换参数不翻面，可直接引用；「参考」= 方向可用、绝对水平要带条件；"
                 "「不可外推」= 样本里没有可检验的对象；「已知边界」= 引用任何数字都要一并声明的数据限制。",
        ) or "全部"
    with _k2:
        st.caption("分层看的是结论对自选参数的敏感程度，不是结论重不重要。"
                   "选某一层只留那一层的条目，选「全部」看完整清单。")

page.kpi_row([
    {"label": "汇率是次口径下最主要的尾部风险来源",
     "value": "4.51 倍",
     "help": "S09 次口径 3.1708% 是判定线 0.7024% 的 4.51 倍，全表最高。"
             "纯汇率情景不加缓冲；同一个情景在美元口径下恒为 0。"},
    {"label": "可以直接引用的结论",
     "value": f"{len(_tiers['稳健']['items'])} 条",
     "help": "对自选参数不敏感：利率主因、汇率次因、库级覆盖、线性累加。"
             "四条判断的方向在所有稳健性检验下不变。"},
    {"label": "方向可用、绝对水平要带条件",
     "value": f"{len(_tiers['参考']['items'])} 条",
     "help": "信用维度整体、假设情景的绝对水平、历史情景的个别读数。"
             "可以引用，但必须连同适用范围一起引用。"},
    {"label": "样本不支持、明确划界",
     "value": f"{len(_tiers['不可外推']['items'])} 条",
     "help": "δ 对信用情景的传导、压力情景的极端档、汇率路径的幅度判断。"
             "这三条不是结论不确定，而是样本里没有可检验的对象。"},
])

# ------------------------------------------------------------ 满宽倍数图
_x = _si[["scenario_id", "scenario_name", "scenario_class", "caliber",
          "x_es99", "loss_after_buffer_pct", "es_1d_99_pct", "x_var99"]].copy()
_x["口径"] = _x["caliber"].map(labels.CALIBER_SHORT)
_CD = [labels.CALIBER_SHORT["主"], labels.CALIBER_SHORT["次"]]

# 行序只按次口径排一次，两个口径共用：行对齐了才能一眼看出哪几条、
# 在哪一侧差得最多。早先两个口径各排各的，同一个情景在左右两图里不在同一行。
_piv = _x.pivot(index="scenario_id", columns="caliber", values="x_es99")
_order = list(_piv.sort_values("次", ascending=False).index)
_hi = float(_x["x_es99"].max())

with page.card(
    "同一情景在两条口径下的倍数，纯汇率情景的美元口径恒为 0",
    "横轴是含缓冲净损失 ÷ 1 日 99% 期望损失，1.0 就是刚好压在判定线上。"
    "汇率变动在美元口径里被计价基准搬运掉了。",
):
    # 判定线只画线不写字：竖线落在 1.0，左侧几条短条正好压在那一带，
    # 文字挂上去必被盖住，值改到卡片下面那句读法里说。
    _bars = alt.Chart(_x).mark_bar(size=11, cornerRadiusEnd=3).encode(
        y=alt.Y("scenario_id:N", title="", sort=_order,
                axis=alt.Axis(labelOverlap=False, labelFontSize=11)),
        yOffset=alt.YOffset("口径:N", sort=_CD),
        x=alt.X("x_es99:Q", title="相当于几倍 1 日 99% 期望损失",
                scale=alt.Scale(domain=[0, _hi * 1.07])),
        color=alt.Color("口径:N", title="", sort=_CD,
                        scale=alt.Scale(domain=_CD,
                                        range=[_tok["series"][0], _tok["series"][1]]),
                        legend=alt.Legend(orient="bottom", direction="horizontal",
                                          title=None, symbolType="square")),
        tooltip=[alt.Tooltip("scenario_id:N", title="编号"),
                 alt.Tooltip("scenario_name:N", title="情景"),
                 alt.Tooltip("scenario_class:N", title="类别"),
                 alt.Tooltip("loss_after_buffer_pct:Q", title="含缓冲净损失 %", format=".4f"),
                 alt.Tooltip("es_1d_99_pct:Q", title="判定线 %", format=".4f"),
                 alt.Tooltip("x_es99:Q", title="对判定线的倍数", format=".3f"),
                 alt.Tooltip("x_var99:Q", title="对 99% VaR 的倍数", format=".3f")],
    )
    st.altair_chart(
        charts.add_rule(_bars, 1.0, orient="vertical").properties(height=440),
        use_container_width=True,
    )
    _fx = _piv.loc[["S07", "S08", "S09"], "次"]
    st.caption(
        "红虚线 = 判定线 1.0×。S07 / S08 / S09 是纯汇率情景：美元口径三条都停在 0，"
        f"人民币口径分别是 {' / '.join(f'{v:.2f}' for v in _fx)} 倍。"
        "全表最高的是 S09 的 4.51 倍。"
    )
    charts.data_drawer(
        _x, key="cc_x", title="数据抽屉 · 压力倍数：各情景对判定线与对 99% VaR 的倍数",
        height=340,
        fmt={"x_es99": "%.3f", "x_var99": "%.3f",
             "loss_after_buffer_pct": "%.4f", "es_1d_99_pct": "%.4f"},
        labels={"scenario_id": "编号", "scenario_name": "情景", "scenario_class": "类别",
                "caliber": "口径", "口径": "口径",
                "loss_after_buffer_pct": "含缓冲净损失 %",
                "es_1d_99_pct": "1 日 99% 期望损失（判定线）%",
                "x_es99": "倍数 · 对判定线", "x_var99": "倍数 · 对 99% VaR"},
        note="倍数由 results/stress_impact.csv 的 x_es99 / x_var99 两列直接读取，"
             "不在页面上重算。",
    )

# ------------------------------------------------------------ 满宽结论表
# 层级并进「结论」这一格，不做单独的列：三列并排时三样加起来比卡片还宽，
# 卡片右缘会吃掉最长那一列的几个字，读到的锚点是不全的。并进去之后只剩两列，
# 锚点那一列才放得下。「全部」视图里靠这个前缀分辨层级。
_rows = []
for _t in labels.CONCLUSION_TIERS:
    for _claim, _anchor in _t["items"]:
        _rows.append({"层级序": TIERS.index(_t["tier"]),
                      "结论": f"{_t['tier']} · {_claim}", "数据锚点": _anchor})
for _name, _lim in labels.KNOWN_LIMITS:
    _rows.append({"层级序": TIERS.index("已知边界"),
                  "结论": f"已知边界 · {_name}", "数据锚点": _lim})
_all = pd.DataFrame(_rows).sort_values("层级序", kind="stable").drop(columns="层级序")
_all = _all.reset_index(drop=True)
_now = _all if tier == "全部" else _all[_all["结论"].str.startswith(f"{tier} · ")].reset_index(drop=True)

with page.card(f"{tier} · {len(_now)} 条结论与各自的数据锚点", _note(tier)):
    # 用 Markdown 表而不是 st.dataframe：数据锚点里最长的一条要 850px，
    # 结论那一列还要 360px，两列加起来比卡片宽，st.dataframe 会把超出的部分
    # 从单元格右缘直接切掉——切掉的正好是「只在此范围内成立」这类限定语，
    # 而限定语是这些结论能不能引用的关键。Markdown 表的单元格会自动折行，
    # 一个字都不会丢。
    _head = "| 结论 | 数据锚点 |\n|---|---|\n"
    _body = "".join(f"| {r['结论']} | {r['数据锚点']} |\n" for _, r in _now.iterrows())
    st.markdown(_head + _body)

page.caliber_note(
    "主口径（复权 USD）与次口径（人民币）必须标清，NAV 除权口径不与复权主口径混用；"
    "95% / 99% 是 Kupiec 检验的显著性水平，不是 VaR 置信度。"
)

# ------------------------------------------------------------ 硬断言清单
# 原「稳健性检验与数据质量」页的断言块，按阶段重组并入本页（阶段四·终期输出）。
# 与上面的交叉：asserts 是工具自带的验收断言，KNOWN_LIMITS 是数据可得性边界，
# 两者不是同一件事，故分别成卡。
with page.card("工具自带的 30 条硬断言",
               "任一不满足即报错终止，不是打印警告后继续——这些断言是结论可信度的地基。"):
    _asserts = pd.DataFrame(
        [("`stress_scenarios.py`", "主/次口径收益差须逐值等于汇率项（log 收益可加性）",
          "双口径差值不再等于汇率项 → 累计不可直接求和，双口径的所有差值类结论失效"),
         ("`stress_impact.py`", "HS250 @ common773 的四个基准配置须与阶段二报告 §7.2 登记值一致",
          "判定阈值与阶段二定案脱钩 → 全部尾部判定失去依据"),
         ("`stress_impact.py`", "2022 残差标定量须与逐事件表算出的值一致",
          "信用区间上端的加项与数据源不符"),
         ("`stress_impact.py`", "情景库中每个情景都须进入测算",
          "有情景被静默漏算（X2 曾因此漏算，直到覆盖度检验才暴露）"),
         ("`stress_impact.py`", "历史情景的信用区间下端须与窗口累计量不同",
          "信用区间退回窗口级量，与 1 日判定线口径错配"),
         ("`stress_impact.py`", "假设情景的 `loss_modeled_pct` 与 `worst_day_modeled_pct` 须重合",
          "假设情景路径不再是单日 → 其信用区间不再是 1 日量"),
         ("`stress_robustness.py`", "主期限须从 `stress_impact.csv` 读出且恒为 1 日",
          "两个脚本的持有期口径不同步"),
         ("`stress_robustness.py`", "窗口敏感性表的最差单日 / 回撤须与基准表逐值对账",
          "两张表的同一量算法分叉"),
         ("`stress_robustness.py`", "缓冲网格中「本报告取值」档须逐条复现基准表的损失与尾部判定",
          "5 档缓冲的分析与 headline 结论不是同一套数"),
         ("`var_backtest.py`", "两个 VaR 产物的日期向量须一致", "参数法与历史模拟法的口径已分叉"),
         ("`var_backtest.py`", "存疑日剔除数须等于预期", "剔除是静默空操作（「剔了但没剔」）"),
         ("`var_backtest.py`", "M1δ 与 M1 的违规日集合须相同", "δ-normal 恒等式被破坏"),
         ("`var_historical.py`", "样本外长度须与 9/15 一致；VaR 不得为负；99% 须 ≥ 95%；前视检查须与手写循环一致",
          "各期分位路径或违规判定与协议不符"),
         ("`var_historical.py`、`var_parametric.py`", "人民币次口径的 NaN 须为尾部连续",
          "次口径序列出现中间断点，需检查并非尾部缺失"),
         ("`var_common.py`", "Christoffersen 的 LR_ind 不得为负", "似然比符号写反")],
        columns=["脚本", "断言", "不满足时的含义"],
    )
    st.dataframe(_asserts, width="stretch", height=330, hide_index=True)
    st.caption(
        "断言处数：`stress_robustness.py` 11 处、`var_historical.py` 8 处、"
        "`stress_impact.py` 5 处、`var_backtest.py` 3 处，"
        "`stress_scenarios.py` / `var_common.py` / `var_parametric.py` 各 1 处，合计 30 处。"
        "断言全部通过是每次改动后的验收前提，不得为了跑通而放宽。"
    )

page.footer(
    "results/stress_impact.csv、results/stress_anchor_sens.csv、"
    "results/stress_window_sens.csv、docs/final_report.md、docs/tool_usage.md",
    "§5.1 三级结论 / §8 表 13 / §2 口径说明",
)
