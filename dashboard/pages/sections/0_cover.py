"""HERO · 本页一览：项目主结论四个 stat tile + 一张锚点表。

排在顶栏之后，接着进入五主题卡片。这里不放图——读者往下滚就是为了看图，
封面只负责把「这个项目最后得出了什么」摆在最上面。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import page  # noqa: E402

page.init("本页一览")

# 不写副标题：当前试算单就摆在右上角的 chip 里，重复一遍只是占地方。
page.section_head("本页一览")
st.markdown("中资投资级美元债市场的 VaR 计量、压力测试与稳健性检验，"
            "全部结论只读既有测算结果，不在看板里重算模型。")

_cols = st.columns(4, gap="small")
for _col, (_label, _value, _sub) in zip(_cols, [
    ("HS250 校准违规率", "1.0349 %",
     "95% 置信带 [0.43, 1.91]；5/5 主口径组合通过 Kupiec。"),
    ("1 日档含缓冲占比", "50.5 %",
     "3 日档 40.2%；超限情景数 17 → 23。"),
    ("主口径判定线", "0.6659 %",
     "HS250 在 common773 / 主 / 99% 下的 1 日 ES。"),
    ("最严重情景倍数", "4.51 ×",
     "H1 含缓冲净损失 3.1708% ÷ 0.6659%。"),
]):
    with _col:
        page.tile_kpi(_label, _value, sub=_sub)

st.dataframe(
    pd.DataFrame(
        [
            ["校准违规率", "1.0349 %", "8 / 773 个交易日越限", "results/backtest_results.csv"],
            ["判定线（1 日 ES）", "0.6659 %", "压力倍数与缓冲占比的分母", "results/var_series_long.csv"],
            ["含缓冲占比", "50.5 %", "缓冲吃掉判定线的一半", "results/stress_buffer_sens.csv"],
            ["H1 净损失", "3.1708 %", "30 个情景里最严重的一个", "results/stress_impact.csv"],
            ["推荐模型", "HS250", "32 组里 8 组通过全部检验", "results/model_scorecard.csv"],
        ],
        columns=["锚点", "数值", "一句话读法", "出处"],
    ),
    hide_index=True,
    use_container_width=True,
)

page.footer(
    "results/ 下 backtest_results.csv、var_series_long.csv、stress_impact.csv、"
    "stress_buffer_sens.csv、model_scorecard.csv"
)