"""看板入口：一张可滚的单页长文档，按主题分五段。

启动（必须在仓库根执行，Streamlit 只按当前工作目录找 .streamlit/config.toml）：

    .venv/bin/streamlit run dashboard/app.py

页面不是一条工作流，取数、计量、检验、加压、判定五件事没有先后——读者从哪一段
开始读都可以，页内不给「下一步」按钮。想换口径 / 报价方案 / 模型 / 置信度 / 缓冲档，
点右上角那颗 chip，弹层里改，五段里的数字一起跟着变。

全部数字只读既有 CSV（results/、clean_data/、events/、factors/），不重算模型。

页面文件在 `pages/sections/` 下，每个可单独 `streamlit run` 打开，也被
`tools/check_dashboard.py` 用 AppTest 逐个执行。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

from lib import page, pipe, theme  # noqa: E402

try:
    st.set_page_config(page_title=page.APP_TITLE,
                       page_icon=":material/query_stats:", layout="wide")
except Exception:
    pass
theme.register()

SECTION_DIR = HERE / "pages" / "sections"
SECTIONS = [
    "0_cover",       # HERO + 本页一览
    "1_data_factor",  # 主题一 · 数据与因子概览
    "2_var_models",   # 主题二 · VaR 模型对比看板
    "3_stress",       # 主题三 · 压力测试交互分析
    "4_robustness",   # 主题四 · 稳健性检验探索
    "5_verdict",      # 主题五 · 结论与边界
]


def run_section(name: str) -> None:
    """按序跑一段主题。段内不写 CSS，只用原生组件拼。"""
    path = SECTION_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"section_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"找不到主题段：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


page.banner("风险计量与压力测试 · 交互看板", pipe.trial_meta())
page.trial_chip()

for _name in SECTIONS:
    run_section(_name)