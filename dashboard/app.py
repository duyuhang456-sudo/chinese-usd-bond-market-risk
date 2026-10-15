"""看板入口。

启动（必须在仓库根执行，Streamlit 只按当前工作目录找 .streamlit/config.toml）：

    .venv/bin/streamlit run dashboard/app.py

导航按项目四阶段组织：阶段一（数据与因子）、阶段二（VaR 计量）、阶段三（压力测试）、
阶段四（结论与终期），一阶段一组，侧栏标题即阶段名。阶段四的「概览与业务结论」是默认
首页，第一屏给一张四阶段进度时间线回答「做到哪了」，下面保留原业务结论。

页面本身各自独立，可以单独 streamlit run dashboard/pages/<name>.py 打开，
也可以被 tools/check_dashboard.py 用 AppTest 逐页执行。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from lib import labels, page, theme  # noqa: E402

try:
    st.set_page_config(page_title=page.APP_TITLE, layout="wide")
except Exception:
    pass
theme.register()

NAV = st.navigation(
    {
        "阶段一 · 数据与因子": [
            st.Page("pages/2_data_factors.py", title="数据与因子概览",
                    icon=":material/table_view:"),
            st.Page("pages/data_quality.py", title="数据质量",
                    icon=":material/database:"),
        ],
        "阶段二 · VaR 计量": [
            st.Page("pages/3_var_backtest.py", title="VaR 模型对比与回测",
                    icon=":material/monitoring:"),
            st.Page("pages/4_model_validation.py", title="模型诊断与验证",
                    icon=":material/fact_check:"),
        ],
        "阶段三 · 压力测试": [
            st.Page("pages/5_stress.py", title="压力测试交互分析",
                    icon=":material/crisis_alert:"),
            st.Page("pages/robustness.py", title="稳健性检验",
                    icon=":material/verified:"),
        ],
        "阶段四 · 结论与终期": [
            st.Page("pages/1_overview.py", title="概览与业务结论",
                    icon=":material/insights:", default=True),
            st.Page("pages/7_conclusions.py", title="结论分级与已知边界",
                    icon=":material/rule:"),
        ],
    }
)

with st.sidebar:
    st.caption(
        "本看板只读 `results/`、`factors/`、`clean_data/` 下既有的测算产物，"
        "不重算任何数字，也不改写 `code/` 与 `figures/`；每页页脚写明该页读了哪张表。"
    )
    with st.expander("四阶段进度 · git 可追溯"):
        for _p in labels.PHASES:
            st.caption(
                f"阶段{_p['no']} {_p['name']}　{_p['span']}　· 已完成　{_p['commit']}"
            )
        st.caption(
            "进度全部来自已发生的 git 历史，`git log --oneline --reverse` 逐条可溯；"
            "图表对应关系见各阶段页、结论文案见结论分级页。"
        )

NAV.run()
