"""看板冒烟检查。

在无浏览器的环境里，这是唯一能真正执行页面逻辑的办法：用 Streamlit 官方的
AppTest 把每个页面文件跑一遍，断言没有抛异常、并抽查关键元素是否渲染。

查的不只是「没报错」，还有这一版定下的几条硬约束：

1. 页面集合与 `app.py` 的四阶段导航一致：命名页六页（1_overview、2_data_factors、
   3_var_backtest、4_model_validation、5_stress、7_conclusions）+ 阶段拆出的
   data_quality（阶段一）、robustness（阶段三）共八页，多一个少一个都算不过。
2. **外观全部走 `.streamlit/config.toml`，页面源码不注入任何 HTML/CSS**——
   源码里出现 `unsafe_allow_html`、`jpm-` 自定义 class、或 `lib/style.py`
   复活，都在这里拦下。这是 Streamlit 官方设计规范的做法。
3. 每页有原生页头（`st.title`）与读数带（`st.metric`），页脚写明数据来源。
4. 每页至少一个数据抽屉，且**抽屉标题全库唯一**——此前 14 个折叠块全叫
   「表格视图」，读者展开前不知道里面是什么。
5. 页面源码里没有 emoji 码位（中文界面，沿报告体例；`:material/` 图标是 SVG
   短码，不在禁令内），不出现 `st.divider`（官方：dividers look heavy）。
6. 页面源码不向 `results/`、`figures/` 等产物目录写入。

用法（仓库根）：

    .venv/bin/python tools/check_dashboard.py

它只读产物、不写任何东西，因此跑完 results/ 与 figures/ 的 git 状态必须仍然干净。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"

from streamlit.testing.v1 import AppTest  # noqa: E402

# 七页的固定清单：文件名 → 侧栏/页内标题
PAGES = {
    "1_overview.py": "概览与业务结论",
    "2_data_factors.py": "数据与因子概览",
    "3_var_backtest.py": "VaR 模型对比与回测",
    "4_model_validation.py": "模型诊断与验证",
    "5_stress.py": "压力测试交互分析",
    "7_conclusions.py": "结论分级与已知边界",
    "data_quality.py": "数据质量",
    "robustness.py": "稳健性检验",
}

# 源码里不该再出现的东西：既包括旧接口，也包括「用 HTML/CSS 补外观」这条路
_FORBIDDEN_SRC = [
    ("unsafe_allow_html", "注入 HTML/CSS（外观只走 config.toml）"),
    ("jpm-", "残留的自定义 class 标记"),
    ("st.divider(", "st.divider（官方：dividers look heavy）"),
    ("page.section(", "page.section 已废，改用 page.card"),
    ("conclusion_band(", "conclusion_band 已废，改用 page.kpi_row"),
    ("stat_tile(", "stat_tile 已废，改用 page.kpi_row"),
    ("table_view(", "table_view 已废，改用 charts.data_drawer"),
    ("card_banner(", "card_banner 已废，改用 page.card"),
    ("style.inject(", "style.inject 已废，外观只走 config.toml"),
]

# emoji 码位：图形符号、杂项符号、装饰符号、区域指示符与变体选择符
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️]")
# 看板只读契约：页面源码里不该出现写产物目录的调用
_WRITE = re.compile(r"(to_csv|savefig|\.write_text|open\([^)]*[\"']w)")


def _count(at: AppTest, kind: str) -> int:
    """数出某类元素。

    1.50 对 arrow_vega_lite_chart 没有专门的访问器（at.arrow_vega_lite_chart
    直接 AttributeError），这类元素在 at.main 里是 UnknownElement，只有 .type
    还留着原型名。所以取不到访问器时退回到按 .type 扫容器。
    """
    try:
        return len(getattr(at, kind))
    except Exception:
        pass
    total = 0
    for container in (getattr(at, "main", []), getattr(at, "sidebar", [])):
        try:
            items = list(container)
        except Exception:
            continue
        total += sum(1 for e in items if getattr(e, "type", None) == kind)
    return total


def _texts(at: AppTest) -> str:
    """页面渲染出来的全部文本，用来找页脚等约定文案。"""
    chunks = []
    for kind in ("markdown", "caption", "title", "header", "subheader"):
        try:
            chunks += [e.value for e in getattr(at, kind)]
        except Exception:
            pass
    return "\n".join(chunks)


def n_controls(at: AppTest) -> int:
    """页内控制带的控件数。

    控制带是这一版「让读者自己跑」的抓手，所以要真的数出来。`segmented_control`
    落在 `button_group` 上而不是 radio——第 5、6 两页的控制带只有它，
    早先漏了这一项，那两页会显示成「控件=0」，看不出控制带在不在。
    """
    kinds = ("radio", "selectbox", "slider", "select_slider", "multiselect", "button_group")
    return sum(len(getattr(at, k)) for k in kinds)


def check_page(name: str, path: Path, seen_titles: set,
               structural: bool = True) -> tuple[bool, str]:
    """跑一页，返回（是否通过, 摘要）。

    structural=False 只查「不抛异常、无 emoji、不写产物」，用于入口 app.py；
    页面文件走全套结构断言。抽屉标题去重靠调用方传入的 seen_titles。
    """
    src = path.read_text(encoding="utf-8")
    problems = []

    hit = _EMOJI.search(src)
    if hit:
        problems.append(f"源码含 emoji {hit.group()!r}")
    hit = _WRITE.search(src)
    if hit:
        problems.append(f"源码疑似写文件 {hit.group()!r}")
    for needle, why in _FORBIDDEN_SRC:
        if needle in src:
            problems.append(f"源码含 {needle!r}：{why}")

    at = AppTest.from_file(str(path), default_timeout=240)
    at.run()
    if at.exception:
        msgs = [f"{e.type}: {e.value}" for e in at.exception]
        return False, "异常 " + " | ".join(msgs)

    drawers = list(at.expander)
    charts = _count(at, "arrow_vega_lite_chart") or _count(at, "vega_lite_chart")
    metrics = _count(at, "metric")
    if structural:
        if not _count(at, "title"):
            problems.append("缺页头（st.title）")
        if not metrics:
            problems.append("缺读数带（st.metric）")
        if "数据来源" not in _texts(at):
            problems.append("缺页脚（数据来源）")
        # 「每张图配一个表格孪生」只对有图的页面成立：概览页没有图，也就没有抽屉
        if charts and not drawers:
            problems.append("有图但没有数据抽屉")
        for ex in drawers:
            if ex.label in seen_titles:
                problems.append(f"抽屉标题重复：{ex.label}")
            seen_titles.add(ex.label)

    parts = [f"读数={metrics}", f"图={charts}", f"抽屉={len(drawers)}",
             f"表={_count(at, 'dataframe')}",
             f"控件={n_controls(at)}"]
    if problems:
        return False, "；".join(problems)
    return True, "、".join(parts)


def main() -> int:
    bad = 0

    # 受控 CSS 那一层已经整个退役；文件复活说明有人在往回走
    if (DASH / "lib" / "style.py").exists():
        print("[FAIL] dashboard/lib/style.py 复活了——外观只走 .streamlit/config.toml")
        bad += 1

    page_dir = DASH / "pages"
    found = {p.name for p in page_dir.glob("*.py")}
    expected = set(PAGES)
    if found != expected:
        for extra in sorted(found - expected):
            print(f"[FAIL] dashboard/pages/{extra}  不在页面清单里（该删还是该改名？）")
            bad += 1
        for missing in sorted(expected - found):
            print(f"[FAIL] dashboard/pages/{missing}  清单里有但文件不存在")
            bad += 1

    targets = [(DASH / "app.py", "入口 app.py", False)] + \
              [(page_dir / n, n, True) for n in sorted(expected)]
    seen_titles: set = set()
    for path, name, structural in targets:
        if not path.exists():
            continue
        ok, msg = check_page(name, path, seen_titles, structural=structural)
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {path.relative_to(ROOT)}  {msg}")
        if not ok:
            bad += 1

    print()
    if bad:
        print(f"{bad} 项未通过。")
        return 1
    print(f"四阶段导航的全部 {len(PAGES)} 个页面通过（实际 {len(found)} 个；"
          f"数据抽屉标题互不重复，共 {len(seen_titles)} 个）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
