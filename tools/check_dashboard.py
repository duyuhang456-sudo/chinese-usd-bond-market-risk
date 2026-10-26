"""看板冒烟检查。

在无浏览器的环境里，这是唯一能真正执行页面逻辑的办法：用 Streamlit 官方的
AppTest 把每个页面文件跑一遍，断言没有抛异常、并抽查关键元素是否渲染。

查的不只是「没报错」，还有这一版定下的几条硬约束：

1. 页面集合与 `app.py` 的单页装配一致：`pages/sections/` 下 0_cover、1_data_factor、
   2_var_models、3_stress、4_robustness、5_verdict 共六段，顺序由 app.py 的
   SECTIONS 列表给定；多一个少一个都算不过。
2. **外观全部走 `.streamlit/config.toml`，页面源码不注入任何 HTML/CSS**——
   源码里出现 `unsafe_allow_html`、`jpm-` 自定义 class、或 `lib/style.py`
   复活，都在这里拦下。这是 Streamlit 官方设计规范的做法。唯一例外是
   `lib/toc.py`：右下角目录胶囊与回到顶部圆钮必须 `position: fixed` 才能钉在
   视口上，Streamlit 没有原生组件能做，于是把 `st.html()` 收紧到该文件的唯一
   一处调用（`_HTML_ALLOWED`），HTML 注入开关仍然全库禁用。
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

# 单页长文档·5 主题分组的固定清单：相对 dashboard/pages 的路径 → 标题
# （子目录 sections/ 是允许的；上一轮的 5 步工作流全部下线，新骨架为单页可滚文档）
PAGES = {
    "sections/0_cover.py": "本页一览",
    "sections/1_data_factor.py": "数据与因子概览",
    "sections/2_var_models.py": "VaR 模型对比看板",
    "sections/3_stress.py": "压力测试交互分析",
    "sections/4_robustness.py": "稳健性检验探索",
    "sections/5_verdict.py": "结论与边界",
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

# 悬浮件的唯一豁免：st.html() 只允许出现在 lib/toc.py，且只出现一次。
# 其余源码里出现 st.html() 或者允许 HTML 注入的开关，仍然算不过。
_HTML_ALLOWED = DASH / "lib" / "toc.py"
_HTML_MAX_CALLS = 1


def check_html_gate() -> list:
    """全 dashboard/ 扫描 HTML 注入，返回问题清单。"""
    problems = []
    for path in sorted(DASH.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        if "unsafe_allow_html" in src:
            problems.append(f"{rel} 出现 unsafe_allow_html：HTML 注入开关全库禁用")
        calls = src.count("st.html(")
        if not calls:
            continue
        if path != _HTML_ALLOWED:
            problems.append(
                f"{rel} 出现 st.html(...)：只允许放 "
                f"{_HTML_ALLOWED.relative_to(ROOT)}")
        elif calls > _HTML_MAX_CALLS:
            problems.append(
                f"{rel} 出现 {calls} 处 st.html(...)：悬浮件只该有一处")
    return problems


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
    """页内控件数。

    主区是零控件的（口径/报价/模型/置信度/缓冲都收在 app.py 顶部那颗 popover 里），
    所以这里数出来的正常值是 0；`segmented_control` 落在 `button_group` 上而不是
    radio，只放它的那几页早先会显示成「控件=0」，看不出控件在不在，所以一并计入。
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
        # 「每张图配一个表格孪生」只对有图的页面成立
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

    for msg in check_html_gate():
        print(f"[FAIL] {msg}")
        bad += 1

    page_dir = DASH / "pages"
    found = {str(p.relative_to(page_dir)) for p in page_dir.rglob("*.py")}
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
    print(f"单页长文档的全部 {len(PAGES)} 段通过（实际 {len(found)} 段；"
          f"数据抽屉标题互不重复，共 {len(seen_titles)} 个）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
