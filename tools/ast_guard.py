"""AST 守卫：批量改写注释时，证明可执行代码一字未改。

把每个脚本解析成 AST、剥离全部 docstring 节点后取 dump，与基线快照比对。
注释不进 AST，docstring 被剥掉，所以剩下的任何差异都意味着动了代码。

比对比对象是**快照基线**（tools/.ast_baseline.json）而不是 git HEAD：
工作区里已有上一轮 docstring 整改与一处图标题改动，拿 HEAD 当基线会把已知差异
误报成新破坏。基线在整改开始前拍一次，之后每批改完比一次。

用法：
    ./.venv/bin/python tools/ast_guard.py --snapshot    # 拍基线（整改开始前跑一次）
    ./.venv/bin/python tools/ast_guard.py               # 比对全部 26 个脚本
    ./.venv/bin/python tools/ast_guard.py code/foo.py   # 只比对指定文件

退出码 0 表示一致；1 表示有文件存在非注释差异或解析失败。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / "code"
BASELINE = Path(__file__).resolve().parent / ".ast_baseline.json"


def strip_docstrings(tree: ast.Module) -> ast.Module:
    """就地剥离全部 docstring 节点，返回同一棵树。

    参数：
        tree: 已解析的 AST 模块对象。

    返回：
        传入的那棵树本身（就地修改），调用方不必重新赋值。
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # 函数体被剥空会变成非法 AST，补一个 Pass 占位
            node.body = body[1:] or [ast.Pass()]
    return tree


def fingerprint(path: Path) -> str:
    """算一个文件的「无注释 AST」指纹。

    参数：
        path: 脚本路径。

    返回：
        ast.dump 后的字符串，已剥离全部 docstring。
    """
    return ast.dump(strip_docstrings(ast.parse(path.read_text())))


def all_scripts() -> list[Path]:
    """列出 code/ 下的全部脚本。

    返回：
        按文件名排序的路径列表。
    """
    return sorted(CODE.glob("*.py"))


def do_snapshot() -> int:
    """把当前全部脚本的 AST 指纹写成基线快照。

    返回：
        0 成功；1 有文件解析失败。
    """
    snaps: dict[str, str] = {}
    for p in all_scripts():
        try:
            snaps[p.name] = fingerprint(p)
        except SyntaxError as e:
            print(f"  [语法错误] {p.name}：{e}")
            return 1
    BASELINE.write_text(json.dumps(snaps, ensure_ascii=False, indent=1))
    print(f"基线已拍：{len(snaps)} 个脚本 → {BASELINE.relative_to(REPO)}")
    return 0


def do_compare(names: list[str]) -> int:
    """比对当前脚本与基线快照的 AST 指纹。

    参数：
        names: 要比对的脚本文件名（可含路径前缀）；为空表示全部。

    返回：
        0 全部一致；1 有差异或解析失败。
    """
    if not BASELINE.exists():
        print(f"基线不存在：{BASELINE.relative_to(REPO)}")
        print("先跑一次 --snapshot 拍基线。")
        return 1
    snaps: dict[str, str] = json.loads(BASELINE.read_text())

    targets = all_scripts()
    if names:
        wanted = {Path(n).name for n in names}
        targets = [p for p in targets if p.name in wanted]
        missing = wanted - {p.name for p in targets}
        for m in sorted(missing):
            print(f"  [跳过] {m}：code/ 下无此文件")
    if not targets:
        print("没有要比对的文件")
        return 1

    bad: list[tuple[str, str]] = []
    for p in targets:
        if p.name not in snaps:
            print(f"  [新文件] {p.name}：基线里没有，未参与比对")
            continue
        try:
            cur = fingerprint(p)
        except SyntaxError as e:
            bad.append((p.name, f"语法错误：{e}"))
            continue
        if cur == snaps[p.name]:
            print(f"  [一致] {p.name}")
        else:
            bad.append((p.name, "除注释与 docstring 外仍有差异"))

    print(f"\n比对 {len(targets)} 个文件")
    if bad:
        print("发现问题：")
        for name, why in bad:
            print(f"   {name} -> {why}")
        return 1
    print("全部一致：可执行代码逐节点相同")
    return 0


def main(argv: list[str]) -> int:
    """按命令行分派到拍基线或比对。

    返回：
        do_snapshot / do_compare 的退出码。
    """
    if "--snapshot" in argv:
        return do_snapshot()
    return do_compare([a for a in argv if not a.startswith("-")])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
