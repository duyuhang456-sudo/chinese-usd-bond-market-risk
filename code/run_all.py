"""一键运行入口：把四个阶段的全部脚本按依赖顺序串成一条命令（阶段四 Day1 · 9/21）

消费：不读数据，只是挨个调用 code/ 下的脚本
产出：各脚本自己的产物；跑完由 code/check_outputs.py 统一对账
口径：没有——本脚本只管调用顺序，不碰任何一个数字
边界：取数类脚本失败只记警告继续跑；计算类脚本失败立即终止
用法：python code/run_all.py                       # 跑 phase1+2+3（默认跳过取数）
      python code/run_all.py --only phase2,phase3  # 只跑计量与压力测试
      python code/run_all.py --download            # 连同取数一起跑（需联网）
      python code/run_all.py --check               # 只做产物对账，不重跑测算
      python code/run_all.py --list                # 列出执行清单后退出

以前复现全链路得照 README 手敲 12 条命令，还得自己保证顺序——脚本之间靠 CSV 传参，顺序错
了不会报错，只会静默拿上一次的旧产物接着算。这份清单把那 12 条固化下来，并把失败分成两类：
取数失败不拦，raw_data/ 里有归档数据就能离线重建；计算失败必须停，带着缺料往下跑，后面
的产物会静默基于旧数据，而且不会自己暴露。

给人看的依赖顺序见 docs/tool_usage.md §3「分阶段运行」；本文件下面的 STEPS 才是这份顺序
的唯一可执行来源，真正跑的就是它，code/check_outputs.py 会核对每步的产物形状。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO, ensure_dirs  # noqa: E402

CODE = Path(__file__).resolve().parent
PY = sys.executable

# (阶段, 脚本名, 类别)
# 类别只有两种：download 要联网，失败记个警告接着跑；compute 失败就停。
# build_factors.py / staleness_remedy.py / descriptive_stats.py 不在关键路径上
# （不喂给阶段二），但也是阶段一的交付物，所以一并重建。
STEPS: list[tuple[str, str, str]] = [
    ("phase1", "download_all", "download"),
    ("phase1", "download_nav", "download"),
    ("phase1", "clean_data", "compute"),
    ("phase1", "outlier_detect", "compute"),
    ("phase1", "adjudicate_outliers", "compute"),
    ("phase1", "build_factors", "compute"),
    ("phase1", "build_factors_nav", "compute"),
    ("phase1", "staleness_remedy", "compute"),
    ("phase1", "build_tr_factors", "compute"),
    ("phase1", "descriptive_stats", "compute"),
    ("phase2", "prep_phase2", "compute"),
    ("phase2", "var_parametric", "compute"),
    ("phase2", "var_historical", "compute"),
    ("phase2", "var_backtest", "compute"),
    ("phase2", "recommend_baseline", "compute"),
    ("phase2", "verify_delta_transmission", "compute"),
    ("phase3", "stress_scenarios", "compute"),
    ("phase3", "stress_impact", "compute"),
    ("phase3", "stress_robustness", "compute"),
]

PHASE_NAMES = {
    "phase1": "一：数据体系搭建与风险因子拆解",
    "phase2": "二：风险计量模型开发与回测验证",
    "phase3": "三：压力测试框架构建与测算分析",
}


def banner(t: str) -> None:
    """打印一条标题，上下各加一道分隔线。t 是要显示的文本。"""
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def list_steps() -> None:
    """按阶段分组打印 STEPS 的清单，只打印，一个脚本都不跑。"""
    banner("执行清单")
    cur = None
    for i, (phase, script, kind) in enumerate(STEPS, 1):
        if phase != cur:
            cur = phase
            print(f"\n  {phase}  {PHASE_NAMES[phase]}")
        tag = "取数" if kind == "download" else "计算"
        print(f"    {i:>2}. [{tag}] code/{script}.py")


def run_step(script: str, kind: str, quiet: bool) -> tuple[bool, float]:
    """开个子进程跑一个脚本，返回它成没成、花了多少秒。

    script 是脚本名，不带目录也不带 .py 后缀，比如 "stress_impact"。kind 是类别
    （"download" 还是 "compute"），这个函数体里其实用不到——失败了要不要中断，是调用方
    按类别判断的，参数留着只是为了跟 STEPS 的三元组对上。quiet 为 True 时把子进程的
    输出收起来，只在失败时打印末尾 2000 字符；为 False 就直接透到终端上。

    返回 (成没成, 秒数) 这样一个二元组；脚本文件不存在就返回 (False, 0.0)，不抛异常。
    """
    path = CODE / f"{script}.py"
    if not path.exists():
        print(f"  !! 脚本不存在：{path}")
        return False, 0.0
    t0 = time.perf_counter()
    cmd = [PY, str(path)]
    if quiet:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        ok = r.returncode == 0
        if not ok:
            print(r.stdout[-2000:] if r.stdout else "")
            print(r.stderr[-2000:] if r.stderr else "")
    else:
        r = subprocess.run(cmd, cwd=REPO)
        ok = r.returncode == 0
    return ok, time.perf_counter() - t0


def main() -> int:
    """一键运行入口：解析参数、按依赖顺序跑脚本、跑完做一次产物对账。

    参数从命令行读，认 --only / --download / --check / --quiet / --list 这几个。
    返回的是进程退出码，只有三种：0 = 全部成功且对账通过；1 = 计算类脚本失败，
    或者产物对账没过；2 = 阶段名写错了。
    """
    ap = argparse.ArgumentParser(
        description="中资投资级美元债风险计量与压力测试预研工具 · 一键运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python code/run_all.py --only phase2,phase3",
    )
    ap.add_argument("--only", default="phase1,phase2,phase3",
                    help="只跑指定阶段，逗号分隔（默认全部）：phase1,phase2,phase3")
    ap.add_argument("--download", action="store_true",
                    help="连同取数脚本一起跑（需联网）。默认跳过取数，"
                         "直接用已归档的 raw_data/ 复现——取数失败不该阻断离线复现。")
    ap.add_argument("--check", action="store_true",
                    help="只对账现有产物（code/check_outputs.py），不重跑任何测算。")
    ap.add_argument("--quiet", action="store_true",
                    help="抑制各脚本的正常输出，只在失败时打印末尾若干行。")
    ap.add_argument("--list", action="store_true", help="列出执行清单后退出。")
    args = ap.parse_args()

    if args.list:
        list_steps()
        return 0

    if args.check:
        from check_outputs import main as check_main
        return check_main()

    phases = [p.strip() for p in args.only.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASE_NAMES]
    if unknown:
        print(f"未知阶段：{unknown}；可选 {list(PHASE_NAMES)}")
        return 2

    steps = [s for s in STEPS if s[0] in phases]
    banner("中资投资级美元债风险计量与压力测试预研工具 · 一键运行")
    print(f"  仓库      {REPO}")
    print(f"  解释器    {PY}")
    print(f"  阶段      {', '.join(phases)}（共 {len(steps)} 步）")
    print(f"  取数      {'跑（--download）' if args.download else '跳过（默认；加 --download 开启）'}")

    ensure_dirs()

    ok_n = skip_n = 0
    warns: list[str] = []
    t_all = time.perf_counter()

    for phase, script, kind in steps:
        if kind == "download" and not args.download:
            print(f"\n[跳过] {script}（取数，未加 --download）")
            skip_n += 1
            continue
        tag = "取数" if kind == "download" else "计算"
        print(f"\n{'-' * 78}\n[{tag}] code/{script}.py\n{'-' * 78}")
        ok, dt = run_step(script, kind, args.quiet)
        if ok:
            ok_n += 1
            print(f"  ✓ {script} 完成  {dt:.1f}s")
            continue
        if kind == "download":
            warns.append(script)
            print(f"  ! {script} 失败（取数类，记警告并继续）——"
                  f"若 raw_data/ 已有归档数据，后续计算不受影响")
            continue
        banner(f"终止：计算类脚本 {script} 失败")
        print(f"  计算类脚本失败必须终止，不得带着缺料往下跑——"
              f"否则后续产物会静默基于旧数据，且不会自我暴露。")
        print(f"  已用 {time.perf_counter() - t_all:.1f}s；请先修复 code/{script}.py。")
        return 1

    # 全部跑完，对一次账
    banner("产物对账")
    from check_outputs import check
    bad = check()

    banner("汇总")
    warn_txt = f" / 取数警告 {len(warns)} 步（{', '.join(warns)}）" if warns else ""
    print(f"  成功 {ok_n} 步 / 跳过 {skip_n} 步{warn_txt}")
    print(f"  耗时 {time.perf_counter() - t_all:.1f}s")
    if bad:
        print(f"  !! 产物对账未通过：{len(bad)} 处不一致（见上）")
        return 1
    print("  ✓ 全部完成，产物对账一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
