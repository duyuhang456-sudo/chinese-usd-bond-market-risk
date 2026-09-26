"""产物对账：核对 results/ 下每张 CSV 的行列数，防「脚本改了、README 没跟」。

全链路的脚本只保证跑得出结果，不保证结果跟文档写的一致。9/25 那次 1 日重做出了 28 处缺陷，
其中一个成因就是数据改了、引用它的文档没同步。这里把 README「结果目录」两节里声明的行×列
抄成期望值逐张核对，哪个脚本改了输出形状而文档没更新，当场就报出来。

期望值的来源有两类，表里逐行标了：
  [R] = README「结果目录」小节明写了行数/列数；
  [M] = README 没写列数，靠实测补上的（行数跟 [R] 一致）。

用法：
    python code/check_outputs.py          # 对账，不一致返回码 1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from common import REPO, RES  # noqa: E402

# (文件名, 期望行数, 期望列数, 依据)
# 阶段二 17 张
PHASE2 = [
    ("baseline_var.csv", 2, 7, "[M]"),
    ("baseline_var_tr.csv", 3, 8, "[M]"),
    ("factor_exposure.csv", 4, 3, "[M]"),
    ("factor_corr.csv", 3, 4, "[M]"),
    ("regimes.csv", 1273, 5, "[M]"),
    ("doubtful_days.csv", 16, 11, "[M]"),
    ("var_parametric.csv", 1023, 30, "[R] 行"),
    ("var_attribution.csv", 4, 3, "[M]"),
    ("garch_refit_trace_main.csv", 1273, 4, "[M]"),
    ("garch_refit_trace_rmb.csv", 1268, 4, "[M]"),
    ("var_historical.csv", 1023, 27, "[R] 行"),
    ("backtest_results.csv", 262, 33, "[R] 行列"),
    ("backtest_exceptions.csv", 832, 12, "[R] 行"),
    ("model_scorecard.csv", 32, 26, "[M]"),
    ("baseline_decision.csv", 16, 17, "[M]"),
    ("delta_transmission_events.csv", 1500, 18, "[R] 行"),
    ("delta_transmission_summary.csv", 24, 18, "[R] 行"),
]

# 阶段三 9 张
PHASE3 = [
    ("stress_scenarios.csv", 112, 15, "[R] 行列"),
    ("stress_impact.csv", 30, 38, "[R] 行列"),
    ("stress_factor_contrib.csv", 150, 10, "[R] 行列"),
    ("stress_coverage.csv", 30, 15, "[R] 行列"),
    ("stress_worst_windows.csv", 80, 8, "[R] 行列"),
    ("stress_window_sens.csv", 50, 12, "[R] 行列"),
    ("stress_shift_coverage.csv", 8, 13, "[R] 行列"),
    ("stress_buffer_sens.csv", 150, 9, "[R] 行列"),
    ("stress_anchor_sens.csv", 12, 10, "[R] 行列"),
]

EXPECTED = PHASE2 + PHASE3


def check(verbose: bool = True) -> list[str]:
    """核对 results/ 下每张 CSV 的行列数，再反查有没有没登记的 CSV。

    verbose=True 就逐张打印 OK 行和汇总，False 只收问题不打字。返回不一致清单，每项是一句
    能直接打印的说明，空列表就是全对上了。清单里是两类问题：形状跟 EXPECTED 对不上，以及
    results/ 下冒出期望表里没有的文件。
    """
    bad: list[str] = []
    if verbose:
        print(f"  对账 results/ 下 {len(EXPECTED)} 张 CSV"
              f"（阶段二 {len(PHASE2)} + 阶段三 {len(PHASE3)}）")
    for name, exp_r, exp_c, src in EXPECTED:
        p = RES / name
        if not p.exists():
            bad.append(f"{name}: 文件不存在")
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: 读取失败 {e}")
            continue
        got = (len(df), len(df.columns))
        if got != (exp_r, exp_c):
            bad.append(f"{name}: 实测 {got[0]} 行 × {got[1]} 列，期望 {exp_r} 行 × {exp_c} 列（{src}）")
        elif verbose:
            print(f"    OK  {name:<32} {got[0]:>5} 行 × {got[1]:>2} 列")

    # 反着再查一遍：results/ 下不该有 README 没写、期望表里也没有的 CSV
    known = {n for n, *_ in EXPECTED}
    extra = sorted(p.name for p in RES.glob("*.csv") if p.name not in known)
    bad += [f"{n}: 存在于 results/ 但未登记（README 或本表须补）" for n in extra]

    if bad:
        print(f"\n  !! 对账不通过，{len(bad)} 处不一致：")
        for b in bad:
            print(f"     - {b}")
    elif verbose:
        print("  ✓ 全部一致")
    return bad


def main() -> int:
    """打印表头，跑一遍 check()。

    返回退出码：0 是全部一致，1 是有对不上的。
    """
    print("=" * 70)
    print("产物对账（code/check_outputs.py）")
    print("=" * 70)
    bad = check()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
