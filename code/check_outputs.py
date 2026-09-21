"""产物对账：核对 results/ 下 CSV 的行列维度，防「脚本改了、README 没跟」。

**为什么需要它**：全链路脚本只保证「跑得出结果」，不保证「结果与文档描述一致」。
9/25 那次 1 日重做后出现 28 处缺陷，成因之一就是数据改了、引用它的文档没同步。
本模块把 README「结果目录」两节里声明的行×列固化成期望值，逐张核对——
一旦某个脚本改了输出的形状而文档未更新，这里立即报错。

期望值的来源分两类，表中逐行标注：
  [R] = README「结果目录」小节明文声明的行数/列数；
  [M] = README 未声明列数、由本次实测补全（行数与 [R] 一致）。

用法：
    python code/check_outputs.py          # 对账，失败返回码 1
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
    """返回不一致清单；空列表 = 全部对上。"""
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

    # 反向检查：results/ 下不应有 README 未登记、也未在期望表里的 CSV
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
    print("=" * 70)
    print("产物对账（code/check_outputs.py）")
    print("=" * 70)
    bad = check()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
