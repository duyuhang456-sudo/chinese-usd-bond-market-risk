"""公共工具：项目路径 / 数据窗口 / 保存函数

原始数据统一存入 raw_data/；本模块只负责「下载并原样归档」，日期只做 ISO 化与窗口截取，
不做清洗（清洗是第 2 步的任务）。

本模块是全仓库路径常量的唯一来源。各脚本一律 `from common import ...`，不再在脚本顶部各自
`REPO / "..."`——此前 `CLEAN`/`FACT`/`RES`/`FIG` 四个常量在多个脚本里重复定义，改一处输出
位置要动十几个文件。逐脚本的原定义位置与收敛依据见 `docs/tool_path_convergence.md`。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 仓库根目录 = 本文件上一级的上一级
REPO: Path = Path(__file__).resolve().parent.parent
RAW: Path = REPO / "raw_data"
CLEAN: Path = REPO / "clean_data"
FACT: Path = REPO / "factors"
RES: Path = REPO / "results"
FIG: Path = REPO / "figures"

# 风险事件时间线：目录与其中的 CSV 分列两个常量。
# 二者曾被写成 `EV`（目录）与 `EVENTS`（CSV 文件）两个相近的名字，
# 看着像同一事物的两种拼写、实际指向不同类型，改名合并会造出同名异指。
EVENTS_DIR: Path = REPO / "events"
EVENTS_CSV: Path = EVENTS_DIR / "risk_events_timeline.csv"

# 脚本会写入的目录（raw_data 由 raw_path() 按需创建，events 是人工维护的输入）
OUT_DIRS: tuple[Path, ...] = (CLEAN, FACT, RES, FIG)

# 近 5 年数据窗口（含 1 个月缓冲，供后续滚动统计/回测使用）
START: str = "2021-08-01"


def ensure_dirs(*paths: Path) -> None:
    """创建输出目录，已存在的跳过。

    参数：
        *paths: 要创建的目录。不传时创建 OUT_DIRS 里的全部输出目录。

    返回：
        None。

    备注：
        parents=True 在此写死。此前 18 处模块级 mkdir 有 4 处漏了它
        （stress_impact / stress_scenarios / stress_robustness / var_backtest），
        因仓库根必然存在、父目录总在才没报错；输出一旦改到仓库外即 FileNotFoundError。
    """
    for p in (paths or OUT_DIRS):
        p.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    """返回 raw_data 下的目标路径，并确保该目录已存在。

    参数：
        name: 文件名，如 "nav_9141HK.csv"。

    返回：
        raw_data/<name> 的完整路径。只保证目录存在，不创建文件。
    """
    ensure_dirs(RAW)
    return RAW / name


def save_csv(df: pd.DataFrame, name: str, date_col: str = "date") -> Path:
    """把原始数据按日期升序、去重后写入 raw_data。

    参数：
        df: 待保存的数据表。
        name: 文件名。
        date_col: 日期列名。该列存在时按它升序排序、按它去重（保留最后一条）
            并格式化为 YYYY-MM-DD；不存在时原样写出。

    返回：
        写出的文件路径。
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    p = raw_path(name)
    df.to_csv(p, index=False)
    print(f"[saved] {p.name}  rows={len(df)}")
    return p


def write_table(df: pd.DataFrame, path: Path) -> None:
    """写出结果表：utf-8-sig（带 BOM）、不带索引，并把 IEEE 负零规整为 0.0。

    参数：
        df: 待写出的结果表。
        path: 输出路径。

    返回：
        None。

    备注：
        负零来自 round(-0.0, 4)，纯汇率情景主口径那个恒为零的分量即是一例。
        它数值上等于 0，却会打印成 "-0.0000"，与文档中「恒为 0」的表述冲突，
        读表时像是笔误。此处只在写盘时规整符号，不改动任何数值。
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c] + 0.0  # 把 -0.0 规整成 0.0，NaN 不变；删掉本行，CSV 会重新写出 -0.0000
    out.to_csv(path, index=False, encoding="utf-8-sig")
