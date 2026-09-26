"""公共工具：项目路径、数据窗口、写盘函数（全仓库共用）

原始数据统一走 raw_data/。本模块只管「下载下来原样存好」：日期统一成 ISO 格式、按窗口截
一下，别的什么都不做，清洗是第 2 步的事。

全仓库的路径常量只在本文件里定义一处，各脚本一律 `from common import ...`，不再自己在脚本
顶部拼 `REPO / "..."`。以前 CLEAN/FACT/RES/FIG 这四个常量在十几个脚本里各写一遍，输出位置
一改就得挨个文件动。哪个脚本原来的定义在哪、凭什么并过来，见 docs/tool_path_convergence.md。

边界：只管归档，不清洗、不复权、不判异常。

用法：各脚本 `from common import ...`，取路径常量、ensure_dirs、save_csv、write_table。
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

# 风险事件时间线：目录一个常量、里面的 CSV 一个常量，别合并。
# 这两个原来叫 EV（目录）和 EVENTS（CSV 文件），名字像，指的却是两样东西，
# 并成一个名字会撞车。
EVENTS_DIR: Path = REPO / "events"
EVENTS_CSV: Path = EVENTS_DIR / "risk_events_timeline.csv"

# 脚本要往里写的目录（raw_data 由 raw_path() 现用现建；events 是人工维护的输入，不在其中）
OUT_DIRS: tuple[Path, ...] = (CLEAN, FACT, RES, FIG)

# 数据窗口起点取近 5 年，再往前多留 1 个月当缓冲，给后面的滚动统计和回测攒天数
START: str = "2021-08-01"


def ensure_dirs(*paths: Path) -> None:
    """建输出目录，已经在的就跳过。

    paths 是要建的目录，不传就把 OUT_DIRS 里那几个都建上。

    parents=True 是写死的。以前模块级有 18 处 mkdir，其中 4 处漏了它（stress_impact /
    stress_scenarios / stress_robustness / var_backtest 四个脚本），当时输出都在仓库里、
    父目录总在，所以没炸；哪天输出改到仓库外面，这几处会直接 FileNotFoundError。
    """
    for p in (paths or OUT_DIRS):
        p.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    """给个文件名，返回它在 raw_data 下的完整路径，顺手把 raw_data 建出来。

    name 是文件名，比如 "nav_9141HK.csv"。只保证目录在，不建文件。
    """
    ensure_dirs(RAW)
    return RAW / name


def save_csv(df: pd.DataFrame, name: str, date_col: str = "date") -> Path:
    """把原始数据写进 raw_data：先按日期升序、去重，再落盘。

    df 是要存的表，name 是文件名，date_col 是日期列名。表里有这一列就按它排序、按它去重
    （同一天留最后一条）、并把日期写成 YYYY-MM-DD；没有这列就原样写出。返回写出的路径。
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
    """写结果表：utf-8-sig（带 BOM）、不带索引，顺手把 IEEE 负零抹成 0.0。

    df 是要写的表，path 是输出路径。

    负零是 round(-0.0, 4) 造出来的，纯汇率情景主口径那个恒为零的分量就是一例。它数值上
    明明等于 0，打出来却是 "-0.0000"，跟文档里「恒为 0」的说法对不上，看表的人容易当成
    笔误。这里只在写盘的时候把符号抹平，数值一个都不动。
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c] + 0.0  # 加 0 就是为了把 -0.0 变成 0.0（NaN 不受影响）。看着多余，千万别删：删了 CSV 里会重新冒出 -0.0000
    out.to_csv(path, index=False, encoding="utf-8-sig")
