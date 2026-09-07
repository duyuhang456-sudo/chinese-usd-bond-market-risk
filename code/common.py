"""
公共工具：项目路径 / 数据窗口 / 保存函数

原始数据统一存入 raw_data/；本模块只负责“下载并原样归档”，
日期只做 ISO 化与窗口截取，不做清洗（清洗是第 2 步的任务）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 仓库根目录 = 本文件上一级的上一级
REPO: Path = Path(__file__).resolve().parent.parent
RAW: Path = REPO / "raw_data"

# 近 5 年数据窗口（含 1 个月缓冲，供后续滚动统计/回测使用）
START: str = "2021-08-01"


def raw_path(name: str) -> Path:
    """返回 raw_data 下的目标路径，并确保目录存在。"""
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / name


def save_csv(df: pd.DataFrame, name: str, date_col: str = "date") -> Path:
    """按 date 升序保存原始 CSV，返回路径。"""
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    p = raw_path(name)
    df.to_csv(p, index=False)
    print(f"[saved] {p.name}  rows={len(df)}")
    return p
