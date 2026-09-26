"""数据清洗（阶段一 · 第 2 天）：日期对齐 + 缺失值处理

消费：raw_data/ 下的 treasury 收益率曲线、FRED 各序列、9141.HK 行情与官方 NAV、3141.HK 对照线
产出：clean_data/ 下各 *_clean.csv（含 flag 列）、fill_log.csv（逐列逐段填充明细）、
      completeness_report.csv（完整率报告）、master_calendar.csv（主日历）
口径：主日历取 treasury 的日期（即美股交易日），其余序列都对齐到它。单日或连续 ≤3 天缺失用
      前值填充（flag=1），连续 >3 天用两端线性插值（flag=2）；头部/尾部缺失、以及任一端没有
      有效锚点的段一律不填充，留空让下游看得见。每类序列只在其自身有效区间与主日历的交集内
      对齐——OAS 利差 2023-09 前没有数据，那段自然缺失不该被填，否则会把「源数据没有」伪装成
      「补全了」。完整率一律取填充前的真实值。treasury 只保留 1 Mo–30 Yr 标准期限列，剔除
      「4 Mo」「1.5 Month」这类近期才发布的列，免得把「发布晚」误记成「缺失」。月度宏观序列
      不转日度，逐列单独标注。
用法：./.venv/bin/python code/clean_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, CLEAN, ensure_dirs

ensure_dirs()

FILL_FFILL = 1  # 前值填充
FILL_LIN = 2    # 线性插值

# treasury 标准期限列（无风险利率曲线用）
TSY_CORE = ["1 Mo", "2 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr",
            "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]


def read_raw(name: str, date_col: str) -> pd.DataFrame:
    """读入 raw_data/ 下的原始 CSV，并把日期列统一成 date。

    参数：
        name: 原始文件名，如 "fred_DEXCHUS.csv"。
        date_col: 该文件里的日期列名（各源不统一，有 Date / date 两种）。

    返回：
        按日期升序、按日期去重（保留第一条）后的表，日期列为 datetime 且名为 date。
    """
    df = pd.read_csv(RAW / name)
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """把指定列强制转为数值，转不动的置为缺失。

    参数：
        df: 待处理的表。就地修改并返回同一个对象，不复制。
        cols: 要转的列名。

    返回：
        传入的 df 本身。
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def record_runs(dates: pd.DatetimeIndex, fl: pd.Series, file: str, col: str) -> list[dict]:
    """把 flag 序列的连续非零段写成 fill_log 的行。

    参数：
        dates: 与 fl 等长的日期索引，用于取每段的起止日。
        fl: 填充标记序列，0 = 未填充，FILL_FFILL / FILL_LIN = 填充方式。
        file: 记入 fill_log 的源文件名。
        col: 记入 fill_log 的列名。

    返回：
        [{file, column, start, end, length, method}] 的行列表。

    备注：
        当前无调用点——fill_gap_runs() 在填充的同时自行拼好了同样的行。
        保留是因为它按 flag 序列反推区段，可用于复核 fill_gap_runs 的区间是否切对。
    """
    rows: list[dict] = []
    flv = fl.to_numpy()
    k, n = 0, len(flv)
    while k < n:
        if flv[k] > 0:
            k2 = k
            while k2 < n and flv[k2] == flv[k]:
                k2 += 1
            rows.append({
                "file": file, "column": col,
                "start": dates[k].date(), "end": dates[k2 - 1].date(),
                "length": k2 - k,
                "method": "前值填充" if flv[k] == FILL_FFILL else "线性插值",
            })
            k = k2
        else:
            k += 1
    return rows


def fill_gap_runs(s: pd.Series, file: str, col: str) -> tuple[pd.Series, pd.Series, list[dict]]:
    """按缺失段的长度决定填充方式，逐段填充单列。

    参数：
        s: 待填充的序列，索引为日期。
        file: 源文件名，用于写 fill_log。
        col: 列名，用于写 fill_log。

    返回：
        (填充后序列, flag 序列, fill_log 行列表) 三元组。
        flag 与原序列等长：0 未填充、1 前值填充、2 线性插值。

    规则：
        连续缺失 ≤3 天用前值填充，>3 天用两端线性插值。
        头部（段首即缺失）与尾部（段尾仍缺失）不填充，任一端无有效值也不填充——
        没有可参照的锚点时插值只是编数，不如留空让下游看见。
    """
    out = s.copy().astype(float)
    flag = pd.Series(0, index=s.index, dtype=int)
    vals = s.isna().to_numpy()
    dates = pd.DatetimeIndex(s.index)
    runs: list[dict] = []
    i, n = 0, len(s)
    while i < n:
        if vals[i]:
            j = i
            while j < n and vals[j]:
                j += 1
            L = j - i
            if i == 0 or j == n:                    # 头部 / 尾部缺失：不填充
                i = j
                continue
            prev, nxt = s.iloc[i - 1], s.iloc[j]
            if pd.isna(prev) or pd.isna(nxt):       # 边界未知：不填充
                i = j
                continue
            if L <= 3:
                out.iloc[i:j] = prev
                flag.iloc[i:j] = FILL_FFILL
            else:
                out.iloc[i:j] = np.linspace(prev, nxt, L + 2)[1:-1]
                flag.iloc[i:j] = FILL_LIN
            runs.append({
                "file": file, "column": col,
                "start": dates[i].date(), "end": dates[j - 1].date(),
                "length": L,
                "method": "前值填充" if L <= 3 else "线性插值",
            })
            i = j
        else:
            i += 1
    return out, flag, runs


def clean_daily(name: str, val_cols: list[str], master: pd.DatetimeIndex,
                date_col: str = "date") -> tuple[pd.DataFrame, dict, list[dict]]:
    """把一条日度序列对齐到主日历并填充缺失。

    参数：
        name: raw_data/ 下的原始文件名。
        val_cols: 要清洗的数值列。
        master: 主日历（美股交易日）。
        date_col: 原始文件里的日期列名。

    返回：
        (清洗后的表, 报告 dict, fill_log 行列表) 三元组。
        报告 dict 含有效区间起止、期望天数、填充前完整单元格数、完整率百分比与各方式填充量。

    规则：
        只在该序列自身的有效区间与主日历的交集内对齐——OAS 利差 2023-09 之前没有数据，
        那段自然缺失不该被当成「缺失值」去填，否则会把「源数据没有」伪装成「补全了」。
        完整率一律取填充前的真实值。
    """
    df = read_raw(name, date_col)
    df = to_num(df, val_cols)

    tmin = max(master.min(), df["date"].min())      # 有效区间 ∩ 主日历
    tmax = min(master.max(), df["date"].max())
    grid = pd.DatetimeIndex([d for d in master if tmin <= d <= tmax])

    panel = df.set_index("date").reindex(grid)      # 缺失在主日历上暴露
    exp = int(len(grid) * len(val_cols))
    present_before = int(panel[val_cols].notna().sum().sum())   # 填充前的真实完整率
    flags = pd.Series(0, index=grid, dtype=int)
    runs: list[dict] = []
    for c in val_cols:
        filled, fl, rr = fill_gap_runs(panel[c], name, c)
        panel[c] = filled
        flags = flags.combine(fl, max)
        runs += rr
    panel["flag"] = flags.values
    panel = panel.reset_index().rename(columns={"index": "date"})

    meta = {
        "file": name, "frequency": "日度",
        "span_start": grid[0].date(), "span_end": grid[-1].date(),
        "expected_days": len(grid),
        "present_before_fill": present_before,
        "completeness_pct": round(present_before / exp * 100, 2),
        "ffill_cells": sum(r["length"] for r in runs if r["method"] == "前值填充"),
        "lin_cells": sum(r["length"] for r in runs if r["method"] == "线性插值"),
        "fill_runs": len(runs),
    }
    return panel, meta, runs


def clean_monthly(name: str, val_cols: list[str]) -> tuple[pd.DataFrame, dict, list[dict]]:
    """清洗月度宏观序列：排序去重、填补缺失、逐列标注。

    参数：
        name: raw_data/ 下的原始文件名。
        val_cols: 要清洗的数值列。

    返回：
        (清洗后的表, 报告 dict, fill_log 行列表) 三元组。
        每列各带一个 `<列名>_flag` 标记列（日度序列是全表共用一个 flag，
        月度这里逐列分标，因为月度序列短、单列的填充情况需要单独看清）。

    备注：
        月度序列不转日度。频率不同，向下展开只会造出并不存在的观测。
    """
    df = read_raw(name, "date")
    df = to_num(df, val_cols)
    exp = int(len(df) * len(val_cols))
    present_before = int(df[val_cols].notna().sum().sum())
    runs: list[dict] = []
    for c in val_cols:
        filled, fl, rr = fill_gap_runs(df[c], name, c)
        df[c] = filled
        df[f"{c}_flag"] = fl.values
        runs += rr
    n_ff = sum(r["length"] for r in runs if r["method"] == "前值填充")
    n_lin = sum(r["length"] for r in runs if r["method"] == "线性插值")
    meta = {
        "file": name, "frequency": "月度",
        "span_start": df["date"].min().date(), "span_end": df["date"].max().date(),
        "expected_days": len(df),
        "present_before_fill": present_before,
        "completeness_pct": round(present_before / exp * 100, 2),
        "ffill_cells": n_ff, "lin_cells": n_lin, "fill_runs": len(runs),
    }
    return df, meta, runs


def save_clean(df: pd.DataFrame, name: str) -> None:
    """把清洗结果写到 clean_data/，日期格式化为 YYYY-MM-DD。

    参数：
        df: 待写出的表。
        name: 输出文件名。

    返回：
        None。
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(CLEAN / name, index=False, encoding="utf-8-sig")
    print(f"  [保存] clean_data/{name}  rows={len(out)}")


def main() -> None:
    """阶段一第 2 步的入口：建主日历，清洗全部日度与月度序列。

    脚本契约：
        消费：raw_data/ 下的 treasury 曲线、FRED 各序列、9141.HK 行情与 NAV、
            3141.HK 备用对照线。
        产出：clean_data/ 下的 *_clean.csv、master_calendar.csv、
            completeness_report.csv、fill_log.csv。
        主日历：以 treasury 的日期为准（即美股交易日），其余序列都对齐到它。

    返回：
        None。
    """
    # 主日历 = treasury 日期（美股交易日）
    tsy_raw = read_raw("treasury_yield_curve.csv", "date")
    master = pd.DatetimeIndex(sorted(tsy_raw["date"]))
    pd.DataFrame({"date": master.strftime("%Y-%m-%d")}).to_csv(
        CLEAN / "master_calendar.csv", index=False)
    print(f"[主日历] {len(master)} 个美股交易日："
          f"{master[0].date()} ~ {master[-1].date()}")

    report: list[dict] = []
    all_runs: list[dict] = []

    # 1) treasury：标准期限列，主日历自身（无需 reindex）
    tsy_cols = [c for c in TSY_CORE if c in tsy_raw.columns]
    tsy = to_num(tsy_raw.copy(), tsy_cols)[["date", *tsy_cols]]
    dates = pd.DatetimeIndex(tsy["date"])
    exp = int(len(tsy) * len(tsy_cols))
    present_before = int(tsy[tsy_cols].notna().sum().sum())
    runs: list[dict] = []
    for c in tsy_cols:
        filled, fl, rr = fill_gap_runs(tsy[c], "treasury_yield_curve.csv", c)
        tsy[c] = filled
        runs += rr
    all_runs += runs
    tsy["flag"] = 0                             # 统一携带 flag 列（本表无缺失则全 0）
    save_clean(tsy, "treasury_yield_curve_clean.csv")
    report.append({
        "file": "treasury_yield_curve.csv", "frequency": "日度(主日历)",
        "span_start": dates[0].date(), "span_end": dates[-1].date(),
        "expected_days": len(tsy), "present_before_fill": present_before,
        "completeness_pct": round(present_before / exp * 100, 2),
        "ffill_cells": sum(r["length"] for r in runs if r["method"] == "前值填充"),
        "lin_cells": sum(r["length"] for r in runs if r["method"] == "线性插值"),
        "fill_runs": len(runs),
    })

    # 2) 其余日度序列对齐主日历（汇率 / OAS 利差 / 标的净值）
    #    (原始文件, 数值列, 日期列, 输出 clean 文件名)
    for name, cols, dc, out in [
        ("fred_DEXCHUS.csv", ["DEXCHUS"], "date", "fred_DEXCHUS_clean.csv"),
        ("fred_BAMLEMIBHGCRPIOAS.csv", ["BAMLEMIBHGCRPIOAS"], "date",
         "fred_BAMLEMIBHGCRPIOAS_clean.csv"),
        ("benchmark_9141HK.csv", ["Close", "Adj Close"], "Date",
         "benchmark_9141HK_clean.csv"),
        # HKD 柜台 3141.HK（备用对照线，用于 9141 报价陈旧度对比实验）
        ("alt_3141HK_AsiaUSDIG_HKD.csv", ["Close", "Adj Close"], "Date",
         "benchmark_3141HK_clean.csv"),
        # 9141.HK 官方日度 NAV（USD/单位，MoneyDJ 镜像，官方锚点已验证）
        ("nav_9141HK_MoneyDJ.csv", ["nav_usd"], "date",
         "nav_9141HK_clean.csv"),
    ]:
        panel, meta, runs = clean_daily(name, cols, master, date_col=dc)
        save_clean(panel, out)
        report.append(meta)
        all_runs += runs
        print(f"    {name}: 填充 {len(runs)} 段 / {meta['ffill_cells'] + meta['lin_cells']} 单元格")

    # 3) 月度宏观序列（单独清洗，不转日度）
    for name, cols in [("fred_FEDFUNDS.csv", ["FEDFUNDS"]),
                       ("fred_CPIAUCSL.csv", ["CPIAUCSL"])]:
        df, meta, runs = clean_monthly(name, cols)
        save_clean(df, name.replace(".csv", "_clean.csv"))
        report.append(meta)
        all_runs += runs

    rep = pd.DataFrame(report)
    rep.to_csv(CLEAN / "completeness_report.csv", index=False, encoding="utf-8-sig")
    if all_runs:
        pd.DataFrame(all_runs).to_csv(CLEAN / "fill_log.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 76)
    print("完整率 / 填充报告  completeness_report.csv")
    print("=" * 76)
    print(rep.to_string(index=False))
    print(f"\n填充明细共 {len(all_runs)} 段 → fill_log.csv")


if __name__ == "__main__":
    main()
