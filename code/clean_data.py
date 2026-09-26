"""数据清洗：日期对齐 + 缺值处理（阶段一 · 第 2 天）

消费：raw_data/ 下的 treasury 收益率曲线、FRED 各序列、9141.HK 行情与官方 NAV、3141.HK 对照线
产出：clean_data/ 下各 *_clean.csv（含 flag 列）、fill_log.csv（逐列逐段的填充明细）、
      completeness_report.csv（完整率报告）、master_calendar.csv（主日历）
口径：主日历拿 treasury 的日期（也就是美股交易日），别的序列都往它上面靠。缺一天、或者连着
      缺不超过 3 天，拿前一个有效值顶上（flag=1）；连着缺超过 3 天，用两头的值拉条直线取中间
      （flag=2）。开头的缺、末尾的缺、以及两端里有一端压根没有有效值的段，一律不填，留空让
      下游看得见。每条序列只在自己的有效区间和主日历的交集里对齐：OAS 利差 2023-09 之前本来
      就没有数据，那段空是源头没给，不是缺值，填了等于把「源数据没有」伪装成「补全了」。完整
      率一律按填充前的真实值算。treasury 只留 1 Mo–30 Yr 这些标准期限列，把「4 Mo」
      「1.5 Month」这类最近才发布的列剔掉，免得把「发布晚」记成「缺失」。月度宏观序列不转日度，
      一列一列单独标。
用法：./.venv/bin/python code/clean_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, CLEAN, ensure_dirs

ensure_dirs()

FILL_FFILL = 1  # 前值填充：拿上一个有效值顶着
FILL_LIN = 2    # 线性插值：两头连条直线取中间

# treasury 标准期限列（无风险利率曲线用）
TSY_CORE = ["1 Mo", "2 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr",
            "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]


def read_raw(name: str, date_col: str) -> pd.DataFrame:
    """读 raw_data/ 里的原始 CSV，把日期列统一改名叫 date。

    name 是原始文件名（如 "fred_DEXCHUS.csv"），date_col 是那个文件里日期列叫什么，各家源
    文件不统一，有 Date 也有 date。返回按日期升序、按日期去重（同一天留第一条）的表，
    日期列是 datetime。
    """
    df = pd.read_csv(RAW / name)
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """把指定几列强转成数值，转不动的算缺失。

    df 是要处理的表，cols 是要转的列名。就地改、返回的还是同一个对象，不另复制一份。
    """
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def record_runs(dates: pd.DatetimeIndex, fl: pd.Series, file: str, col: str) -> list[dict]:
    """照着 flag 序列，把一段段连续的填充还原成 fill_log 里的行。

    dates 是跟 fl 等长的日期索引，用来取每段起止哪一天；fl 是填充标记，0 是没填，
    FILL_FFILL / FILL_LIN 是填充方式；file 和 col 是要记进 fill_log 的源文件名和列名。
    返回一行一段的列表，字段是 file / column / start / end / length / method。

    现在没人调它，fill_gap_runs() 填充的时候顺手就把这些行拼好了。留着是因为它走的是反
    方向：从 flag 序列倒推区间，可以拿来核对 fill_gap_runs 的段切得对不对。
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
    """看缺失段有多长，一段一段地填一列。

    s 是要填的序列（索引是日期），file 和 col 用来写 fill_log。返回三样东西：填完的序列、
    跟原序列等长的 flag（0 没填 / 1 前值填充 / 2 线性插值）、fill_log 的行列表。

    规则是连着缺不超过 3 天拿前值填，超过 3 天用两头线性插值，中间拉条直线取点。开头就缺、
    末尾还缺、以及某一端压根没有有效值的段，都不填：两头都没有锚点，插值就成了编数，
    不如留空让下游看见。
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
            if i == 0 or j == n:                    # 开头或末尾缺：不填
                i = j
                continue
            prev, nxt = s.iloc[i - 1], s.iloc[j]
            if pd.isna(prev) or pd.isna(nxt):       # 某一端没有有效值：不填
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
    """把一条日度序列对齐到主日历，再补缺失。

    name 是 raw_data/ 下的原始文件名，val_cols 是要清洗的数值列，master 是主日历（美股
    交易日），date_col 是原始文件里的日期列名。返回清洗后的表、一份报告、fill_log 行列表；
    报告里有有效区间起止、应有天数、填充前的完整格数、完整率百分比，以及两种填充各填了多少。

    只在这条序列自己的有效区间和主日历的交集里对齐：OAS 利差 2023-09 之前没有数据，那段空
    是源头就没有，不该当缺失去填，填了等于把「源数据没有」伪装成「补全了」。完整率一律按
    填充前的真实值算。
    """
    df = read_raw(name, date_col)
    df = to_num(df, val_cols)

    tmin = max(master.min(), df["date"].min())      # 起止都取「序列有效区间」和「主日历」里更窄的那头
    tmax = min(master.max(), df["date"].max())
    grid = pd.DatetimeIndex([d for d in master if tmin <= d <= tmax])

    panel = df.set_index("date").reindex(grid)      # 摆到主日历上，哪天缺一目了然
    exp = int(len(grid) * len(val_cols))
    present_before = int(panel[val_cols].notna().sum().sum())   # 填充前的真实完整率，先数下来
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
    """清洗月度宏观序列：排序去重、补缺失、逐列标注。

    name 是 raw_data/ 下的原始文件名，val_cols 是要清洗的数值列。返回清洗后的表、报告、
    fill_log 行列表。每列配一个 `<列名>_flag` 标记列。日度序列是全表共用一个 flag，月度
    这里一列一个，因为月度序列短，哪一列填过得单独看清。

    月度序列不转日度。频率不一样，硬往下摊只会造出根本不存在的观测。
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
    """把清洗结果写进 clean_data/，日期写成 YYYY-MM-DD。

    df 是要写的表，name 是输出文件名。
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(CLEAN / name, index=False, encoding="utf-8-sig")
    print(f"  [保存] clean_data/{name}  rows={len(out)}")


def main() -> None:
    """阶段一第 2 步的入口：先把主日历建出来，再清洗全部日度与月度序列。

    消费 raw_data/ 下的 treasury 曲线、FRED 各序列、9141.HK 的行情与 NAV、3141.HK 那条备用
    对照线；产出 clean_data/ 下的 *_clean.csv、master_calendar.csv、
    completeness_report.csv、fill_log.csv。主日历以 treasury 的日期为准，也就是美股交易日，
    其余序列一律对齐到它。
    """
    # 主日历就拿 treasury 的日期（美股交易日）
    tsy_raw = read_raw("treasury_yield_curve.csv", "date")
    master = pd.DatetimeIndex(sorted(tsy_raw["date"]))
    pd.DataFrame({"date": master.strftime("%Y-%m-%d")}).to_csv(
        CLEAN / "master_calendar.csv", index=False)
    print(f"[主日历] {len(master)} 个美股交易日："
          f"{master[0].date()} ~ {master[-1].date()}")

    report: list[dict] = []
    all_runs: list[dict] = []

    # 1) treasury：只取标准期限列。它自己就是主日历，不用再往哪对齐
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
    tsy["flag"] = 0                             # 也带一列 flag 跟别的表对齐（这张表没缺就全 0）
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
