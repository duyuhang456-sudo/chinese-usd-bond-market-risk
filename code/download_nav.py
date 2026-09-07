"""
9141.HK 官方日度 NAV 下载（MoneyDJ 镜像源，净値经官方锚点验证一致）

背景：9141.HK（USD 柜台）报价陈旧（62.7% 相邻真实报价不变）——价格更新滞后，
但该基金的「每单位资产净值(NAV, USD)」由基金管理人逐日披露，是更接近真实市值的
日度估值序列，可作为陈旧问题的对照与阶段二 VaR 的候选输入。

来源：华夏基金(香港)官网只给近期；MoneyDJ 理财网的净值表页面
  (basic0003.xdjhtm?etfid=9141.HK) 内嵌的走势图接口
  /ETF/X/xdjbcd/Basic0003BCD.xdjbcd?etfid=9141.HK&b=YYYYMMDD&c=YYYYMMDD
  返回「日期 + 净値-美元 + 市价-美元」两段日度序列。
验证：净値序列在 2026-04-14=1.8930、2026-07-03=1.8737，与 MoneyDJ 净值表标注的
  官方净値一致（±0.0001），视为官方 NAV 的可靠镜像。

用法： ./.venv/bin/python code/download_nav.py
输出： raw_data/nav_9141HK_MoneyDJ.csv   (date, nav_usd, mkt_usd, source)
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import requests

from common import RAW, REPO

URL = "http://210.17.21.214/ETF/X/xdjbcd/Basic0003BCD.xdjbcd"


def fetch(begin: str, end: str) -> str:
    r = requests.get(URL, params={"etfid": "9141.HK", "b": begin, "c": end},
                     timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def parse(raw: str) -> pd.DataFrame:
    # MoneyDJ 返回单行文本，偶有逗号丢失（日期/数值被空格拼合）→ 统一按 [,\s] 切分
    toks = re.split(r"[, ]", raw)
    toks = [t for t in toks if t]
    dates = [t for t in toks if re.fullmatch(r"\d{8}", t)]
    floats = [float(t) for t in toks if re.fullmatch(r"\d+\.\d+", t)]
    if len(floats) % 2 != 0 or len(floats) != 2 * len(dates):
        raise ValueError(f"结构异常 dates={len(dates)} floats={len(floats)}")
    half = len(floats) // 2
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "nav_usd": floats[:half],        # 净値-美元
        "mkt_usd": floats[half:],        # 市价-美元
    })
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    today = dt.date.today()
    df = fetch("20210101", today.strftime("%Y%m%d"))
    df = parse(df)
    df["source"] = "MoneyDJ(净値=华夏基金(香港)官方NAV镜像)"
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(RAW / "nav_9141HK_MoneyDJ.csv", index=False, encoding="utf-8-sig")
    print(f"[已写] raw_data/nav_9141HK_MoneyDJ.csv  rows={len(df)}  "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    # 官方锚点验证
    for ymd, exp in [("2026-04-14", 1.8930), ("2026-07-03", 1.8737),
                     ("2026-09-04", 1.8602)]:
        row = df[df["date"] == ymd]
        if len(row):
            nav = float(row["nav_usd"].iloc[0])
            print(f"  验证 {ymd}: nav_usd={nav:.4f}  (官方/期望 {exp})  "
                  f"{'✓' if abs(nav-exp) < 5e-4 else '✗ 偏离!'}")


if __name__ == "__main__":
    main()
