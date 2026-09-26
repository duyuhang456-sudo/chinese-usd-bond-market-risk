"""9141.HK 官方日度 NAV 下载（MoneyDJ 镜像源，净值经官方锚点验证一致）

消费：无（需联网）
产出：raw_data/nav_9141HK_MoneyDJ.csv（date, nav_usd, mkt_usd, source）
口径：9141.HK（USD 柜台）报价陈旧（62.7% 相邻真实报价不变），价格更新滞后；而该基金的每单位
      资产净值(NAV, USD) 由管理人逐日披露，更接近真实市值的日度估值，用作陈旧问题的对照与
      阶段二 VaR 的候选输入。来源：华夏基金(香港)官网只给近期，改用 MoneyDJ 理财网净值表页面
      (basic0003.xdjhtm?etfid=9141.HK) 内嵌的走势图接口
      /ETF/X/xdjbcd/Basic0003BCD.xdjbcd?etfid=9141.HK&b=YYYYMMDD&c=YYYYMMDD
      返回「日期 + 净值-美元 + 市价-美元」两段日度序列。
边界：净值序列在 2026-04-14=1.8930、2026-07-03=1.8737，与 MoneyDJ 净值表标注的官方净值一致
      （±0.0001），视为可用的镜像。日期数与数值数不成 2:1 时抛 ValueError。
用法：./.venv/bin/python code/download_nav.py
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import requests

from common import RAW, REPO

URL = "http://210.17.21.214/ETF/X/xdjbcd/Basic0003BCD.xdjbcd"


def fetch(begin: str, end: str) -> str:
    """向 MoneyDJ 的走势图接口拉取原始返回文本。

    参数：
        begin: 起始日，YYYYMMDD。
        end: 结束日，YYYYMMDD。

    返回：
        接口返回的原始文本（未解析）。HTTP 状态异常时抛 requests 的异常。
    """
    r = requests.get(URL, params={"etfid": "9141.HK", "b": begin, "c": end},
                     timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def parse(raw: str) -> pd.DataFrame:
    """把接口返回的原始文本解析成日度 NAV 表。

    参数：
        raw: fetch() 返回的单行文本，形如「日期,净値,市价,日期,净値,市价,…」。

    返回：
        date / nav_usd / mkt_usd 三列，按日期升序。前一半数值为净値、后一半为市价。

    异常：
        日期数与数值数不成 2:1 时抛 ValueError，避免把错位的序列当成数据。
    """
    # MoneyDJ 返回单行文本，偶有逗号丢失（日期与数值被空格拼合），故按逗号与空格切分
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
    """下载 9141.HK 的完整日度 NAV 并写入 raw_data/nav_9141HK_MoneyDJ.csv。

    脚本契约：
        消费：MoneyDJ 接口（需联网），起始日固定 20210101、结束日取当天。
        产出：raw_data/nav_9141HK_MoneyDJ.csv（date / nav_usd / mkt_usd / source）。
        断言：三个官方 NAV 锚点（2026-04-14 = 1.8930、2026-07-03 = 1.8737、
            2026-09-04 = 1.8602）偏离超过 5e-4 时打印「✗ 偏离!」。
            此处只打印不中断——镜像源异常需要人看一眼再决定，不宜让取数类脚本挂掉全链路。

    返回：
        None。
    """
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
