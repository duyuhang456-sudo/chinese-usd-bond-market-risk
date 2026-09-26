"""异常逐条判定：候选日到底算不算真冲击（阶段一 Day3 · 9/9 下午）

消费：clean_data/outlier_candidates.csv、events/risk_events_timeline.csv（人工维护）
产出：clean_data/outlier_judgment.csv
口径：候选日前后各 3 个自然日以内命中事件表里的某一条，判「真实冲击」，处理方式是「保留」，
      把事件名附上；一条都没命中，判「存疑（无事件支撑）」，处理方式是「仅标注、不改数」。
      极少数特殊日期（9141 跟 3141 反向之类）在备注列里人工复核。
用法：./.venv/bin/python code/adjudicate_outliers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, EVENTS_CSV, ensure_dirs

ensure_dirs()

# 人工备注：9141 极个别的大波动日跟 3141.HK（HKD 柜台）反着走。
# 3141 大多数日子报价是陈旧的，只能当参考线索，不当证据。
SPECIAL = {
    "2022-11-14": "与3141反向但命中中国防疫优化+地产三支箭，且 CNY 当日 -1.6%，判真实冲击(3141当日陈旧)",
    "2022-11-29": "无独立事件强支撑且与3141反向 → 存疑(流动性/交易所价噪声候选)",
    "2024-08-02": "与3141反向，但命中8/2美国就业走弱+套息平仓(美债利率大跌)，ETF+1.4%可由久期解释，判真实冲击",
    "2024-09-10": "无事件强支撑且与3141反向 → 存疑",
    "2025-07-23": "命中2025-07-23股债切换/中国长债大跌(低置信)，判真实冲击，但单日-1.0%偏大需留意",
}


def main() -> None:
    """把候选异常挨个跟风险事件时间线做窗口匹配，给出判定和处理方式。

    消费 clean_data/outlier_candidates.csv 和 events/risk_events_timeline.csv；产出
    clean_data/outlier_judgment.csv，列里有 verdict / handling / 命中事件 / gap_days /
    note / special。

    判定就一条规则：候选日前后 3 个自然日内命中任何一条事件，判「真实冲击」、处理「保留」；
    没命中就判「存疑(无事件支撑)」、处理「仅标注,不改数」。SPECIAL 里那几个日期另外挂一段
    人工复核备注（9141 跟 3141 反向之类）。
    """
    cand = pd.read_csv(CLEAN / "outlier_candidates.csv", dtype={"date": str})
    ev = pd.read_csv(EVENTS_CSV, dtype={"date": str})
    ev["d"] = pd.to_datetime(ev["date"])
    evd = ev.sort_values("d")

    def match(day: str) -> pd.DataFrame:
        d = pd.Timestamp(day)
        win = evd[(evd["d"] >= d - pd.Timedelta(days=3)) &
                  (evd["d"] <= d + pd.Timedelta(days=3))]
        if len(win) == 0:
            return None
        # 窗口里挑离得最近的那条事件；距离一样就按日期先后取第一条，不再另设平局规则
        win = win.copy()
        win["_gap"] = (win["d"] - d).dt.days.abs()
        win = win.sort_values(["_gap", "d"])
        return win.iloc[0]

    rows = []
    for _, r in cand.iterrows():
        m = match(r["date"])
        if m is not None:
            verdict = "真实冲击"
            handling = "保留"
            event = m["event_cn"]
            ev_date = m["date"]
            gap = int(abs(pd.Timestamp(ev_date) - pd.Timestamp(r["date"])).days)
            note = f"事件窗口内(±3日,偏差{gap}日)"
        else:
            verdict = "存疑(无事件支撑)"
            handling = "仅标注,不改数"
            event, ev_date, gap, note = "", "", "", ""
        rows.append({
            "series": r["series"], "date": r["date"],
            "value": r["value"],
            "hit": r["hit"],
            "verdict": verdict, "handling": handling,
            "event_date": ev_date, "event": event, "gap_days": gap,
            "note": note, "special": SPECIAL.get(r["date"], ""),
        })

    out = pd.DataFrame(rows).sort_values(["date", "series"])
    # 打一份汇总
    print("判定汇总:")
    print(out["verdict"].value_counts().to_string())
    n_shock = len(out[out.verdict == "真实冲击"])
    n_doubt = len(out[out.verdict.str.startswith("存疑")])
    print(f"\n真实冲击 {n_shock} 条（保留）；存疑 {n_doubt} 条（仅标注不改数）。")
    print("\n存疑清单（待人工复核/或属 2025-26 待事件补录）:")
    dq = out[out.verdict.str.startswith("存疑")]
    for _, r in dq.iterrows():
        print(f"  {r['date']}  {r['series']:<12} val={r['value']:+.1f}  {r['special']}")
    out.to_csv(CLEAN / "outlier_judgment.csv", index=False, encoding="utf-8-sig")
    print("\n[已写] clean_data/outlier_judgment.csv")


if __name__ == "__main__":
    main()
