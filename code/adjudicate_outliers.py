"""异常逐条判定（阶段一 9/9 下午）

消费：clean_data/outlier_candidates.csv、events/risk_events_timeline.csv（人工维护）
产出：clean_data/outlier_judgment.csv
口径：候选日前后各 3 日内命中事件 → 判「真实冲击」，处理「保留」并附事件名；未命中任何事件
      → 判「存疑（无事件支撑）」，处理「仅标注、不改数」。少量特殊日期（9141 与 3141 反向
      等）手工在备注列复核。
用法：./.venv/bin/python code/adjudicate_outliers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, EVENTS_CSV, ensure_dirs

ensure_dirs()

# 特殊备注：极少数 9141 极端日与 3141.HK(HKD 柜台) 反向（后者多数日陈旧，仅作参考线索）
SPECIAL = {
    "2022-11-14": "与3141反向但命中中国防疫优化+地产三支箭，且 CNY 当日 -1.6%，判真实冲击(3141当日陈旧)",
    "2022-11-29": "无独立事件强支撑且与3141反向 → 存疑(流动性/交易所价噪声候选)",
    "2024-08-02": "与3141反向，但命中8/2美国就业走弱+套息平仓(美债利率大跌)，ETF+1.4%可由久期解释，判真实冲击",
    "2024-09-10": "无事件强支撑且与3141反向 → 存疑",
    "2025-07-23": "命中2025-07-23股债切换/中国长债大跌(低置信)，判真实冲击，但单日-1.0%偏大需留意",
}


def main() -> None:
    """把候选异常逐条与风险事件时间线做窗口匹配，给出判定与处理方式。

    脚本契约：
        消费：clean_data/outlier_candidates.csv、events/risk_events_timeline.csv。
        产出：clean_data/outlier_judgment.csv（含 verdict / handling / 命中事件 /
             gap_days / note / special 列）。
        判定规则：候选日前后 3 个自然日内命中任一事件，判「真实冲击」、处理「保留」；
            未命中判「存疑(无事件支撑)」、处理「仅标注,不改数」。SPECIAL 里的少数日期
            附人工复核备注（9141 与 3141 反向等情形）。

    返回：
        None。
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
        # 窗口内取距离最近的事件；gap 相同时按日期先后取一条，不设额外的平局裁决规则
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
    # 汇总
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
