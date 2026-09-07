"""
9141.HK 报价陈旧的三条出路对照实验（阶段一 · 为阶段二 VaR 选输入口径）

问题：9141.HK（USD 柜台，正式标的）62.7% 相邻真实报价不变 → 日度收益大量为 0，
      直接喂日度 VaR 会系统性低估波动/稀释久期。
出路（都试，最后给推荐）：
  A) 周度频率  —— 隔周价格必然变化，陈旧消失；代价=观测数减半、日度 VaR 需 /√5 折算
  B) 官方 NAV  —— 9141.HK 每单位资产净值(USD)，基金管理人逐日披露（MoneyDJ 镜像、
                   官方锚点 2026-04-14=1.8930 / 2026-07-03=1.8737 已验证）→ 真正日度、低陈旧
  C) 真实久期口径修正 —— 用「仅变动日」标定的真实久期(≈3.3y) β，把陈旧日的
                   利率面收益补上（利率因子本身逐日可得、永不停更）；
                   另给因子法风险估计（利率由真久期×日度因子方差、利差用变动日残差方差）

对照尺子（全部为同一把）：
  零收益占比 / n / 年化波动 / 日等价σ / 利率剥离 R² / 等效久期 / 利差代理 vs ΔOAS
  （2023-09+，滞后一日）的 ρ 与命中率 / 正态 1 日 99% VaR。
  其中 官方 NAV 是「真实日度估值」，作为其它方案的最优参照（ground truth）。

用法： ./.venv/bin/python code/staleness_remedy.py
输出： factors/staleness_remedy_comparison.csv、figures/staleness_remedy.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import REPO

CLEAN = REPO / "clean_data"
FACT = REPO / "factors"
FIG = REPO / "figures"
FACT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

VAR99 = 2.326  # 正态 1 日 99% 分位数


def zshare(x: pd.Series) -> tuple[int, float]:
    """零收益占比(%)——只在该序列自己的真实观测上数。"""
    v = x.dropna()
    if len(v) == 0:
        return 0, float("nan")
    return int(len(v)), float((v == 0).mean() * 100.0)


def daily_ols(price: pd.Series, d5y: pd.Series, d10y: pd.Series,
              lag: int = 1) -> dict:
    """日度利差剥离：etf_ret(t)=α+β5·Δy5(t-1)+β10·Δy10(t-1)+ε。"""
    ret = np.log(price).diff() * 100
    l5, l10 = d5y.shift(lag), d10y.shift(lag)
    X = sm.add_constant(pd.DataFrame({"b5": l5, "b10": l10}))
    j = pd.concat([ret.rename("y"), X], axis=1).dropna()
    m = sm.OLS(j["y"], sm.add_constant(j[["b5", "b10"]])).fit()
    beta = m.params["b5"] + m.params["b10"]
    return {"r2": m.rsquared, "dur": -beta * 100, "n": int(m.nobs),
            "resid": (j["y"] - m.fittedvalues), "ret": ret, "m": m}


def weekly_ols(price: pd.Series, y5l: pd.Series, y10l: pd.Series) -> dict:
    """周度利差剥离（W-FRI 周末取值；同周 Δy，日度 T+1 错位在周度桶内可忽略）。
    传收益率水平序列 y5l/y10l。"""
    p = price.resample("W-FRI").last()
    r = np.log(p).diff() * 100
    dy5 = y5l.resample("W-FRI").last().diff() * 100.0
    dy10 = y10l.resample("W-FRI").last().diff() * 100.0
    X = sm.add_constant(pd.DataFrame({"b5": dy5, "b10": dy10}))
    j = pd.concat([r.rename("y"), X], axis=1).dropna()
    m = sm.OLS(j["y"], sm.add_constant(j[["b5", "b10"]])).fit()
    return {"r2": m.rsquared, "dur": -(m.params["b5"] + m.params["b10"]) * 100,
            "n": int(m.nobs), "resid": (j["y"] - m.fittedvalues),
            "ret": r, "m": m}


def oas_check(resid, oas_bp, mode: str, oas_lag: int = 1):
    """利差代理(残差, %) vs ΔOAS（重叠窗 2023-09+），按口径选：
    price 日度 → ΔOAS 滞后一日（价格=前一日美股）；NAV 日度 → 同日（lag=0）；
    weekly → 同周本地差分。"""
    if resid is None:
        return {}
    if mode == "daily":
        do = oas_bp.diff().shift(oas_lag)
    else:  # weekly
        idx = resid.index
        wlab = idx.to_series().dt.strftime("%G-%V")
        do = pd.Series(oas_bp, index=idx).groupby(wlab.values).last().diff()
        do = do.reindex(idx)
    dd = pd.concat([pd.Series(resid).rename("sp"), do.rename("do")], axis=1).dropna()
    if len(dd) < 30:
        return {}
    rho = dd["sp"].corr(dd["do"])
    hit = ((dd["sp"] < 0) & (dd["do"] > 0)) | ((dd["sp"] > 0) & (dd["do"] < 0))
    return {"rho": float(rho), "hit": float(hit.mean() * 100), "n_oas": len(dd)}


def ann_vol(ret: pd.Series, per_year: float) -> float:
    return float(ret.std() * np.sqrt(per_year))


def main() -> None:
    def c(n): return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")
    tsy = c("treasury_yield_curve_clean.csv")
    etf = c("benchmark_9141HK_clean.csv")
    nav = c("nav_9141HK_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")
    oas_bp = oas["BAMLEMIBHGCRPIOAS"] * 100.0

    # 收益率水平（用于周度差分与日度因子）
    y5l, y10l = tsy["5 Yr"], tsy["10 Yr"]
    d5y = y5l.diff() * 100.0     # bp
    d10y = y10l.diff() * 100.0
    px9141, pxnav = etf["Adj Close"], nav["nav_usd"]

    # ---------- 候选日度序列 ----------
    ret9141 = np.log(px9141).diff() * 100
    retnav = np.log(pxnav).diff() * 100

    # C1) 真实久期修正序列：陈旧(零)日按利率面补，变动日保留原值
    m_active = None
    r = ret9141
    act = pd.DataFrame({"r": r, "l5": d5y.shift(1), "l10": d10y.shift(1)})
    act = act[(act["r"] != 0)].dropna()
    ma = sm.OLS(act["r"], sm.add_constant(act[["l5", "l10"]])).fit()
    rate_all = ma.params["l5"] * d5y.shift(1) + ma.params["l10"] * d10y.shift(1)
    rate_all = rate_all.reindex(r.index)
    ret_fix = r.where(r != 0, rate_all).dropna()
    # 因子法风险估计：利率部分用日度因子方差×真久期，利差用变动日残差
    spread_active = (act["r"] - ma.predict(sm.add_constant(act[["l5", "l10"]])))
    sig_rate_d = rate_all.std()                       # 利率日波动(%/日)
    sig_spread_d = spread_active.std()                # 利差日波动(%/日,变动日)
    sig_model_d = float(np.sqrt(sig_rate_d**2 + sig_spread_d**2))

    # ---------- 周度序列（价格 & NAV 都做，便于对照） ----------
    wo9141 = weekly_ols(px9141, y5l, y10l)
    wonav = weekly_ols(pxnav, y5l, y10l)

    # ---------- 指标表 ----------
    rows = []
    def add(name, rets, dur_r2, resid, per_year, mode, oas_lag, extra=""):
        n0, z = zshare(rets)
        vol = ann_vol(rets, per_year)          # 年化σ%：跨频率直接可比
        sd1 = vol / np.sqrt(252.0)             # 1 日 σ（持有 1 日）
        rows.append({
            "方案": name, "n": n0, "零收益占比%": round(z, 1),
            "年化波动%": round(vol, 2),
            "1日σ%": round(sd1, 3),
            "1日99%VaR(正态)%": round(sd1 * VAR99, 3),
            "利率剥离R²": round((dur_r2 or {}).get("r2", float("nan")), 3),
            "等效久期(年)": round((dur_r2 or {}).get("dur", float("nan")), 2),
            "OAS-ρ": "", "OAS命中%": "", "备注": extra,
        })
        oc = oas_check(resid, oas_bp, mode, oas_lag)
        if oc:
            rows[-1]["OAS-ρ"] = round(oc["rho"], 3)
            rows[-1]["OAS命中%"] = round(oc["hit"], 1)

    # 日度：价格 / NAV / 真实久期修正（NAV 与美股同日→lag=0；价格滞后 1 日→lag=1）
    d9141 = daily_ols(px9141, d5y, d10y, lag=1)
    add("D1 价格9141(现状)", ret9141, d9141, d9141["resid"], 252, "daily", 1,
        "基准：~63%零收益")
    nav_d = daily_ols(pxnav, d5y, d10y, lag=0)
    add("D2 官方NAV(日度)", retnav, nav_d, nav_d["resid"], 252, "daily", 0,
        "真实日度估值★(lag0)")
    add("D3 真久期修正(日度)", ret_fix, None, None, 252, "daily", 1,
        f"零日按利率面补(因子法σ年化{sig_model_d*np.sqrt(252):.1f}%,偏高),非观测→仅供参考")
    # 周度：价格 / NAV（同周 Δy）
    add("W1 价格周度", wo9141["ret"], wo9141, wo9141["resid"], 52, "weekly", 0,
        "周度去陈旧")
    add("W2 NAV周度", wonav["ret"], wonav, wonav["resid"], 52, "weekly", 0,
        "NAV 周度参照")

    df = pd.DataFrame(rows)
    print("=" * 100)
    print("9141.HK 报价陈旧处理方案对照（n/零收益为各序列自身口径；VaR 均折算为 1 日）")
    print("=" * 100)
    print(df.to_string(index=False))
    df.to_csv(FACT / "staleness_remedy_comparison.csv", index=False,
              encoding="utf-8-sig")

    # NAV vs 价格日度收益一致度（对官方 NAV 是否可作为 9141 的日度无偏代理）
    j = pd.concat([ret9141.rename("price"), retnav.rename("nav")], axis=1)
    both = j.dropna()
    bothm = both[(both["price"] != 0) & (both["nav"] != 0)]
    print("\nNAV vs 价格 日度收益：两序列同日都动", len(bothm),
          "天；相关", round(bothm["price"].corr(bothm["nav"]), 3),
          "；价格有值日与NAV相关", round(both["price"].corr(both["nav"]), 3))
    print(f"NAV 官方日度：真实观测零收益 {zshare(retnav)[1]:.1f}% "
          f"(价格 {zshare(ret9141)[1]:.1f}%)；NAV 年化波动 {ann_vol(retnav,252):.2f}% "
          f"(价格 {ann_vol(ret9141,252):.2f}%)")
    print(f"因子法(真久期) σ_model={sig_model_d:.4f}%/日 → 年化 {sig_model_d*np.sqrt(252):.2f}% ；"
          f"利率部分贡献={sig_rate_d:.3f}%/日、利差={sig_spread_d:.3f}%/日")

    # ---------- 图 ----------
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    # 1) 归一化净值：NAV vs 价格 收盘(周度平滑显示趋势一致)
    base = pd.concat([pxnav.rename("nav"), px9141.rename("px")], axis=1)
    for col, lab, colr in [("nav", "官方NAV(USD)", "tab:green"), ("px", "9141价格(USD)", "tab:blue")]:
        s = base[col]
        axs[0].plot(s.index, s / s.iloc[0] * 100, lw=1.0, color=colr, label=lab)
    axs[0].set_ylabel("归一(起点=100)"); axs[0].set_title(
        "9141.HK 官方NAV vs 市价水平（NAV 逐日、市价陈旧→呈台阶）")
    axs[0].legend(); axs[0].grid(alpha=.3)
    # 2) 日收益分布：NAV vs 价格（对数Y 看 0 质量）
    for col, lab, colr in [("nav", "NAV日收益%", "tab:green"), ("price", "价格日收益%", "tab:blue")]:
        a = (j[col].abs().dropna() + 1e-6)
        axs[1].hist(np.log(a), bins=60, alpha=.45, color=colr, label=lab)
    axs[1].set_xlabel("ln(|日收益|+eps)"); axs[1].set_ylabel("频数(对数)")
    axs[1].set_title("日收益量级分布：价格有 ~63% 精确 0%（最左峰），NAV 逐日变化")
    axs[1].legend(); axs[1].grid(alpha=.3)
    # 3) 周度与 NAV 年化滚窗对比
    rw = wo9141["ret"]; rw_nav = wonav["ret"]
    axs[2].plot(rw.index, rw.rolling(26).std() * np.sqrt(52), color="tab:blue",
                label="价格周度 年化σ(26周滚窗)")
    axs[2].plot(rw_nav.index, rw_nav.rolling(26).std() * np.sqrt(52),
                color="tab:green", label="NAV周度 年化σ(26周滚窗)")
    axs[2].axhline(sig_model_d * np.sqrt(252), color="tab:red", ls="--",
                   label=f"因子法(真久期)年化σ={sig_model_d*np.sqrt(252):.2f}%")
    axs[2].set_ylabel("年化波动%"); axs[2].legend(); axs[2].grid(alpha=.3)
    axs[2].set_title("波动率量级对比：周度价格 / 周度NAV / 因子法(真久期)参照")
    fig.tight_layout()
    fig.savefig(FIG / "staleness_remedy.png", dpi=150)
    print("\n[已写] figures/staleness_remedy.png；factors/staleness_remedy_comparison.csv")


if __name__ == "__main__":
    main()
