"""9141.HK 报价陈旧的三条出路对照实验（阶段一 · 为阶段二 VaR 选输入口径）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、fred_DEXCHUS_clean.csv、
      fred_BAMLEMIBHGCRPIOAS_clean.csv、benchmark_9141HK_clean.csv、nav_9141HK_clean.csv
产出：factors/staleness_remedy_comparison.csv、figures/staleness_remedy.png
口径：问题是 9141.HK（USD 柜台，正式标的）62.7% 相邻真实报价不变，日度收益大量为 0，直接
      喂日度 VaR 会系统性低估波动、稀释久期。三条出路都试：A) 周度频率，隔周价格必然变化、
      陈旧消失，代价是观测数减半、日度 VaR 需 /√5 折算；B) 官方 NAV，管理人逐日披露的每
      单位资产净值(USD)，是真正日度、低陈旧的序列；C) 真实久期口径修正，用仅变动日标定的
      真实久期（≈3.3y）β 把陈旧日的利率面收益补上，另给因子法风险估计（利率用真久期 ×
      日度因子方差，利差用变动日残差方差）。对照尺子共用同一把：零收益占比 / n / 年化波动 /
      日等价 σ / 利率剥离 R² / 等效久期 / 利差代理 vs 滞后一日 ΔOAS（2023-09 起）的 ρ 与
      命中率 / 正态 1 日 99% VaR。
边界：官方 NAV 当作其它方案的最优参照（ground truth），但它本身来自 MoneyDJ 镜像源，锚点
      校验见 code/download_nav.py。
用法：./.venv/bin/python code/staleness_remedy.py
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

from common import CLEAN, FACT, FIG, ensure_dirs

ensure_dirs()

VAR99 = 2.326  # 正态 1 日 99% 分位数


def zshare(x: pd.Series) -> tuple[int, float]:
    """统计序列中恰好等于 0 的观测占比。

    参数：
        x: 收益序列（%）。

    返回：
        (n, z) 二元组。n 为去掉 NaN 后的观测数（int）；z 为零收益占 n 的百分比（float）。
        序列全为空时返回 (0, nan)。

    备注：
        只在各序列自身的真实观测上计数，不补、不插值。三个候选口径的 n 本来就不同，
        补齐到同一长度会把「零收益多」和「样本期长」混成一个数。
    """
    v = x.dropna()
    if len(v) == 0:
        return 0, float("nan")
    return int(len(v)), float((v == 0).mean() * 100.0)


def daily_ols(price: pd.Series, d5y: pd.Series, d10y: pd.Series,
              lag: int = 1) -> dict:
    """用滞后利率变动对日度收益做 OLS，剥离利率面并取残差作利差代理。

    回归式：etf_ret(t) = α + β5·Δy5(t-lag) + β10·Δy10(t-lag) + ε。

    参数：
        price: 价格或净值序列（USD），索引为交易日。
        d5y: 5 年期收益率的日度差分（bp）。
        d10y: 10 年期收益率的日度差分（bp）。
        lag: 两个利率因子的滞后阶数。价格口径取 1（美股 T 日收盘对应境内 T+1 日利率变动），
            NAV 口径取 0（同日）。

    返回：
        dict，键为
        r2: 回归 R²；
        dur: 等效久期（年），取 -(β5+β10)×100；
        n: 回归样本数；
        resid: 残差序列（%），即利差代理；
        ret: 日度对数收益全序列（%），未按因子做 NaN 对齐；
        m: statsmodels 回归结果对象。
    """
    ret = np.log(price).diff() * 100
    l5, l10 = d5y.shift(lag), d10y.shift(lag)
    X = sm.add_constant(pd.DataFrame({"b5": l5, "b10": l10}))
    j = pd.concat([ret.rename("y"), X], axis=1).dropna()
    m = sm.OLS(j["y"], sm.add_constant(j[["b5", "b10"]])).fit()
    beta = m.params["b5"] + m.params["b10"]
    return {"r2": m.rsquared, "dur": -beta * 100, "n": int(m.nobs),
            "resid": (j["y"] - m.fittedvalues), "ret": ret, "m": m}


def weekly_ols(price: pd.Series, y5l: pd.Series, y10l: pd.Series) -> dict:
    """先按 W-FRI 取周末值，再对周度收益做利率剥离 OLS。

    参数：
        price: 价格或净值序列（USD），日度；内部按 W-FRI 重采样取每周最后一个值。
        y5l: 5 年期收益率水平序列（%），差分由函数内部完成。
        y10l: 10 年期收益率水平序列（%），差分由函数内部完成。

    返回：
        dict，键同 daily_ols（r2 / dur / n / resid / ret / m）。ret（周度对数收益，%）与
        resid（周度残差，%）都落在 W-FRI 的周度索引上；dur 同样取 -(β5+β10)×100。

    备注：
        利率用同周 Δy，不做滞后。日度口径里 T+1 的错位在一周的时间桶内已被吸收，
        周度再错位一周反而会把当周的价格变动配到下周的利率上。
    """
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
    """把利差代理残差与 ΔOAS 对齐，算相关系数与方向命中率。

    参数：
        resid: 利差剥离得到的残差序列（%）。传 None 时直接返回空 dict。
        oas_bp: OAS 水平序列（bp）。
        mode: 取 "daily" 时 ΔOAS 按日差分并滞后 oas_lag 日；取其它值（调用处传 "weekly"）
            时按 %G-%V 周标签取周内最后一个值再做周间差分，并重排回原索引。
        oas_lag: 日度口径下 ΔOAS 的滞后阶数。价格口径传 1（收盘价是前一日美股），
            NAV 口径传 0（同日）。周度口径不使用该参数。

    返回：
        dict，键为 rho（残差与 ΔOAS 的相关系数）、hit（两者符号相反的比例，%）、
        n_oas（对齐后的重叠观测数）。resid 为 None，或对齐后不足 30 个观测时，返回空 dict {}。

    备注：
        OAS 序列自 2023-09-05 起，实际重叠窗口取决于 resid 与之的交集，本函数不另行截窗。
    """
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
    """把单期收益标准差折算成年化波动率。

    参数：
        ret: 收益序列（%），日度或周度均可。
        per_year: 年化因子，日度取 252，周度取 52。

    返回：
        年化波动率（%），等于样本标准差（ddof=1）乘以 per_year 的平方根。
    """
    return float(ret.std() * np.sqrt(per_year))


def main() -> None:
    """跑三条出路与两种频率的对照实验，输出方案对照表与三联图。

    脚本契约：
        消费：clean_data/treasury_yield_curve_clean.csv（5 Yr / 10 Yr 收益率水平）、
              clean_data/benchmark_9141HK_clean.csv（Adj Close）、
              clean_data/nav_9141HK_clean.csv（nav_usd）、
              clean_data/fred_BAMLEMIBHGCRPIOAS_clean.csv（OAS 水平，自 2023-09-05 起）。
        产出：factors/staleness_remedy_comparison.csv，六个方案各一行（D1 价格、D2 官方 NAV、
              D3 真久期修正、W1 价格周度、W2 NAV 周度），列含 n、零收益占比%、年化波动%、
              1 日σ%、1 日 99% 正态 VaR%、利率剥离 R²、等效久期(年)、OAS-ρ、OAS 命中%、备注；
              figures/staleness_remedy.png，三联图（NAV 与价格归一净值、日收益量级分布、
              周度滚窗年化波动对照）。
        断言/边界：无硬断言。日度与周度方案的年化 σ 统一除以 √252 折成 1 日口径再比较，
              年化因子分别取 252 与 52。D3 是构造序列而非真实观测（零收益日按利率面补，
              变动日保留原值），其残差传 None，故 OAS 两列留空；因子法 σ 由利率日波动与
              变动日残差合成，数值偏高，表里已注明仅供参考。周度方案的 OAS 对齐用同周本地差分。
    """
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

    # ---- 候选日度序列 ----
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

    # ---- 周度序列（价格 & NAV 都做，便于对照） ----
    wo9141 = weekly_ols(px9141, y5l, y10l)
    wonav = weekly_ols(pxnav, y5l, y10l)

    # ---- 指标表 ----
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

    # 日度：价格 / NAV / 真实久期修正（NAV 与美股同日用 lag=0，价格滞后 1 日用 lag=1）
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

    # NAV 与价格的日度收益一致度：验「官方 NAV 可否当作 9141 的日度无偏代理」
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

    # ---- 图 ----
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
