"""9141.HK 报价陈旧的三条出路对照实验（阶段一 · 为阶段二 VaR 选输入口径）

消费：clean_data/ 下的 treasury_yield_curve_clean.csv、fred_DEXCHUS_clean.csv、
      fred_BAMLEMIBHGCRPIOAS_clean.csv、benchmark_9141HK_clean.csv、nav_9141HK_clean.csv
产出：factors/staleness_remedy_comparison.csv、figures/staleness_remedy.png
口径：9141.HK 是 USD 柜台，也是正式标的。它的报价有 62.7% 的情况是相邻两次真实报价一分
      不动，日收益一大片是 0；这样的序列直接喂日度 VaR，波动会被系统性压低，久期也被稀释。
      三条出路都试一遍：A) 改周度，隔一周价格总会动，陈旧就没了，代价是观测数减半，日度
      VaR 得除以 √5 折回来；B) 改用官方 NAV，管理人每天披露的每单位资产净值(USD)，真正
      日度、几乎不陈旧；C) 用真实久期把账补回来，拿只在变动日标定出来的真实久期（≈3.3y）
      当 β，把陈旧日缺的利率面收益补上，另外给一个因子法风险估计（利率用真久期乘日度因子
      方差，利差用变动日残差方差）。几个方案用同一把尺子量：零收益占比、n、年化波动、
      日等价 σ、利率剥离 R²、等效久期、利差代理对滞后一日 ΔOAS（2023-09 起）的 ρ 和命中率、
      正态 1 日 99% VaR。算这些的时候 NAV 口径配 lag=0、市价口径配 lag=1，两个不能换，
      换了不报错，但整套因子暴露都会变。
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

VAR99 = 2.326  # 正态分布 1 日 99% 的分位数


def zshare(x: pd.Series) -> tuple[int, float]:
    """数一串收益里有百分之多少是精确的 0。

    x 是收益序列（%）。返回 (n, z)：n 是去掉空值后的观测数，z 是零收益占 n 的百分比。
    序列全空就给 (0, nan)。

    只在各序列自己的真实观测上数，不补、不插值。三个候选口径的天数本来就不一样，补齐成
    一样长，等于把「零收益多」和「样本期长」搅成一个数，看不出到底是哪个。
    """
    v = x.dropna()
    if len(v) == 0:
        return 0, float("nan")
    return int(len(v)), float((v == 0).mean() * 100.0)


def daily_ols(price: pd.Series, d5y: pd.Series, d10y: pd.Series,
              lag: int = 1) -> dict:
    """拿滞后一期的利率变动去解释日收益，把利率那块剥掉，剩下的残差当利差代理。

    回归式是 etf_ret(t) = α + β5·Δy5(t-lag) + β10·Δy10(t-lag) + ε。

    price 是价格或净值序列（USD），索引是交易日；d5y、d10y 是 5 年和 10 年收益率的日变化（bp）。
    lag 是两个利率因子的滞后期数：价格口径给 1，因为美股 T 日收盘对应的是境内 T+1 日的
    利率变动；NAV 口径给 0，同一天，别拿错。

    返回一个 dict：r2 是回归 R²，dur 是等效久期（年，等于 -(β5+β10)×100），n 是回归样本数，
    resid 是残差序列（%），也就是利差代理，ret 是没按因子做过对齐的日度对数收益全序列（%），
    m 是 statsmodels 的回归结果对象。
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
    """按 W-FRI 取每周末的值，再对周度收益做同一套利率剥离。

    price 是日度价格或净值序列（USD），函数内部自己按 W-FRI 取每周最后一个值；y5l、y10l
    是 5 年和 10 年收益率的水平序列（%），差分在函数里做。

    返回的 dict 键和 daily_ols 一样（r2 / dur / n / resid / ret / m），dur 同样取
    -(β5+β10)×100。ret（周度对数收益，%）和 resid（周度残差，%）都落在 W-FRI 的周度索引上。

    利率用同周的 Δy，不滞后。日度口径里那个 T+1 的错位，装进一周的桶里已经被抹平了，
    周度再滞后一周反而会把当周的价格变动配到下一周的利率上。
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
    """把利差代理残差和 ΔOAS 摆到一起，算相关系数和方向命中率。

    resid 是剥出来的利差残差（%），传 None 就直接返回空 dict；oas_bp 是 OAS 水平序列（bp）。
    mode 给 "daily" 时 ΔOAS 按日差分、滞后 oas_lag 天；给别的值（调用处传 "weekly"）就按
    %G-%V 周标签取周内最后一个值、做周间差分，再排回原来的索引。oas_lag 只在日度口径下用：
    价格口径给 1（收盘价是前一日美股），NAV 口径给 0（同日），周度口径不用它。

    返回 dict 三个键：rho 是残差和 ΔOAS 的相关系数，hit 是两者符号相反的占比（%），
    n_oas 是对齐后重叠的观测数。resid 是 None、或者对齐后不到 30 个观测，就给 {}。

    OAS 序列从 2023-09-05 才有，实际能对上多少天得看 resid 和它的交集，这里不另外截窗。
    """
    if resid is None:
        return {}
    if mode == "daily":
        do = oas_bp.diff().shift(oas_lag)
    else:  # 周度
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
    """把单期收益的标准差折成年化波动。

    ret 是收益序列（%），日度周度都行；per_year 是年化因子，日度给 252，周度给 52。
    返回年化波动率（%），就是样本标准差（ddof=1）乘 per_year 的平方根。
    """
    return float(ret.std() * np.sqrt(per_year))


def main() -> None:
    """三条出路、两种频率都跑一遍做对照，出一张方案对照表和一张三联图。

    读 clean_data/treasury_yield_curve_clean.csv（5 Yr / 10 Yr 收益率水平）、
    clean_data/benchmark_9141HK_clean.csv（Adj Close）、clean_data/nav_9141HK_clean.csv
    （nav_usd）、clean_data/fred_BAMLEMIBHGCRPIOAS_clean.csv（OAS 水平，2023-09-05 起）。

    写 factors/staleness_remedy_comparison.csv，六个方案各一行（D1 价格、D2 官方 NAV、
    D3 真久期修正、W1 价格周度、W2 NAV 周度），列有 n、零收益占比%、年化波动%、1 日 σ%、
    1 日 99% 正态 VaR%、利率剥离 R²、等效久期(年)、OAS-ρ、OAS 命中%、备注；另写
    figures/staleness_remedy.png，三联图（NAV 与价格的归一净值、日收益量级分布、
    周度滚窗年化波动对照）。

    没有硬断言。日度和周度都先把年化 σ 除以 √252 折成 1 日口径再比，年化因子分别取 252
    和 52。D3 是拼出来的序列、不是真观测（零收益日按利率面补，变动日留原值），残差传 None，
    所以 OAS 两列空着；它的因子法 σ 是利率日波动和变动日残差合出来的，数偏高，表里已注明
    只作参考。周度方案的 OAS 对齐用同周内的本地差分。
    """
    def c(n): return pd.read_csv(CLEAN / n, parse_dates=["date"]).set_index("date")
    tsy = c("treasury_yield_curve_clean.csv")
    etf = c("benchmark_9141HK_clean.csv")
    nav = c("nav_9141HK_clean.csv")
    oas = c("fred_BAMLEMIBHGCRPIOAS_clean.csv")
    oas_bp = oas["BAMLEMIBHGCRPIOAS"] * 100.0

    # 收益率水平，周度差分和日度因子都要用
    y5l, y10l = tsy["5 Yr"], tsy["10 Yr"]
    d5y = y5l.diff() * 100.0     # 转成 bp
    d10y = y10l.diff() * 100.0
    px9141, pxnav = etf["Adj Close"], nav["nav_usd"]

    # ---- 几个候选的日度序列 ----
    ret9141 = np.log(px9141).diff() * 100
    retnav = np.log(pxnav).diff() * 100

    # C1) 真久期修正序列：零收益日拿利率面补上，有变动的日子保留原值
    m_active = None
    r = ret9141
    act = pd.DataFrame({"r": r, "l5": d5y.shift(1), "l10": d10y.shift(1)})
    act = act[(act["r"] != 0)].dropna()
    ma = sm.OLS(act["r"], sm.add_constant(act[["l5", "l10"]])).fit()
    rate_all = ma.params["l5"] * d5y.shift(1) + ma.params["l10"] * d10y.shift(1)
    rate_all = rate_all.reindex(r.index)
    ret_fix = r.where(r != 0, rate_all).dropna()
    # 因子法风险估计：利率部分用日度因子方差乘真久期，利差部分用变动日的残差
    spread_active = (act["r"] - ma.predict(sm.add_constant(act[["l5", "l10"]])))
    sig_rate_d = rate_all.std()                       # 利率那条的日波动(%/日)
    sig_spread_d = spread_active.std()                # 利差那条的日波动(%/日，只在变动日上算)
    sig_model_d = float(np.sqrt(sig_rate_d**2 + sig_spread_d**2))

    # ---- 周度序列，价格和 NAV 各跑一份，好对照 ----
    wo9141 = weekly_ols(px9141, y5l, y10l)
    wonav = weekly_ols(pxnav, y5l, y10l)

    # ---- 指标表：每加一个方案就往 rows 里塞一行 ----
    rows = []
    def add(name, rets, dur_r2, resid, per_year, mode, oas_lag, extra=""):
        n0, z = zshare(rets)
        vol = ann_vol(rets, per_year)          # 年化 σ%，日度周度放一起也能直接比
        sd1 = vol / np.sqrt(252.0)             # 折成 1 日 σ（持有期就是 1 日）
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

    # 日度三个：价格、NAV、真久期修正。NAV 和美股同一天估值所以用 lag=0，价格用 lag=1，别换
    d9141 = daily_ols(px9141, d5y, d10y, lag=1)
    add("D1 价格9141(现状)", ret9141, d9141, d9141["resid"], 252, "daily", 1,
        "基准：~63%零收益")
    nav_d = daily_ols(pxnav, d5y, d10y, lag=0)
    add("D2 官方NAV(日度)", retnav, nav_d, nav_d["resid"], 252, "daily", 0,
        "真实日度估值★(lag0)")
    add("D3 真久期修正(日度)", ret_fix, None, None, 252, "daily", 1,
        f"零日按利率面补(因子法σ年化{sig_model_d*np.sqrt(252):.1f}%,偏高),非观测→仅供参考")
    # 周度再来两个：价格和 NAV，利率都用同周的 Δy
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

    # 对一下 NAV 和价格的日度收益：看官方 NAV 能不能当 9141 的日度无偏代理
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
    # 1) 归一化净值：NAV 和价格放一起，看长期趋势对不对得上
    base = pd.concat([pxnav.rename("nav"), px9141.rename("px")], axis=1)
    for col, lab, colr in [("nav", "官方NAV(USD)", "tab:green"), ("px", "9141价格(USD)", "tab:blue")]:
        s = base[col]
        axs[0].plot(s.index, s / s.iloc[0] * 100, lw=1.0, color=colr, label=lab)
    axs[0].set_ylabel("归一(起点=100)"); axs[0].set_title(
        "9141.HK 官方NAV vs 市价水平（NAV 逐日、市价陈旧→呈台阶）")
    axs[0].legend(); axs[0].grid(alpha=.3)
    # 2) 日收益的量级分布，Y 取对数，好看清价格那一大堆 0
    for col, lab, colr in [("nav", "NAV日收益%", "tab:green"), ("price", "价格日收益%", "tab:blue")]:
        a = (j[col].abs().dropna() + 1e-6)
        axs[1].hist(np.log(a), bins=60, alpha=.45, color=colr, label=lab)
    axs[1].set_xlabel("ln(|日收益|+eps)"); axs[1].set_ylabel("频数(对数)")
    axs[1].set_title("日收益量级分布：价格有 ~63% 精确 0%（最左峰），NAV 逐日变化")
    axs[1].legend(); axs[1].grid(alpha=.3)
    # 3) 周度价格、周度 NAV 的滚窗年化波动，再加一条因子法作参照
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
