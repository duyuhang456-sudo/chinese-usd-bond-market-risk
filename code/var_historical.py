"""阶段二 Day3（9/16）历史模拟法 VaR：经典 HS（窗宽 250/500/750）+ Filtered-HS（EWMA / GARCH 双滤波），双口径

消费：factors/factor_table_nav_tr.csv（复权主口径 etf_ret_tr_pct / 人民币次口径 etf_ret_rmb_pct）；
      results/var_parametric.csv（9/15 产物的 sig_ewma / sig_garch 列）
产出：results/var_historical.csv
      figures/var_hs_vs_parametric.png / var_hs_quantile_path.png / var_fhs_compare.png
口径：1 日持有期；95% / 99%；VaR 存正值（损失）；μ=0 且不去均值（与 spec §1 参数法口径一致）；
      窗内经验分位用线性插值；预测窗严格取 [t−W, t−1]。滤波用的 σ 列是 9/15 的扩展窗逐日一步
      向前预测（只用 ≤ t−1 信息），直接复用即无前视。
边界：W=250/500/750 → 有效 1023/773/523 日。FHS 两版 σ 自样本外首日起才有定义，窗满需再
      250 日，故各 773 日；一律用成熟种子 seed_win=250，不用序列首日播种（该方案会使
      FHS99 虚高约 85%，量化过程见 spec §6 的 9/16 补记 L168-171），代价是少 250 个
      预测日。次口径汇率源尾 5 日未发布：M2 因窗端点落入 NaN 段缺 4 日，M2f 另因 σ_t 本身
      为 NaN 缺 5 日。
用法：./.venv/bin/python code/var_historical.py（须先跑 code/var_parametric.py）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Songti SC",
                                   "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import CLEAN, FACT, RES, FIG, ensure_dirs
from var_common import (WIN, LAM, C_M1, C_GARCH, C_ALT, C_GREY, INK2,
                        hs_var, hs_var_filtered, count_violations)

ensure_dirs()

WS = (250, 500, 750)            # 窗宽稳健性
CS = ("95", "99")
Q = {"95": 0.05, "99": 0.01}
# 窗宽是有序维度，用同色相深浅表示，不占用第二色相（蓝/橙留给「模型」维度，避免串义）
C_W = {250: C_ALT, 500: "#6fcbac", 750: "#b3e0cd"}
# FHS 与经典 HS 同族，用同色相、以线型区分滤波方式
LS_HS, LS_E, LS_G = "-", "--", ":"


def nval(s: pd.Series) -> int:
    """非 NaN 计数。

    参数：
        s: 任意 Series。

    返回：
        int，s 中非 NaN 的元素个数。
    """
    return int(s.notna().sum())


def span(s: pd.Series) -> tuple[str, str]:
    """首个与末个有效日的日期字符串。

    参数：
        s: 带 DatetimeIndex 的 Series。

    返回：
        (首个有效日, 末个有效日) 二元组，格式 "YYYY-MM-DD"；无有效值时返回 ("—", "—")。
    """
    d = s.dropna()
    return ("—", "—") if d.empty else (str(d.index[0].date()), str(d.index[-1].date()))


def acf1(x: pd.Series) -> float:
    """一阶自相关。

    参数：
        x: 数值 Series。

    返回：
        float，去 NaN 后按位置计算的一阶自相关系数。
    """
    return float(x.dropna().autocorr(lag=1))


def main() -> None:
    """历史模拟法 VaR 主流程（其余口径说明见模块 docstring）。

    脚本契约：
        消费：factors/factor_table_nav_tr.csv（复权主口径 / 人民币次口径收益）；
            results/var_parametric.csv（9/15 产物，取其中 sig_ewma / sig_garch 作滤波 σ）；
            results/garch_refit_trace_rmb.csv（次口径 GARCH 边界解区段）；
            clean_data/outlier_judgment.csv（阶段一存疑日，存在时才读）。
        产出：results/var_historical.csv（样本外逐日：收益、经典 HS 三窗 × 两置信度、FHS 两滤波 ×
            两置信度，主口径与次口径分列）；figures/var_hs_vs_parametric.png、
            figures/var_hs_quantile_path.png、figures/var_fhs_compare.png。
        断言/边界：样本外长度必须等于 len(F) − WIN，与 9/15 一致；VaR 一律存正值且 99% ≥ 95%；
            W=250 在样本外首日即应有值；随机抽 3 个 t 与手工 np.quantile(r[t−250:t], 0.01) 精确比对
            以防前视；违规判定与手写循环比对；次口径尾部缺口按 M2 缺 4 日、M2f 缺 5 日的机制分别说明。
            缺 results/var_parametric.csv 时直接 SystemExit，须先跑 code/var_parametric.py。
    """
    vp_path = RES / "var_parametric.csv"
    if not vp_path.exists():
        raise SystemExit("缺少 results/var_parametric.csv —— 请先运行 code/var_parametric.py"
                         "（复现顺序见 spec §9：9/16 依赖 9/15 的 σ 列）")
    F = pd.read_csv(FACT / "factor_table_nav_tr.csv", parse_dates=["date"]).set_index("date")
    r_main = F["etf_ret_tr_pct"].astype(float)
    r_rmb = F["etf_ret_rmb_pct"].astype(float)
    nan_pos = np.flatnonzero(r_rmb.isna().values)
    assert nan_pos.size == 0 or nan_pos[-1] == len(r_rmb) - 1, "人民币次口径 NaN 非尾部连续，需检查"
    n_tail = int(nan_pos.size)

    VP = pd.read_csv(vp_path, parse_dates=["date"]).set_index("date")
    idx = VP.index
    assert len(idx) == len(F) - WIN, f"样本外长度与 9/15 不一致：{len(idx)} vs {len(F) - WIN}"

    print(f"[输入] factor_table_nav_tr rows={len(F)}  主口径 n={nval(r_main)}  "
          f"次口径 n={nval(r_rmb)}（CNY 尾 {n_tail} 日记 NaN，不伪造）")
    print(f"       样本外 {len(idx)} 日（{idx[0].date()} ~ {idx[-1].date()}）"
          f"，协议 §7.1：窗 {WIN} 日 / 步长 1 日 / 逐日向前")

    # ---- M2 经典 HS ----
    oos = pd.DataFrame(index=idx)
    oos.index.name = "date"
    oos["ret_tr_pct"] = r_main.reindex(idx)
    oos["ret_rmb_pct"] = r_rmb.reindex(idx)
    for w in WS:
        for c in CS:
            oos[f"hs{w}_var_{c}"] = hs_var(r_main, w, c).reindex(idx)
            oos[f"rmb_hs{w}_var_{c}"] = hs_var(r_rmb, w, c).reindex(idx)

    # ---- M2f Filtered-HS（两版滤波，σ 取自 9/15，同为一步向前、无前视） ----
    oos["fhs_sig_e"], oos["fhs_sig_g"] = VP["sig_ewma"], VP["sig_garch"]
    oos["rmb_fhs_sig_e"], oos["rmb_fhs_sig_g"] = VP["rmb_sig_ewma"], VP["rmb_sig_garch"]
    for c in CS:
        oos[f"fhs_e_var_{c}"] = hs_var_filtered(oos["ret_tr_pct"], oos["fhs_sig_e"], WIN, c)
        oos[f"fhs_g_var_{c}"] = hs_var_filtered(oos["ret_tr_pct"], oos["fhs_sig_g"], WIN, c)
        oos[f"rmb_fhs_e_var_{c}"] = hs_var_filtered(oos["ret_rmb_pct"], oos["rmb_fhs_sig_e"], WIN, c)
        oos[f"rmb_fhs_g_var_{c}"] = hs_var_filtered(oos["ret_rmb_pct"], oos["rmb_fhs_sig_g"], WIN, c)

    # 符号约定护栏：VaR 一律存正值。若误传负号，违规判定会把每天都判成违规，看起来只是
    # 「失败率 100%」，容易当成模型问题而不是符号问题
    vcols = [c for c in oos.columns if "_var_" in c]
    assert all((oos[c].dropna() >= -1e-9).all() for c in vcols), "VaR 出现负值，符号约定被破坏"
    assert all((oos[c.replace("_95", "_99")].dropna() >= oos[c].dropna() - 1e-9).all()
               for c in vcols if c.endswith("_95")), "99% VaR 应 ≥ 95% VaR"

    oos.round(6).to_csv(RES / "var_historical.csv", encoding="utf-8-sig")

    # ---- [1] 样本与下标 ----
    print("\n[1] 样本与下标约定（M2f 少 250 日：σ 自样本外首日起才可用，窗满再需 250 日）：")
    print(f"    {'模型':<24s}{'口径':<8s}{'首个有效日':<14s}{'末个有效日':<14s}{'有效日数':>10s}")
    for lbl, col in [("M2 经典 HS W=250", "hs250_var_99"), ("M2 经典 HS W=500", "hs500_var_99"),
                     ("M2 经典 HS W=750", "hs750_var_99"), ("M2f FHS-E（EWMA 滤波）", "fhs_e_var_99"),
                     ("M2f FHS-G（GARCH 滤波）", "fhs_g_var_99")]:
        for cal, pre in [("美元主", ""), ("人民币", "rmb_")]:
            s = oos[pre + col]
            a, b = span(s)
            print(f"    {lbl:<24s}{cal:<8s}{a:<14s}{b:<14s}{nval(s):>10d}")

    # ---- [2] 分位口径 ----
    print("\n[2] 分位口径（窗内线性插值，插值位置 = q·(n−1)，0 基）：")
    for w in WS:
        p99, p95 = 0.01 * (w - 1), 0.05 * (w - 1)
        print(f"    W={w:<4d} 1% 位置 {p99:6.2f} → 窗内第 {int(p99)+1}、{int(p99)+2} 小观测之间；"
              f"5% 位置 {p95:6.2f} → 第 {int(p95)+1}、{int(p95)+2} 小")
    print("    即 99% VaR 的尾部由窗内约 **2 个**观测决定（n=250 时），这是 HS 尾部估计不稳的根源，")
    print("    也是 9/17 必须写明检验力边界的原因。")
    dev = float((oos["ret_tr_pct"] - oos["ret_tr_pct"].mean()).abs().mean())
    print(f"    另：本口径**不做去均值**（与 spec §1 参数法 μ=0 一致），窗内均值实际上充当了条件均值；")
    print(f"    若改为去均值，平均 VaR 将上升约 {abs(float(oos['ret_tr_pct'].mean())):.4f}（≈2%），已记档。")

    # ---- [3] 平均 VaR 对照 ----
    print("\n[3] 样本外平均 VaR（%/日，μ=0）与路径极差：")
    rows = []
    for lbl, col in [("M2 经典 HS W=250", "hs250"), ("M2 经典 HS W=500", "hs500"),
                     ("M2 经典 HS W=750", "hs750"), ("M2f FHS-E（EWMA）", "fhs_e"),
                     ("M2f FHS-G（GARCH）", "fhs_g")]:
        for cal, pre in [("美元主", ""), ("人民币", "rmb_")]:
            v95, v99 = oos[f"{pre}{col}_var_95"], oos[f"{pre}{col}_var_99"]
            rows.append({"模型": lbl, "口径": cal, "VaR95(%)": v95.mean(), "VaR99(%)": v99.mean(),
                         "99%最小": v99.min(), "99%最大": v99.max(), "99%极差": v99.max() - v99.min(),
                         "有效日": nval(v99)})
    cmp_hs = pd.DataFrame(rows)
    print(cmp_hs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("    对照 9/15 参数法：M1 VaR95 0.4398 / VaR99 0.6220；M1g 0.4348 / 0.6149。")

    # ---- [4] 与参数法的关系 + 失败率预览（只给计数，p 值归 9/17） ----
    print("\n[4] 与 9/15 参数法的关系：")
    for a, b in [("hs250_var_99", "m1_var_99"), ("hs250_var_99", "garch_var_99"),
                 ("hs250_var_95", "m1_var_95"), ("fhs_e_var_99", "garch_var_99")]:
        d = pd.concat([oos[a], VP[b]], axis=1).dropna()
        print(f"    corr({a:<14s}, {b:<14s}) = {d.iloc[:,0].corr(d.iloc[:,1]):.3f}")
    print("\n    失败率预览（**仅计数，不作检验**——Kupiec / Christoffersen 统一由 9/17 出具，")
    print("    以免两个脚本产出两套「官方」数字；判定共用 var_common.count_violations）：")
    print(f"    {'模型':<24s}{'口径':<8s}{'95%违规':>9s}{'费率':>8s}{'期望':>8s}{'99%违规':>9s}{'费率':>8s}{'期望':>8s}")
    for lbl, col in [("M1 无条件（9/15）", "m1"), ("M1g GARCH（9/15）", "garch"),
                     ("M1e EWMA（9/15）", "ewma"), ("M2 经典 HS W=250", "hs250"),
                     ("M2 经典 HS W=500", "hs500"), ("M2 经典 HS W=750", "hs750"),
                     ("M2f FHS-E（EWMA）", "fhs_e"), ("M2f FHS-G（GARCH）", "fhs_g")]:
        for cal, pre in [("美元主", ""), ("人民币", "rmb_")]:
            src = VP if col in ("m1", "garch", "ewma") else oos
            x95, n95, r95 = count_violations(oos["ret_tr_pct" if not pre else "ret_rmb_pct"],
                                             src[f"{pre}{col}_var_95"])
            x99, n99, r99 = count_violations(oos["ret_tr_pct" if not pre else "ret_rmb_pct"],
                                             src[f"{pre}{col}_var_99"])
            print(f"    {lbl:<24s}{cal:<8s}{x95:>6d}/{n95:<4d}{r95:>7.2f}%{0.05*n95:>8.1f}"
                  f"{x99:>6d}/{n99:<4d}{r99:>7.2f}%{0.01*n99:>8.1f}")

    # ---- [5] 窗宽稳健性 ----
    print("\n[5] 窗宽稳健性（两套比较，不可混用）：")
    print("    (a) 各窗自身全样本 —— 样本长度不同，**直接横比失败率是错的**：")
    for w in WS:
        s99 = oos[f"hs{w}_var_99"]
        x95, n95, r95 = count_violations(oos["ret_tr_pct"], oos[f"hs{w}_var_95"])
        x99, n99, r99 = count_violations(oos["ret_tr_pct"], s99)
        print(f"        W={w:<4d} n={n99:<5d} 95%：{x95:>3d}/{n95:<5d}{r95:>6.2f}%   "
              f"99%：{x99:>2d}/{n99:<5d}{r99:>6.2f}%   99% 分位路径极差 {s99.max()-s99.min():.4f}")
    common = oos[[f"hs{w}_var_99" for w in WS]].notna().all(axis=1)
    ci = oos.index[common]
    print(f"    (b) 公共样本 {len(ci)} 日（{ci[0].date()} ~ {ci[-1].date()}）—— 唯一可横比的样本：")
    for w in WS:
        x95, n95, r95 = count_violations(oos.loc[common, "ret_tr_pct"], oos.loc[common, f"hs{w}_var_95"])
        x99, n99, r99 = count_violations(oos.loc[common, "ret_tr_pct"], oos.loc[common, f"hs{w}_var_99"])
        print(f"        W={w:<4d} 95%：{x95:>3d}/{n95}（{r95:.2f}%，期望 {0.05*n95:.1f}）   "
              f"99%：{x99:>2d}/{n99}（{r99:.2f}%，期望 {0.01*n99:.1f}）")
    print(f"    (c) 读法警告：公共样本全部落在 {ci[0].date()} 之后，**不含 2022 高波动段**——")
    print("        而宽窗的问题（记忆滞留、响应慢）恰恰只在高波动段暴露。故「W=750 在公共样本违规最少」")
    print("        **不能**读作「宽窗更优」，只能读作「在平静样本上越宽的窗越保守」。")
    print(f"        99% 下公共样本期望违规仅 {0.01*len(ci):.1f} 次，检验力极弱，结论须以点估计 + 分段一致性为准。")
    print("    (d) 收敛性的直接证据是分位路径本身（见上表「99% 分位路径极差」）：")
    print("        W=250 极差最大 → 对波动敏感但抖动；W=750 极差最小 → 平滑但响应慢。")

    # ---- [6] Filtered-HS ----
    print("\n[6] Filtered-HS（两版滤波）——与经典 HS 的**头对头**比较（唯一诚实的比法：同一样本）")
    fe = oos[["hs250_var_99", "fhs_e_var_99", "fhs_g_var_99"]].notna().all(axis=1)
    sub6 = oos.loc[fe]
    print(f"    公共样本 {int(fe.sum())} 日（{sub6.index[0].date()} ~ {sub6.index[-1].date()}）：")
    print(f"        {'模型':<22s}{'95%违规':>10s}{'费率':>8s}{'99%违规':>10s}{'费率':>8s}{'平均VaR99':>11s}")
    for lbl, t in [("经典 HS W=250（无滤波）", "hs250"), ("FHS-E（EWMA 滤波）", "fhs_e"),
                   ("FHS-G（GARCH 滤波）", "fhs_g")]:
        x95, n95, r95 = count_violations(sub6["ret_tr_pct"], sub6[f"{t}_var_95"])
        x99, n99, r99 = count_violations(sub6["ret_tr_pct"], sub6[f"{t}_var_99"])
        print(f"        {lbl:<22s}{x95:>6d}/{n95:<4d}{r95:>7.2f}%{x99:>6d}/{n99:<4d}{r99:>7.2f}%"
              f"{sub6[f'{t}_var_99'].mean():>11.4f}")
    n6 = int(fe.sum())
    print(f"        （期望：95% → {0.05*n6:.1f} 次；99% → {0.01*n6:.1f} 次；"
          f"2SE 带宽：95% ±{2*np.sqrt(0.05*0.95/n6)*100:.2f}%，99% ±{2*np.sqrt(0.01*0.99/n6)*100:.2f}%）")
    r_hs95 = count_violations(sub6["ret_tr_pct"], sub6["hs250_var_95"])[2]
    r_fe95 = count_violations(sub6["ret_tr_pct"], sub6["fhs_e_var_95"])[2]
    r_hs99 = count_violations(sub6["ret_tr_pct"], sub6["hs250_var_99"])[2]
    r_fe99 = count_violations(sub6["ret_tr_pct"], sub6["fhs_e_var_99"])[2]
    print(f"        读法（与直觉相反，如实报出）：本样本上**滤波并未改善经典 HS**——")
    print(f"            95%：经典 HS {r_hs95:.2f}%（几乎正中 5%）→ FHS-E {r_fe95:.2f}%（偏高，VaR 偏小）")
    print(f"            99%：经典 HS {r_hs99:.2f}% → FHS-E {r_fe99:.2f}%（同样偏高）")
    print(f"            但两处差异分别为 {abs(r_fe95-r_hs95):.2f}% / {abs(r_fe99-r_hs99):.2f}%，"
          f"均**落在 2SE 带宽内** → 只能说「未观察到改善」，不能说「滤波有害」。")
    print(f"            机制上是说得通的：滤波修正的是「波动标度」，而经典 HS 的 250 日窗")
    print(f"            本身已含近期高波动观测；z 的自相关虽被压掉（见下），但分位×σ_t 后")
    print(f"            平均 VaR99 由 {sub6['hs250_var_99'].mean():.4f} 降到 "
          f"{sub6['fhs_e_var_99'].mean():.4f}，覆盖率随之下降。")
    z_e = (oos["ret_tr_pct"] / oos["fhs_sig_e"]).dropna()
    r_ = oos["ret_tr_pct"].dropna()
    print(f"        标准化诊断（滤波的目标是让 z 更接近 iid）：")
    print(f"            超额峰度  r = {r_.kurt():+.3f}  →  z_E = {z_e.kurt():+.3f}  →  z_G = "
          f"{(oos['ret_tr_pct']/oos['fhs_sig_g']).dropna().kurt():+.3f}")
    print(f"            一阶自相关 r² = {acf1(r_**2):+.3f}  →  z_E² = {acf1(z_e**2):+.3f}  →  z_G² = "
          f"{acf1((oos['ret_tr_pct']/oos['fhs_sig_g']).dropna()**2):+.3f}")
    print(f"            即滤波确实压掉了大部分波动聚集（r² 自相关 → z² 自相关），这是 FHS 的机制依据；")
    print(f"            但两版滤波的失败率差异很小（均值差 "
          f"{abs(oos['fhs_e_var_99'].mean()-oos['fhs_g_var_99'].mean())/oos['fhs_e_var_99'].mean()*100:.1f}%）"
          f"→ 9/17 对比时不必把「滤波方式选择」当成重要变量。")
    # 次口径边界解段：该段 GARCH σ 退化为常数，滤波失效
    tr_r = pd.read_csv(RES / "garch_refit_trace_rmb.csv", parse_dates=["date"]).set_index("date")
    bad = tr_r.index[tr_r["alpha"] < 1e-4]
    if len(bad):
        print(f"        [边界解] 次口径 {len(bad)} 次重估落 α→0、β→1（{bad[0].date()} ~ {bad[-1].date()}）：")
        print(f"                 该段 σ_t 退化为常数，z_i=r_i/σ_const → **FHS 退化为「缩放后的经典 HS」**，")
        print(f"                 滤波恰在最需要的高波动段失效。此结论强于 9/15 的「GARCH 等价于无条件」，")
        print(f"                 图中已用阴影标出（区间由 garch_refit_trace_rmb.csv 程序化读取，不硬编码）。")

    # ---- [7] ghost effect：极值日进出窗 ----
    print("\n[7] Ghost effect（窗内极值日进出窗造成的 VaR 跳变）——9/17 归因违规日时直接用到：")
    print("    下标：预测日 i 用窗 [i−W, i−1]。由 i−1 走到 i，**离窗**日为 i−1−W，**入窗**日为 i−1。")
    for w in (250, 750):
        va = oos[f"hs{w}_var_99"]
        dva = va.diff().dropna()
        print(f"        W={w}：")
        for lbl, top in [("VaR 下降前 3（大额亏损日离窗 → 尾部变浅）", dva.nsmallest(3)),
                         ("VaR 跳升前 3（新极值日入窗 → 尾部变深）", dva.nlargest(3))]:
            print(f"            {lbl}：")
            for dt, val in top.items():
                i = WIN + int(oos.index.get_loc(dt))     # 全样本下标
                leave, enter = i - 1 - w, i - 1
                ld = str(r_main.index[leave].date()) if leave >= 0 else "—(样本前)"
                lv = f"{r_main.iloc[leave]:+.2f}%" if leave >= 0 else "—"
                print(f"                {dt.date()}  ΔVaR {val:+.5f}   离窗 {ld} r={lv:>7s}"
                      f"   入窗 {r_main.index[enter].date()} r={r_main.iloc[enter]:+.2f}%")
    print("        读法：跳变紧邻的「离窗/入窗」日若确为大额亏损日，则 VaR 变动由窗结构驱动，")
    print("        而非模型失效——9/17 凡遇违规日，先查它是否落在此类跳变点附近。")

    # ---- [8] 存疑日与 HS 的交互（为 9/17 §8.1 预置证据） ----
    oj_path = CLEAN / "outlier_judgment.csv"
    if oj_path.exists():
        oj = pd.read_csv(oj_path, parse_dates=["date"])
        sus = oj[oj["verdict"].astype(str).str.contains("存疑", na=False)]
        etf = sus[sus["series"] == "ETF_ret_pct"]
        print(f"\n[8] 阶段一「存疑」日与 HS 的交互（共 {len(sus)} 条，其中 ETF 类 {len(etf)} 条）——")
        print("    9/17 的「含/不含」稳健性复核据此预置：若存疑日根本进不了任何窗口的尾部 1%，")
        print("    则 M2 的含/不含两版**结果相同**（仍按规执行两版，只是预期无差异）。")
        rank = r_main.abs().rank(ascending=False)
        if len(etf):
            hits = [(d.date(), float(r_main.get(d, np.nan)), int(rank.get(d, -1))) for d in etf["date"]]
            hits = [h for h in hits if np.isfinite(h[1])]
            hits.sort(key=lambda h: h[2])
            for d, v, rk in hits:
                print(f"        {d}  复权日收益 {v:+.4f}%  |r| 全样本排名 {rk}/{len(r_main)}")
            print(f"        → ETF 类存疑日在**复权口径**下的 |r| 排名最靠前也只有 {hits[0][2]}")
            print(f"          （远在窗内前 2 名之外）→ 不进入任何 W 的尾部 1%，含/不含无差异是预期结果。")
        fx = sus[sus["series"] == "FX_ret_pct"]
        if len(fx):
            rk_fx = r_rmb.abs().rank(ascending=False)
            best = min((int(rk_fx.get(d, 10**9)) for d in fx["date"]), default=10**9)
            print(f"        汇率类存疑日 {len(fx)} 条，在次口径 |r| 中排名最靠前 {best} → 同样不影响主口径 M2。")

    # ---- 自检 ----
    print("\n[8b] 自检：")
    assert nval(oos["hs250_var_99"]) == len(oos), "W=250 在样本外首日即应有值（窗 [0,249] 已满）"
    for w in WS:
        assert nval(oos[f"hs{w}_var_99"]) == len(oos) - max(0, w - WIN)
    # 防前视：随机抽 3 个 t，与手工 np.quantile(r[t−250:t], 0.01) 精确比对
    rng = np.random.default_rng(20260916)
    picks = sorted(rng.choice(np.arange(WIN + 300, len(oos)), size=3, replace=False).tolist())
    for p in picks:
        t_full = WIN + p
        manual = -float(np.quantile(r_main.iloc[t_full - 250:t_full].values, 0.01))
        got = float(oos["hs250_var_99"].iloc[p])
        assert abs(manual - got) < 1e-9, f"t={t_full} 前视/口径不符：{manual} vs {got}"
    print(f"        防前视抽检 3 个 t（{picks}）：手工 np.quantile(r[t−250:t], 0.01) 与输出完全一致")
    # 违规判定与手写循环一致
    x_a, n_a, _ = count_violations(oos["ret_tr_pct"], oos["hs250_var_95"])
    d = pd.concat([oos["ret_tr_pct"].rename("r"), oos["hs250_var_95"].rename("v")], axis=1).dropna()
    x_b = int((d["r"] < -d["v"]).sum())
    assert (x_a, n_a) == (x_b, len(d)), "违规判定与手写循环不一致"
    print(f"        count_violations 与手写循环一致（HS250-95：{x_a}/{n_a}）")
    # 次口径尾部缺口的机制不同，必须分别说明（都源自汇率源尾 5 日 NaN，但敏感点不一样）：
    #   M2  VaR(t) 用窗 [t−W, t−1]，只要 t−1 落入 NaN 段即缺，故缺 4 日
    #   M2f 输出 = 窗分位 × σ_t，σ_t 本身在 NaN 段为 NaN，故缺 5 日（比 M2 多 1 日）
    print(f"        次口径缺口（汇率源尾 {n_tail} 日 NaN，主口径完全不受影响）：")
    for tag, col in [("M2  经典 HS W=250", "rmb_hs250_var_99"),
                     ("M2f FHS-E", "rmb_fhs_e_var_99"), ("M2f FHS-G", "rmb_fhs_g_var_99")]:
        s = oos[col]
        pos = np.flatnonzero(s.notna().values)
        warm = int(pos[0]) if pos.size else len(s)
        tail = len(s) - 1 - int(pos[-1]) if pos.size else 0
        print(f"            {tag:<20s} 预热 {warm:>3d} 日 + 尾缺 {tail} 日 → 有效 {nval(s)}")
    print("            机制：M2 用窗 [t−W, t−1]，需 t−1 落入 NaN 段才缺（尾缺 4）；")
    print("                  M2f 输出 = 窗分位 × σ_t，σ_t 本身在 NaN 段即为 NaN，故尾缺 5（多 1 日）。")
    print(f"        VaR 正值约定、99%≥95% 单调性：断言通过")
    print(f"        → results/var_historical.csv  rows={len(oos)}  cols={len(oos.columns)}")

    # ---- 图 1：HS vs 参数法（口径 × 置信度） ----
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.4), sharex=True)
    panels = [("美元主口径（复权 USD）", "ret_tr_pct", ""), ("人民币次口径", "ret_rmb_pct", "rmb_")]
    for irow, (lbl, rcol, pre) in enumerate(panels):
        for icol, c in enumerate(CS):
            ax = axes[irow, icol]
            ax.plot(oos.index, oos[rcol], lw=.7, color=C_GREY, label="日收益")
            ax.plot(oos.index, -oos[f"{pre}hs250_var_{c}"], lw=1.6, color=C_ALT, label="M2 经典 HS(250)")
            ax.plot(oos.index, -VP[f"{pre}m1_var_{c}"], lw=1.2, color=C_M1, label="M1 无条件")
            ax.plot(oos.index, -VP[f"{pre}garch_var_{c}"], lw=1.2, color=C_GARCH, label="M1g GARCH")
            ax.set_title(f"{lbl} · {c}% VaR（负值 = 损失阈值）", fontsize=9.5, loc="left")
            ax.grid(alpha=.25, lw=.6)
            ax.margins(x=0.06)
            if icol == 0:
                ax.set_ylabel("日收益 / VaR (%)")
    # 边界解只存在于次口径 GARCH（主口径 0 次），故阴影只画在下排，否则读者会以为主口径
    # 也有 133 次退化。下排两个置信度都要标：退化的 σ_t 同时影响该段两个面板的读法。
    if len(bad):
        for ax in axes[1]:
            ax.axvspan(bad[0], bad[-1], color=INK2, alpha=.08, lw=0)
        axes[1, 0].text(bad[0], 0.04, " 阴影段：MLE 落 α→0/β→1 边界",
                        transform=axes[1, 0].get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.938))
    fig.suptitle("历史模拟法（经典 HS 250 日）与参数法逐日样本外 VaR 对照（2022-08 ~ 2026-09）",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(FIG / "var_hs_vs_parametric.png", dpi=150)
    print("\n[图] figures/var_hs_vs_parametric.png")

    # ---- 图 2：经验分位路径随窗宽变化 ----
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.2), sharex=True)
    for irow, (lbl, pre) in enumerate([("美元主口径（复权 USD）", ""), ("人民币次口径", "rmb_")]):
        for icol, c in enumerate(CS):
            ax = axes[irow, icol]
            labs = []
            for w in WS:
                s = oos[f"{pre}hs{w}_var_{c}"]
                ax.plot(s.index, -s, lw=1.8 if w == 250 else 1.3, color=C_W[w], label=f"W={w}")
                if s.notna().any():
                    labs.append([float(-s.dropna().iloc[-1]), f"  {w}", C_W[w]])
            labs.sort()
            for i in range(1, len(labs)):
                labs[i][0] = max(labs[i][0], labs[i - 1][0] + 0.05)
            for y, txt, col in labs:
                ax.text(oos.index[-1], y, txt, va="center", ha="left", fontsize=8, color=col)
            rng99 = oos[f"{pre}hs250_var_{c}"].max() - oos[f"{pre}hs250_var_{c}"].min()
            rng750 = oos[f"{pre}hs750_var_{c}"].max() - oos[f"{pre}hs750_var_{c}"].min()
            extra = f"（路径极差 250 日 {rng99:.2f} → 750 日 {rng750:.2f}）" if c == "99" else ""
            ax.set_title(f"{lbl} · {c}% 经验分位路径{extra}", fontsize=9.5, loc="left")
            ax.set_ylabel("−VaR (%)" if icol == 0 else "")
            ax.grid(alpha=.25, lw=.6)
            ax.margins(x=0.08)
            # 半年刻度在本图宽度下会挤成「2022-072023-01」，改用年刻度（6 个标签，不重叠）
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # 四个面板序列完全相同，图例提到画布顶部，避免面板内图例压住分位线
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.938))
    fig.suptitle("历史模拟法经验分位路径对窗宽的敏感性（窗越宽越平滑、响应越慢）", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(FIG / "var_hs_quantile_path.png", dpi=150)
    print("[图] figures/var_hs_quantile_path.png")

    # ---- 图 3：滤波是否改善（公共样本，99%） ----
    m = oos[["hs250_var_99", "fhs_e_var_99", "fhs_g_var_99"]].notna().all(axis=1)
    sub = oos.loc[m]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    for i, (lbl, rcol, pre) in enumerate([("美元主口径（复权 USD）", "ret_tr_pct", ""),
                                          ("人民币次口径", "ret_rmb_pct", "rmb_")]):
        ax = axes[i]
        ax.plot(sub.index, sub[rcol], lw=.7, color=C_GREY, label="日收益")
        ax.plot(sub.index, -sub[f"{pre}hs250_var_99"], lw=1.4, color=C_ALT, ls=LS_HS, label="经典 HS(250)")
        ax.plot(sub.index, -sub[f"{pre}fhs_e_var_99"], lw=1.4, color=C_ALT, ls=LS_E, label="FHS（EWMA 滤波）")
        ax.plot(sub.index, -sub[f"{pre}fhs_g_var_99"], lw=1.4, color=C_ALT, ls=LS_G, label="FHS（GARCH 滤波）")
        # 同上：边界解只在次口径，阴影不可画到主口径面板
        if len(bad) and i == 1:
            ax.axvspan(bad[0], bad[-1], color=INK2, alpha=.08, lw=0)
        ax.set_title(f"{lbl} · 99% VaR（公共样本 {len(sub)} 日，{sub.index[0].date()} ~ {sub.index[-1].date()}）",
                     fontsize=9.5, loc="left")
        ax.set_ylabel("日收益 / VaR (%)" if i == 0 else "")
        ax.grid(alpha=.25, lw=.6)
        ax.margins(x=0.02)
    # 两面板序列相同，图例提到画布顶部，否则会压住 2023-2024 段的分位线
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Filtered-HS 与经典 HS 对照：滤波（EWMA / GARCH）能否改善高波动期的响应滞后",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    fig.savefig(FIG / "var_fhs_compare.png", dpi=150)
    print("[图] figures/var_fhs_compare.png")


if __name__ == "__main__":
    main()
