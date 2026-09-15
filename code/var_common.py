"""
阶段二 VaR 公共机制：常量 / 配色 / σ 估计器 / 历史模拟分位 / 违规判定。

抽出来是为了让 9/15（参数法）、9/16（历史模拟）、9/17（回测检验）三个脚本**共用同一套口径**——
尤其是违规判定：若 9/16 的预览表和 9/17 的正式回测各写一份，很容易出现「同一份 VaR 序列
数出不同违规数」的口径分叉，届时结论无法归因。

所有 σ / 分位序列都遵守同一条时序纪律：**在位置 t 只能使用 ≤ t−1 的信息**。
pandas 的 `rolling(...)` 在位置 t 含当日，故一律 `.shift(1)`（历史教训见 var_parametric.py 的 docstring）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量
WIN = 250                       # 估计窗（交易日），与回测协议 §7.1 一致
Z = {"95": 1.6449, "99": 2.3263}
LAM = 0.94                      # EWMA 衰减因子（RiskMetrics）

# 分类色：参考调色板已文档化通过 all-pairs 校验的前三槽（固定顺序，不轮转）。
# 按**槽位**命名而非按模型命名，实体→槽位的映射在各脚本内固定：
#   槽1 蓝   = 基础/无条件口径（9/15 M1；9/16 对照线）
#   槽2 橙   = 条件波动口径（9/15 GARCH）
#   槽3 水绿 = 第三序列（9/15 EWMA；9/16 经典 HS / Filtered-HS）
C_M1, C_GARCH = "#2a78d6", "#eb6834"
C_ALT = "#1baf7a"
C_GREY = "#9a9a96"              # 收益序列：中性语境，不作分类色
INK2 = "#52514e"


# ---------------------------------------------------------------- σ 估计器
def sigma_uncond(r: pd.Series, win: int = WIN) -> pd.Series:
    """M1：滚动 win 日无条件 σ。

    注意 pandas rolling 在位置 t 含当日 → 必须再 shift(1)，使 σ_t 只用 [t−win, t−1]，
    与 GARCH/δ-normal/EWMA 一致；否则预测日 t 的收益会进入自己的 σ（前视偏差）。
    """
    return r.rolling(win).std().shift(1)


def sigma_ewma(r: pd.Series, lam: float = LAM, seed_win: int = WIN) -> pd.Series:
    """EWMA 条件 σ，σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}，以首个窗的样本方差为种子。

    seed_win=1 时改用「首个收益的平方」作种子（单点求样本方差为 NaN）——这样 σ 自第 0 天
    即有定义，9/16 的 Filtered-HS 才能用满 1023 天、与参数法同窗可比。
    实测两种播种到第 250 日差异仅 2.4e-08（相对 8.8e-08），种子只影响最初约 30 天。
    """
    v = np.full(len(r), np.nan)
    if seed_win < 2:
        prev = float(r.iloc[0]) ** 2
        v[0] = prev
        start = 1
    else:
        prev = float(r.iloc[:seed_win].var())
        v[seed_win - 1] = prev
        start = seed_win
    for i in range(start, len(r)):
        x = r.iloc[i - 1]
        if np.isfinite(x):
            prev = lam * prev + (1.0 - lam) * x * x
        v[i] = prev
    return pd.Series(np.sqrt(v), index=r.index)


def var_cols(sig: pd.Series, tag: str) -> dict[str, pd.Series]:
    """σ 序列 → {tag_var_95, tag_var_99}（μ=0，spec §1）。"""
    return {f"{tag}_var_{c}": Z[c] * sig for c in Z}


# ---------------------------------------------------------------- 历史模拟
def hs_var(r: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2 经典历史模拟：经验分位 VaR_c(t) = −quantile(r[t−win : t], 1−c)。

    分位**显式**指定线性插值（pandas 默认值虽同为 linear，但版本间不保证，口径应钉在代码里）。
    插值位置 = q·(n−1)（0 基）：n=250、c=99% → 2.49，即落在窗内**第 3、4 小**观测之间
    （权重 0.51/0.49）；c=95% → 12.45，第 13、14 小。故 99% VaR 的尾部实际只由约 2 个观测决定，
    9/17 报告检验力时必须写明这一点。
    """
    return -r.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)


def hs_var_filtered(r: pd.Series, sig: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2f Filtered-HS：窗内收益先除以各自 σ 标准化，再乘当日 σ_t。

    z_i = r_i/σ_i（σ 只含 ≤i−1 信息）→ 窗内 z 的经验分位 × σ_t（σ_t 只含 ≤t−1 信息）。
    目的是让分位在「波动标度」上估计，改善经典 HS 在高波动期的响应滞后。
    """
    z = r / sig
    return -(z.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)) * sig


# ---------------------------------------------------------------- 回测判定
def count_violations(ret: pd.Series, var: pd.Series) -> tuple[int, int, float]:
    """违规判定（spec §7.1）：第 t 日实际收益 r_t < −VaR_t 记一次违规。

    返回 (违规数, 有效日数, 失败率%)。两侧任一为 NaN 的日子直接剔除（次口径汇率尾 5 日即此种），
    不填补、不替换为其它口径——否则失败率的分母会被静默改写。
    """
    d = pd.concat([ret.rename("r"), var.rename("v")], axis=1).dropna()
    n = int(len(d))
    x = int((d["r"] < -d["v"]).sum())
    return x, n, (x / n * 100.0 if n else float("nan"))
