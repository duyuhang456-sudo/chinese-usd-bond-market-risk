"""阶段二 VaR 公共机制：常量 / 配色 / σ 估计器 / 历史模拟分位 / 违规判定。

被 9/15 参数法、9/16 历史模拟、9/17 回测三处共用。抽成公共模块是为了让违规判定只有一处
定义：两条路径各写一份，就会出现「同一份 VaR 序列数出不同违规数」的口径分叉，届时结论
无法归因。

所有 σ 与分位序列都遵守同一条时序纪律：位置 t 只能使用 ≤ t−1 的信息。pandas 的
`rolling(...)` 在位置 t 含当日，故一律 `.shift(1)`——少这一步，预测日 t 的收益会进入
自己的 σ，构成前视偏差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- 常量 ----
WIN = 250                       # 估计窗（交易日），与回测协议 §7.1 一致
Z = {"95": 1.6449, "99": 2.3263}
LAM = 0.94                      # EWMA 衰减因子（RiskMetrics）

# 分类色：参考调色板已文档化通过 all-pairs 校验的前三槽（固定顺序，不轮转）。
# 按槽位命名而非按模型命名，实体到槽位的映射在各脚本内固定：
#   槽1 蓝   = 基础/无条件口径（9/15 M1；9/16 对照线）
#   槽2 橙   = 条件波动口径（9/15 GARCH）
#   槽3 水绿 = 第三序列（9/15 EWMA；9/16 经典 HS / Filtered-HS）
C_M1, C_GARCH = "#2a78d6", "#eb6834"
C_ALT = "#1baf7a"
C_GREY = "#9a9a96"              # 收益序列：中性语境，不作分类色
INK2 = "#52514e"


# ---- σ 估计器 ----
def sigma_uncond(r: pd.Series, win: int = WIN) -> pd.Series:
    """M1：滚动 win 日无条件 σ。

    参数：
        r: 日收益序列（%）。
        win: 滚动窗长（交易日），默认 250，与回测协议 §7.1 同窗。

    返回：
        与 r 同索引的 σ 序列，前 win 个位置为 NaN。

    备注：
        pandas rolling 在位置 t 含当日，故末尾再 shift(1)，使 σ_t 只用 [t−win, t−1]，
        与 GARCH/δ-normal/EWMA 一致；少了这一步，预测日 t 的收益会进入自己的 σ（前视偏差）。
    """
    return r.rolling(win).std().shift(1)


def sigma_ewma(r: pd.Series, lam: float = LAM, seed_win: int = WIN) -> pd.Series:
    """EWMA 条件 σ，σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}，以首个窗的样本方差为种子。

    参数：
        r: 日收益序列（%）。
        lam: 衰减因子 λ，默认 0.94（RiskMetrics）。
        seed_win: 播种窗长；≥2 时以 r 前 seed_win 个观测的样本方差为种子（σ 自第 seed_win−1 个位置起有定义），
            <2 时以首个收益的平方为种子（σ 自第 0 天即有定义）。

    返回：
        与 r 同索引的 σ 序列；种子窗之前的位置为 NaN。

    备注：
        seed_win=1 是为让 σ 自第 0 天即有定义（单点求样本方差为 NaN），9/16 的 Filtered-HS 才能用满
        1023 天、与参数法同窗可比。实测两种播种到第 250 日差异仅 2.4e-08（相对 8.8e-08），
        种子只影响最初约 30 天。
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
    """σ 序列 → {tag_var_95, tag_var_99}（μ=0，spec §1）。

    参数：
        sig: σ 序列（%）。
        tag: 列名前缀，如 "m1"、"rmb_garch"。

    返回：
        dict，键为 f"{tag}_var_95" 与 f"{tag}_var_99"，值为 z·σ 的 VaR 序列（存正值）。
    """
    return {f"{tag}_var_{c}": Z[c] * sig for c in Z}


# ---- 历史模拟 ----
def hs_var(r: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2 经典历史模拟：经验分位 VaR_c(t) = −quantile(r[t−win : t], 1−c)。

    参数：
        r: 日收益序列（%）。
        win: 滚动窗长（交易日），默认 250。
        c: 置信度字符串，"95" 或 "99"。

    返回：
        与 r 同索引的 VaR 序列（存正值），前 win 个位置为 NaN。

    备注：
        分位显式指定线性插值（pandas 默认值虽同为 linear，但版本间不保证，口径应钉在代码里）。
        插值位置 = q·(n−1)（0 基）：n=250、c=99% → 2.49，即落在窗内第 3、4 小观测之间（权重 0.51/0.49）；
        c=95% → 12.45，第 13、14 小。故 99% VaR 的尾部实际只由约 2 个观测决定，
        9/17 报告检验力时必须写明这一点。
    """
    return -r.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)


def hs_var_filtered(r: pd.Series, sig: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2f Filtered-HS：窗内收益先除以各自 σ 标准化，再乘当日 σ_t。

    参数：
        r: 日收益序列（%）。
        sig: 条件 σ 序列（%），须只含 ≤ i−1 的信息。
        win: 滚动窗长（交易日），默认 250。
        c: 置信度字符串，"95" 或 "99"。

    返回：
        与 r 同索引的 VaR 序列（存正值）；r 与 sig 任一为 NaN 的位置、以及窗未满的位置为 NaN。

    备注：
        先把收益除以同期条件 σ 得到标准化残差 z_i（σ 只含 ≤ i−1 的信息），取窗内 z 的经验分位，
        再乘回 t 时刻的条件 σ_t（只含 ≤ t−1 的信息）。分位在「波动标度」上估计，
        以改善经典 HS 在高波动期的响应滞后。
    """
    z = r / sig
    return -(z.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)) * sig


# ---- 回测判定 ----
def hit_series(ret: pd.Series, var: pd.Series) -> pd.Series:
    """逐日违规指示（spec §7.1）：`r_t < −VaR_t` → True。

    参数：
        ret: 日收益序列（%）。
        var: VaR 序列（%），存正值。

    返回：
        布尔 Series，名为 "hit"；任一侧为 NaN 的日子剔除（不填补），索引只保留两侧均非 NaN 的日期。

    备注：
        与 count_violations 共用同一条 r < −v 判定——两条路径若各写一份，极易出现
        「同一份 VaR 序列数出不同违规数」的口径分叉（本模块存在的理由）。
    """
    d = pd.concat([ret.rename("r"), var.rename("v")], axis=1).dropna()
    return (d["r"] < -d["v"]).rename("hit")


def count_violations(ret: pd.Series, var: pd.Series) -> tuple[int, int, float]:
    """违规判定（spec §7.1）：第 t 日实际收益 r_t < −VaR_t 记一次违规。

    参数：
        ret: 日收益序列（%）。
        var: VaR 序列（%），存正值。

    返回：
        (违规数 x, 有效日数 n, 失败率%) 三元组；n=0 时失败率返回 NaN。

    备注：
        两侧任一为 NaN 的日子直接剔除（次口径汇率尾 5 日即此种），不填补、不替换为其它口径——
        否则失败率的分母会被静默改写。
    """
    h = hit_series(ret, var)
    n = int(len(h))
    x = int(h.sum())
    return x, n, (x / n * 100.0 if n else float("nan"))


# ---- 覆盖检验（9/17） ----
def _kln(k: int, q: float) -> float:
    """k·ln q，约定 k=0 时为 0 —— 即 0·ln0 = 0 的限制约定。

    参数：
        k: 计数（int）。
        q: 概率。

    返回：
        float；k=0 时返回 0.0，否则为 k·ln(q)（q=0 且 k>0 时为 −inf）。

    备注：
        Kupiec 公式在 x=0（分子项 p^x）与 x=N（分母项 (1−x/N)^{N−x}）两处都靠这条约定收敛；
        漏掉任一处都会得到 nan（实测「全 1」情形即因此报 nan）。
    """
    return 0.0 if k == 0 else k * float(np.log(q))

def _kx(k: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """`_kln` 的向量化版：逐元素算 k·ln(prob)。

    参数：
        k: 计数数组（np.ndarray）。
        prob: 概率数组，与 k 同形。

    返回：
        与 k 同形的数组；k≤0 或 prob≤0 的位置取 0。内部的除零/无效值告警被屏蔽。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((k > 0) & (prob > 0), k * np.log(np.where(prob > 0, prob, 1.0)), 0.0)


def kupiec_lr(x: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec 无条件覆盖 LR_uc ~ χ²(1)（spec §7.2 第 2 条）。

    参数：
        x: 违规次数。
        n: 有效日数。
        p: 名义违规率（1−c），如 0.05 / 0.01。

    返回：
        (LR_uc, p 值) 二元组，p 值取自 χ²(1) 的生存函数；n≤0 或 x 越界（x<0、x>n）时返回 (NaN, NaN)。

    备注：
        LR_uc = −2·ln[ (1−p)^{N−x}·p^x / (1−x/N)^{N−x}·(x/N)^x ]，即「名义 p 的伯努利似然」对
        「以 x/N 为率的伯努利似然」的似然比。双边检验：覆盖不足与过度覆盖都会拒绝。
        x=0 与 x=n 是合法输入（LR_uc 有限，见 _kln 的 0·ln0 约定）。χ² 是渐近近似，
        本样本下须与 lr_null 给出的有限样本零分布并列报告。
    """
    if n <= 0 or x < 0 or x > n:
        return float("nan"), float("nan")
    from scipy import stats  # 局部导入：只有检验类函数用 scipy，模块其余部分不依赖它
    lr = -2.0 * ((n - x) * np.log(1.0 - p) + _kln(x, p)
                 - (_kln(n - x, 1.0 - x / n) + _kln(x, x / n)))
    lr = max(float(lr), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_lr(hits) -> tuple[float, float, int, int, int, int]:
    """Christoffersen 独立性 LR_ind ~ χ²(1)（spec §7.2 第 3 条，必做）。

    参数：
        hits: 违规指示序列，元素可转 bool。

    返回：
        (LR_ind, p 值, n00, n01, n10, n11) 六元组；长度 < 2 时返回 (NaN, NaN, 0, 0, 0, 0)。
        n00/n01 为前一日 0 时「次日 0 / 次日 1」的次数，n10/n11 同理。

    备注：
        转移计数取 hits[:-1] → hits[1:]（严格按序列相邻，不跳过任何日）。
        π̂01 = n01/(n00+n01)（0 后接 1 的条件概率）、π̂11 = n11/(n10+n11)、
        π̂ = (n01+n11)/(n00+n01+n10+n11)（无条件率），
        LR_ind = −2·[ ln L_无条件 − ln L_一阶马尔可夫 ]。
        分母为 0 的项其似然贡献恒为 1（0^0=1）故取 0 不影响结果；返回前 max(lr, 0) 防浮点负零。
        违规聚集时 π̂11 > π̂01、LR_ind 增大——聚集说明 VaR 未跟上波动聚集（条件覆盖失效）。
    """
    h = np.asarray([bool(v) for v in hits], dtype=bool)
    if len(h) < 2:
        return float("nan"), float("nan"), 0, 0, 0, 0
    a, b = h[:-1], h[1:]
    return christoffersen_from_counts(int(np.sum(~a & ~b)), int(np.sum(~a & b)),
                                      int(np.sum(a & ~b)), int(np.sum(a & b)))


def christoffersen_from_counts(n00: int, n01: int, n10: int, n11: int
                               ) -> tuple[float, float, int, int, int, int]:
    """由转移计数直接算 LR_ind —— 供「剔除存疑日后丢弃跨日转移对」的情形使用。

    参数：
        n00, n01, n10, n11: 转移计数，相邻关系须按剔除前的原始序列判定。

    返回：
        (LR_ind, p 值, n00, n01, n10, n11) 六元组；四个计数全为 0 时返回 (NaN, NaN, 0, 0, 0, 0)。

    异常：
        raw < −1e-9 时抛 AssertionError（似然比符号写反）。

    备注：
        与 christoffersen_lr 的差别只在入口：那条路径上相邻关系必须按剔除前的原始序列判定
        （见 var_backtest.transition_counts），届时候选序列已不是一段连续日历，无法再交给
        christoffersen_lr 从序列重数一遍。本函数先断言 raw 不为实质负值，再把浮点负零钳到
        0——若符号写反成 2·(ll_unc − ll_ind)，断言会直接中断，而不是被钳位抹成 0、
        伪装成「所有模型 LR_ind 都恰好等于 0」。
    """
    from scipy import stats
    if n00 + n01 + n10 + n11 == 0:
        return float("nan"), float("nan"), n00, n01, n10, n11
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind = (_kln(n00, 1.0 - pi01) + _kln(n01, pi01)
              + _kln(n10, 1.0 - pi11) + _kln(n11, pi11))
    ll_unc = _kln(n00 + n10, 1.0 - pi) + _kln(n01 + n11, pi)
    raw = -2.0 * (ll_unc - ll_ind)
    # 一阶马尔可夫似然恒 ≥ 无条件似然，raw 理论非负。先按符号断言，再把浮点负零钳到 0：
    # 若符号写反成 2·(ll_unc − ll_ind)，raw 是实质负值，断言会直接中断，而不是被钳位抹成
    # 0、伪装成「所有模型 LR_ind 都恰好等于 0」。
    # 向量化孪生 _lr_vec 只有钳位、没有这条断言，护栏只存在于本标量路径。
    assert raw > -1e-9, f"LR_ind 出现负值 {raw}——似然比符号写反了"
    lr = max(raw, 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1)), n00, n01, n10, n11


def accept_region(n: int, p: float, alpha: float = 0.05) -> tuple[int, int]:
    """Kupiec 精确接受区间：LR_uc(x) ≤ χ²(1) 的 1−α 分位数的 x 集合边界（闭区间）。

    参数：
        n: 有效日数。
        p: 名义违规率。
        alpha: 检验水平，默认 0.05（χ²(1) 的 95% 分位 = 3.8415）。

    返回：
        (x_lo, x_hi) 二元组，可接受的违规次数闭区间；区间为空时返回 (0, 0)。

    备注：
        用于 spec §7.2 L265 要求的「点估计 + 区间」表述——单看失败率或单看 p 值都不足以
        区分「差一点」与「差很远」，区间宽度本身还直接暴露检验力。
    """
    from scipy import stats
    crit = stats.chi2.ppf(1.0 - alpha, 1)
    xs = [x for x in range(n + 1) if kupiec_lr(x, n, p)[0] <= crit]
    return (min(xs), max(xs)) if xs else (0, 0)


def _lr_vec(hits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化 LR_uc 与 LR_ind，供蒙特卡洛零分布批量求值。

    参数：
        hits: 布尔矩阵，形状 (B, n)，B 为模拟样本数、n 为日数。

    返回：
        (x, lr_ind) 二元组，均为 (B,) 数组。x 为各样本的违规次数（LR_uc 中与 (n, p) 有关的项
        由调用处补入，见 lr_null）；lr_ind 用 _kx 向量化计算，负值钳为 0。

    备注：
        与逐样本标量实现（kupiec_lr / christoffersen_lr）同式；实测 1023 点随机序列上两者
        一致到 1e-10 以内（var_backtest 的自检段在跑这条比对）。
    """
    n = hits.shape[1]
    x = hits.sum(1).astype(float)   # 回传 x：LR_uc 里含 (n, p) 的项由调用处补入，见 lr_null
    # ---- LR_ind
    a, b = hits[:, :-1], hits[:, 1:]
    n00 = (~a & ~b).sum(1).astype(float); n01 = (~a & b).sum(1).astype(float)
    n10 = (a & ~b).sum(1).astype(float);  n11 = (a & b).sum(1).astype(float)
    d01 = n00 + n01; d11 = n10 + n11; tot = d01 + d11
    with np.errstate(divide="ignore", invalid="ignore"):
        pi01 = np.where(d01 > 0, n01 / np.where(d01 > 0, d01, 1.0), 1.0)
        pi11 = np.where(d11 > 0, n11 / np.where(d11 > 0, d11, 1.0), 1.0)
        pi = np.where(tot > 0, (n01 + n11) / np.where(tot > 0, tot, 1.0), 1.0)
        # 合并同类项：π 取 1 时分母为 0 的对数项被 k=0 约定消掉，故安全
        ll_ind = (_kx(n00, 1 - pi01) + _kx(n01, pi01)
                  + _kx(n10, 1 - pi11) + _kx(n11, pi11))
        ll_unc = _kx(n00 + n10, 1 - pi) + _kx(n01 + n11, pi)
    lr_ind = np.maximum(-2.0 * (ll_unc - ll_ind), 0.0)
    return x, lr_ind


_LR_NULL_CACHE: dict = {}


def lr_null(n: int, p: float, B: int = 20000, seed: int = 20260917,
            base_seed: int = 20260917) -> dict:
    """iid Bernoulli(p) 下 LR_uc 与 LR_ind 的有限样本零分布（spec §7.2 L265 的量化）。

    参数：
        n: 日数（样本长度）。
        p: 名义违规率（1−c），如 0.05 / 0.01。
        B: 蒙特卡洛重复数，默认 20000。
        seed: 随机数种子，默认 20260917。
        base_seed: 只进缓存键，不参与随机数生成（生成用 seed）。

    返回：
        dict，五键：lr_uc 与 lr_ind 为 (B,) 零分布样本；crit 为 χ²(1) 的 95% 分位（3.8415），
        size_uc 与 size_ind 为名义 5% 下的实际拒绝率（零分布中 LR ≥ crit 的比例）。
        同一组 (n, p, B, seed, base_seed) 的结果缓存在 _LR_NULL_CACHE。

    备注：
        χ²(1) 是渐近近似，本样本（99% 下仅约 10 次期望违规）并不满足——实测名义 5% 的实际
        拒绝率在不同 (N, p) 下从 1.3% 到 8.3% 不等，且方向不一致。故用本函数的零分布算 p 值
        （p_mc），与 χ² p 值并列报告，并据同一零分布披露 size 表。
    """
    key = (n, p, B, seed, base_seed)
    if key in _LR_NULL_CACHE:
        return _LR_NULL_CACHE[key]
    rng = np.random.default_rng(seed)
    from scipy import stats
    lr_uc_a, lr_ind_a = [], []
    blk = max(1, min(2000, B))
    for s in range(0, B, blk):
        m = min(blk, B - s)
        hits = rng.random((m, n)) < p
        x, lr_ind = _lr_vec(hits)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr_uc = -2.0 * ((n - x) * np.log(1.0 - p) + np.where(x > 0, x * np.log(p), 0.0)
                            - (n - x) * np.log(np.where(x < n, 1.0 - x / n, 1.0))
                            - np.where(x > 0, x * np.log(np.where(x > 0, x / n, 1.0)), 0.0))
        lr_uc_a.append(np.maximum(lr_uc, 0.0)); lr_ind_a.append(lr_ind)
    lr_uc = np.concatenate(lr_uc_a); lr_ind = np.concatenate(lr_ind_a)
    crit = float(stats.chi2.ppf(0.95, 1))
    out = {"lr_uc": lr_uc, "lr_ind": lr_ind, "crit": crit,
           "size_uc": float(np.mean(lr_uc >= crit)), "size_ind": float(np.mean(lr_ind >= crit))}
    _LR_NULL_CACHE[key] = out
    return out


def p_mc(null: np.ndarray, observed: float) -> float:
    """蒙特卡洛 p 值 = P(LR_sim ≥ LR_obs)。

    参数：
        null: 零分布样本数组（如 lr_null 的 lr_uc / lr_ind）。
        observed: 观测到的 LR 统计量。

    返回：
        float；observed 非有限时返回 NaN。

    备注：
        未取 +1 修正（B=20000 下影响 < 5e-5）。
    """
    if not np.isfinite(observed):
        return float("nan")
    return float(np.mean(null >= observed))


def lr_ind_null_conditional(n: int, x: int, p: float = float("nan"), B: int = 20000,
                            seed: int = 20260917) -> np.ndarray:
    """条件零分布：以观测到的违规数 x 为率模拟 iid Bernoulli(x/n)，返回 LR_ind 样本。

    参数：
        n: 日数。
        x: 观测到的违规次数。
        p: 未参与计算，仅保留以与 lr_null 的签名对齐。
        B: 蒙特卡洛重复数，默认 20000。
        seed: 随机数种子，实际生成用 seed+1。

    返回：
        (B,) 数组，条件零分布下的 LR_ind 样本；结果按 ("cond", n, x, B, seed) 缓存。

    备注：
        LR_ind 问的是「给定违规次数，违规是否扎堆」，故条件于实际发生的 x 是该问法在有限样本下
        的正确定标参照。它不单调放大 p 值：x 低于期望时条件零分布比无条件更集中，观测到的聚集
        反而更意外，p 更小（实测次口径 HS500 95%：χ² p=0.0342 → p_cond=0.0119）。
    """
    key = ("cond", n, x, B, seed)
    if key in _LR_NULL_CACHE:
        return _LR_NULL_CACHE[key]
    rate = x / n if n else 0.0
    rng = np.random.default_rng(seed + 1)
    out = []
    blk = max(1, min(2000, B))
    for s in range(0, B, blk):
        m = min(blk, B - s)
        hits = rng.random((m, n)) < rate
        out.append(_lr_vec(hits)[1])
    arr = np.concatenate(out)
    _LR_NULL_CACHE[key] = arr
    return arr


def lr_cc(lr_uc: float, lr_ind: float) -> tuple[float, float]:
    """条件覆盖 LR_cc = LR_uc + LR_ind ~ χ²(2)（spec §7.2 L262）。

    参数：
        lr_uc: Kupiec 无条件覆盖统计量。
        lr_ind: Christoffersen 独立性统计量。

    返回：
        (LR_cc, p 值) 二元组，p 值取自 χ²(2) 的生存函数；非有限的输入按 0 计入。
    """
    from scipy import stats
    s = (lr_uc if np.isfinite(lr_uc) else 0.0) + (lr_ind if np.isfinite(lr_ind) else 0.0)
    return float(s), float(stats.chi2.sf(s, 2))
