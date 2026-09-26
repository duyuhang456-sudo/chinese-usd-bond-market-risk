"""阶段二 VaR 的公共零件：常量、配色、σ 估计、历史模拟分位、违规判定。

9/15 参数法、9/16 历史模拟、9/17 回测都从这里取东西。抽成公共模块的理由只有一个：
违规判定只能有一处定义。两边各写一份，就会出现同一份 VaR 序列数出不同违规数的情况，
到时候分不清是模型的问题还是代码的问题。

这里所有 σ 和分位序列都守同一条规矩：位置 t 只能用 t−1 及更早的信息。pandas 的
`rolling(...)` 默认把当天算进去，所以每处末尾都接一下 `.shift(1)`。少这一下，
算第 t 天的时候第 t 天自己的收益已经进了 σ，回测数字就不再是样本外成绩。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- 常量 ----
WIN = 250                       # 估计窗，250 个交易日，见 spec §7.1
Z = {"95": 1.6449, "99": 2.3263}
LAM = 0.94                      # EWMA 衰减因子，RiskMetrics 的常规取值

# 三个分类色，取自调色板里已过 all-pairs 校验的前三槽，顺序固定不轮转。
# 变量名按槽位取，不按模型取——具体哪个模型用哪个色，各脚本里各自固定：
#   蓝   = 基础口径（9/15 的 M1；9/16 的对照线）
#   橙   = 条件波动口径（9/15 的 GARCH）
#   水绿 = 第三条序列（9/15 的 EWMA；9/16 的经典 HS 与 Filtered-HS）
C_M1, C_GARCH = "#2a78d6", "#eb6834"
C_ALT = "#1baf7a"
C_GREY = "#9a9a96"              # 画收益序列用的中性色，不进分类色
INK2 = "#52514e"


# ---- σ 估计器 ----
def sigma_uncond(r: pd.Series, win: int = WIN) -> pd.Series:
    """M1：过去 win 天收益的标准差，一天一天往前滚。

    r 是日收益（%），win 是天数，默认 250，和 spec §7.1 的回测窗一致。
    返回一串和 r 等长的 σ，前 win 个位置是空的——天数没攒够就没有值，别当 0 用。

    末尾的 shift(1) 别删。rolling 默认把当天也算进去，不往后挪一天的话，
    算第 t 天的 σ 时第 t 天自己的收益已经在里面了，回测数字就不再是样本外成绩。
    挪完这一下，σ_t 只用到 [t−win, t−1]，和 GARCH、δ-normal、EWMA 三处对齐。
    """
    return r.rolling(win).std().shift(1)


def sigma_ewma(r: pd.Series, lam: float = LAM, seed_win: int = WIN) -> pd.Series:
    """EWMA 条件 σ：σ²_t = λσ²_{t−1} + (1−λ)·r²_{t−1}，开头一段的样本方差当种子。

    r 是日收益（%），lam 是衰减因子 λ，默认 0.94（RiskMetrics 的常规取值）。
    seed_win 是播种窗长：给 250 就用前 250 天的样本方差起步，σ 从第 250 个位置才有值；
    给 1 就用第一个收益的平方起步，σ 从第 0 天就有值。返回和 r 等长的 σ，
    种子窗之前的位置是空的。

    播种给 1 是特意为了让 σ 从第 0 天就有定义——只拿一个点求样本方差是 NaN。
    这样 9/16 的 Filtered-HS 才能用满 1023 天，和参数法同窗可比。
    两种播种到第 250 天只差 2.4e-08，实际只影响最前面 30 天左右。
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
    """把一串 σ 变成两串 VaR：z 乘 σ，95% 和 99% 各一串。

    sig 是 σ 序列（%），tag 是列名前缀，比如 "m1"、"rmb_garch"。
    返回 dict，键是 f"{tag}_var_95" 和 f"{tag}_var_99"，值是 VaR 序列，存正值。
    μ 固定取 0，也就是不去均值，见 spec §1。
    """
    return {f"{tag}_var_{c}": Z[c] * sig for c in Z}


# ---- 历史模拟 ----
def hs_var(r: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2 经典历史模拟：直接取过去 win 天收益的经验分位，不看分布长什么样。

    r 是日收益（%），win 是天数（默认 250），c 是置信度（"95" 或 "99"）。
    返回和 r 等长的 VaR 序列，存正值，前 win 个位置是空的。

    分位这里写死了线性插值。pandas 的默认值现在也是 linear，但版本之间不保证，
    口径得钉在代码里，不能靠默认值。

    99% 这一档要留意尾部有多薄：250 天里插值位置落在第 3、4 小的观测之间，
    等于说 VaR 的尾部实际由大约 2 个观测决定。9/17 讲检验力时必须写明这一点。
    """
    return -r.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)


def hs_var_filtered(r: pd.Series, sig: pd.Series, win: int = WIN, c: str = "95") -> pd.Series:
    """M2f Filtered-HS：收益先按各自的 σ 折算成标准化残差，取完分位再乘回来。

    r 是日收益（%），sig 是条件 σ 序列（%），win 是天数，c 是置信度。
    sig 只能含 i−1 及更早的信息，传进来之前就得 shift 好，这个函数不再补。
    返回和 r 等长的 VaR 序列，存正值；r 或 sig 缺值的位置、以及窗没满的位置都是空的。

    做法是 r 除以同期 σ 得 z，取窗内 z 的经验分位，再乘回 t 时刻的 σ_t。
    分位等于估在波动标度上，用来缓解经典 HS 在高波动期反应慢半拍的问题。
    """
    z = r / sig
    return -(z.rolling(win).quantile(1.0 - float(c) / 100.0, interpolation="linear").shift(1)) * sig


# ---- 回测判定 ----
def hit_series(ret: pd.Series, var: pd.Series) -> pd.Series:
    """逐日违规标记：r 跌破 −VaR 记 True，否则 False（spec §7.1）。

    ret 是日收益（%），var 是 VaR 序列（%），存正值。
    返回一个名为 "hit" 的布尔序列。两边只要缺一边，那天整条去掉，不填补。

    这个函数和 count_violations 必须共用同一条判定。各写一份的话，同一份 VaR 序列
    会被数出两个不同的违规数——公共模块存在的理由就是这个。
    """
    d = pd.concat([ret.rename("r"), var.rename("v")], axis=1).dropna()
    return (d["r"] < -d["v"]).rename("hit")


def count_violations(ret: pd.Series, var: pd.Series) -> tuple[int, int, float]:
    """数违规次数：r 跌破 −VaR 记一次（spec §7.1）。

    ret 是日收益（%），var 是 VaR 序列（%），存正值。
    返回 (违规数, 有效天数, 失败率%)，有效天数为 0 时失败率给 NaN，不给 0。

    两边缺值的那天直接去掉——次口径汇率尾部少 5 天就是这种情况。
    不要为了凑齐天数把缺的填上、或换成别的口径，那等于把失败率的分母悄悄改掉。
    """
    h = hit_series(ret, var)
    n = int(len(h))
    x = int(h.sum())
    return x, n, (x / n * 100.0 if n else float("nan"))


# ---- 覆盖检验（9/17） ----
def _kln(k: int, q: float) -> float:
    """k 乘 ln q，另外规定 k 等于 0 时直接给 0。

    这是 0·ln0 = 0 的约定。Kupiec 公式有两处会撞上它：x=0 时的 p^x，
    以及 x=N 时的 (1−x/N)^{N−x}。两处漏掉任何一处都会算出 nan，
    实测「全 1」的情形就是这么报 nan 的。
    """
    return 0.0 if k == 0 else k * float(np.log(q))

def _kx(k: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """_kln 的向量版：一次算一整排 k·ln(prob)，给蒙特卡洛用。

    k 和 prob 同形，返回同样形状的数组。k 或 prob 不为正的位置一律取 0，
    和 _kln 的约定一致。里面的除零和无效值告警是主动屏蔽的，不是漏了。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((k > 0) & (prob > 0), k * np.log(np.where(prob > 0, prob, 1.0)), 0.0)


def kupiec_lr(x: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec 无条件覆盖检验：实际违规数离名义违规率有多远（spec §7.2 第 2 条）。

    x 是违规次数，n 是有效天数，p 是名义违规率（也就是 1−c，比如 0.05、0.01）。
    返回 (LR_uc, p 值)，p 值取自 χ²(1)。n≤0 或 x 不在 [0, n] 里时返回 (NaN, NaN)。

    式子是把「名义率 p 的伯努利似然」和「实际率 x/n 的伯努利似然」比一下再取 −2·ln。
    两边都查：违规太少（过度覆盖）和违规太多（覆盖不足）都会拒绝。
    x=0 和 x=n 都是合法输入，靠 _kln 的约定收敛，不算异常。

    χ² 是渐近近似，本样本并不满足。所以这个 p 值要和 lr_null 给的有限样本零分布
    并列报告，不能只报这一个。
    """
    if n <= 0 or x < 0 or x > n:
        return float("nan"), float("nan")
    from scipy import stats  # 局部导入：全模块只有检验类函数用 scipy，其余部分不依赖它
    lr = -2.0 * ((n - x) * np.log(1.0 - p) + _kln(x, p)
                 - (_kln(n - x, 1.0 - x / n) + _kln(x, x / n)))
    lr = max(float(lr), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_lr(hits) -> tuple[float, float, int, int, int, int]:
    """Christoffersen 独立性检验：违规是不是扎堆出现（spec §7.2 第 3 条，必做）。

    hits 是违规指示序列，能转成 bool 就行。
    返回 (LR_ind, p 值, n00, n01, n10, n11)；序列短于 2 天时返回 (NaN, NaN, 0, 0, 0, 0)。
    四个计数是转移次数：n01 是「前一天没违规、今天违规」的次数，n10 反过来，
    n00 和 n11 是连着两天同状态的次数。

    相邻关系严格按序列本身取（hits[:-1] 配 hits[1:]），中间不跳天。
    拿条件概率 π̂01、π̂11 和无条件率 π̂ 各算一个似然，比一下再取 −2·ln。
    分母为 0 的项按 0^0=1 处理，取 0 不影响结果；返回前的 max(lr, 0) 是防浮点负零。

    违规扎堆时 π̂11 大于 π̂01，LR_ind 跟着变大。扎堆说明 VaR 没跟上波动聚集，
    也就是条件覆盖失效——这正是这个检验要问的事。
    """
    h = np.asarray([bool(v) for v in hits], dtype=bool)
    if len(h) < 2:
        return float("nan"), float("nan"), 0, 0, 0, 0
    a, b = h[:-1], h[1:]
    return christoffersen_from_counts(int(np.sum(~a & ~b)), int(np.sum(~a & b)),
                                      int(np.sum(a & ~b)), int(np.sum(a & b)))


def christoffersen_from_counts(n00: int, n01: int, n10: int, n11: int
                               ) -> tuple[float, float, int, int, int, int]:
    """转移计数已经在手时，直接算 LR_ind——给剔除存疑日之后用的。

    四个计数是转移次数，而且相邻关系必须按剔除前的原始序列判定。
    返回 (LR_ind, p 值, n00, n01, n10, n11)；四个计数全是 0 时返回 (NaN, NaN, 0, 0, 0, 0)。
    raw 小于 −1e-9 会抛 AssertionError，意思是似然比的符号写反了。

    和 christoffersen_lr 只差在入口。那条路径上相邻关系必须按剔除前的原始序列定
    （见 var_backtest.transition_counts），走到这一步序列已经不是一段连续日历，
    没法再交给 christoffersen_lr 从头数一遍，只能把计数传进来。

    断言在前、钳位在后的次序不能换。符号要是写反成 2·(ll_unc − ll_ind)，断言会当场
    中断；换成只钳位，就会被抹成 0，看起来像「所有模型的 LR_ind 恰好都等于 0」。
    向量化的孪生 _lr_vec 只有钳位没有断言，这道护栏只在本标量路径上有。
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
    # 一阶马尔可夫似然恒 ≥ 无条件似然，raw 理论非负，所以负数一定是符号写反（理由见 docstring）
    assert raw > -1e-9, f"LR_ind 出现负值 {raw}——似然比符号写反了"
    lr = max(raw, 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1)), n00, n01, n10, n11


def accept_region(n: int, p: float, alpha: float = 0.05) -> tuple[int, int]:
    """Kupiec 的接受区间：违规次数落在哪一段里算通过。

    n 是有效天数，p 是名义违规率，alpha 是检验水平，默认 0.05，
    对应 χ²(1) 的 95% 分位 3.8415。
    返回 (下限, 上限)，两端都算通过；区间为空时返回 (0, 0)。

    做法就是逐个试 x，看哪些 x 的 LR_uc 不超过临界值，取其中最小和最大的。

    spec §7.2 要求「点估计 + 区间」的写法，靠的就是它。只看失败率或者只看 p 值，
    分不出「差一点」和「差很远」；区间本身有多宽，也直接说明了检验力还剩多少。
    """
    from scipy import stats
    crit = stats.chi2.ppf(1.0 - alpha, 1)
    xs = [x for x in range(n + 1) if kupiec_lr(x, n, p)[0] <= crit]
    return (min(xs), max(xs)) if xs else (0, 0)


def _lr_vec(hits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量版 LR_uc 与 LR_ind：一次算一整批模拟序列，给蒙特卡洛零分布用。

    hits 是布尔矩阵，形状 (B, n)，B 是模拟条数、n 是天数。
    返回 (x, lr_ind) 两个长度为 B 的数组：x 是每条序列的违规次数，
    lr_ind 是独立性统计量，负值一律钳到 0。

    LR_uc 里含 (n, p) 的项这里不算，由调用处补上，见 lr_null。
    式子与上面两个标量函数相同，实测在 1023 个点的随机序列上两版一致到 1e-10 以内，
    var_backtest 的自检段每次都会跑这条比对。
    """
    n = hits.shape[1]
    x = hits.sum(1).astype(float)   # x 单独回传：LR_uc 里含 (n, p) 的项由 lr_null 那边补
    # ---- LR_ind
    a, b = hits[:, :-1], hits[:, 1:]
    n00 = (~a & ~b).sum(1).astype(float); n01 = (~a & b).sum(1).astype(float)
    n10 = (a & ~b).sum(1).astype(float);  n11 = (a & b).sum(1).astype(float)
    d01 = n00 + n01; d11 = n10 + n11; tot = d01 + d11
    with np.errstate(divide="ignore", invalid="ignore"):
        pi01 = np.where(d01 > 0, n01 / np.where(d01 > 0, d01, 1.0), 1.0)
        pi11 = np.where(d11 > 0, n11 / np.where(d11 > 0, d11, 1.0), 1.0)
        pi = np.where(tot > 0, (n01 + n11) / np.where(tot > 0, tot, 1.0), 1.0)
        # π 取 1 时分母为 0 的那些对数项被 k=0 的约定消掉，所以这里不会有 nan
        ll_ind = (_kx(n00, 1 - pi01) + _kx(n01, pi01)
                  + _kx(n10, 1 - pi11) + _kx(n11, pi11))
        ll_unc = _kx(n00 + n10, 1 - pi) + _kx(n01 + n11, pi)
    lr_ind = np.maximum(-2.0 * (ll_unc - ll_ind), 0.0)
    return x, lr_ind


_LR_NULL_CACHE: dict = {}


def lr_null(n: int, p: float, B: int = 20000, seed: int = 20260917,
            base_seed: int = 20260917) -> dict:
    """模拟出 LR_uc 和 LR_ind 在「真值就是 p」时候的分布（spec §7.2 的量化）。

    n 是天数，p 是名义违规率，B 是模拟条数（默认 20000），seed 是随机种子。
    返回五键的 dict：lr_uc 和 lr_ind 是两条零分布样本，crit 是 χ²(1) 的 95% 分位
    （3.8415），size_uc 和 size_ind 是名义 5% 之下实际的拒绝率。
    同一组参数的结果会缓存，重复调用不会重算。

    base_seed 只进缓存的键，不参与生成随机数，别指望改它能换一批样本。

    为什么要模拟：χ²(1) 是渐近近似，本样本（99% 下只有约 10 次期望违规）根本不满足。
    实测名义 5% 的实际拒绝率在不同 (n, p) 下从 1.3% 到 8.3% 都有，方向还不一致。
    所以 p 值用这里的零分布算（见 p_mc），和 χ² 那个并列报，size 表也从同一批样本出。
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
    """蒙特卡洛 p 值：模拟里有多少条不小于实际算出来的那个数。

    null 是零分布样本（lr_null 给的 lr_uc 或 lr_ind），observed 是实际观测到的统计量。
    返回占比；observed 不是有限数时给 NaN。

    没做 +1 修正。B=20000 下影响不到 5e-5，不必管。
    """
    if not np.isfinite(observed):
        return float("nan")
    return float(np.mean(null >= observed))


def lr_ind_null_conditional(n: int, x: int, p: float = float("nan"), B: int = 20000,
                            seed: int = 20260917) -> np.ndarray:
    """条件零分布：按住实际发生的违规条数 x，重新模拟 LR_ind 的分布。

    n 是天数，x 是实际观测到的违规次数，B 是模拟条数，seed 是种子（生成时用的是 seed+1）。
    p 不参与计算，留着只是为了让签名和 lr_null 对得上。
    返回长度为 B 的数组，结果按 ("cond", n, x, B, seed) 缓存。

    为什么要条件化：LR_ind 问的是「违规条数已经定了，它们扎不扎堆」，所以把实际发生的
    x 按住，才是这个问法在有限样本下的正确参照。

    它不单调放大 p 值，方向要看情况：x 低于期望时，条件零分布比无条件更集中，
    观测到的扎堆反而更意外，p 更小——实测次口径 HS500 的 95% 就是 χ² 给 0.0342、
    条件化之后给 0.0119。
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
    """条件覆盖 LR_cc：把两个检验的统计量加起来，服从 χ²(2)（spec §7.2）。

    lr_uc 是 Kupiec 的统计量，lr_ind 是 Christoffersen 的统计量。
    返回 (LR_cc, p 值)，p 值取自 χ²(2)；输入不是有限数时按 0 计入。
    """
    from scipy import stats
    s = (lr_uc if np.isfinite(lr_uc) else 0.0) + (lr_ind if np.isfinite(lr_ind) else 0.0)
    return float(s), float(stats.chi2.sf(s, 2))
