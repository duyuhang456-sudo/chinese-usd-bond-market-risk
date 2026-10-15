"""产物加载器。

全部是只读的 read_csv —— 看板绝不向 results/ 或 figures/ 写任何东西，
否则 code/check_outputs.py 的产物对账会把多余文件报为不通过。

几个统一约定：
  * 一律用 utf-8-sig 读，否则首列列名会带上 BOM（\\ufeff），后续按列名取数会静默取空；
  * 数值列一律 pd.to_numeric(errors="coerce")，不 fillna —— 缺口要留成断口，不能补 0；
  * 带日期的表统一把 date 解析成 datetime 并排序。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import labels, paths

# ---------------------------------------------------------------- 基础读取


def _read(path) -> pd.DataFrame:
    return pd.read_csv(paths.require(path), encoding="utf-8-sig")


def _read_dated(path) -> pd.DataFrame:
    df = _read(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 因子表


@st.cache_data(show_spinner=False)
def factor_table(name: str) -> pd.DataFrame:
    """四个因子表之一：factor_table_nav_tr / factor_table_nav / factor_table / factor_table_3141HK。"""
    return _read_dated(paths.factor(name))


@st.cache_data(show_spinner=False)
def descriptive_stats_nav() -> pd.DataFrame:
    return _read(paths.factor("descriptive_stats_nav"))


@st.cache_data(show_spinner=False)
def factor_corr() -> pd.DataFrame:
    df = _read(paths.result("factor_corr"))
    df = df.rename(columns={df.columns[0]: "factor"})
    long = df.melt(id_vars="factor", var_name="factor_b", value_name="corr")
    return long


@st.cache_data(show_spinner=False)
def benchmark_comparison() -> pd.DataFrame:
    return _read(paths.factor("benchmark_comparison"))


@st.cache_data(show_spinner=False)
def staleness_remedy_comparison() -> pd.DataFrame:
    return _read(paths.factor("staleness_remedy_comparison"))


# ---------------------------------------------------------------- 模型与回测


@st.cache_data(show_spinner=False)
def var_historical() -> pd.DataFrame:
    return _read_dated(paths.result("var_historical"))


@st.cache_data(show_spinner=False)
def var_parametric() -> pd.DataFrame:
    return _read_dated(paths.result("var_parametric"))


# 模型代号 -> (来源表, 主口径列前缀)。次口径列一律加 rmb_ 前缀。
_MODEL_SRC = {
    "M1": ("parametric", "m1_var_"),
    "M1g": ("parametric", "garch_var_"),
    "M1e": ("parametric", "ewma_var_"),
    "M1d": ("parametric", "dnorm_var_"),
    "M1g-t": ("parametric", "garch_t_var_"),
    "HS250": ("historical", "hs250_var_"),
    "HS500": ("historical", "hs500_var_"),
    "HS750": ("historical", "hs750_var_"),
    "FHS-E": ("historical", "fhs_e_var_"),
    "FHS-G": ("historical", "fhs_g_var_"),
}


@st.cache_data(show_spinner=False)
def var_series_long(caliber: str, conf: str, models: tuple) -> pd.DataFrame:
    """把逐日 VaR 拼成一张长表：date / model / var_pct / ret_pct。

    caliber ∈ {主, 次}；conf ∈ {95, 99}；models 是模型代号元组（须可哈希才能进缓存）。
    缺口原样保留为 NaN，交给 Altair 画成断口。
    """
    hist = var_historical()
    para = var_parametric()
    ret_col = "ret_tr_pct" if caliber == "主" else "ret_rmb_pct"
    frames = []
    for model in models:
        if model not in _MODEL_SRC:
            continue
        src, stem = _MODEL_SRC[model]
        tbl = para if src == "parametric" else hist
        col = stem + conf
        if caliber == "次":
            col = "rmb_" + col
        if col not in tbl.columns:
            continue
        part = tbl[["date", col, ret_col]].rename(
            columns={col: "var_pct", ret_col: "ret_pct"}
        )
        part["model"] = model
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["date", "model", "var_pct", "ret_pct"])
    out = pd.concat(frames, ignore_index=True)
    out["var_pct"] = pd.to_numeric(out["var_pct"], errors="coerce")
    out["ret_pct"] = pd.to_numeric(out["ret_pct"], errors="coerce")
    # 名义置信度 → 判定线口径的反向符号：VaR 存正值，画 −VaR 由作图侧处理
    return out.sort_values(["model", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def backtest_results() -> pd.DataFrame:
    return _read(paths.result("backtest_results"))


# 逐日 GARCH 重估轨迹：两张表是「次口径边界解」这条结论的唯一锚点，
# 此前看板一次都没读过。
_GARCH_TRACE = {"主": "garch_refit_trace_main", "次": "garch_refit_trace_rmb"}


@st.cache_data(show_spinner=False)
def garch_refit(caliber: str) -> pd.DataFrame:
    """逐日重估轨迹：date / alpha / beta / persist，另加一列 boundary。

    boundary 标出 α ≤ GARCH_BOUNDARY_ALPHA（0.001）的日子——判据与文档 §3.1
    的「次口径 133 次」是同一把尺子。α 为 NaN 的重估失败日不算边界解。
    """
    df = _read_dated(paths.result(_GARCH_TRACE[caliber]))
    for c in ("alpha", "beta", "persist"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["boundary"] = df["alpha"] <= labels.GARCH_BOUNDARY_ALPHA
    return df


def garch_boundary_summary(caliber: str) -> dict:
    """边界解的天数、跨度与段内极值。

    页面与冒烟脚本共用同一套口径，免得两处各算一遍、算出两个数。
    """
    df = garch_refit(caliber)
    b = df[df["boundary"]]
    out = {
        "n_total": int(len(df)),
        "n_boundary": int(len(b)),
        "persist_ge1": int((df["persist"] >= 1).sum()),
        "beta_ge_0999": int((df["beta"] >= 0.999).sum()),
        "persist_max": float(df["persist"].max()),
    }
    if len(b):
        out["start"] = b["date"].min()
        out["end"] = b["date"].max()
        out["beta_mean"] = float(b["beta"].mean())
    else:
        out["start"] = out["end"] = out["beta_mean"] = None
    return out


@st.cache_data(show_spinner=False)
def backtest_exceptions() -> pd.DataFrame:
    df = _read(paths.result("backtest_exceptions"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def model_scorecard() -> pd.DataFrame:
    return _read(paths.result("model_scorecard"))


@st.cache_data(show_spinner=False)
def baseline_decision() -> pd.DataFrame:
    return _read(paths.result("baseline_decision"))


@st.cache_data(show_spinner=False)
def baseline_var_tr() -> pd.DataFrame:
    return _read(paths.result("baseline_var_tr"))


@st.cache_data(show_spinner=False)
def var_attribution() -> pd.DataFrame:
    """因子波动贡献份额，转成长表：caliber / factor / share_pct。"""
    df = _read(paths.result("var_attribution"))
    df = df.rename(columns={df.columns[0]: "factor"})
    long = df.melt(id_vars="factor", var_name="caliber_raw", value_name="share_pct")
    name_map = {"主口径（复权 USD）": "主", "人民币次口径": "次"}
    long["caliber"] = long["caliber_raw"].map(name_map).fillna(long["caliber_raw"])
    long["share_pct"] = pd.to_numeric(long["share_pct"], errors="coerce")
    return long.dropna(subset=["share_pct"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def factor_exposure() -> pd.DataFrame:
    return _read(paths.result("factor_exposure"))


@st.cache_data(show_spinner=False)
def regimes() -> pd.DataFrame:
    return _read_dated(paths.result("regimes"))


@st.cache_data(show_spinner=False)
def delta_transmission_events() -> pd.DataFrame:
    df = _read(paths.result("delta_transmission_events"))
    for c in ("event_date", "e_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def delta_transmission_summary() -> pd.DataFrame:
    return _read(paths.result("delta_transmission_summary"))


# ---------------------------------------------------------------- 压力测试


@st.cache_data(show_spinner=False)
def stress_impact() -> pd.DataFrame:
    return _read(paths.result("stress_impact"))


@st.cache_data(show_spinner=False)
def stress_scenarios() -> pd.DataFrame:
    return _read(paths.result("stress_scenarios"))


@st.cache_data(show_spinner=False)
def stress_buffer_sens() -> pd.DataFrame:
    df = _read(paths.result("stress_buffer_sens"))
    df["loss_net_pct"] = pd.to_numeric(df["loss_net_pct"], errors="coerce")
    df["es_1d_99_pct"] = pd.to_numeric(df["es_1d_99_pct"], errors="coerce")
    df["is_tail"] = df["tail_flag"].astype(str).str.strip() == labels.FLAG_YES
    return df


@st.cache_data(show_spinner=False)
def stress_factor_contrib() -> pd.DataFrame:
    df = _read(paths.result("stress_factor_contrib"))
    df["share_pct"] = pd.to_numeric(df["share_pct"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def stress_window_sens() -> pd.DataFrame:
    return _read(paths.result("stress_window_sens"))


@st.cache_data(show_spinner=False)
def stress_anchor_sens() -> pd.DataFrame:
    return _read(paths.result("stress_anchor_sens"))


@st.cache_data(show_spinner=False)
def stress_coverage() -> pd.DataFrame:
    return _read(paths.result("stress_coverage"))


@st.cache_data(show_spinner=False)
def stress_shift_coverage() -> pd.DataFrame:
    return _read(paths.result("stress_shift_coverage"))


@st.cache_data(show_spinner=False)
def stress_worst_windows() -> pd.DataFrame:
    return _read(paths.result("stress_worst_windows"))


# ---------------------------------------------------------------- 数据质量



@st.cache_data(show_spinner=False)
def completeness_report() -> pd.DataFrame:
    return _read(paths.clean("completeness_report"))


@st.cache_data(show_spinner=False)
def fill_log() -> pd.DataFrame:
    return _read(paths.clean("fill_log"))


@st.cache_data(show_spinner=False)
def doubtful_days() -> pd.DataFrame:
    df = _read(paths.result("doubtful_days"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def outlier_candidates() -> pd.DataFrame:
    df = _read(paths.clean("outlier_candidates"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def outlier_judgment() -> pd.DataFrame:
    df = _read(paths.clean("outlier_judgment"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def risk_events() -> pd.DataFrame:
    df = _read(paths.EVENTS / "risk_events_timeline.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 公共工具


def slice_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """按日期区间切片；无 date 列就原样返回。"""
    if "date" not in df.columns or df.empty:
        return df
    lo = pd.Timestamp(start) if start is not None else df["date"].min()
    hi = pd.Timestamp(end) if end is not None else df["date"].max()
    return df[(df["date"] >= lo) & (df["date"] <= hi)]


def date_bounds(*frames) -> tuple:
    """若干张表 date 列的公共最小区间。"""
    los, his = [], []
    for f in frames:
        if f is not None and not f.empty and "date" in f.columns:
            los.append(f["date"].min())
            his.append(f["date"].max())
    if not los:
        return None, None
    return max(los), min(his)
