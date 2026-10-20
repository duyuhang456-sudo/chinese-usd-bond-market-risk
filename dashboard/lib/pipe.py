"""整张看板共用的试算状态：读者改的那五个选择。

五个选择是汇率口径、报价方案、VaR 模型、置信度档、缓冲档位。它们不是「五步流程」
——看板已经是一张单页长文档，这五件事互相独立，读者从哪一段开始读都行，改了哪一
个也只影响全页读数，不决定「下一步该做什么」。控件全部收在 app.py 顶部那颗
`st.popover` 里（`page.trial_chip`），关掉时主区零控件。

状态存在 st.session_state 里，键名统一加 `pipe_` 前缀，与页内其它 widget key 隔开。

每个取值都带默认回落：直接打开页面（AppTest 逐段跑也是这样），
状态等于报告基准——主口径 · D2 官方 NAV · HS250 · 99% · 0.3365 登记缓冲。

为什么用 commit() 而不是把控件 key 直接设成 pipe_*：这是上一轮五步分页时代的
结论——Streamlit 切到别的页面后，本页 widget 不再挂载，它上报的 widget-state
会从下一次 rerun 的状态里消失。现在是一根长文档、切页这件事没有了，但 commit()
这层依然无害且保持幂等，先留着，勿在其它清理里顺手删掉。

本模块只做**查表**，不做模型算术。唯一的现场换算在压力测试那张卡：
净损失 ÷ 选中模型的 ES（两数都在既有产物里）。
"""

from __future__ import annotations

import streamlit as st

from . import data, labels

# --- session_state 键与基准默认 ---------------------------------------------
K_CALIBER = "pipe_caliber"
K_QUOTE = "pipe_quote"
K_MODEL = "pipe_model"
K_CONF = "pipe_conf"
K_BUFFER = "pipe_buffer"

# 报告基准：与 model_scorecard / stress_buffer_sens 的登记行逐字对应
BASELINE = {
    "caliber": "主",
    "quote": "D2",
    "model": "HS250",
    "conf": "99",
    "buffer": "0.3365 本报告取值",
}

# 报价方案是「日度三案 + 周度两案」共五案，进计量链的只有 D2
QUOTE_CHOSEN = "D2"
QUOTE_DAILY = ("D1", "D2", "D3")


def _choose(key: str, default: str, allowed) -> str:
    """读当前选择；没选过或值已不合法（旧 session 残留）时回落默认。"""
    v = st.session_state.get(key)
    if v in allowed:
        return v
    return default


def commit(key: str, value) -> None:
    """把控件当前返回值写进跨页持久 key（见模块 docstring 的切页丢状态说明）。

    None（segmented 尚未选）不覆盖，保留已持久化的值或基准回落。
    """
    if value is not None:
        st.session_state[key] = value


# --- 五个选择 ---------------------------------------------------------------


def pipe_caliber() -> str:
    """汇率口径：主（复权 USD）/ 次（人民币），基准主。"""
    return _choose(K_CALIBER, BASELINE["caliber"], ("主", "次"))


def pipe_conf() -> str:
    """VaR 置信度：95 / 99，基准 99。"""
    return _choose(K_CONF, BASELINE["conf"], ("95", "99"))


def pipe_model() -> str:
    """八个参选模型之一（M1d / M1g-t 是对照，不在此列），基准 HS250。"""
    return _choose(K_MODEL, BASELINE["model"], tuple(labels.MODEL_ORDER))


def pipe_quote() -> str:
    """报价方案代号 D1/D2/D3/W1/W2，基准 D2（官方 NAV 日度）。"""
    return _choose(K_QUOTE, BASELINE["quote"], tuple(qc[0] for qc in quote_options()))


def pipe_buffer() -> str:
    """缓冲档位标签（stress_buffer_sens 的 5 个登记值），基准 0.3365。"""
    return _choose(K_BUFFER, BASELINE["buffer"], tuple(buffer_labels()))


def consume_chart_model(key: str, param: str) -> bool:
    """消费 st.altair_chart(on_select="rerun") 的图上点模型事件。

    Streamlit 1.50 的 on_select 不支持 layer 多视图，所以承接点击的只能是
    单视图图（②页的平均 VaR 柱条）。只在**页顶、pipe_model() 之前**调：
    命中唯一参选模型且与当前不同 → 写管线并返回 True，调用方立即 st.rerun()。

    读完就把该 widget 的 state 删掉——否则图表里残留的选择会在读者之后
    改用 selectbox 时把模型又拽回来。对照模型（M1δ / M1g-t）与多选一律忽略。
    """
    evt = st.session_state.get(key)
    val = None
    if isinstance(evt, dict):
        val = (evt.get("selection") or {}).get(param)
    picked = None
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], dict):
        picked = val[0].get("model")
    elif isinstance(val, dict):
        picked = val.get("model")
    if picked in tuple(labels.MODEL_ORDER) and picked != pipe_model():
        st.session_state.pop(key, None)
        commit(K_MODEL, picked)
        return True
    return False


def consume_table_model(key: str, ordered_models: list[str]) -> bool:
    """消费 st.dataframe(on_select="rerun", selection_mode="single-row") 的行选择。

    ordered_models 必须与传入 st.dataframe 的表逐行对齐（③页评分卡按
    ORDER10 重排）。其余语义同 consume_chart_model。
    """
    evt = st.session_state.get(key)
    rows = None
    if isinstance(evt, dict):
        rows = (evt.get("selection") or {}).get("rows")
    if isinstance(rows, list) and len(rows) == 1:
        i = rows[0]
        if isinstance(i, int) and 0 <= i < len(ordered_models):
            picked = ordered_models[i]
            if picked in tuple(labels.MODEL_ORDER) and picked != pipe_model():
                st.session_state.pop(key, None)
                commit(K_MODEL, picked)
                return True
    return False


def reset_to_baseline() -> None:
    """第 ⑤ 步「用报告基准参数重走」：五个选择一键复位。"""
    st.session_state[K_CALIBER] = BASELINE["caliber"]
    st.session_state[K_QUOTE] = BASELINE["quote"]
    st.session_state[K_MODEL] = BASELINE["model"]
    st.session_state[K_CONF] = BASELINE["conf"]
    st.session_state[K_BUFFER] = BASELINE["buffer"]


# --- 控件层：试用 / 重置 --------------------------------------------------------


def controls_row() -> None:
    """`page.trial_chip()` 的 popover 内五控 + 重置按钮。

    各控件无 key（避免 widget-state 随切页消失，参考模块 docstring 的跨页坑），
    每次 rerun 把控件返回值 commit 到 pipe_* 持久键。所有控件用 `index=` 显式
    给默认值，AppTest 单开也能跑出基准态。
    """
    _cal = st.radio("汇率口径", ("主", "次"),
                    index=("主", "次").index(pipe_caliber()),
                    help="主 = 复权 USD 口径；次 = 人民币口径",
                    horizontal=True)
    commit(K_CALIBER, _cal)

    opts = quote_options()
    codes = [c for c, _ in opts]
    labels_q = [lab for _, lab in opts]
    _q = st.selectbox("报价方案", opts, index=codes.index(pipe_quote()),
                      format_func=lambda x: x[1] if isinstance(x, tuple) else x)
    commit(K_QUOTE, _q[0] if isinstance(_q, tuple) else _q)

    conf_idx = ("95", "99").index(pipe_conf()) if pipe_conf() in ("95", "99") else 1
    _cf = st.radio("置信度", ("95", "99"), index=conf_idx,
                   help="99% 是 Kupiec 主结论依据；95% 仅供对照",
                   horizontal=True)
    commit(K_CONF, _cf)

    _mo = st.selectbox("试算模型", list(labels.MODEL_ORDER),
                       index=list(labels.MODEL_ORDER).index(pipe_model()),
                       format_func=lambda c: labels.MODEL_LABEL.get(c, c))
    commit(K_MODEL, _mo)

    bls = buffer_labels()
    _buf = st.selectbox("缓冲档位", bls,
                        index=bls.index(pipe_buffer()) if pipe_buffer() in bls else 0)
    commit(K_BUFFER, _buf)

    if st.button("用报告基准参数重走", use_container_width=True):
        reset_to_baseline()
        st.rerun()


# --- 报价方案（factors/staleness_remedy_comparison.csv） --------------------


def quote_options() -> list[tuple[str, str]]:
    """按表内顺序给出（代号, 完整标签）：D1 价格9141(现状) 等。"""
    df = data.staleness_remedy_comparison()
    out = []
    for raw in df["方案"]:
        code, _, name = str(raw).partition(" ")
        out.append((code, f"{code} {name}".rstrip()))
    return out


def quote_label(code: str) -> str:
    for c, lab in quote_options():
        if c == code:
            return lab
    return code


def quote_row(code: str):
    """某方案在四案对照表里的整行（Series）；找不到给 None。"""
    df = data.staleness_remedy_comparison()
    for _, row in df.iterrows():
        if str(row["方案"]).split(" ", 1)[0] == code:
            return row
    return None


# --- 缓冲档位（results/stress_buffer_sens.csv） -----------------------------


def buffer_labels() -> list[str]:
    return list(dict.fromkeys(data.stress_buffer_sens()["buffer_label"]))


def buffer_value(label: str) -> float:
    df = data.stress_buffer_sens()
    return float(df.loc[df["buffer_label"] == label, "buffer_value"].iloc[0])


# --- 查表 -------------------------------------------------------------------


def scorecard_row(model: str | None = None, caliber: str | None = None,
                  conf: str | None = None):
    """评分卡 32 行中指定的那一行（8 模型 × 2 口径 × 2 置信度）。

    参数缺省即取管线当前选择。verdict / mean_var_base_pct / es_base_pct /
    rate_base_pct / cal_dev / cap_idx_vs_M1 / n_pass_uc_core 都从这里取。
    """
    model = model or pipe_model()
    caliber = caliber or pipe_caliber()
    conf = conf or pipe_conf()
    df = data.model_scorecard()
    hit = df[(df["model"] == model) & (df["caliber"] == caliber)
             & (df["conf"].astype(str) == str(conf))]
    if hit.empty:
        return None
    return hit.iloc[0]


def backtest_row(scope: str = labels.BASE_SCOPE, model: str | None = None,
                 caliber: str | None = None, conf: str | None = None):
    """回测明细表中指定「模型 × 口径 × 置信度 × 范围」的那一行。"""
    model = model or pipe_model()
    caliber = caliber or pipe_caliber()
    conf = conf or pipe_conf()
    df = data.backtest_results()
    hit = df[(df["model"] == model) & (df["caliber"] == caliber)
             & (df["conf"].astype(str) == str(conf)) & (df["scope"] == scope)]
    if hit.empty:
        return None
    return hit.iloc[0]


def selected_es() -> float | None:
    """当前试算的 ES 分母（scorecard.es_base_pct，common773）。

    95% 档没有登记 1 日 ES 判定线，返回 None——第 ④ 步据此回退到登记线
    （HS250 的 0.6659 / 0.7024），不现场外推。
    """
    if pipe_conf() != "99":
        return None
    row = scorecard_row()
    if row is None:
        return None
    v = float(row["es_base_pct"])
    return v if v == v else None  # NaN 防护


# --- 眉带元信息 -------------------------------------------------------------


def trial_meta(*, with_buffer: bool = False) -> str:
    """眉带右侧的试算单摘要，如「试算 主口径 · D2 · HS250 · 99% · 0.3365」。"""
    parts = [
        f"试算 {labels.CALIBER_SHORT[pipe_caliber()]}",
        pipe_quote(),
        pipe_model(),
        f"{pipe_conf()}%",
    ]
    if with_buffer:
        parts.append(f"缓冲 {buffer_value(pipe_buffer()):.4f}")
    return " · ".join(parts)
