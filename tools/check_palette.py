#!/usr/bin/env python
"""配色校验：dataviz 五项可计算检查的标准库移植，外加本看板的令牌自检。

为什么单独有这个文件：dataviz 的校验脚本是 node 写的，本机没有 node，云端也不装。
而那套规则里最要紧的一条是「配色可计算就不要用眼睛看」——换底色就必须重算，
所以把它的数学逐条移植过来，标准库即可运行，无第三方依赖：

  2. 明度带   OKLCH L 落在该模式的区间内
  3. 彩度下限 OKLCH C >= 0.10（低于它就发灰，读不出色相）
  4. 色觉分离 Machado-Oliveira-Fernandes(2009) severity=1.0 模拟下的 OKLab ΔE×100
              （取 protan/deutan 里较差的那个；tritan 一并报出）
  4b.常视觉下限 不模拟时的最差配对 ΔE×100，硬门槛 15——二次编码不能豁免这一条
  5. 对比度   每个色标对图表底色的 WCAG 比值

阈值、Machado 矩阵与判定逻辑照抄原脚本，**改阈值必须两边同步**。
原脚本的 `--pairs all` 对应散点/小倍数这类全对比形态，这里默认两种都跑。

用法：
    .venv/bin/python tools/check_palette.py          # 校验本看板的两套令牌
    .venv/bin/python tools/check_palette.py --quiet   # 只打印失败项
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- 阈值（照抄 validate_palette.js）-----------------------------------------
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}   # OKLCH L
CHROMA_FLOOR = 0.10                                    # OKLCH C
CVD_TARGET, CVD_FLOOR = 8.0, 6.0                       # OKLab ΔE×100，相邻配对
NORMAL_FLOOR = 15.0                                    # 硬门槛
CONTRAST_MIN = 3.0                                     # WCAG，对底色
ORDINAL_MIN_DL = 0.06                                  # 有序色阶相邻步的最小 ΔL
ORDINAL_LIGHT_FLOOR = 2.0                              # 有序色阶最浅步对底色的 WCAG

# Machado, Oliveira & Fernandes (2009)，severity 1.0，作用在线性 RGB 上
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


# --- 色彩换算 ----------------------------------------------------------------
def _srgb(hex_str: str) -> tuple[float, float, float]:
    """#rrggbb -> 0..1 的 sRGB 三元组。前端已保证格式合法。"""
    h = hex_str.strip().lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(hex_str: str) -> tuple[float, float, float]:
    return tuple(_s2lin(c) for c in _srgb(hex_str))  # type: ignore[return-value]


def _cbrt(x: float) -> float:
    """Python 3.9 没有 math.cbrt；这里输入恒为非负，取幂即可。"""
    cbrt = getattr(math, "cbrt", None)
    return cbrt(x) if cbrt else x ** (1.0 / 3.0)


def _rel_lum(hex_str: str) -> float:
    r, g, b = _lin(hex_str)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _oklab(hex_str: str) -> tuple[float, float, float]:
    return _oklab_from_lin(_lin(hex_str))


def _oklch(hex_str: str) -> tuple[float, float]:
    lab = _oklab(hex_str)
    return lab[0], math.hypot(lab[1], lab[2])


def _okhue(hex_str: str) -> float:
    _, a, b = _oklab(hex_str)
    return (math.degrees(math.atan2(b, a)) % 360 + 360) % 360


def _clamp(c: float) -> float:
    return max(0.0, min(1.0, c))


def _simulate(hex_str: str, kind: str) -> tuple[float, float, float]:
    r, g, b = _lin(hex_str)
    m = MACHADO[kind]
    return (_clamp(m[0][0] * r + m[0][1] * g + m[0][2] * b),
            _clamp(m[1][0] * r + m[1][1] * g + m[1][2] * b),
            _clamp(m[2][0] * r + m[2][1] * g + m[2][2] * b))


def delta_e(c1: str, c2: str, kind: str | None = None) -> float:
    """OKLab 上的欧氏距离 ×100。kind 为空即不模拟（常视觉）。"""
    a = _oklab_from_lin(_simulate(c1, kind) if kind else _lin(c1))
    b = _oklab_from_lin(_simulate(c2, kind) if kind else _lin(c2))
    return 100 * math.dist(a, b)


# --- 检查 --------------------------------------------------------------------
def validate_categorical(palette: list[str], mode: str, surface: str,
                         pairs: str = "adjacent") -> tuple[list[tuple], bool]:
    lo, hi = BAND[mode]
    report: list[tuple] = []
    ok = True

    offband = [(c, round(_oklch(c)[0], 3)) for c in palette
               if not lo <= _oklch(c)[0] <= hi]
    ok &= not offband
    report.append(("明度带", "PASS" if not offband else "FAIL",
                   f"越界 {offband}" if offband
                   else f"{len(palette)} 槽全在 L {lo}–{hi}"))

    lowc = [(c, round(_oklch(c)[1], 3)) for c in palette if _oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("彩度下限", "PASS" if not lowc else "FAIL",
                   f"低于下限（发灰）{lowc}" if lowc
                   else f"{len(palette)} 槽全 >= {CHROMA_FLOOR}"))

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]
    label = "全配对" if pairs == "all" else "相邻"

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist),
              default=99.0)
    wd = worst[0] if worst else 99.0
    state = "PASS" if wd >= CVD_TARGET else ("WARN" if wd >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    report.append(("色觉分离", state,
                   f"最差{label} {worst[3]}↔{worst[2]} ΔE {wd:.1f}（{worst[1]}）"
                   f" · tritan {tri:.1f}" if worst else "n/a"))

    nworst = None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if nworst is None or d < nworst[0]:
            nworst = (d, palette[i], palette[j])
    nd = nworst[0] if nworst else 99.0
    nstate = "PASS" if nd >= NORMAL_FLOOR else "FAIL"
    ok &= nstate == "PASS"
    detail = (f"最差{label} {nworst[2]}↔{nworst[1]} ΔE {nd:.1f}（常视觉）" if nworst else "n/a")
    if nworst and nstate == "FAIL":
        detail += f" —— 低于 {NORMAL_FLOOR:.0f}，全彩视觉也难分辨"
    report.append(("常视觉下限", nstate, detail))

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    report.append(("对比度", "WARN" if low else "PASS",
                   f"低于 {CONTRAST_MIN}:1 —— 须有可见标签或表格视图兜底：{low}" if low
                   else f"{len(palette)} 槽全 >= {CONTRAST_MIN}:1（底色 {surface}）"))
    return report, ok


def validate_ordinal(palette: list[str], mode: str,
                     surface: str) -> tuple[list[tuple], bool]:
    """有序色阶的检查与分类色不同：分类检查会「正确地」判错一条色阶
    （它本就跨越明度带、浅步低于彩度下限）。这里验的是它读起来像不像一条色阶。"""
    report: list[tuple] = []
    ok = True
    ls = [_oklch(c)[0] for c in palette]

    order = sorted(range(len(ls)), key=lambda i: ls[i])
    mono = order == list(range(len(ls))) or order == list(reversed(range(len(ls))))
    ok &= mono
    report.append(("明度单调", "PASS" if mono else "FAIL",
                   "浅→深成序" if mono
                   else f"次序错乱 —— L 值 {[round(v, 3) for v in ls]}"))

    gaps = [abs(ls[i + 1] - ls[i]) for i in range(len(ls) - 1)]
    thin = [(palette[i], palette[i + 1], round(g, 3))
            for i, g in enumerate(gaps) if g < ORDINAL_MIN_DL]
    ok &= not thin
    report.append((f"相邻 ΔL", "PASS" if not thin else "FAIL",
                   f"步距过近 {thin}" if thin else f"各步均 >= {ORDINAL_MIN_DL}"))

    by_l = sorted(palette, key=_oklch)
    lightest = by_l[-1] if mode == "light" else by_l[0]
    cr = contrast(lightest, surface)
    ok &= cr >= ORDINAL_LIGHT_FLOOR
    report.append(("浅端对比度", "PASS" if cr >= ORDINAL_LIGHT_FLOOR else "FAIL",
                   f"{lightest} 对底色 {cr:.2f}:1"
                   + ("" if cr >= ORDINAL_LIGHT_FLOOR
                      else f" —— 低于 {ORDINAL_LIGHT_FLOOR}:1 下限")))

    hues = [_okhue(c) for c in palette]
    spread = max(hues) - min(hues) if hues else 0.0
    if spread > 180:
        spread = 360 - spread
    one_hue = spread <= 40
    ok &= one_hue
    report.append(("单色相", "PASS" if one_hue else "FAIL",
                   f"色相跨度 {spread:.0f}°"
                   + ("" if one_hue else " —— >40°，不是单色相色阶")))
    return report, ok


# --- 本看板的令牌自检 --------------------------------------------------------
def _load_tokens() -> dict:
    sys.path.insert(0, str(ROOT / "dashboard"))
    from lib import theme  # noqa: PLC0415

    return {
        "light": theme.tokens("light"),
        "dark": theme.tokens("dark"),
        "series_light": list(theme.SERIES_LIGHT),
        "series_dark": list(theme.SERIES_DARK),
        "ramp": list(theme.BLUE_RAMP),
    }


def _config_surfaces() -> dict[str, str]:
    """从 .streamlit/config.toml 取底色，用来和令牌对账（两处必须一致）。"""
    text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    out = {}
    for key in ("backgroundColor", "secondaryBackgroundColor", "textColor",
                "primaryColor", "borderColor"):
        m = re.search(rf'^\s*{key}\s*=\s*"([^"]+)"', text, re.M)
        if m:
            out[key] = m.group(1)
    return out


def _config_chart_colors() -> dict[str, list]:
    """从 config.toml 取两个图表色数组。

    原生图表（st.line_chart 等）走这两个键，Altair 图走 lib/theme.py 注册的主题；
    两边必须同色，否则同一个页面里会出现两套配色。config.toml 的注释声称它们
    与令牌一致，这里就是那句声称的检查。
    """
    text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    out: dict[str, list] = {}
    for key in ("chartCategoricalColors", "chartSequentialColors"):
        m = re.search(rf"^\s*{key}\s*=\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            out[key] = [c.lower() for c in re.findall(r'"([^"]+)"', m.group(1))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="只打印失败与警告项")
    args = ap.parse_args()

    tok = _load_tokens()
    cfg = _config_surfaces()
    # 页面钉的是哪一档，config 对账就比哪一档——两档都要分别过自己的五项检查，
    # 但「config.toml 与令牌一致」只有一条，读者看到的那一档才算数。
    active = re.search(r'^\s*base\s*=\s*"(light|dark)"', 
                       (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"),
                       re.M)
    active = active.group(1) if active else "light"
    failures: list[str] = []

    def show(title: str, report: list[tuple], ok: bool) -> None:
        print(f"\n{title}")
        for name, state, detail in report:
            if args.quiet and state == "PASS":
                continue
            print(f"  [{state:4}] {name:<10} {detail}")
        if not ok:
            failures.append(title)

    for mode in ("light", "dark"):
        surface = tok[mode]["page"]
        series = tok["series_light"] if mode == "light" else tok["series_dark"]
        # 相邻配对：折线、柱、堆叠这类相邻形态，8 槽都得过。
        rep, ok = validate_categorical(series, mode, surface, "adjacent")
        show(f"分类色 · {mode} · 相邻配对 · 底色 {surface}", rep, ok)
        # 全配对：散点、小倍数这类任意两色都会并排出现的形态。
        # 规则是「全对比形态最多 3 个着色序列」，所以只验前 3 槽；
        # 第 4 槽起就会跌破常视觉下限（浅色 13.7、深色 10.6），这正是不许用的理由。
        rep, ok = validate_categorical(series[:3], mode, surface, "all")
        show(f"分类色 · {mode} · 全配对（散点上限 3 槽）· 底色 {surface}", rep, ok)

        rep, ok = validate_ordinal(tok[mode]["blue_ramp_ordinal"], mode, surface)
        show(f"离散蓝阶 · {mode} · 底色 {surface}", rep, ok)

    # 状态色是保留色，只用于状态、且永远配图标或文字，所以它不参与分类色的
    # 「相邻 ΔE」那套——四个状态是一条有序的严重度阶梯，不是四个并列的类别
    # （参考档自身的 warning↔serious 也只有 ΔE 13.6）。
    # 能计算的是它作为色块/色条时的可见性：对底色 >= 3:1。
    print("\n状态色对底色的可见性（色块 >= 3:1；作为需读的文字则要 >= 4.5:1）")
    for mode in ("light", "dark"):
        surface = tok[mode]["page"]
        for name, hexv in tok[mode]["status"].items():
            cr = contrast(hexv, surface)
            flag = "PASS" if cr >= 4.5 else ("WARN" if cr >= 3.0 else "FAIL")
            if flag == "FAIL":
                failures.append(f"状态色 {mode}/{name} 对底色仅 {cr:.2f}:1")
            note = "" if cr >= 4.5 else "  （仅作色块，须配图标或文字）"
            print(f"  [{flag:4}] {mode:5} {name:<9} {hexv}  {cr:.2f}:1{note}")

    # config.toml 与令牌必须指同一个底色，否则页面与图会分裂成两套观感
    print("\nconfig.toml 与 theme.py 的令牌对账")
    pairs = [("backgroundColor", "page"), ("secondaryBackgroundColor", "surface"),
             ("textColor", "ink2")]
    if "borderColor" in cfg:
        pairs.append(("borderColor", "border"))
    for cfg_key, tok_key in pairs:
        a, b = cfg.get(cfg_key), tok[active].get(tok_key)
        same = (a or "").lower() == (b or "").lower()
        print(f"  [{'PASS' if same else 'FAIL':4}] {cfg_key:<26} "
              f"{a} vs tokens['{active}']['{tok_key}'] {b}")
        if not same:
            failures.append(f"config.{cfg_key} 与令牌不一致")

    # 两个图表色数组：原生图表与 Altair 图必须同色
    chart = _config_chart_colors()
    cat = chart.get("chartCategoricalColors")
    want_cat = [c.lower() for c in tok[f"series_{active}"]]
    if cat is None:
        print("  [FAIL] chartCategoricalColors      缺失——原生图表会退回 Streamlit 默认色")
        failures.append("config.chartCategoricalColors 缺失")
    else:
        same = cat == want_cat
        print(f"  [{'PASS' if same else 'FAIL':4}] chartCategoricalColors      "
              f"{len(cat)} 槽 vs SERIES_{active.upper()} {len(want_cat)} 槽")
        if not same:
            failures.append("config.chartCategoricalColors 与当前档 SERIES 不一致")

    seq = chart.get("chartSequentialColors")
    ramp = [c.lower() for c in tok["light" if active == "light" else "dark"]["blue_ramp"]]
    if seq is None:
        print("  [FAIL] chartSequentialColors       缺失")
        failures.append("config.chartSequentialColors 缺失")
    else:
        # Streamlit 的硬要求：不是正好十个就整组弃用
        n_ok = len(seq) == 10
        # 每一个都必须出自 BLUE_RAMP，且按浅→深成序（否则不是单色相递进）
        idx = [ramp.index(c) if c in ramp else -1 for c in seq]
        in_ramp = all(i >= 0 for i in idx)
        ordered = in_ramp and idx == sorted(idx) and len(set(idx)) == len(idx)
        ok = n_ok and in_ramp and ordered
        detail = f"{len(seq)} 槽"
        if not n_ok:
            detail += " —— 必须是正好 10 个，否则 Streamlit 整组弃用"
        elif not in_ramp:
            detail += f" —— 有槽位不在 BLUE_RAMP 里：{[c for c, i in zip(seq, idx) if i < 0]}"
        elif not ordered:
            detail += " —— 不是按 BLUE_RAMP 的浅→深成序"
        else:
            detail += f"，取自 BLUE_RAMP 第 {idx} 步"
        print(f"  [{'PASS' if ok else 'FAIL':4}] chartSequentialColors       {detail}")
        if not ok:
            failures.append("config.chartSequentialColors 与 BLUE_RAMP 对不上")

    print()
    if failures:
        print("未通过：" + "；".join(failures))
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
