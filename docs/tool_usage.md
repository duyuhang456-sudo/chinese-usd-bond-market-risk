# 工具使用说明

> 中资投资级美元债风险计量与压力测试预研工具 · 轻量化原型
> 面向对象：接手本仓库的后续同学、答辩评审、以及需要复现结果的任何人。

本工具是一个**命令行脚本包**，不是服务、也没有图形界面。全部功能通过一条命令或
分阶段命令驱动，输入是 `raw_data/` 里的公开数据，输出是 `results/` 的 26 张 CSV 与
`figures/` 的 22 张 PNG。

---

## 1. 环境搭建

### 1.1 版本要求

| 项 | 版本 | 说明 |
|---|---|---|
| Python | **3.9.6** | 本仓库在 3.9 上开发与验证；代码中未使用 3.10+ 语法（如 `match`） |
| 依赖 | 见 `requirements.txt` | 已全部锁定版本 |

**表 1　运行环境**

表 1 说明：Python 版本不是随便写的——本仓库代码在 3.9 下编写，且刻意避开了
3.10 才支持的嵌套同类引号 f-string（该写法在 3.9 下是 `SyntaxError`）。
若要迁移到更新的版本，需重新验证全链路。

### 1.2 核心依赖

| 包 | 版本 | 用途 |
|---|---|---|
| pandas | 2.3.3 | 全部表格处理 |
| numpy | 2.0.2 | 数值计算 |
| scipy | 1.13.1 | 统计分布与检验 |
| statsmodels | 0.14.6 | OLS 回归（δ 估计、信用系数区间） |
| arch | 7.2.0 | GARCH(1,1) 建模（参数法 VaR 的条件波动率） |
| matplotlib | 3.9.4 | 全部绘图 |
| requests / yfinance / beautifulsoup4 / curl_cffi | — | 仅取数脚本需要（联网） |
| pypdf / openpyxl | — | 文档读取与 Excel 输出（辅助） |

**表 2　核心依赖**

表 2 说明：**计算类脚本只依赖前六项**。后两组仅在 `--download` 取数时才会用到——
这意味着离线复现（默认模式）不需要网络、也不需要这几个包能正常访问外部服务。

### 1.3 安装

```bash
# 1) 先确认解释器是 3.9.x——这一步不能省
python3 -V

# 2) 建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**第 1 步不能省，是因为 `python3` 不一定指向 3.9。** 常见的坑：装了 conda 或
Homebrew 的机器上，`python3` 往往指向 3.12+，此时 `python3 -m venv` 建出的环境
版本不符。macOS 的 Command Line Tools 自带 3.9.6，可在没有其他 3.9 时直接用它：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 -m venv .venv
```

若只想跑离线复现（不重新取数），实测最小可运行集为：
`pandas`、`numpy`、`scipy`、`statsmodels`、`arch`、`matplotlib`。

---

## 2. 一键运行

```bash
python code/run_all.py
```

这是**推荐的默认入口**。它把 19 个步骤按依赖顺序串起来，默认跳过取数、直接用
已归档的 `raw_data/` 复现，因此不需要联网。

### 2.1 命令行参数

| 参数 | 作用 |
|---|---|
| （无） | 跑 phase1 + phase2 + phase3，跳过取数（默认） |
| `--only phase2,phase3` | 只跑指定阶段，逗号分隔 |
| `--download` | 连同两个取数脚本一起跑（**需联网**） |
| `--check` | 只对账现有产物，不重跑任何测算 |
| `--quiet` | 抑制各脚本的正常输出，只在失败时打印末尾若干行 |
| `--list` | 列出执行清单后退出 |

**表 3　`run_all.py` 参数**

### 2.2 两类失败的处理方式不同

这是本入口最重要的一条设计，直接决定复现结论是否可信：

- **取数类脚本失败**（`download_all.py` / `download_nav.py`，需联网）→
  记警告并**继续**。因子与结果都可从已归档的 `raw_data/` 重建，
  取数失败不该阻断离线复现。
- **计算类脚本失败** → **立即终止**并返回非零码。带着缺料往下跑会污染后续所有产物，
  且这种污染不会自我暴露——脚本会静默使用上一次的旧产物，报告读起来一切正常。

### 2.3 跑完后的产物对账

全链路结束后，入口会自动调用 `code/check_outputs.py` 核对 `results/` 下每张 CSV 的
行数 × 列数与登记值是否一致，并**反向检查**有无未登记的 CSV。任一不符即返回非零码。

### 2.4 实测耗时与产物

| 项 | 已装好的开发 venv | **另建的干净 venv** |
|---|---|---|
| 步骤 | 19 步（17 步计算 + 2 步取数，默认跳过取数） | 同左 |
| 耗时 | 约 51 ~ 55 秒 | **76.7 秒**（首次运行，冷启动） |
| 产物 | `results/` 26 张 CSV + `figures/` 22 张 PNG，共 48 个 | 同左 |
| 对账 | 26/26 一致 | 26/26 一致 |
| 与清空前比对 | 48/48 逐字节相同 | 48/48 逐字节相同 |

**表 4　全量运行的实测结果**（2026-09-21，Python 3.9.6）

表 4 说明：两列是同一件事在两个环境里各做一遍。**干净 venv 那一列是本节的关键
证据**——它按 §1.3 的步骤从零建环境、只装 `requirements.txt` 里声明的包，装完后
清空 `results/*.csv` 与 `figures/*.png` 再跑全链路，48 个产物与清空前**逐字节相同**
（`shasum -a 256 -c` 全部通过）。这同时证明三件事：依赖清单是完整的（没有偷偷
依赖开发环境里恰好装着的包）、全链路是确定性的（同输入必得同输出）、以及产物
确实由本次运行产生（清空后重建，不是旧文件蒙混过关）。

两列耗时之差（51~55 秒 对 76.7 秒）来自冷启动：干净 venv 首次运行时
`pandas` / `scipy` / `matplotlib` 等需从磁盘冷读并编译字节码。第二次起与开发条件
基本持平，不构成性能差异。

### 2.5 重建时的两个坑

1. **不能用 `--only phase2,phase3` 重建全部产物。**
   `results/baseline_var_tr.csv` 是**阶段一**脚本 `build_tr_factors.py` 的产物，
   而 `var_backtest.py` 依赖它。只跑二三阶段必然中途缺料失败。清空后重建请用全量命令。
2. **必须「先清空再重建」来验证。**
   在已有产物上做校验无法发现「脚本什么都没写」——脚本静默退出时退出码仍为 0、
   旧文件原封不动，校验会全部通过。本项目在开发中正是这样漏掉过一个
   `var_backtest.py` 的空转缺陷。

---

## 3. 分阶段运行

需要单步调试或只重算某一段时，按下面的顺序手动执行。**顺序不可打乱**：脚本之间
靠 CSV 传参，顺序错了会静默用到上一次的旧产物。

### 3.1 阶段一：数据体系搭建与风险因子拆解

```bash
python code/clean_data.py
python code/outlier_detect.py
python code/adjudicate_outliers.py
python code/build_factors.py
python code/build_factors_nav.py
python code/staleness_remedy.py
python code/build_tr_factors.py
python code/descriptive_stats.py
```

### 3.2 阶段二：风险计量模型开发与回测验证

```bash
python code/prep_phase2.py
python code/var_parametric.py
python code/var_historical.py          # 依赖 var_parametric.py 产出的 σ 列
python code/var_backtest.py
python code/recommend_baseline.py
python code/verify_delta_transmission.py
```

### 3.3 阶段三：压力测试框架构建与测算分析

```bash
python code/stress_scenarios.py
python code/stress_impact.py           # 消费 stress_scenarios.csv
python code/stress_robustness.py       # 消费 stress_impact.csv
```

阶段三三步须严格按序；但三者**只读因子表与阶段二结果 CSV、不重算 VaR**，
故在阶段二产物已存在时可独立重跑。

### 3.4 取数（需联网，可选）

```bash
python code/download_all.py
python code/download_nav.py
```

---

## 4. 脚本职责表

`code/` 下共 26 个 `.py`，分为三类：**库 2 个**、**工具 2 个**、**执行脚本 22 个**。

### 4.1 库与工具

| 文件 | 类型 | 职责 |
|---|---|---|
| `common.py` | 库 | **全仓库路径常量的唯一来源**（`RAW` / `CLEAN` / `FACT` / `RES` / `FIG` / `EVENTS_CSV`）+ `ensure_dirs()` / `write_table()` / `save_csv()`。任何脚本都不应再自行定义路径 |
| `var_common.py` | 库 | 阶段二 VaR 公共机制：常量、配色、σ 估计器、历史模拟分位、违规判定 |
| `run_all.py` | 工具 | 一键运行入口（见 §2） |
| `check_outputs.py` | 工具 | 产物对账：核对 `results/` 下 CSV 的行列维度，并反向检查未登记文件 |

**表 5　库与工具**

### 4.2 阶段一脚本

| 文件 | 职责 |
|---|---|
| `download_all.py` | 一键下载全部数据源（取数） |
| `download_treasury.py` | 美债收益率曲线（treasury.gov） |
| `download_fred.py` | FRED 数据（CSV 直下，无需 API key） |
| `download_benchmark.py` | 标的下载：ChinaAMC 亚洲美元投资级债 ETF（**9141.HK 美元柜台**） |
| `download_nav.py` | 9141.HK 官方日度 NAV（MoneyDJ 镜像，净値经官方锚点验证） |
| `clean_data.py` | 日期对齐 + 缺失值处理 |
| `outlier_detect.py` | 异常值识别 |
| `adjudicate_outliers.py` | 异常逐条判定（结合风险事件时间线） |
| `build_factors.py` | 三大风险因子构建 + 信用利差剥离（含 9141 vs 3141 对照） |
| `build_factors_nav.py` | 三大风险因子正式表（**NAV 口径**，阶段二默认输入） |
| `build_tr_factors.py` | **复权（总收益）+ 人民币双口径**正式因子表（阶段二 VaR 输入） |
| `staleness_remedy.py` | 报陈旧三条出路对照实验（为 VaR 选输入口径） |
| `descriptive_stats.py` | 描述性统计 + 质量复核（第一阶段收口） |

**表 6　阶段一脚本**

### 4.3 阶段二脚本

| 文件 | 职责 |
|---|---|
| `prep_phase2.py` | 建模准备：基线 VaR / 因子暴露 / 协方差校验 / 波动分段 / 存疑日清单 |
| `var_parametric.py` | 参数法 VaR：无条件正态 / GARCH(1,1) / 因子协方差 δ-normal |
| `var_historical.py` | 历史模拟法 VaR：经典 HS（窗宽 250/500/750）+ Filtered-HS |
| `var_backtest.py` | 滚动窗口回测：失败率 + Kupiec + Christoffersen 检验 |
| `recommend_baseline.py` | 基准口径定案（95% / 99% 各定一版，供阶段三对标） |
| `verify_delta_transmission.py` | 验证「因子冲击 → 组合损失」的线性传导精度 |

**表 7　阶段二脚本**

### 4.4 阶段三脚本

| 文件 | 职责 |
|---|---|
| `stress_scenarios.py` | 压力情景库构建（3 历史 + 2 补充 + 10 假设 = 15 个情景） |
| `stress_impact.py` | 三路径情景映射 + 压力测试测算 + 因子贡献度分解 |
| `stress_robustness.py` | 五项稳健性检验：覆盖度 / 窗口 / 缓冲 / 信用区间 / 锚点窗长 |

**表 8　阶段三脚本**

---

## 5. 输出物清单

### 5.1 `results/`　26 张 CSV

| 阶段 | 张数 | 代表文件 |
|---|---|---|
| 一 | 1 | `baseline_var_tr.csv` |
| 二 | 17 | `var_parametric.csv`、`var_historical.csv`、`backtest_results.csv`、`model_scorecard.csv`、`baseline_decision.csv` |
| 三 | 9 | `stress_scenarios.csv`、`stress_impact.csv`、`stress_coverage.csv`、`stress_buffer_sens.csv` |

**表 9　结果文件按阶段分布**

表 9 说明：**阶段一的 1 张（`baseline_var_tr.csv`）是易漏项**——它由
`build_tr_factors.py` 产出，却常被误当作阶段二产物，因此在「只跑二三阶段」的
重建命令下会导致缺料中断。`docs/phase2_spec.md` 与 README 的「结果目录」两节
逐张登记了行数与列数，`check_outputs.py` 把这份登记固化为可执行的断言。

### 5.2 `figures/`　22 张 PNG

见 §6 的三列对应表（图 ← 产出脚本 ← 消费数据）。

### 5.3 文档与产物的对应

| 文档 | 覆盖的产物 |
|---|---|
| `docs/data_sources.md`、`docs/data_cleaning.md` | 阶段一 `raw_data/` → `clean_data/` |
| `docs/outlier_check.md` | `clean_data/outlier_candidates.csv`、`outlier_judgment.csv` |
| `docs/factors.md` | `factors/factor_table*.csv` |
| `docs/staleness_remedy.md` | `figures/staleness_remedy.png` |
| `docs/phase1_data_description.md` | 阶段一全部产物（正式交付） |
| `docs/phase2_spec.md` | 阶段二口径与回测协议 |
| `docs/phase2_model_validation.md` | 阶段二 `results/*.csv`（正式交付） |
| `docs/phase3_scenario_library.md` | `results/stress_scenarios.csv` |
| `docs/phase3_stress_testing.md` | 阶段三全部 `results/*.csv` 与 `figures/*.png` |
| `docs/tool_path_convergence.md` | `code/` 的路径常量收敛记录 |

**表 10　文档与产物的对应**

表 10 说明：**报表里的每个数字都应能指到上表某一份 CSV 的具体行与列**。
这是本项目的取值纪律：凡现有数据算不出的量，一律写「目前资料不足，无法确认」，
不以外推或估算填充。

---

## 6. 可视化模块的定位

需求要求工具「含可视化模块与使用说明」。本工具**不把绘图抽成独立模块**——
绘图逻辑内嵌在各自的测算脚本里，与产生该图所需的数据同处一个作用域，
抽离会引入回归风险而无实益。取而代之，下表把「哪张图、由谁产出、消费什么」
固定下来，使可视化部分**可定位、可单独复现**。

| 图（`figures/`） | 产出脚本（`code/`） | 消费数据 |
|---|---|---|
| `spread_factor_vs_oas_nav.png` | `build_factors_nav.py` | `clean_data/` |
| `spread_factor_vs_oas.png`、`spread_factor_vs_oas_3141HK.png` | `build_factors.py` | `clean_data/` |
| `tr_return_series.png` | `build_tr_factors.py` | `factors/factor_table_nav.csv`、`clean_data/nav_9141HK_clean.csv`、`raw_data/dividends_9141HK.csv` |
| `descriptive_hist_qq_nav.png`、`portfolio_return_decomp_nav.png` | `descriptive_stats.py` | `factors/factor_table_nav.csv`、`clean_data/master_calendar.csv` |
| `staleness_remedy.png` | `staleness_remedy.py` | `clean_data/` |
| `phase2_rollvol_regimes.png` | `prep_phase2.py` | `factors/factor_table_nav_tr.csv`、`clean_data/outlier_judgment.csv`、`events/` |
| `var_series_compare.png`、`var_garch_sigma.png` | `var_parametric.py` | `factors/factor_table_nav_tr.csv` |
| `var_hs_quantile_path.png`、`var_hs_vs_parametric.png`、`var_fhs_compare.png` | `var_historical.py` | `factors/factor_table_nav_tr.csv`、`results/var_parametric.csv`、`results/garch_refit_trace_rmb.csv` |
| `backtest_exceptions.png`、`backtest_coverage.png` | `var_backtest.py` | `results/var_parametric.csv`、`var_historical.csv`、`regimes.csv`、`doubtful_days.csv`、`factors/factor_table_3141HK.csv`、`events/` |
| `baseline_model_tradeoff.png` | `recommend_baseline.py` | `results/backtest_results.csv` |
| `delta_transmission.png` | `verify_delta_transmission.py` | `factors/factor_table_nav_tr.csv`、`results/model_scorecard.csv`、`events/` |
| `stress_scenario_library.png` | `stress_scenarios.py` | `factors/factor_table_nav_tr.csv` |
| `stress_mapping.png`、`stress_loss_vs_var.png`、`stress_factor_contrib.png` | `stress_impact.py` | `factors/factor_table_nav_tr.csv`、`results/stress_scenarios.csv`、`factor_exposure.csv`、`backtest_results.csv`、`delta_transmission_events.csv` |
| `stress_coverage.png` | `stress_robustness.py` | `factors/factor_table_nav_tr.csv`、`results/stress_impact.csv`、`stress_scenarios.csv`、`factor_exposure.csv` |

**表 11　图 ← 脚本 ← 数据 三列对应**（14 个脚本产出 22 张图）

表 11 说明：由表可见**绘图与测算是同一次运行完成的**——只要跑通对应脚本，
图就会随之更新，不存在「数据更新了但图还是旧的」的情况。若要单独重绘某张图，
执行该行的产出脚本即可（阶段三三张图的脚本只读 CSV、不重算 VaR，可独立重跑）。
图中若有与文档表述相关的标注（阈值线、口径标签），一律随数据一起算出，
不写死在绘图代码里。

两点补充：其一，22 张图对应 **21 处 `savefig` 调用**——`build_factors.py` 的绘图写成
带参函数 `run(tag, ..., out_fig)`，被 9141.HK 与 3141.HK 各调用一次，产出两张对照图；
这也正是它的图上带着标的口径来源。其二，`var_backtest.py` 会读
`factor_table_3141HK.csv`，只为在违规时间线上标注 3141 同向性——
**3141 不参与任何计量**，仅作旁证。

---

## 7. 硬断言的语义与报错含义

脚本内置多处硬断言，任一不满足即**报错终止**（不是打印警告后继续）。
这是本项目刻意的选择：宁可停在一个明确的地方，也不要带着错误的中间结果往下跑。

### 7.1 断言清单

| 脚本 | 断言 | 不满足时的含义 |
|---|---|---|
| `stress_scenarios.py` | 主/次口径收益差须逐值等于汇率项（log 收益可加性） | 双口径差值不再等于汇率项 → 累计不可直接求和，双口径的所有差值类结论失效 |
| `stress_impact.py` | HS250 @ common773 的四个基准配置须与阶段二报告 §7.2 登记值一致 | 判定阈值与阶段二定案脱钩 → 全部尾部判定失去依据 |
| `stress_impact.py` | 2022 残差标定量须与逐事件表算出的值一致 | 信用区间上端的加项与数据源不符 |
| `stress_impact.py` | **情景库中每个情景都须进入测算** | 有情景被静默漏算（此前情景清单被硬编码，新增的 X2 曾因此漏算，直到覆盖度检验才暴露） |
| `stress_impact.py` | 历史情景的信用区间下端须与窗口累计量不同 | 信用区间**退回窗口级量**，与 1 日判定线口径错配 |
| `stress_impact.py` | 假设情景的 `loss_modeled_pct` 与 `worst_day_modeled_pct` 须重合 | 假设情景路径不再是单日 → 其信用区间不再是 1 日量 |
| `stress_robustness.py` | 主期限须从 `stress_impact.csv` 读出且恒为 1 日 | 两个脚本的持有期口径不同步 |
| `stress_robustness.py` | 窗口敏感性表的最差单日 / 回撤须与基准表逐值对账 | 两张表的同一量算法分叉 |
| `stress_robustness.py` | 缓冲网格中「本报告取值」档须逐条复现基准表的损失与尾部判定 | 5 档缓冲的分析与 headline 结论不是同一套数 |
| `var_backtest.py` | 两个 VaR 产物的日期向量须一致 | 参数法与历史模拟法的口径已分叉 |
| `var_backtest.py` | 存疑日剔除数须等于预期 | 剔除是静默空操作（"剔了但没剔"） |
| `var_backtest.py` | M1δ 与 M1 的违规日集合须相同 | δ-normal 恒等式被破坏 |
| `var_historical.py` | 样本外长度须与 9/15 一致；VaR 不得为负；99% 须 ≥ 95%；前视检查须与手写循环一致 | 各期分位路径或违规判定与协议不符 |
| `var_historical.py`、`var_parametric.py` | 人民币次口径的 NaN 须为尾部连续 | 次口径序列出现中间断点，需检查并非尾部缺失 |
| `var_common.py` | Christoffersen 的 LR_ind 不得为负 | 似然比符号写反 |

**表 12　硬断言清单**（合计 30 处，覆盖 7 个脚本）

表 12 说明：这些断言是**结论可信度的地基**，不是防御性代码。表中加粗的两条
都对应真实发生过的事故：情景漏算与口径退回。断言的报错消息一律写明
「期望值 vs 实际值」与「这意味着什么」，便于直接定位。

表中各脚本的断言处数：`stress_robustness.py` 11 处、`var_historical.py` 8 处、
`stress_impact.py` 5 处、`var_backtest.py` 3 处、`stress_scenarios.py` /
`var_common.py` / `var_parametric.py` 各 1 处；本表按语义归并，同一脚本的多条
若指向同一类风险则合并为一行。断言全部通过是每次改动后的验收前提——
**不得为了跑通而放宽断言**，放宽等于把结论的地基悄悄拆掉。

### 7.2 对账（非终止）

`check_outputs.py` 的对账**不终止其他脚本**，只在末尾汇总并返回非零码。
它核对的是「脚本产出的形状」与「文档声明的形状」是否一致——用于捕捉
「脚本改了、README 没跟」这一类不同步。本项目曾出现 28 处此类缺陷，
成因之一就是数据变了而引用它的文档没同步。

---

## 8. 已知边界与不可外推的范围

工具能算的不等于能说的。以下边界在使用结果时必须一并声明：

**表 13　已知边界**

| # | 边界 | 影响 |
|---|---|---|
| 1 | **信用路径自 2023-09 起**，OAS 序列 2022 年整年缺失 | 信用维度无法覆盖 2022 年；2022 残差是**代理量**，不得表述为「已用 2022 年信用维度残差标定」 |
| 2 | 假设情景是**外推档、不是预测** | 冲击幅度参考历史极端值的 1 / 1.5 / 2 倍，绝对水平不可当作预期损失 |
| 3 | **1 日持有期来自老师口头指示** | 需求文档全文未出现「持有期」；不得写作「需求文档要求 1 日持有期」 |
| 4 | 压力情景按构造即应超过 VaR | 1 日口径下 30 条组合有 19 条命中，二值「是否超限」区分度低，应以「相当于几倍 99% 期望损失」为主导统计量 |
| 5 | √t 缩放**未做独立回测** | 多期参考值系统性低于同期限经验值，不宜直接采用 |
| 6 | δ 线性传导的精度有边界 | 显著头部冲击下线性近似会低估；2022 年残差不可拆解为纯信用维度 |
| 7 | 标的为 ETF（9141.HK）**代理**中资投资级美元债组合 | 净值含费用与跟踪误差，不等同于债券组合本身 |
| 8 | 多期（3 / 5 / 6 日）覆盖度读数**不参与判定** | 移入附录并标注为不同量纲 |

表 13 说明：第 1、3 两条是**归属正确性**问题——写错来源比写错数字更严重，
因为它把外部指示或代理数据说成了有据可依的结论。第 4 条是 1 日重做的直接后果：
阈值下降快于损失下降，命中率自然升高，这是口径变化而非风险变化。

---

## 9. 常见操作

```bash
# 只对账，不重跑
python code/run_all.py --check

# 只重算压力测试三个脚本
python code/run_all.py --only phase3

# 看执行清单
python code/run_all.py --list

# 完整重建（含取数，需联网）
python code/run_all.py --download
```
