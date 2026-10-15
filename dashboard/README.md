# 交互看板（Streamlit）

把 `figures/` 下 22 张静态图迁移成可探索的看板。静态图是测算脚本跑到该行顺手
`savefig` 出来的，参数写死在绘图代码里，一张图一个视角；看板只做一件事，
**把已经落库的产物切成可交互的切片**，让每张图都能换口径、换窗口、换档位，
并且不悬停也能从数据抽屉里取到每个值。

## 只读契约

看板只 `read_csv`，不向 `results/` 或 `figures/` 写任何东西，也不改动
`code/` 下的 26 个生产脚本。这条不是风格偏好：`code/check_outputs.py` 会
glob `results/*.csv` 并把多余文件报为对账不通过，写回一个文件就会打破一键复现的验收。

同理，看板里**不做任何重算**——没有 OLS，没有滚动分位，所有数字都来自既有 CSV。
唯一一处现场除法是「含缓冲净损失 ÷ 判定线」，那就是压力页主导统计量 `x_es99`
的定义本身，`results/stress_impact.csv` 里也存着同一个量。

## 跑起来

```bash
pip install -r requirements.txt            # 已含 -r requirements-dashboard.txt
.venv/bin/streamlit run dashboard/app.py   # 必须在仓库根执行
```

**为什么要在仓库根执行**：Streamlit 只按当前工作目录找 `.streamlit/config.toml`，
配置放在仓库根而不是 `dashboard/` 下。数据路径全部由 `lib/paths.py` 从
`__file__` 解析，所以从别的目录启动只会丢主题、不会读错数据。

看板读的是 `results/`、`factors/`、`clean_data/`、`events/` 下已经入库的产物。
若某个产物缺失，`lib/paths.py` 会抛出一句中文 `FileNotFoundError`，提示先跑
`python code/run_all.py`。

## 七个页面

按项目**四阶段**组织，侧栏一组一个阶段，一页回答一个问题、页内不再套第二层筛选。
默认首页是阶段四的「概览与业务结论」：第一屏给一条四阶段进度时间线（来自 git 历史，
回答「做到哪了」），下面保留原业务结论。八页是这样落在四阶段里的——原「稳健性检验与
数据质量」一页跨三个阶段，按阶段拆分并入各阶段（稳健性→阶段三、数据质量→阶段一、
断言→并入结论分级）。

| 页面 | 阶段 | 回答的问题 | 读什么 |
|---|---|---|---|
| 概览与业务结论（默认首页） | 四 | 做到哪了、结论是什么 | `lib/labels.py` 的 `PHASES`（git 历史）+ 终期报告业务结论 + 四个关键读数 |
| 数据与因子概览 | 一 | 读数建立在什么数据上 | 四个因子表、`factor_corr`、`descriptive_stats_nav`、`var_attribution`、`benchmark_comparison`、`doubtful_days` |
| 数据质量 | 一 | 喂进去的数本身靠不靠得住 | `completeness_report`、`fill_log`、`staleness_remedy_comparison`、`risk_events_timeline`、`outlier_judgment`、`doubtful_days` |
| VaR 模型对比与回测 | 二 | 哪个模型该用、它准不准 | `regimes`、`var_historical`、`var_parametric`、`backtest_results`、`backtest_exceptions`、`model_scorecard`、`baseline_decision`、`delta_transmission_*`、`factor_exposure` |
| 模型诊断与验证 | 二 | 检验做了没有、结论站得住吗 | `backtest_results` 的转移四格与三个检验列、`var_parametric` 的五条 σ、`var_historical` 的 HS 分位、`garch_refit_trace_*` |
| 压力测试交互分析 | 三 | 极端情形下亏多少、越线没有 | `stress_scenarios`、`stress_impact`、`stress_buffer_sens`、`stress_factor_contrib` |
| 稳健性检验 | 三 | 结论对口径与设定敏感吗 | `stress_coverage`、`stress_shift_coverage`、`stress_window_sens`、`stress_buffer_sens`、`stress_impact`、`stress_anchor_sens`、`stress_worst_windows` |
| 结论分级与已知边界 | 四 | 哪些能当真、哪些只能参考、哪些不能外推 | `stress_impact`、`var_historical` 的锚点 + `lib/labels.py` 的三层结论文案 + 30 条硬断言 |

这样就补上了此前没有任何落位的六张：`phase2_rollvol_regimes` 与 `var_hs_vs_parametric`
进 VaR 对比回测页，`var_garch_sigma`、`var_hs_quantile_path`、`var_fhs_compare` 进模型诊断页，
`stress_mapping` 进压力测试页。

剩下十六张与看板里的图**不是逐一对应**：同一张静态图的内容可能拆到两节
（`stress_scenario_library` 的清单与映射分在压力测试页的两处），也可能换一种更硬的表达
（`descriptive_hist_qq_nav` 只重画了直方图那半，正态性改由描述统计抽屉里的超额峰度
与 JB 检验 p 值给结论；`backtest_exceptions` 的散点在页面上是失败率与 Kupiec 点区间图）。

每页顶部有**一行**筛选器统辖本页全部图表（概览、结论分级、数据质量三页无筛选器——
前两者是纯展示/结论文案页，后者内容与计量口径无关），图卡里不再放单独的筛选器；
**每张图配一个数据抽屉**——折叠块标题写明这张表是什么，不再像上一版那样十几个折叠块
全叫「表格视图」。

## 每页的骨架

骨架全部由 Streamlit 原生组件搭出来，`lib/page.py` 提供三个函数，页内不再有第二层筛选器：

1. `page.header(标题, 定位, crumb=, caliber=, span=)`——面包屑、页标题、一句定位，
   外加口径与数据区间两枚元信息，读者不必翻到页脚就知道本页读数建立在什么上。
2. `page.kpi_row([...])`——一排 `st.metric(border=True)`。标签写的是**判断**而不是名词，
   读标题那一行就知道这一页主张什么；展开说明放 `help`，不占版面。
3. `page.card(标题, 读法)`——`st.container(border=True)` + `st.subheader()`。卡片本身
   就是分节单位，卡里放图、一句读法、以及 `charts.data_drawer()` 数据抽屉。
   同一行卡片长短不一时用 `stretch=True` 拉平。
4. `page.footer(来源, 读取列)`——页脚写明这一页读的是哪几个 CSV 的哪几列。

## 结构

```
dashboard/
  app.py                入口：st.navigation 装配 8 个页面 + 四阶段侧栏 + 进度自述
  pages/                8 个页面，文件名即页面；导航分组见 app.py
  lib/paths.py          从 __file__ 推仓库根；缺料时抛中文错误
  lib/data.py           全部 CSV 的 @st.cache_data 加载器
  lib/theme.py          配色令牌 + Altair 主题注册（light / dark 两套）
  lib/charts.py         可复用图元：折线准星、点+区间、堆叠份额、散点、热图、
                        结论标注、数据抽屉
  lib/page.py           版式骨架：页头、读数带、图卡、页脚
  lib/labels.py         口径 / 符号 / 期限 / 阶段进度（PHASES）的标签文案，单一来源
```

**代码里没有一行 CSS。** 外观全部落在 `.streamlit/config.toml` 的 `[theme]` 里，
图表外观由 `lib/theme.py` 注册的 Altair 主题负责。这是 Streamlit 官方设计规范的路子
（「Do not use custom CSS to polish an app unless the user explicitly asks for it」）：
原生主题跨元素一致，也不会因为 Streamlit 换了内部 class 名就整块失效。
同理，右上角的 Deploy 按钮与开发者菜单是用 `[client] toolbarMode = "minimal"`
这个原生开关关掉的，不是用 CSS 去藏元素。

## 校验

```bash
.venv/bin/python tools/check_dashboard.py   # 逐页 AppTest + 结构断言
.venv/bin/python tools/check_palette.py     # 配色的五项可计算检查
```

`check_dashboard.py` 用 `streamlit.testing.v1.AppTest` 逐页执行，它除了断言零异常，
还查这几条结构约束：页面集合与 `app.py` 的四阶段导航恰好一致（命名页六页 +
阶段拆出的 data_quality / robustness 共八页，改名或漏删会在这里露出来）、
每页都有原生页头（`st.title`）与读数带（`st.metric`）与写明数据来源的页脚、
有图的页面必须配数据抽屉、**抽屉标题全库唯一**、页面源码里没有 emoji 码位与
`unsafe_allow_html` / `st.divider(`、页面源码不写产物目录。

`check_palette.py` 是 dataviz 规范里那个 node 校验脚本（本机没装 node）的标准库移植，
在 `#F7F5F0` 与 `#101F30` 两个底色上各跑五个可计算的检查：OKLCH 明度带、彩度下限、
protan/deutan 模拟下的 CVD ΔE、常视觉 ΔE 下限、以及对底色的 WCAG 对比度；
它同时把 `.streamlit/config.toml` 的键与 `lib/theme.py` 的令牌对一遍账。
状态色在浅底上只过「色块 >= 3:1」这一档、够不上「可读文字 >= 4.5:1」，
所以看板里状态色一律同时给图标或文字，不靠颜色单独表意。

## 口径纪律

看板沿用终期报告的读数纪律，这几条在每一页都成立：

- 损失类读数一律**正值 = 损失**，轴标题写明方向；VaR 存正值，在图上画成负。
- 每一张图都带**主口径 / 次口径**标签。两个口径的差别是计价基准的差别，不是精度差别。
- NAV 除权口径（`factor_table_nav.csv`）与复权主口径（`factor_table_nav_tr.csv`）
  来自不同文件、数值接近但含义不同，`lib/labels.py` 把口径文案收成单一来源。
- 95% / 99% 在回测页指的是 **Kupiec 检验水平**，不是 VaR 置信度。
- 空白处不补零：`oas_bp` 自 2023-09-05 起才有值，`doas_bp` 是它的一阶差分，
  首日无差分可算，自 **2023-09-06** 起才有值——两者起点差一天，在图上都画成断口。

## 设计系统

视觉语言与结项汇报稿共用一套，但**全部由 `.streamlit/config.toml` 表达**：米白
`#F7F5F0` 底、白卡片、深蓝 `#16283C` 焦点色、暖灰 `#E3DDD1` 边、正文 `#465667`。
金色 `#C9A66B` 对米白底只有约 2:1 对比度，做焦点环或正文都不合格，所以它只做
装饰（图与卡片里的强调），需要读的金色文字一律用 `#8C6832`（约 5.3:1）；
`primaryColor` 因此给深蓝而不是金色。

深色档不是把浅色反转，而是另给一组值，运行时按浏览器实际主题切换：`lib/theme.py`
里 `SERIES_DARK` 与 `SURFACE_DARK` 是在深蓝底 `#101F30` 上重新校验过的，
不是浅色配色的机械取反。

分类色按固定顺序取，不循环：`#2a78d6` 蓝 → `#eb6834` 橙 → `#1baf7a` 水绿 →
`#eda100` 黄 → `#e87ba4` 品红 → `#008300` 绿 → `#4a3aa7` 紫 → `#e34948` 红。
前三个正好是 `code/var_common.py` 里 `M1 / GARCH / 备选` 用的三个色，
所以看板与既有静态图是同一套视觉语言。

散点这类全对比形态最多用 3 个着色序列，折线与堆叠可以用到 5 个以上；
状态色（超限 / 未超限、真实冲击 / 存疑）是保留色，只用于状态，且一律同时给
文字或形状，不靠颜色单独表意。单色阶用蓝，连续量级不用彩虹。
需要区分「加总」与「并列序列」时，加总线用中性墨而不是再占一个分类色
（压力页的缓冲档位曲线就是这么画的）。

形状：线 2px，标记带 2px 底色描边，柱不超过 24px 且数据端 4px 圆角，
堆叠段之间留 2px 底色缝，网格是实线发丝，**虚线只留给真正的阈值线**。
文字一律用文字色，不穿序列色。不用双 Y 轴。

图上结论标注用 `charts.annotate()`：把一个读数直接标在它发生的位置，
读者不必自己从图里找。它走的是数值轴，要标在分类轴上的直接用 `mark_text` 层。

## 部署到 Streamlit Community Cloud

看板是本地原型，同时带着部署配置，用你自己的账号可以发布成一条链接。

1. 把仓库推到 GitHub（public 或 private 均可；private 需要在 Cloud 里授权该仓库）。
2. 打开 `share.streamlit.io`（部署控制台，不是 `docs.streamlit.io` 那个文档站），
   用 GitHub 账号登录，选 **New app** → **Deploy a public app from GitHub**。
3. Repository 选本仓库，Branch 选 `main`，**Main file path 填 `dashboard/app.py`**。
   入口在子目录是支持的；`.streamlit/config.toml` 必须留在仓库根，也只有它能留在根。
4. **Advanced settings** 的 Python version **保持默认的 3.12**，不要选 3.9。
   本项目冻结的依赖在 3.12 上全部有 wheel（已用
   `pip install --dry-run --only-binary=:all: --python-version 3.12 -r requirements.txt`
   验过，从零可解）；而 3.9 已停止安全更新，Cloud 只保留仍在维护的版本，
   选不到、或选到了也会被强制换掉。
5. Requirements file 用默认的 `requirements.txt`。Cloud 找依赖文件的顺序是
   「先入口文件所在目录，再仓库根」，只认 `uv.lock` / `Pipfile` / `environment.yml` /
   `requirements.txt` / `pyproject.toml` 这几个固定名字，**不支持指到任意路径**。
   本仓库只在根放了一个 `requirements.txt`，末尾有 `-r requirements-dashboard.txt`，
   Cloud 会顺着装到 streamlit。首次构建因此会连带装上计量链路的全部依赖
   （matplotlib / statsmodels / arch 等，几百 MB），构建偏慢但可用。
6. **云端只装看板这一层**：`dashboard/requirements.txt`（只含
   `streamlit==1.50.0`）已入库。入口文件所在目录的依赖文件优先于仓库根的那份，
   所以 Cloud 只装 streamlit 及其传递依赖，不碰计量链路的包，构建不在 oven 卡。
   看板不 import 任何一个计量包，此瘦身安全。本地离线自检不受影响（仍用仓库根
   `requirements.txt`）。
7. 点 Deploy。之后每次 push 到该分支会自动重新部署。

`.streamlit/config.toml` 放在仓库根，Cloud 从仓库根读它，部署后 `streamlit run` 的
工作目录也是仓库根——与本地要求的启动方式一致。`results/`、`factors/`、`clean_data/`、
`events/` 下的 CSV 都已入库，所以云端构建出来的看板与本地读的是同一份数据。

本地等价的自检（部署前建议跑一遍）：

```bash
python tools/check_dashboard.py            # 逐页执行 + 结构断言
python tools/check_palette.py              # 配色五项检查
python tools/visual_check.py /tmp/shots    # 真的渲染一遍，量几何、留截图
.venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8502
curl -s -o /dev/null -w '%{http_code}' localhost:8502   # 应为 200
```

## 已知限制

- `AppTest` 证明的是「页面不报错、取得到底层表」，**证明不了「画出来了」**。
  真正量渲染的是 `tools/visual_check.py`：它驱动本机 Chrome，逐页量每张图的
  几何并截图。已经出过的三类问题它都能查——图渲染成空盒子、分面图撑破容器
  盖住右边一栏、页面长到四五屏——这三类在 AppTest 里一声不响。
  但观感（好不好看）仍然只能由人打开页面定夺。
- 压力测试页的缓冲档位选择器只在 `stress_buffer_sens.csv` 登记的 5 个档位上取值，
  不做插值——档位是登记值，不是连续参数。
- 压力测试页的族内梯度画的是**已登记情景**本身（轻度 1×、中度 1.5×、极端 2×），
  三点严格落在那条直线上，页面不再提供外推滑块；不外推就没有新假设。
- 稳健性检验页的 √t 缩放附表不参与任何判定：多期读数与 1 日基准列不同量纲。
- 模型诊断页的「五条 σ 路径」与「HS 分位路径」是参数法机理层的中间量，
  与 VaR 对比回测页的 VaR 序列不是同一个量，两页之间不能互相印证。
