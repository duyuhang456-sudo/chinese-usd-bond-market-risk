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
唯一的现场算术在压力测试那张卡：把「含缓冲净损失 ÷ 判定线」的分母从登记线
（HS250 ES）换成读者所选模型的 ES，两个输入都在 `model_scorecard.csv`
与 `stress_buffer_sens.csv` 里；`stress_impact.csv` 登记口径的同一个量
（`x_es99`）始终并列在图上。95% 档不换分母，退回登记线。

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

## 单页长文档 · 5 主题分组

**看板不是一条工作流。** 取数、计量、检验、加压、判定五件事没有先后，读者从
哪一段开始读都可以，读到一半改口径也不丢位置。整页是一根可以一直往下滚的长文档，
没有侧栏目录，没有「下一步」按钮，没有 tab。

顶部只有一颗 `st.popover` 做的**试算 chip**，收起时主区零控件。点开是五个选择：
汇率口径、报价方案、VaR 模型、置信度档、缓冲档位。它们不是「流程步骤」，是全页
共用的五个参数——改任何一个，全页读数一起变，再点收起。默认是报告基准：
主口径 · D2 官方 NAV · HS250 · 99% · 0.3365 登记缓冲。
选了未入链的报价方案（D1/D3/W1/W2）只在主题一出提示，不做假的后续结果。

| 段 | 回答什么 | 张数 | 主路径读什么 |
|---|---|---|---|
| 封面 | 五个数字锚点 + 出处表 | — | 五个主题各自的结论数字 |
| 一 · 数据与因子概览 | 五个报价方案选了哪一个、收益率怎么算、哪些日子存疑、因子拆成哪几块 | 9 图卡 | `staleness_remedy_comparison`、`factor_table_nav_tr`、`completeness_report` |
| 二 · VaR 模型对比看板 | 八个模型逐日比：谁给得小、谁经得起检验、波动来自哪 | 10 图卡 | `var_series_long`、`model_scorecard`、`var_attribution` |
| 三 · 压力测试交互分析 | 30 个情景重演：哪个最狠、缓冲吃掉多少、库级覆盖够不够 | 8 图卡 + 附录 6 卡 | `stress_impact`、`stress_buffer_sens`、`stress_factor_contrib` |
| 四 · 稳健性检验探索 | 失败率、两道闸、GARCH 重估、滤波、体制期，全拿出来确认推荐不是偶然 | 7 图卡 | `backtest_results`、`garch_refit_trace_*`、`regimes` |
| 五 · 结论与边界 | 八条业务结论 + 两条已知边界，每条挂一个能回查的数字 | 10 文本卡 | `docs/final_report.md`、`lib/labels.py` |

全页 77 张图、47 个数据抽屉，抽屉标题全库互不重复。主题五是纯文本段，一张图都没有。

**卡片排两栏。** 通铺单栏时每张图会被 `use_container_width` 撑成 1500×320（4.7:1），
又扁又长，页高也一路堆到二十几屏。现在 `page.card()` 自动两两并排，每张图收在
约 700px 的栏里，比例回到 2:1 上下，整页从 24.6 屏降到 17.7 屏。

需要横向空间的图（30 个情景的条图、10 个模型的小倍数网格）走通栏，在
`page.card(..., wide=True)` 上标出来。段的开头再用 `page.grid_plan("H","H","W",...)`
声明一次每张卡是半栏还是通栏——**落在通栏卡前面的那张半栏卡要标成 W**，否则它
孤零零占着左半屏、右边一片空。计划写少了会退回按 `wide=` 参数走，不会崩。

## 版式组件

骨架全部由 Streamlit 原生组件搭出来，`lib/page.py` 提供版式函数：

1. `page.banner( crumb, meta)`——页顶的期刊式 running head，一条发丝横线 + 小号衬线。
   整页只在最顶上打一条，各主题块不再重复。定位用一张装饰性 Altair 图，不写 CSS。
2. `page.trial_chip()`——右上 popover，包住 `pipe.controls_row()` 的五个选择 +
   「用报告基准参数重走」复位钮。关掉时主区真的零控件。
3. `page.section_head(标题, subtitle=, n_cards=)`——主题块头：caps 小标题 +
   原生 `st.title` + 一句导读。五个主题依次吃色板的前五槽色，单调不跳。
4. `page.tile_kpi(标签, 数值, sub=)`——参考图风格的 stat tile，一个判断 + 一个大数字。
   底下就是 `st.metric(border=True)`，第二句人话进 `help`，鼠标悬停才展开，正文不堆字。
5. `page.card(标题, 读法, wide=)`——`st.container(border=True)` + `st.subheader()`。
   卡片本身就是分节单位，卡里放图、一句读法、以及 `charts.data_drawer()` 数据抽屉。
   默认与上一张并排占半栏，`wide=True` 通栏。高度按内容收边、不拉平——同一行常是
   一高一矮，拉平之后矮的那张框里会空出四百像素，像没画完。
6. `page.text_card(序号, 结论, anchor=)`——主题五的结论卡：卡片底 + caps 序号 +
   一句话结论 + 行尾数据锚点，零绘图成本，同样两两并排。
7. `page.footer(来源, 读取列)`——页脚写明这一段读的是哪几个 CSV 的哪几列。

全页状态在 `lib/pipe.py`：五个选择（`pipe_caliber` / `pipe_quote` / `pipe_model` /
`pipe_conf` / `pipe_buffer`）全部带默认回落，单独用 AppTest 跑任一段也有数；
`reset_to_baseline()` 供 chip 里的复位钮。

## 结构

```
dashboard/
  app.py                入口：按 SECTIONS 顺序装配六段，打 running head 与试算 chip
  pages/sections/
    0_cover.py          封面：五个数字锚点 + 出处表
    1_data_factor.py    主题一 · 数据与因子概览（9 张卡）
    2_var_models.py     主题二 · VaR 模型对比看板（10 张卡）
    3_stress.py         主题三 · 压力测试交互分析（8 张卡 + 情景库附录 6 卡）
    4_robustness.py     主题四 · 稳健性检验探索（7 张卡）
    5_verdict.py        主题五 · 结论与边界（10 张文本卡）
  lib/pipe.py           全页共用的五个选择（pipe_*）与查表 helper
  lib/paths.py          从 __file__ 推仓库根；缺料时抛中文错误
  lib/data.py           全部 CSV 的 @st.cache_data 加载器
  lib/theme.py          配色令牌 + Altair 主题注册（light / dark 两套）
  lib/charts.py         可复用图元：折线准星、点+区间、堆叠份额、散点、热图、
                        结论标注、数据抽屉
  lib/page.py           版式骨架：running head、试算 chip、块头、读数 tile、图卡、页脚
  lib/labels.py         口径 / 符号 / 期限等标签文案，单一来源
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

`check_dashboard.py` 用 `streamlit.testing.v1.AppTest` 逐段执行，它除了断言零异常，
还查这几条结构约束：段文件集合与 `app.py` 的 `SECTIONS` 清单恰好一致
（六段，改名或漏删会在这里露出来）、每段都有原生页头（`st.title`）与读数带
（`st.metric`）与写明数据来源的页脚、有图的段必须配数据抽屉、**抽屉标题全库唯一**、
段源码里没有 emoji 码位与 `unsafe_allow_html` / `st.divider(`、不写产物目录。

`check_palette.py` 是 dataviz 规范里那个 node 校验脚本（本机没装 node）的标准库移植，
在 `#FCFBF8`（浅色档纸白）与 `#0A1120`（当前在用的深海蓝）两个底色上各跑五个可计算的检查：
OKLCH 明度带、彩度下限、protan/deutan 模拟下的 CVD ΔE、常视觉 ΔE 下限、
以及对底色的 WCAG 对比度；
它同时把 `.streamlit/config.toml` 的键与 `lib/theme.py` 的令牌对一遍账。
状态色在浅底上只过「色块 >= 3:1」这一档、够不上「可读文字 >= 4.5:1」，
所以看板里状态色一律同时给图标或文字，不靠颜色单独表意。

## 口径纪律

看板沿用终期报告的读数纪律，这几条在每一页都成立：

- 损失类读数一律**正值 = 损失**，轴标题写明方向；VaR 存正值，在图上画成负。
- 每一张图都带**主口径 / 次口径**标签。两个口径的差别是计价基准的差别，不是精度差别。
- NAV 除权口径（`factor_table_nav.csv`）与复权主口径（`factor_table_nav_tr.csv`）
  来自不同文件、数值接近但含义不同，`lib/labels.py` 把口径文案收成单一来源。
- 95% / 99% 在主题二、四指的是 **Kupiec 检验水平**（尾部概率 5% / 1%），不是 VaR 置信度。
- `backtest_results.csv` 的 `rate_pct` 与评分卡 `p` 列读的是**真实违规率**，不是名义水平；
  名义水平永远是 5% / 1%。这条误读在报告里出现过两次，卡片 caption 里带着说明。
- 表 3-1 一类「临界值」永远取 5% 分位，因为判据是 `p ≤ 5%`。
- 空白处不补零：`oas_bp` 自 2023-09-05 起才有值，`doas_bp` 是它的一阶差分，
  首日无差分可算，自 **2023-09-06** 起才有值——两者起点差一天，在图上都画成断口。

## 设计系统

视觉语言是**深海军蓝的学术版式**，全部由 `.streamlit/config.toml` 表达：
深海蓝黑 `#0A1120` 底、卡片 `#101A2C`、冷白 `#D6DEEB` 正文、深蓝发丝边
`#22314C`。字体走衬线栈，西文 Source Serif / Charter / Georgia、中文回落宋体
（Songti SC / SimSun），不拉 Google Fonts。页头是一条发丝横线 + 小号衬线 running
head（`page.banner()` 用装饰性 Altair 图定位，不写 CSS），不是实色色带。

**这套看板只出深色一档。** `config.toml` 把 `base` 钉成 `dark`，
`lib/theme.py` 的 `base_mode()` 也优先读这个钉死的值，不跟浏览器的系统主题走——
页面底是深蓝、图底是浅纸会分裂成两层。浅色那套令牌仍留在 `_TOKENS["light"]` 里
（`check_palette.py` 两档都校验），但页面不使用。

深色档的色板不是浅色的机械取反：`SERIES_DARK` 是同色相的提亮版，在深海底上重新
过了五项校验。热图用的 `BLUE_RAMP_DARK` 从离卡片 1.6:1 的深靛走到亮蓝——低端不能
贴着卡片底色，否则值为 0 的格子会整块消失，分不出「值为 0」和「没有数据」。

分类色按固定顺序取，不循环：`#4784C2` 板蓝 → `#CA5553` 板红 → `#B18F2C` 金黄 →
`#039E93` 青绿 → `#C57127` 橙 → `#8C77C4` 紫 → `#599C4F` 绿 → `#CA7392` 粉。
顺序是按 protan/deutan 色觉模拟算出来的（最差相邻 ΔE 过线），不要手工重排。

散点这类全对比形态最多用 3 个着色序列，折线与堆叠可以用到 5 个以上；
状态色（推荐 / 淘汰、超限 / 未超限）是保留色，只用于状态，且一律同时给
文字或形状，不靠颜色单独表意。单色阶用藏青，连续量级不用彩虹。
需要区分「加总」与「并列序列」时，加总线用中性墨而不是再占一个分类色
（压力页的缓冲档位曲线就是这么画的）。

形状：线 1.6px，标记带 1.5px 底色描边，柱为直角（数据端无圆角），
堆叠段之间留底色缝，网格是实线发丝，**虚线只留给真正的阈值线**。
文字一律用文字色，不穿序列色。不用双 Y 轴。

图上结论标注用 `charts.annotate()`：把一个读数直接标在它发生的位置，
读者不必自己从图里找。它走的是数值轴，要标在分类轴上的直接用 `mark_text` 层。

## 交互

所有交互原语都在 `lib/charts.py` 收着，不在页面里写裸的 vega 参数：

- `charts.zoom_x()` / `zoom_xy()`：返回 `alt.selection_interval(bind="scales", encodings=["x"])`，
  拖移平移、滚轮缩放、双击复位。绑定只走 x 轴，y 轴尺度不动。
- `charts.legend_filter(chart, field)`：`alt.selection_point(fields=[field], bind="legend")`，
  点图例文字隔离单系列，其余降 opacity。
- `line_crosshair(zoom=, legend=)`：折线图的悬停准星，参数按需叠加。
- `st.dataframe(..., on_select="rerun", selection_mode="single-row", key=...)` +
  `pipe.consume_table_model(key, ordered_models)`：**全页唯一一处**「点一下就改全页」的
  交互——主题二 2.4 评分卡上勾选行首方框即切换试算模型，顶部读数、2.1 的蓝线与红点、
  后面所有主题一起跟着换。`1.50` 的行选择只认行首方框（约 30px 宽），点行体不触发。
- `charts.data_drawer(...)` 抽屉内统一加「下载本表 CSV」按钮，导出与页面同列同
  中文列名，全库一处生效。

上一轮那三条跨页联动（柱条点选切模型、锚图 on_select、判定书四选一）已经下线：
页面合成一根之后，「点了要跳到另一页才知道结果」这件事本身就没意义了。

金融红线（逐条守）：

- 缩放 / 选择**只改视图，不改数字**：读数 tile、判定线、p 值等全样本口径数字绝不随
  视图区间重算。需要提示处用现有 caption 一行话说清。
- 交互态只用 opacity（0.15/0.2）不用新色；坐标轴不动画缩放。
- 缩放绑在 vega `bind="scales"`，是视图层面的平移/缩放，与选段无关。
- 下钻图默认就显示当前档位最极端情景，点任一情景条即时切换，**视图不重算**。
- 全部原始数据可经抽屉下载钮导出（utf-8-sig，Excel 直接打开不乱码）。

行首方框命中区经验：约 30px 宽；首行 y 偏移约 51，表头高 34，行高 35。

**图不出框的规矩：全页不写 `facet()`，也不写固定 `width=`。** 分面 spec 的顶层
`width` 是「每一格」的宽度，Streamlit 按格数翻倍，k 格就撑出 k 倍压出卡片；把宽度
定死能躲开这一条，但换个视口就失效（1280 下 1400 的图必溢）。所以多面板一律拆成
若干张单图，用 `st.columns()` 排，每张 `use_container_width=True`——宽度永远跟着
卡片走。小倍数的轴标签按惯例省：y 轴只在每行第一格、x 轴只在最后一行。
实测 1280 / 1440 / 1600 三档视口零出框。

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
  真正量渲染的是 `tools/visual_check.py`：它驱动本机 Chrome，量每张图的几何并截图。
  已经出过的三类问题它都能查——图渲染成空盒子、分面图撑破容器盖住右边一栏、
  某一段过长（护栏按主题块量，单页形态下整页屏数没有参考价值）——这三类在 AppTest
  里一声不响。但观感（好不好看）仍然只能由人打开页面定夺。
- 缓冲档位只在 `stress_buffer_sens.csv` 登记的 5 个档位上取值，不做插值——
  档位是登记值，不是连续参数。
- 族内梯度画的是**已登记情景**本身（轻度 1×、中度 1.5×、极端 2×），
  三点严格落在那条直线上，不提供外推滑块；不外推就没有新假设。
- √t 缩放附表不参与任何判定：多期读数与 1 日基准列不同量纲。
- 五条 σ 路径与 HS 分位路径是参数法机理层的中间量，与主路径的 VaR 序列不是同一个量，
  两处不能互相印证。
- AppTest 驱动不了 `st.dataframe` 的行选择，2.4 评分卡勾行切模型只能在浏览器里点验。
