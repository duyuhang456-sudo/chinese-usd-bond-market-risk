# 工具整合前置盘点：路径常量收敛清单

**用途**：第四阶段（9/28–10/2）Day 1「路径收敛 + 一键入口」的施工依据。
**盘点时间**：2026-09-21　**盘点对象**：`code/` 下全部 24 个 `.py`（6179 行）。

**结论一句话**：`REPO` 已正确单点定义于 `code/common.py:14`，21 个脚本按 `from common import REPO`
取用；但派生于 `REPO` 的六个路径常量共 **48 处定义、散落在 17 个脚本**里，需收敛到 `common.py`；
另有 `var_backtest.py` 无 `__main__` 守卫，对一键入口构成硬约束。

---

## 一、总体盘点

| 常量 | 指向 | 定义脚本数 | 带 `mkdir` | 无 `mkdir` |
|---|---|---|---|---|
| `CLEAN` | `clean_data/` | 8 | 1 | 7 |
| `FACT` | `factors/` | 12 | 4 | 8 |
| `RES` | `results/` | 10 | 2 | 8 |
| `FIG` | `figures/` | 14 | 11 | 3 |
| `RAW` | `raw_data/` | 1 | 0 | 1 |
| `EVENTS` | `events/risk_events_timeline.csv`（**文件**） | 1 | — | — |
| `EV` | `events/`（**目录**） | 2 | — | — |

> 表 1　六个路径常量在 `code/` 下的重复定义分布（不含 `common.py`）

表 1 显示两点。其一，重复面最大的是 `FIG`（14 个脚本各自定义），最小的是 `RAW`（仅 1 处，
且该处是遮蔽而非新增，见 §三）。其二，`mkdir` 的覆盖与定义数完全不同步：`FIG` 14 处定义里
有 11 处顺手建了目录，而 `CLEAN` 8 处定义里只有 1 处建目录——这说明 `mkdir` 是各脚本按自身
需要临时加的，不是一条落地的约定。

七个常量的定义处数为 48，分布在 17 个脚本里；其余 7 个脚本（`common.py`、`var_common.py`
与 5 个 `download_*.py` 中的 4 个）不定义路径常量。

---

## 二、逐脚本定义清单

| # | 脚本 | 定义行（常量@行号） | `mkdir` 行 |
|---|---|---|---|
| 1 | `adjudicate_outliers.py` | `CLEAN`@19、`EVENTS`@20 | 无 |
| 2 | `build_factors.py` | `CLEAN`@39、`FACT`@40、`FIG`@41 | 42、43 |
| 3 | `build_factors_nav.py` | `CLEAN`@35、`FACT`@36、`FIG`@37 | 38、39 |
| 4 | `build_tr_factors.py` | `RAW`@36、`CLEAN`@37、`FACT`@38、`RES`@39、`FIG`@40 | **全无** |
| 5 | `clean_data.py` | `CLEAN`@26 | 27 |
| 6 | `descriptive_stats.py` | `FACT`@27、`FIG`@28 | 29、30 |
| 7 | `outlier_detect.py` | `CLEAN`@29 | 无 |
| 8 | `prep_phase2.py` | `FACT`@33、`CLEAN`@34、`EV`@35、`RES`@36、`FIG`@37 | 38、39 |
| 9 | `recommend_baseline.py` | `RES`@63、`FIG`@64 | 无 |
| 10 | `staleness_remedy.py` | `CLEAN`@37、`FACT`@38、`FIG`@39 | 40、41 |
| 11 | `stress_impact.py` | `FACT`@169、`RES`@170、`FIG`@171 | 172 |
| 12 | `stress_robustness.py` | `FACT`@108、`RES`@109、`FIG`@110 | 111 |
| 13 | `stress_scenarios.py` | `FACT`@93、`RES`@94、`FIG`@95 | 96 |
| 14 | `var_backtest.py` | `RES`、`FIG`@53（**同行元组赋值**） | 54 |
| 15 | `var_historical.py` | `FACT`@48、`RES`@49、`FIG`@50 | 51 |
| 16 | `var_parametric.py` | `FACT`@44、`RES`@45、`FIG`@46 | 47、48 |
| 17 | `verify_delta_transmission.py` | `FACT`@81、`EV`@82、`RES`@83、`FIG`@84 | **全无** |

> 表 2　17 个脚本的路径常量定义位置与 `mkdir` 位置

表 2 是周一动工时的逐行改单。第 4 行与第 17 行是两个极端：`build_tr_factors.py` 一个脚本
定义了五个常量却一个目录都不建，`verify_delta_transmission.py` 定义四个也一个不建；
而第 2、3、16 行连 `mkdir` 成对出现。收敛时须把 `mkdir` 一并统一，否则
「清空 `results/` 后一条命令重建」的验收会栽在目录不存在上。

---

## 三、需要处理的六类问题

### 3.1 五个常量重复定义（主体工作）

`CLEAN` / `FACT` / `RES` / `FIG` 共 44 处定义需删至 0 处，改为 `from common import ...`。
`EV` / `EVENTS` 的处理见 3.3。

> **对计划的更正**：`下周计划_第四阶段_….md` §二 9/28 写的是「`RES` / `FACT` 两个路径常量
> 在 **13 个脚本**里各自重复定义」。实测为 `RES` **10** 个、`FACT` **12** 个、二者并集
> **14** 个。原数字估计偏低，已按实测更正该计划文档。

### 3.2 `RAW` 的遮蔽（唯一的命名冲突）

`common.py:15` 已定义 `RAW = REPO / "raw_data"`。而 `build_tr_factors.py:36` **又定义了一次
同名的 `RAW`**。这不是「另一个常量」，是对既有常量的**遮蔽**——两处当前指向同一路径，
但两份定义各自演进后就会分叉。该脚本应从 `common` 一并 import `RAW`，删掉本地这份。

### 3.3 `EV` 与 `EVENTS` 名字相近、目标不同（**不可直接改名合并**）

| 名字 | 定义处 | 实际指向 | 类型 |
|---|---|---|---|
| `EVENTS` | `adjudicate_outliers.py:20` | `events/risk_events_timeline.csv` | **文件** |
| `EV` | `prep_phase2.py:35`、`verify_delta_transmission.py:82` | `events/` | **目录** |

> 表 3　`EV` 与 `EVENTS` 的指向差异

表 3 是本次盘点里最容易出错的一处：两个名字看着像同一个东西的不同拼写，实际一个指文件、
一个指目录。**切勿把 `EV` 直接重命名为 `EVENTS`** ——`adjudicate_outliers.py` 里那个
`EVENTS` 已经占用了这个名字且指的是 CSV，改名会造成同名异指。正确做法是拆成两个明确的
常量名，例如 `EVENTS_DIR = REPO / "events"` 与 `EVENTS_CSV = REPO / "events" / "risk_events_timeline.csv"`，
三处引用各按自己实际需要取用。

### 3.4 `var_backtest.py` 的三处内联路径 + 元组赋值

该脚本是唯一一处不按「一常量一行」写的：

- `:53` `RES, FIG = REPO / "results", REPO / "figures"` —— **同行元组赋值**。
  按行首 `[A-Z_]+\s*=` 的正则扫描会整行漏掉，是本次盘点第一遍就漏掉的一处。
- `:115` `EVT = pd.read_csv(REPO / "events" / "risk_events_timeline.csv", ...)` —— 内联，
  与 `adjudicate_outliers.py:20` 的 `EVENTS` 指向同一个文件、却各写一份。
- `:116` `B31 = pd.read_csv(REPO / "factors" / "factor_table_3141HK.csv", ...)` —— 内联。

### 3.5 `mkdir` 的缺失与两种写法

**写法不统一**：18 处 `mkdir` 里，14 处写作 `mkdir(parents=True, exist_ok=True)`，
4 处写作 `mkdir(exist_ok=True)`（`stress_impact.py:172`、`stress_scenarios.py:96`、
`stress_robustness.py:111`、`var_backtest.py:54`）。后者不带 `parents=True`，
当前能跑通只是因为 `REPO` 必然存在、`results`/`figures` 的父目录一定在；
一旦 Day 1 的 `--out-dir` 参数把输出指到仓库外的新路径，这四处会直接
`FileNotFoundError`。收敛时统一为带 `parents=True` 的写法。

**缺失面**：`RES` 10 处定义里 8 处不建目录、`CLEAN` 8 处定义里 7 处不建。
现状不报错是**靠执行顺序**——`clean_data.py`（建 `CLEAN`）与 `prep_phase2.py`（建 `RES`）
在链路里先跑，后面的脚本才写盘。这是隐式的顺序依赖，不是设计。

### 3.6 `var_backtest.py` 无 `__main__` 守卫（对一键入口的硬约束）

24 个脚本里 21 个有 `if __name__ == "__main__":`；`common.py`、`var_common.py` 是纯模块，
本就不需要；**唯独 `var_backtest.py` 两者皆无**——它没有 `main()` 函数，
`:110` 之后的测算语句是**裸的顶层代码**。

后果有二。其一，一键入口**只能以子进程方式调用它**，不能 `import`（一旦 import 就会
立即触发整条回测、写盘并画图）。其二，它目前之所以没出事，只是因为**仓库里没有任何
模块 import 它**（已 grep 确认 0 处）——这是一个尚未触发的隐患，不是无害的写法。
Day 1 写 `run_all.py` 时须同时给它补上 `main()` + `__main__` 守卫，否则 `run_all.py`
无论怎么组织都会踩到。

---

## 四、收敛方案

`code/common.py` 由现有两行扩展为路径常量的唯一来源：

```python
REPO: Path = Path(__file__).resolve().parent.parent
RAW: Path = REPO / "raw_data"
CLEAN: Path = REPO / "clean_data"
FACT: Path = REPO / "factors"
RES: Path = REPO / "results"
FIG: Path = REPO / "figures"
EVENTS_DIR: Path = REPO / "events"
EVENTS_CSV: Path = EVENTS_DIR / "risk_events_timeline.csv"
```

同时在 `common.py` 里提供一个建目录的辅助函数（如 `ensure_dirs(*paths)`），
把 18 处散落的 `mkdir` 收到一处，`parents=True, exist_ok=True` 写死一次。
`run_all.py` 在启动时先建全部输出目录，脚本自身不再各自建目录——
这样「清空 `results/`」后的重建不再依赖执行顺序。

**`download_*.py` 与 `clean_data.py` 的 `RAW` / `CLEAN` 归并**：5 个下载脚本目前
`from common import RAW, REPO` 或 `from common import START, save_csv`，
本身不定义路径常量，无需改动。

---

## 五、改动清单

| 动作 | 文件数 | 具体 |
|---|---|---|
| `common.py` 增补常量 + `ensure_dirs` | 1 | 新增 6 个常量与 1 个函数 |
| 删本地定义改 import | 17 | 共 44 处定义 + 1 处 `RAW` 遮蔽 |
| `EV` / `EVENTS` 拆分归一 | 3 | `adjudicate_outliers.py:20`、`prep_phase2.py:35`、`verify_delta_transmission.py:82` |
| 内联路径改走常量 | 3 | `var_backtest.py:115`、`:116`；`descriptive_stats.py:132`、`var_historical.py:277` |
| 元组赋值拆开 | 1 | `var_backtest.py:53` |
| `mkdir` 收敛 | 18 行 | 删除全部模块级 `mkdir`，改由 `common.ensure_dirs` 统一 |
| 补 `main()` + 守卫 | 1 | `var_backtest.py` |

> 表 4　收敛动作与涉及文件数

表 4 里最需要留意的是「内联路径」一行：除 `var_backtest.py` 的两处外，另有两个脚本
在**函数体内**用内联路径读文件——`descriptive_stats.py:132` 读
`clean_data/master_calendar.csv`（该脚本 `:27` 附近已定义 `FACT`/`FIG`，却没有 `CLEAN`）、
`var_historical.py:277` 读 `clean_data/outlier_judgment.csv`（定义了 `FACT`/`RES`/`FIG`，
同样没有 `CLEAN`）。这两处按行首正则扫描扫不到，只有把函数体一起过一遍才会现形，
是继 `var_backtest.py:53` 之后**第二处会被漏掉的内联路径**。

---

## 六、验收

1. `grep -c "^[A-Z_]* = REPO /" code/*.py` 在 `common.py` 之外返回 0；
2. `grep -rn 'REPO /' code/*.py` 只应命中 `common.py` 内部；
3. `python -c "import var_backtest"` 不产生任何输出、不写任何文件（守卫生效）；
4. 清空 `results/`、`figures/` 后一条命令重建全部产物，连跑两次逐字节一致；
5. 全部硬断言仍通过（`stress_scenarios.py` 的 log 收益可加性、`stress_impact.py` 的
   HS250 四配置对账与情景完整性、`stress_robustness.py` 的 1 日口径一致性三组）。

### 第 4 条的实测结果与一处修正

**修正**：重建命令**不能**加 `--only phase2,phase3`。原验收条件写的是
「清空 `results/` 后 `python code/run_all.py --only phase2,phase3` 一条命令重建全部产物」，
但实测 `results/baseline_var_tr.csv` 由**阶段一**脚本 `build_tr_factors.py` 产出
（写盘点在 `code/build_tr_factors.py:126`）。加上 `--only phase2,phase3` 后该文件不会被重建，
而 `var_backtest.py` 依赖它——重建必然半途失败。正确命令是全量 `python code/run_all.py`
（取数默认跳过，不影响离线复现）。

**实测（2026-09-21）**：

| 检查 | 结果 |
|---|---|
| 清空 `results/*.csv` + `figures/*.png` 后 `python code/run_all.py` | 退出码 0；17 步计算成功 / 2 步取数跳过；耗时 51.5s |
| `check_outputs` 对账 | 26/26 全部一致（阶段二 17 + 阶段三 9） |
| 产物数 | 26 CSV + 22 PNG = 48 |
| **与重构前基线比**（`shasum -a 256 -c`，基线取自重构前落盘产物） | **48/48 逐字节相同**——重构行为等价，无一字节改动 |
| 连跑第二次 | 48/48 逐字节一致——脚本可确定性复现 |

「与重构前基线逐字节相同」是本轮最有力的一条证据：它同时排除了「路径改错写到别处」
「`ensure_dirs()` 误建目录导致输出分叉」「包装 `main()` 时漏搬或重复执行语句」三类改坏方式。
上一版 `var_backtest.py` 的包装缺陷正是**没被这一条拦住**的——成因是当时只用
`shasum -c` 比对了**已存在**的文件，而脚本实际什么都没写、退出码仍为 0；
补上「清空后重建」这一步才暴露。教训：验收必须**先清空再重建**再比对，
在已有产物上做校验，跳过执行的一步与真正跑通的一步无法区分。

---

## 七、本次盘点确认**不需要**处理的两项

盘点同时排除了两类改动，记录下来以免周一动工时重复排查：

- **相对路径字面量：0 处**。已 grep 全部 `read_csv` / `to_csv` / `savefig` / `open` /
  `write_table` / `save_csv` 调用的第一个参数，无一处使用裸字符串路径，
  全部经 `REPO` 派生的常量或内联 `REPO / ...` 表达。这意味着 Day 1 的
  `--out-dir` 改造是**可行的**——所有写盘点已经统一锚在 `REPO` 之下，
  改一处 `RES` 即可全局改道。
- **脚本间的模块依赖：仅 1 处**。`stress_robustness.py:106` 的
  `from stress_impact import CR_2022_RESID, horizon_es` 是全仓库唯一的真模块依赖，
  且 `stress_impact.py` 的模块级代码只有导入与常量定义（`main()` 有守卫），
  import 安全。其余全部脚本彼此独立，一键入口可按文件名排定顺序、逐个以子进程调用。
