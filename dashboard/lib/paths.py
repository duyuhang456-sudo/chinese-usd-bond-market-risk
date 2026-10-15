"""仓库路径解析。

全部路径从本文件位置反推，不依赖当前工作目录——这样从任意目录启动
streamlit 都能读到同一批产物（只有 .streamlit/config.toml 依赖 CWD）。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS = REPO_ROOT / "results"
FACTORS = REPO_ROOT / "factors"
CLEAN_DATA = REPO_ROOT / "clean_data"
EVENTS = REPO_ROOT / "events"
FIGURES = REPO_ROOT / "figures"
DOCS = REPO_ROOT / "docs"


def result(name: str) -> Path:
    """results/ 下的 csv。"""
    return RESULTS / f"{name}.csv"


def factor(name: str) -> Path:
    """factors/ 下的 csv。"""
    return FACTORS / f"{name}.csv"


def clean(name: str) -> Path:
    """clean_data/ 下的 csv。"""
    return CLEAN_DATA / f"{name}.csv"


def require(path: Path) -> Path:
    """产物缺失时给出一句人话，而不是让 pandas 抛裸 FileNotFoundError。"""
    if not path.exists():
        raise FileNotFoundError(
            f"缺少产物 {path}。看板只读既有测算结果，请先在仓库根执行 "
            f"python code/run_all.py 生成全部产物。"
        )
    return path
