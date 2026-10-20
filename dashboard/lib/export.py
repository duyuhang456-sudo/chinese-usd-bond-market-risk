"""把页面上已加载的产物表导出成下载字节。

只在内存里编码，绝不落盘——看板的只读契约不变（不写 results/ factors/ 等
任何产物目录）。utf-8-sig 让 Excel 直接双击打开时中文列名不乱码。
"""

from __future__ import annotations

import pandas as pd


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
