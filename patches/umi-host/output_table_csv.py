# 输出「结构表格」CSV（真·行列），utf-8-sig，供 Excel 直接打开
# 与清单型 OutputCsv（Name/OCR/Path）分离，键名 tableCsv。

import csv

from umi_log import logger
from .output import Output
from .tools import getDataText

try:
    from ..tbpu.parser_tools.table_grid import (
        cells_from_plain_text,
    )
except ImportError:  # 测试脚本按文件加载时无包上下文
    from ocr.tbpu.parser_tools.table_grid import (  # type: ignore
        cells_from_plain_text,
    )


class OutputTableCsv(Output):
    def __init__(self, argd):
        self.dir = argd["outputDir"]
        self.fileName = argd["outputFileName"]
        self.outputPath = f"{self.dir}/{self.fileName}_table.csv"
        self.ignoreBlank = argd["ignoreBlank"]
        self.tables = []  # list of (title, cells)

    def _cells_from_res(self, res):
        """
        已选定的结构表是唯一事实源，导出时只复制并序列化。
        只有完全没有结构时才明确降级为文本重建。
        """
        table = res.get("table")
        if table and table.get("cells"):
            return [list(r) for r in table["cells"]]
        if res.get("code") == 100 and isinstance(res.get("data"), list):
            text = getDataText(res["data"])
            # Explicit text fallback: no selected structural table was supplied.
            if text and ("\t" in text or "\n" in text):
                return cells_from_plain_text(text)
        return []

    def print(self, res):
        if res.get("code") != 100 and self.ignoreBlank:
            return
        title = res.get("fileName") or res.get("path") or ""
        cells = self._cells_from_res(res)
        if cells:
            self.tables.append((title, cells))

    def onEnd(self):
        if not self.tables:
            logger.info("tableCsv: no tables to write")
            return
        try:
            with open(self.outputPath, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                for i, (title, cells) in enumerate(self.tables):
                    if i > 0:
                        w.writerow([])
                    if title:
                        w.writerow([f"# {title}"])
                    for row in cells:
                        w.writerow([("" if c is None else c) for c in row])
        except Exception as e:
            raise Exception(f"Failed to write table csv. {e}\n写入结构表格csv失败。")
