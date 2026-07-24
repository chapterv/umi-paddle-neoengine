# 单层纯文本 PDF

import math
import fitz  # PyMuPDF

from umi_log import logger
from .output_pdf_layered import OutputPdfLayered


class OutputPdfOneLayer(OutputPdfLayered):
    def __init__(self, argd):
        super().__init__(argd)
        self.opacity = 1  # 文本不透明
        self.textDirectionMode = argd.get("pdfOneLayerDirection", "auto")
        if self.textDirectionMode not in ("auto", "horizontal", "vertical"):
            self.textDirectionMode = "auto"
        self.outputPath = f"{self.dir}/{self.fileName}.text.pdf"  # 输出路径

    def _insertVerticalText(self, page, box, text, fontsize, protation):
        """Write visible upright CJK text with the font's vertical metrics."""
        horizontalXref = page.insert_font(
            fontname="cjkv", fontbuffer=self.font.buffer
        )
        topCenter = fitz.Point(
            (box[0][0] + box[1][0]) / 2,
            (box[0][1] + box[1][1]) / 2,
        )
        bottomCenter = fitz.Point(
            (box[3][0] + box[2][0]) / 2,
            (box[3][1] + box[2][1]) / 2,
        )
        longDx = bottomCenter.x - topCenter.x
        longDy = bottomCenter.y - topCenter.y
        longLength = math.hypot(longDx, longDy)
        if longLength <= 0:
            return
        topCenter.x += longDx / longLength * fontsize * 0.12
        topCenter.y += longDy / longLength * fontsize * 0.12
        point = topCenter * page.derotation_matrix
        localAngle = math.degrees(math.atan2(longDx, longDy))
        morph = None
        if abs(localAngle) > 0.01:
            morph = (point, fitz.Matrix(localAngle))
        page.insert_text(
            point,
            text,
            fontsize,
            fontname="cjkv",
            rotate=protation,
            morph=morph,
            stroke_opacity=self.opacity,
            fill_opacity=self.opacity,
        )
        self._bindVerticalFont(page, horizontalXref)

    # 创建空白 PDF
    def _getPDF(self, path):
        source_doc = fitz.open(path)  # 打开原文档
        # 如果已加密，则尝试解密
        if source_doc.is_encrypted and not source_doc.authenticate(self.password):
            raise Exception(
                f'The document is encrypted, and the password "{self.password}" is incorrect.\n文档已加密，输入密码不正确。'
            )
        pdf = fitz.open()  # 创建空白PDF文档对象
        # 复制原始文档的元数据（如作者、标题等）
        meta = source_doc.metadata
        if not meta["producer"]:
            meta["producer"] = "Umi-OCR & PyMuPDF v" + fitz.VersionBind
        if not meta["creator"]:
            meta["creator"] = "Umi-OCR & PyMuPDF PDF converter"
        pdf.set_metadata(meta)
        # 生成空白的每一页
        for page in source_doc:
            rect = page.rect  # 原文档渲染尺寸
            pdf.new_page(width=rect.width, height=rect.height)
        # 尝试复制原始文档的目录数据
        try:
            pdf.set_toc(source_doc.get_toc())
        except Exception:
            logger.warning(
                f"pdf.set_toc error. path: {path}", exc_info=True, stack_info=True
            )
        source_doc.close()  # 释放原文档
        return pdf
