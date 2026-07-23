# 双层可搜索 searchable pdf
# https://github.com/pymupdf/PyMuPDF/discussions/2299

import os
import math
import re
import fitz  # PyMuPDF

from umi_log import logger
from .output import Output


class OutputPdfLayered(Output):
    def __init__(self, argd):
        self.dir = argd["outputDir"]  # 输出路径（文件夹）
        self.originPath = argd["originPath"]  # 原始文件路径
        self.fileName = argd["outputFileName"]  # 文件名
        self.password = argd["password"]  # 密码
        self.outputPath = f"{self.dir}/{self.fileName}.layered.pdf"  # 输出路径
        self.pdf = None
        self.existentPages = []  # 已处理的页数
        self.isInsertFont = False  # 是否有字体嵌入
        self.opacity = 0  # 文本透明度为0
        # 双层 PDF 必须始终按 OCR 几何自动决定文字方向。
        self.textDirectionMode = "auto"
        self._verticalFontXref = None
        self._verticalFontPages = set()
        self._logicalFontIndex = 0
        try:
            self.font = fitz.Font("cjk")  # 字体
        except Exception as e:
            raise Exception(f"Failed to load cjk font. {e}\n无法加载cjk字体。")
        try:
            self.pdf = self._getPDF(self.originPath)  # 加载pymupdf对象
        except Exception as e:
            raise Exception(
                f"Failed to load doc file. {e}\n无法加载文档。\n{self.originPath}"
            )

    # 获取pdf文档对象，或将其它类型的文档转为PDF对象
    def _getPDF(self, path):
        # https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/convert-document/convert.py
        doc = fitz.open(path)
        # 如果已加密，则尝试解密
        if doc.is_encrypted and not doc.authenticate(self.password):
            raise Exception(
                f'The document is encrypted, and the password "{self.password}" is incorrect.\n文档已加密，输入密码不正确。'
            )
        if doc.is_pdf:
            return doc
        b = doc.convert_to_pdf()  # 转换为PDF格式的二进制数据
        pdf = fitz.open("pdf", b)  # 创建PDF文档对象
        try:
            pdf.set_toc(doc.get_toc())  # 复制原始文档的目录
        except Exception:
            logger.warning("pdf.set_toc error", exc_info=True, stack_info=True)
        # 复制原始文档的元数据（如作者、标题等）
        meta = doc.metadata
        if not meta["producer"]:
            meta["producer"] = "Umi-OCR & PyMuPDF v" + fitz.VersionBind
        if not meta["creator"]:
            meta["creator"] = "Umi-OCR & PyMuPDF PDF converter"
        pdf.set_metadata(meta)
        # 复制原始文档的链接
        for pinput in doc:
            links = pinput.get_links()
            pout = pdf[pinput.number]
            for link in links:
                if link["kind"] == fitz.LINK_NAMED:  # 不处理 named links
                    continue
                pout.insert_link(link)  # 写入新文档
        doc.close()  # 释放原文档
        return pdf

    # 计算填满宽和高的一行字体大小
    def _calculateFontSize(self, text, w, h):
        if h > w:  # 竖排转为横排计算
            w, h = h, w
        fontsize = round(h)  # 字体大小初值，假设为行高
        minSize = 5  # 大小下限
        getLen = lambda text, s: self.font.text_length(text, fontsize=s)
        while getLen(text, fontsize) > w and fontsize >= minSize:
            fontsize -= 1  # 尝试减小字体，直到行宽刚好小于界限
        while getLen(text, fontsize) < w:
            fontsize += 1  # 尝试增大字体，直到行宽刚好超过界限
        while getLen(text, fontsize) > w and fontsize >= minSize:
            fontsize -= 0.1  # 再次减小字体，将精度提升到 0.1
        return fontsize

    @staticmethod
    def _edgeLength(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _getBoxSize(self, box):
        width = (self._edgeLength(box[0], box[1]) + self._edgeLength(box[3], box[2])) / 2
        height = (self._edgeLength(box[0], box[3]) + self._edgeLength(box[1], box[2])) / 2
        return width, height

    def _isVerticalText(self, text, box):
        if len(text) <= 1:
            return False
        if self.textDirectionMode == "horizontal":
            return False
        if self.textDirectionMode == "vertical":
            return True
        width, height = self._getBoxSize(box)
        return width > 0 and height >= width * 1.5

    def _bindVerticalFont(self, page, horizontalXref):
        if page.number in self._verticalFontPages:
            return
        if self._verticalFontXref is None:
            fontObj = self.pdf.xref_object(horizontalXref, compressed=False)
            if "/Identity-H" not in fontObj:
                raise RuntimeError("CJK font does not expose an Identity-H encoding.")
            self._verticalFontXref = self.pdf.get_new_xref()
            self.pdf.update_object(
                self._verticalFontXref,
                fontObj.replace("/Identity-H", "/Identity-V", 1),
            )

        oldRef = rf"(/cjkv\s+){horizontalXref}\s+0\s+R"
        newRef = rf"\g<1>{self._verticalFontXref} 0 R"
        resourcesType, resourcesValue = self.pdf.xref_get_key(
            page.xref, "Resources"
        )
        if resourcesType == "dict":
            resourcesValue, count = re.subn(oldRef, newRef, resourcesValue, count=1)
            if count != 1:
                raise RuntimeError("Unable to bind vertical font in page resources.")
            self.pdf.xref_set_key(page.xref, "Resources", resourcesValue)
        elif resourcesType == "xref":
            resourcesXref = int(resourcesValue.split()[0])
            fontType, fontValue = self.pdf.xref_get_key(resourcesXref, "Font")
            if fontType == "dict":
                fontValue, count = re.subn(oldRef, newRef, fontValue, count=1)
                if count != 1:
                    raise RuntimeError("Unable to bind vertical font dictionary.")
                self.pdf.xref_set_key(resourcesXref, "Font", fontValue)
            elif fontType == "xref":
                fontXref = int(fontValue.split()[0])
                self.pdf.xref_set_key(
                    fontXref, "cjkv", f"{self._verticalFontXref} 0 R"
                )
            else:
                raise RuntimeError("Page font resources are unavailable.")
        else:
            raise RuntimeError("Page resources are unavailable.")
        self._verticalFontPages.add(page.number)

    def _insertVerticalText(self, page, box, text, fontsize, protation):
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
        # Identity-V 的默认纵向原点距字框顶边约 0.12em。
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
        return page.get_contents()[-1]

    def _newStream(self, data):
        xref = self.pdf.get_new_xref()
        self.pdf.update_object(xref, "<</Length 0>>")
        self.pdf.update_stream(xref, data)
        return xref

    def _bindPageFont(self, page, fontName, fontXref):
        """把低层创建的 Type3 字体加入当前页资源字典。"""
        fontRef = f"/{fontName} {fontXref} 0 R"
        resourcesType, resourcesValue = self.pdf.xref_get_key(
            page.xref, "Resources"
        )
        if resourcesType == "dict":
            if "/Font<<" in resourcesValue:
                resourcesValue = resourcesValue.replace(
                    "/Font<<", f"/Font<<{fontRef} ", 1
                )
            else:
                resourcesValue = resourcesValue[:-2] + f"/Font<<{fontRef}>>>>"
            self.pdf.xref_set_key(page.xref, "Resources", resourcesValue)
            return
        if resourcesType != "xref":
            raise RuntimeError("Page resources are unavailable.")
        resourcesXref = int(resourcesValue.split()[0])
        fontType, fontValue = self.pdf.xref_get_key(resourcesXref, "Font")
        if fontType == "dict":
            fontValue = fontValue[:-2] + f"{fontRef}>>"
            self.pdf.xref_set_key(resourcesXref, "Font", fontValue)
        elif fontType == "xref":
            fontDictXref = int(fontValue.split()[0])
            self.pdf.xref_set_key(
                fontDictXref, fontName, f"{fontXref} 0 R"
            )
        else:
            self.pdf.xref_set_key(
                resourcesXref, "Font", f"<<{fontRef}>>"
            )

    @staticmethod
    def _rawTextLines(page):
        lines = []
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                chars = [
                    char
                    for span in line.get("spans", [])
                    for char in span.get("chars", [])
                ]
                if chars:
                    lines.append(
                        {
                            "text": "".join(char["c"] for char in chars),
                            "chars": chars,
                            "bbox": fitz.Rect(line["bbox"]),
                        }
                    )
        return lines

    @staticmethod
    def _targetRect(page, box):
        points = [
            fitz.Point(point[0], point[1]) * page.derotation_matrix
            for point in box
        ]
        return fitz.Rect(
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    def _matchLogicalChars(self, page, groups):
        """按文字与几何位置把 OCR block 对应到 PyMuPDF 的实际字符框。"""
        lines = self._rawTextLines(page)
        used = set()
        for group in groups:
            for item in group:
                target = self._targetRect(page, item["box"])
                candidates = []
                for index, line in enumerate(lines):
                    start = 0
                    while True:
                        start = line["text"].find(item["text"], start)
                        if start < 0:
                            break
                        end = start + len(item["text"])
                        charKeys = {(index, pos) for pos in range(start, end)}
                        if not (charKeys & used):
                            chars = line["chars"][start:end]
                            charRect = fitz.Rect(
                                min(char["bbox"][0] for char in chars),
                                min(char["bbox"][1] for char in chars),
                                max(char["bbox"][2] for char in chars),
                                max(char["bbox"][3] for char in chars),
                            )
                            distance = (
                                (
                                    charRect.x0
                                    + charRect.x1
                                    - target.x0
                                    - target.x1
                                )
                                ** 2
                                + (
                                    charRect.y0
                                    + charRect.y1
                                    - target.y0
                                    - target.y1
                                )
                                ** 2
                            )
                            candidates.append(
                                (distance, index, start, end, chars)
                            )
                        start += 1
                if not candidates:
                    nearest = sorted(
                        lines,
                        key=lambda line: (
                            line["bbox"].x0
                            + line["bbox"].x1
                            - target.x0
                            - target.x1
                        )
                        ** 2
                        + (
                            line["bbox"].y0
                            + line["bbox"].y1
                            - target.y0
                            - target.y1
                        )
                        ** 2,
                    )[:3]
                    raise RuntimeError(
                        "Unable to locate vertical OCR text geometry: "
                        f"{item['text']!r}; nearest="
                        f"{[line['text'] for line in nearest]!r}"
                    )
                _, index, start, end, chars = min(
                    candidates, key=lambda value: value[0]
                )
                used.update((index, pos) for pos in range(start, end))
                item["chars"] = chars

    @staticmethod
    def _pdfBBox(page, bbox):
        inverse = ~page.transformation_matrix
        points = [
            fitz.Point(bbox[0], bbox[1]) * inverse,
            fitz.Point(bbox[2], bbox[1]) * inverse,
            fitz.Point(bbox[2], bbox[3]) * inverse,
            fitz.Point(bbox[0], bbox[3]) * inverse,
        ]
        return (
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points),
            max(point.y for point in points),
        )

    def _createLogicalFont(self, page, glyphs):
        """创建最多 254 个唯一字形框的 Type3 逻辑字体。"""
        self._logicalFontIndex += 1
        fontName = f"ocrlogical{self._logicalFontIndex}"
        charProcRefs = []
        glyphNames = []
        widths = []
        cmapRows = []
        encoded = bytearray()
        for code, glyph in enumerate(glyphs, 1):
            text, bbox = glyph
            x0, y0, x1, y1 = self._pdfBBox(page, bbox)
            proc = (
                f"0.01 0 {x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f} d1\n"
                f"{x0:.4f} {y0:.4f} {x1 - x0:.4f} {y1 - y0:.4f} re f\n"
            ).encode("ascii")
            procXref = self._newStream(proc)
            glyphName = f"g{code:03d}"
            charProcRefs.append(f"/{glyphName} {procXref} 0 R")
            glyphNames.append(f"/{glyphName}")
            widths.append("0.01")
            cmapRows.append(
                f"<{code:02X}> <{text.encode('utf-16-be').hex().upper()}>"
            )
            encoded.append(code)

        cmap = (
            "/CIDInit /ProcSet findresource begin\n"
            "12 dict begin\n"
            "begincmap\n"
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            f"/CMapName /{fontName} def\n"
            "/CMapType 2 def\n"
            "1 begincodespacerange\n"
            "<00> <FF>\n"
            "endcodespacerange\n"
            f"{len(cmapRows)} beginbfchar\n"
            + "\n".join(cmapRows)
            + "\nendbfchar\n"
            "endcmap\n"
            "CMapName currentdict /CMap defineresource pop\n"
            "end\n"
            "end\n"
        ).encode("ascii")
        cmapXref = self._newStream(cmap)
        fontXref = self.pdf.get_new_xref()
        fontObject = (
            "<</Type/Font/Subtype/Type3"
            f"/Name/{fontName}"
            f"/FontBBox[0 0 {page.mediabox.width:.4f} {page.mediabox.height:.4f}]"
            "/FontMatrix[1 0 0 1 0 0]"
            f"/CharProcs<<{' '.join(charProcRefs)}>>"
            f"/Encoding<</Type/Encoding/Differences[1 {' '.join(glyphNames)}]>>"
            f"/FirstChar 1/LastChar {len(glyphs)}"
            f"/Widths[{' '.join(widths)}]"
            "/Resources<<>>"
            f"/ToUnicode {cmapXref} 0 R>>"
        )
        self.pdf.update_object(fontXref, fontObject)
        self._bindPageFont(page, fontName, fontXref)
        return fontName, encoded

    def _replaceVerticalGroupsWithLogicalText(self, page, groups):
        """双层 PDF：用 OCR 顺序的逻辑段落替换浏览器会倒排的竖栏文本。"""
        if not groups:
            return
        self._matchLogicalChars(page, groups)
        for group in groups:
            glyphs = []
            for item in group:
                glyphs.extend(
                    (char["c"], char["bbox"]) for char in item["chars"]
                )
                logicalEnd = item["end"].replace("\r", "").replace("\n", "")
                if logicalEnd:
                    glyphs.extend(
                        (ending, item["chars"][-1]["bbox"])
                        for ending in logicalEnd
                    )

            textOperations = []
            for start in range(0, len(glyphs), 254):
                fontName, encoded = self._createLogicalFont(
                    page, glyphs[start : start + 254]
                )
                textOperations.append(
                    f"/{fontName} 1 Tf\n<{encoded.hex().upper()}> Tj"
                )
            logicalStream = (
                "\nq\n/fitzca0000 gs\nBT\n1 0 0 1 0 0 Tm\n"
                + "\n".join(textOperations)
                + "\nET\nQ\n"
            ).encode("ascii")
            self.pdf.update_stream(group[0]["contentXref"], logicalStream)
            for item in group[1:]:
                self.pdf.update_stream(item["contentXref"], b"")

    def print(self, res):  # 输出图片结果
        if not self.pdf:
            logger.error("self.pdf 未初始化。")
            return
        pno = res["page"] - 1  # 当前页数
        self.existentPages.append(pno)  # 记录已处理的页面
        if not res["code"] == 100:
            return  # 忽略空白

        page = self.pdf[pno]  # 当前页对象
        page.clean_contents()  # 内容流清理、语法更正，减少错误
        protation = page.rotation  # 获取页面旋转角度
        isInsertFont = False  # 当前是否进行过字体注入
        verticalGroups = []
        currentVerticalGroup = []
        # 插入文本，用shape.insert_text（可编辑）或page.insert_text（不可编辑）
        for tb in res["data"]:
            if self.opacity == 0 and "from" in tb and tb["from"] == "text":
                continue  # 双层（透明文字）模式下，跳过直接提取的文本，只写入OCR文本
            if not isInsertFont:  # 页面插入字体
                self.isInsertFont = isInsertFont = True
                page.insert_font(fontname="cjk", fontbuffer=self.font.buffer)
            text = tb["text"]
            box = tb["box"]
            x0, y0 = box[0]
            x2, y2 = box[2]
            w, h = self._getBoxSize(box)
            isVertical = self._isVerticalText(text, box)
            if isVertical:
                fontsize = self._calculateFontSize(text, w, h)
                contentXref = self._insertVerticalText(
                    page, box, text, fontsize, protation
                )
                if self.opacity == 0:
                    currentVerticalGroup.append(
                        {
                            "text": text,
                            "end": tb.get("end", "") or "",
                            "box": box,
                            "contentXref": contentXref,
                        }
                    )
                    if currentVerticalGroup[-1]["end"]:
                        verticalGroups.append(currentVerticalGroup)
                        currentVerticalGroup = []
                continue
            if currentVerticalGroup:
                verticalGroups.append(currentVerticalGroup)
                currentVerticalGroup = []
            # 横排时还要受插入点右侧页面宽度约束，避免强制横排的竖栏
            # 从页面右边溢出，进而造成复制文本缺字。
            availableWidth = max(page.rect.width - x0, 1)
            fontsize = self._calculateFontSize(
                text,
                min(max(w, h), availableWidth),
                min(w, h),
            )
            # 插入点的 旋转后的坐标
            point = fitz.Point(x0, y2) * page.derotation_matrix
            page.insert_text(
                point,
                text,
                fontsize,
                fontname="cjk",
                rotate=protation,  # 文本角度设定
                stroke_opacity=self.opacity,  # 描边透明度
                fill_opacity=self.opacity,  # 填充（字体）透明度
            )
        if currentVerticalGroup:
            verticalGroups.append(currentVerticalGroup)
        if self.opacity == 0:
            self._replaceVerticalGroupsWithLogicalText(page, verticalGroups)

    def onEnd(self):  # 结束时保存。
        if not self.pdf:
            return
        # 删除未处理的页数
        for i in range(len(self.pdf) - 1, -1, -1):
            if i not in self.existentPages:
                self.pdf.delete_page(i)
        logger.info(f"保存{len(self.pdf)}页PDF：{self.outputPath}")
        if self.isInsertFont:  # 有任意页面嵌入字体，则构建字体子集
            try:  # 对于部分PDF，如用txt直接打印的，构建字体子集会失败。
                self.pdf.subset_fonts()  # 构建字体子集，减小文件大小。需要 fontTools 库
            except Exception:  # TODO: 失败原因？可能文件中实际并没有字体？
                logger.error("构建字体子集失败。", exc_info=True, stack_info=True)
            # 保存：压缩并进行3级垃圾回收。等同 ez_save
            self.save(self.pdf, self.outputPath, deflate=True, garbage=3)
        else:
            # 无嵌入字体，则直接保存，不压缩
            self.save(self.pdf, self.outputPath)

    def save(self, pdf, path, **options):  # 保存并关闭 pdf 对象
        try:
            # 尝试保存到指定路径
            pdf.save(path, **options)
        except Exception:
            # 保存失败，尝试保存到 ".temp" 路径
            tempPath = self.outputPath + ".temp"
            logger.warning(f"保存PDF失败。 path: {path}", exc_info=True)
            try:
                pdf.save(tempPath, **options)
                pdf.close()
            except Exception as e1:
                logger.error(
                    f"保存PDF到临时路径失败。 tempPath: {tempPath}", exc_info=True
                )
                raise Exception(f"[Error] Unable to save PDF to [{tempPath}]: {e1}")
            # 已保存到 .temp 并 close 原对象，尝试替换文件
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tempPath, path)
            except Exception as e2:
                logger.warning(
                    f"保存PDF文件替换失败。保存到临时文件: {tempPath}", exc_info=True
                )

                raise Exception(
                    f"[Warning] Unable to save PDF: [{path}]. Exception: {e2}. Saved to temporary path: [{tempPath}]."
                )
        else:  # 正常结束
            pdf.close()
