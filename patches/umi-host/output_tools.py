# 从data中提取、拼接文本
def getDataText(data):
    textOut = ""
    l = len(data) - 1
    for i, tb in enumerate(data):
        textOut += tb["text"]
        if i < l:
            textOut += tb["end"]
    return textOut


def extract_table_from_data(data):
    """从 textBlocks 列表提取 table 字典；没有则 None。"""
    if not data or not isinstance(data, list):
        return None
    for tb in data:
        if not isinstance(tb, dict):
            continue
        if "_table" in tb:
            return tb["_table"]
        if tb.get("table"):
            return tb["table"]
    return None


def ensure_geometry_table_on_res(res, enabled=False, table_builder=None):
    """
    在普通 tbpu 改写 textBlocks 前，为结构导出保留原始几何表。

    结构模型已经给出 ``res["table"]`` 时绝不覆盖；未请求 tableCsv 时
    也不增加默认 OCR 的工作量。table_builder 作为公开 seam 注入，生产
    环境传 ``table_grid.build_table``，测试可使用轻量替身。
    """
    if (
        not enabled
        or not isinstance(res, dict)
        or res.get("code") != 100
        or res.get("table") is not None
        or not isinstance(res.get("data"), list)
    ):
        return res
    if table_builder is None:
        from ..tbpu.parser_tools.table_grid import build_table

        table_builder = build_table
    table = dict(table_builder(res["data"]))
    table["source"] = "geometry_auto"
    res["table"] = table
    return res


def promote_table_on_res(res, tbpu_list=None):
    """
    将表格结构提升到 res['table']（ADR-001 D4）。
    tbpu_list: msnInfo['tbpu']，优先读 parser.last_table。
    已由结构模型或 tableCsv 原始几何入口选定的 table 优先；提升过程
    不再二次 normalize，遵守 ADR-001 D5 的单一事实来源。
    """
    if not isinstance(res, dict) or res.get("code") != 100:
        return res
    table = res.get("table")
    if table is None and tbpu_list:
        for tbpu in tbpu_list:
            t = getattr(tbpu, "last_table", None)
            if t is not None and t.get("cells") is not None:
                table = t
                break
    if table is None:
        table = extract_table_from_data(res.get("data"))
    if table is not None:
        res["table"] = table
        data = res.get("data")
        if isinstance(data, list):
            for tb in data:
                if isinstance(tb, dict) and "_table" in tb:
                    try:
                        del tb["_table"]
                    except Exception:
                        pass
    return res
