# -*- coding: utf-8 -*-
"""PaddleX 官方模型下载源：国内优先，Hugging Face 最后备用。"""
from __future__ import annotations

import os


# 必须在 import paddleocr / paddlex 前设置，才会覆盖 PaddleX 的默认 huggingface。
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")


def configure_domestic_model_sources() -> None:
    """将 PaddleX 的实际回退顺序固定为国内三站优先、HF 最后。

    PaddleX 3.7.2 默认候选列表把 Hugging Face 放在首位；仅设置
    PADDLE_PDX_MODEL_SOURCE=modelscope 只能改变首选项，不能让 HF 位于
    所有国内站点之后。因此在 PaddleX 已导入、首次加载模型前重排候选类。
    """
    try:
        from paddlex.inference.utils import official_models

        official_models._ModelManager.hoster_candidates = [
            official_models._ModelScopeModelHoster,
            official_models._AIStudioModelHoster,
            official_models._BosModelHoster,
            official_models._HuggingFaceModelHoster,
        ]
    except Exception:
        # PaddleX 未安装或未来内部接口变动时，不阻断 OCR；其默认机制仍可下载模型。
        pass
