# -*- coding: utf-8 -*-
"""检查或预下载 P1 TableRecognitionPipelineV2 可选模型。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "paddlex"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def dependency_status() -> dict:
    import lxml
    import openpyxl
    import paddleocr
    import paddlex
    import scipy
    import sklearn
    from paddleocr import TableRecognitionPipelineV2

    del TableRecognitionPipelineV2
    return {
        "paddleocr": paddleocr.__version__,
        "paddlex": paddlex.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "lxml": lxml.__version__,
        "openpyxl": openpyxl.__version__,
        "cache": str(CACHE),
    }


def engine_kwargs() -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"engine": "paddle"}
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    return {
        "engine": "onnxruntime",
        "engine_config": {"providers": providers},
    }


def download_models() -> dict:
    from paddleocr import TableRecognitionPipelineV2

    kwargs = engine_kwargs()
    TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=False,
        **kwargs,
    )
    model_root = CACHE / "official_models"
    models = sorted(
        path.name
        for path in model_root.iterdir()
        if path.is_dir()
        and any(
            marker in path.name
            for marker in (
                "table_cls",
                "SLANeXt_wired",
                "SLANeXt_wireless",
                "wired_table_cell_det",
                "wireless_table_cell_det",
            )
        )
    )
    if len(models) < 5:
        raise RuntimeError(
            f"表格模型缓存不完整：期望至少 5 个，实际 {len(models)} 个"
        )
    return {"engine": kwargs["engine"], "models": models}


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="仅检查可选依赖")
    modes.add_argument("--download", action="store_true", help="下载并校验模型")
    args = parser.parse_args()

    try:
        result = {"dependencies": dependency_status()}
        if args.download:
            result["download"] = download_models()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
