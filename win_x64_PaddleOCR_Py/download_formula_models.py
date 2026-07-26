# -*- coding: utf-8 -*-
"""检查或预下载 P1 公式识别 Spike 可选模型。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "paddlex"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE))

from install_status import mark_optional
from model_sources import configure_domestic_model_sources

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_MODEL = "PP-FormulaNet_plus-S"


def dependency_status() -> dict:
    import paddleocr
    import paddlex
    from paddleocr import FormulaRecognitionPipeline

    configure_domestic_model_sources()
    del FormulaRecognitionPipeline
    return {
        "paddleocr": paddleocr.__version__,
        "paddlex": paddlex.__version__,
        "cache": str(CACHE),
        "default_model": DEFAULT_MODEL,
    }


def engine_kwargs() -> dict:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return {"engine": "paddle"}
    return {
        "engine": "onnxruntime",
        "engine_config": {"providers": ["CPUExecutionProvider"]},
    }


def _download_pipeline(*, mode: str, model_name: str) -> dict:
    from paddleocr import FormulaRecognitionPipeline

    configure_domestic_model_sources()
    kwargs = engine_kwargs()
    FormulaRecognitionPipeline(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=(mode == "layout"),
        formula_recognition_model_name=model_name,
        **kwargs,
    )
    model_root = CACHE / "official_models"
    models = sorted(path.name for path in model_root.iterdir() if path.is_dir())
    formula_models = [name for name in models if "FormulaNet" in name]
    layout_models = [name for name in models if "DocLayout" in name]
    if not formula_models:
        raise RuntimeError("公式模型缓存为空")
    if mode == "layout" and not layout_models:
        raise RuntimeError("混排模式缺少版面模型缓存")
    return {
        "engine": kwargs["engine"],
        "mode": mode,
        "model_name": model_name,
        "formula_models": formula_models,
        "layout_models": layout_models,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="仅检查公式依赖")
    modes.add_argument("--download", action="store_true", help="下载并校验公式模型")
    parser.add_argument(
        "--mode",
        choices=("whole_image", "layout"),
        default="layout",
        help="whole_image=整图公式，layout=混排公式区域（默认）",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="默认 PP-FormulaNet_plus-S；可切 plus-M / plus-L 做对照",
    )
    args = parser.parse_args()

    try:
        result = {"dependencies": dependency_status()}
        if args.download:
            result["download"] = _download_pipeline(
                mode=args.mode, model_name=args.model_name
            )
            mark_optional(
                "formula_p1",
                "complete",
                detail=f"mode={args.mode}; model={args.model_name}",
            )
    except Exception as exc:
        mark_optional(
            "formula_p1",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            detail=f"mode={getattr(args, 'mode', '')}; model={getattr(args, 'model_name', '')}",
        )
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
