# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
STATUS_FILE = PLUGIN_DIR / "install_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_status() -> dict:
    return {
        "schema_version": 1,
        "updated_at": _now(),
        "envs": {
            "gpu": {},
            "cpu": {},
        },
        "optional": {
            "table_p1": {},
            "formula_p1": {},
        },
    }


def load_status(path: Path = STATUS_FILE) -> dict:
    if not path.exists():
        return _default_status()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = _default_status()
    data.setdefault("schema_version", 1)
    data.setdefault("updated_at", _now())
    data.setdefault("envs", {})
    data["envs"].setdefault("gpu", {})
    data["envs"].setdefault("cpu", {})
    data.setdefault("optional", {})
    data["optional"].setdefault("table_p1", {})
    data["optional"].setdefault("formula_p1", {})
    return data


def save_status(data: dict, path: Path = STATUS_FILE) -> None:
    data["updated_at"] = _now()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _truthy_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def mark_env(
    env_name: str,
    status: str,
    *,
    backend: str = "",
    python_version: str = "",
    providers: list[str] | None = None,
    models: str = "",
    error: str = "",
    imports: dict[str, bool] | None = None,
    path: Path = STATUS_FILE,
) -> dict:
    data = load_status(path)
    entry = data["envs"].setdefault(env_name, {})
    entry.update(
        {
            "status": status,
            "backend": backend,
            "python_version": python_version,
            "providers": providers or [],
            "models": models,
            "error": error,
            "imports": imports or {},
            "updated_at": _now(),
        }
    )
    save_status(data, path)
    return data


def mark_optional(name: str, status: str, *, error: str = "", detail: str = "", path: Path = STATUS_FILE) -> dict:
    data = load_status(path)
    entry = data["optional"].setdefault(name, {})
    entry.update(
        {
            "status": status,
            "error": error,
            "detail": detail,
            "updated_at": _now(),
        }
    )
    save_status(data, path)
    return data


def check_env(env_name: str, *, backend: str = "", models: str = "", path: Path = STATUS_FILE) -> tuple[dict, int]:
    imports = {"paddle": False, "paddleocr": False, "onnxruntime": False}
    providers: list[str] = []
    python_version = sys.version.split()[0]
    error = ""
    code = 0
    try:
        import paddle  # noqa: F401

        imports["paddle"] = True
    except Exception as exc:
        error = f"missing paddle: {exc}"
        code = 2
    try:
        import paddleocr  # noqa: F401

        imports["paddleocr"] = True
    except Exception as exc:
        error = error or f"missing paddleocr: {exc}"
        code = code or 3
    try:
        import onnxruntime as ort  # noqa: F401

        imports["onnxruntime"] = True
        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:
            error = error or f"provider probe failed: {exc}"
            code = code or 4
    except Exception as exc:
        error = error or f"missing onnxruntime: {exc}"
        code = code or 5

    status = "complete" if all(imports.values()) else "failed"
    data = mark_env(
        env_name,
        status,
        backend=backend,
        python_version=python_version,
        providers=providers,
        models=models,
        error=error,
        imports=imports,
        path=path,
    )
    return data, code


def summarize_status(path: Path = STATUS_FILE) -> str:
    data = load_status(path)
    lines = [
        f"status_file={path}",
        f"updated_at={data.get('updated_at', '')}",
    ]
    for env_name in ("gpu", "cpu"):
        entry = data["envs"].get(env_name, {})
        imports = entry.get("imports") or {}
        lines.append(
            "env.{name}: status={status} backend={backend} py={py} "
            "providers={providers} models={models} imports[paddle={paddle},paddleocr={paddleocr},onnxruntime={onnx}] error={error}".format(
                name=env_name,
                status=entry.get("status", ""),
                backend=entry.get("backend", ""),
                py=entry.get("python_version", ""),
                providers=_truthy_text(entry.get("providers")),
                models=entry.get("models", ""),
                paddle=imports.get("paddle", ""),
                paddleocr=imports.get("paddleocr", ""),
                onnx=imports.get("onnxruntime", ""),
                error=entry.get("error", ""),
            )
        )
    table = data["optional"].get("table_p1", {})
    formula = data["optional"].get("formula_p1", {})
    lines.append(
        "optional.table_p1: status={status} detail={detail} error={error}".format(
            status=table.get("status", ""),
            detail=table.get("detail", ""),
            error=table.get("error", ""),
        )
    )
    lines.append(
        "optional.formula_p1: status={status} detail={detail} error={error}".format(
            status=formula.get("status", ""),
            detail=formula.get("detail", ""),
            error=formula.get("error", ""),
        )
    )
    return "\n".join(lines)


def build_init_fail_message(configs: dict, raw_error: object, path: Path = STATUS_FILE) -> str:
    raw = str(raw_error or "")
    lower = raw.lower()
    data = load_status(path)
    engine = str(configs.get("engine") or "onnxruntime").strip().lower()
    envs = data.get("envs", {})
    gpu = envs.get("gpu") or {}
    cpu = envs.get("cpu") or {}

    def _is_complete(entry: dict) -> bool:
        return entry.get("status") == "complete"

    def _models_ready(entry: dict) -> bool:
        return entry.get("models") == "ready"

    def _has_cuda(entry: dict) -> bool:
        return "CUDAExecutionProvider" in (entry.get("providers") or [])

    if not _is_complete(gpu) and not _is_complete(cpu):
        return (
            "依赖安装未完成：未发现可用的 OCR 运行环境。\n"
            "请先运行 setup.bat，或双击 repair_install.bat 选择“检查/修复基础 OCR 环境”。\n"
            f"原始错误：{raw}"
        )

    if "no module named 'onnxruntime'" in lower or "missing onnxruntime" in lower:
        return (
            "推理后端未安装：当前环境缺少 onnxruntime / onnxruntime-gpu。\n"
            "请运行 repair_install.bat 修复基础 OCR 环境，或重跑 setup.bat 完成推理后端安装。\n"
            f"原始错误：{raw}"
        )

    if "no module named 'paddleocr'" in lower or "missing paddleocr" in lower:
        return (
            "依赖安装未完成：当前环境缺少 paddleocr。\n"
            "请运行 repair_install.bat 修复基础 OCR 环境，或重跑 setup.bat。\n"
            f"原始错误：{raw}"
        )

    if engine == "onnxruntime-gpu" and not _has_cuda(gpu):
        return (
            "Provider 不可用：已安装 GPU 路线，但当前环境没有可用的 CUDAExecutionProvider。\n"
            "可改用 ONNX CPU，或运行 repair_install.bat / setup.bat 重装 1.26.0 + CUDA 12.9。\n"
            f"原始错误：{raw}"
        )

    if ("inference.json" in lower or "no such file or directory" in lower or "cannot open file" in lower):
        return (
            "模型缺失或损坏：运行环境已存在，但模型文件未准备完整。\n"
            "请运行 repair_install.bat 先检查状态；必要时重跑 setup.bat 重新预下载模型。\n"
            f"原始错误：{raw}"
        )

    if engine in ("onnxruntime", "onnxruntime-gpu") and not (_models_ready(gpu) or _models_ready(cpu)):
        return (
            "模型可能未准备完成：依赖已装好，但模型预下载状态不是 ready。\n"
            "首次识别会自动下载；若当前仍失败，请运行 repair_install.bat 检查或重跑 setup.bat。\n"
            f"原始错误：{raw}"
        )

    return (
        "OCR 初始化异常：依赖与状态记录未指向明确的缺依赖/缺模型/Provider 问题。\n"
        "请先运行 repair_install.bat 查看状态摘要，再决定是否重跑 setup.bat。\n"
        f"原始错误：{raw}"
    )


def _cmd_summary(args: argparse.Namespace) -> int:
    print(summarize_status(Path(args.status_file or STATUS_FILE)), flush=True)
    return 0


def _cmd_mark_env(args: argparse.Namespace) -> int:
    imports = {
        "paddle": args.import_paddle,
        "paddleocr": args.import_paddleocr,
        "onnxruntime": args.import_onnxruntime,
    }
    mark_env(
        args.env,
        args.status,
        backend=args.backend,
        python_version=args.python_version,
        providers=list(args.provider or []),
        models=args.models,
        error=args.error,
        imports=imports,
        path=Path(args.status_file or STATUS_FILE),
    )
    print(summarize_status(Path(args.status_file or STATUS_FILE)), flush=True)
    return 0


def _cmd_mark_optional(args: argparse.Namespace) -> int:
    mark_optional(
        args.name,
        args.status,
        error=args.error,
        detail=args.detail,
        path=Path(args.status_file or STATUS_FILE),
    )
    print(summarize_status(Path(args.status_file or STATUS_FILE)), flush=True)
    return 0


def _cmd_check_env(args: argparse.Namespace) -> int:
    _, code = check_env(
        args.env,
        backend=args.backend,
        models=args.models,
        path=Path(args.status_file or STATUS_FILE),
    )
    print(summarize_status(Path(args.status_file or STATUS_FILE)), flush=True)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", default=str(STATUS_FILE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    summary = sub.add_parser("summary")
    summary.set_defaults(func=_cmd_summary)

    mark_env_parser = sub.add_parser("mark-env")
    mark_env_parser.add_argument("--env", choices=("gpu", "cpu"), required=True)
    mark_env_parser.add_argument("--status", required=True)
    mark_env_parser.add_argument("--backend", default="")
    mark_env_parser.add_argument("--python-version", default="")
    mark_env_parser.add_argument("--provider", action="append")
    mark_env_parser.add_argument("--models", default="")
    mark_env_parser.add_argument("--error", default="")
    mark_env_parser.add_argument("--import-paddle", action="store_true")
    mark_env_parser.add_argument("--import-paddleocr", action="store_true")
    mark_env_parser.add_argument("--import-onnxruntime", action="store_true")
    mark_env_parser.set_defaults(func=_cmd_mark_env)

    check_env_parser = sub.add_parser("check-env")
    check_env_parser.add_argument("--env", choices=("gpu", "cpu"), required=True)
    check_env_parser.add_argument("--backend", default="")
    check_env_parser.add_argument("--models", default="")
    check_env_parser.set_defaults(func=_cmd_check_env)

    mark_optional_parser = sub.add_parser("mark-optional")
    mark_optional_parser.add_argument("--name", default="table_p1")
    mark_optional_parser.add_argument("--status", required=True)
    mark_optional_parser.add_argument("--detail", default="")
    mark_optional_parser.add_argument("--error", default="")
    mark_optional_parser.set_defaults(func=_cmd_mark_optional)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
