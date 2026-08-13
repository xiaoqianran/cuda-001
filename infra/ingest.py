# -*- coding: utf-8 -*-
"""把 Modal 产出写入 docs/gallery，并更新 data.json 供 Pages 展示。"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from catalog import BY_ID, CATEGORIES, PROJECTS, category_label

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GALLERY = DOCS / "gallery"
DATA = DOCS / "data.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".webp"}
SKIP_EXTS = {".pfm", ".o", ".so", ".a", ".txt"}
MAX_IMAGES = 12
MAX_EDGE = 1280


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_record(spec: dict) -> dict:
    return {
        "id": spec["id"],
        "title": spec["title"],
        "level": spec["level"],
        "category": spec["category"],
        "category_label": category_label(spec["category"]),
        "stack": spec["stack"],
        "kind": spec["kind"],
        "entry": spec["entry"],
        "status": "pending",
        "elapsed_sec": None,
        "gpu": None,
        "images": [],
        "files": [],
        "log": "",
        "error": None,
        "updated_at": None,
    }


def load_data() -> dict:
    if DATA.exists():
        return json.loads(DATA.read_text(encoding="utf-8"))
    return seed_data()


def seed_data() -> dict:
    return {
        "title": "GPU 加速渲染 · 50 实验",
        "gpu_target": "NVIDIA Tesla T4",
        "updated_at": None,
        "completed": 0,
        "failed": 0,
        "pending": len(PROJECTS),
        "categories": CATEGORIES,
        "projects": [empty_record(p) for p in PROJECTS],
    }


def save_data(data: dict) -> None:
    statuses = [p["status"] for p in data["projects"]]
    data["completed"] = sum(s == "ok" for s in statuses)
    data["failed"] = sum(s == "fail" for s in statuses)
    data["pending"] = sum(s in {"pending", "running"} for s in statuses)
    data["updated_at"] = _now()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _convert_image(src: Path, dest: Path) -> bool:
    from PIL import Image

    try:
        img = Image.open(src)
        img.load()
    except Exception:
        return False
    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB" if img.mode != "L" else "L")
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG", optimize=True)
    return True


def ingest_result(result: dict) -> dict:
    """解包 archive，写 gallery/{id}/，更新 data.json。返回该项目记录。"""
    pid = result["id"]
    spec = BY_ID[pid]
    dest = GALLERY / pid
    dest.mkdir(parents=True, exist_ok=True)

    archive = result.get("archive") or b""
    if archive:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:
                tar.extractall(dest)

    images: list[str] = []
    keep_files: list[str] = []
    names = sorted(p.name for p in dest.iterdir() if p.is_file()) if dest.exists() else []
    if pid == "032":
        keep_frames = {
            n
            for n in names
            if n.startswith(("frame_000", "frame_004", "frame_009", "frame_014", "frame_019"))
        }
        names = [n for n in names if not n.startswith("frame_") or n in keep_frames]
        for extra in dest.iterdir():
            if extra.is_file() and extra.name.startswith("frame_") and extra.name not in keep_frames:
                extra.unlink(missing_ok=True)
    for name in names:
        path = dest / name
        keep_files.append(name)
        ext = path.suffix.lower()
        if ext in SKIP_EXTS or ext not in IMAGE_EXTS:
            continue
        if len(images) >= MAX_IMAGES:
            continue
        png_name = path.stem + ".png"
        png_path = dest / png_name
        if _convert_image(path, png_path) and png_name not in images:
            images.append(png_name)
            if ext in {".ppm", ".pgm"}:
                try:
                    path.unlink()
                except OSError:
                    pass

    rec = empty_record(spec)
    rec.update(
        {
            "status": "ok" if result.get("ok") else "fail",
            "elapsed_sec": result.get("elapsed_sec"),
            "gpu": result.get("gpu"),
            "images": images,
            "files": result.get("files") or keep_files,
            "log": (result.get("log") or "")[-8000:],
            "error": None if result.get("ok") else f"exit {result.get('returncode')}",
            "updated_at": _now(),
        }
    )

    data = load_data()
    by_id = {p["id"]: i for i, p in enumerate(data["projects"])}
    if pid in by_id:
        data["projects"][by_id[pid]] = rec
    else:
        data["projects"].append(rec)
    save_data(data)

    meta = dest / "meta.json"
    meta.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec
