# -*- coding: utf-8 -*-
"""在 Modal 最便宜的 NVIDIA T4 上依次跑 cuda-001 的每个实验。"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/proj/infra")

PROJECT_ROOT = "/proj"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("g++", "make", "ca-certificates")
    .pip_install(
        "numpy==2.1.3",
        "numba==0.61.2",
        "cuda-python==12.2.1",
        "pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "NUMBA_CACHE_DIR": "/tmp/numba_cache",
            "NUMBA_CUDA_USE_NVIDIA_BINDING": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "PYTHONPATH": "/proj:/proj/infra",
        }
    )
    .add_local_dir(
        str(Path(__file__).resolve().parents[1]),
        remote_path=PROJECT_ROOT,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/output/**",
            "docs/gallery/**",
            "**/*.o",
            ".github/**",
        ],
    )
)

app = modal.App("cuda-001-t4", image=image)


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def _pack_output(out_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if not out_dir.exists():
            return buf.getvalue()
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".o", ".so", ".a"}:
                continue
            if path.name in {"bilateral", "fft_filter", "simple_rt", "scene_rt", "mesh_bvh",
                             "aa_dof", "materials", "path_gi", "rt_opt", "volume_mb",
                             "mini_pbrt", "soft_gl", "texture_normal", "shadow_map",
                             "pbr", "deferred", "terrain", "mini_engine"}:
                continue
            tar.add(path, arcname=path.name)
    return buf.getvalue()


def _run(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


@app.function(
    gpu="T4",
    timeout=45 * 60,
    memory=8192,
    cpu=2.0,
)
def run_project(spec: dict) -> dict:
    """在 T4 上跑单个实验，返回日志 + output 压缩包。"""
    pid = spec["id"]
    cwd = Path(PROJECT_ROOT) / pid
    out_dir = cwd / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_ARCH"] = "-arch=sm_75"

    gpu_info = _nvidia_smi()
    header = (
        f"project={pid} title={spec['title']}\n"
        f"kind={spec['kind']} entry={spec['entry']}\n"
        f"gpu={gpu_info}\n"
        f"cwd={cwd}\n"
    )
    log_parts = [header]
    t0 = time.perf_counter()
    rc = 0

    try:
        if spec["kind"] in {"cuda", "cpp"}:
            rc_make, make_log = _run(["make", "-j2"], cwd, env, timeout=600)
            log_parts.append("----- make -----\n" + make_log)
            if rc_make != 0:
                rc = rc_make
            else:
                cmd = [f"./{spec['entry']}"] + list(spec.get("args") or [])
                rc, run_log = _run(cmd, cwd, env, timeout=2400)
                log_parts.append("----- run -----\n" + run_log)
        else:
            cmd = ["python3", "-X", "faulthandler", spec["entry"]]
            rc, run_log = _run(cmd, cwd, env, timeout=2400)
            log_parts.append("----- run -----\n" + run_log)
    except subprocess.TimeoutExpired as exc:
        rc = 124
        log_parts.append(f"TIMEOUT: {exc}\n")
    except Exception as exc:
        rc = 1
        log_parts.append(f"EXCEPTION: {type(exc).__name__}: {exc}\n")

    elapsed = time.perf_counter() - t0
    archive = _pack_output(out_dir)
    files = []
    if out_dir.exists():
        files = sorted(p.name for p in out_dir.iterdir() if p.is_file())

    return {
        "id": pid,
        "ok": rc == 0,
        "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "gpu": gpu_info,
        "log": "".join(log_parts)[-120_000:],
        "files": files,
        "archive": archive,
    }


@app.local_entrypoint()
def main(pid: str = "001"):
    """modal run infra/modal_app.py --pid 001"""
    from catalog import BY_ID

    result = run_project.remote(BY_ID[pid])
    print(f"[{result['id']}] ok={result['ok']} rc={result['returncode']} "
          f"{result['elapsed_sec']}s files={result['files']}")
    print(result["log"][-4000:])
