# -*- coding: utf-8 -*-
"""依次在 Modal T4 上跑每个实验，每完成一个就写入 docs/ 并 git push，触发 Pages Action。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import PROJECTS, BY_ID  # noqa: E402
from ingest import ingest_result, load_data, save_data, seed_data  # noqa: E402


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def commit_and_push(pid: str, rec: dict) -> None:
    git("add", "docs")
    status = git("status", "--porcelain", "docs")
    if not status.stdout.strip():
        print(f"[{pid}] docs 无变更，跳过 commit")
        return
    title = rec.get("title", "")
    st = rec.get("status", "?")
    elapsed = rec.get("elapsed_sec")
    msg = f"gallery({pid}): {st} {title} ({elapsed}s on T4)"
    c = git("commit", "-m", msg)
    print(c.stdout)
    if c.returncode != 0:
        print(f"[{pid}] commit 失败")
        return
    p = git("push", "-u", "origin", "HEAD")
    print(p.stdout)
    if p.returncode != 0:
        print(f"[{pid}] push 失败，重试…")
        for wait in (4, 8, 16, 32):
            import time as _t
            _t.sleep(wait)
            p = git("push", "-u", "origin", "HEAD")
            print(p.stdout)
            if p.returncode == 0:
                break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="只跑一个 id，如 001")
    parser.add_argument("--from-id", default="", help="从某个 id 开始（含）")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    if not (ROOT / "docs" / "data.json").exists():
        save_data(seed_data())

    ids = [p["id"] for p in PROJECTS]
    if args.only:
        ids = [args.only]
    elif args.from_id:
        if args.from_id not in ids:
            print(f"unknown id {args.from_id}", file=sys.stderr)
            return 2
        ids = ids[ids.index(args.from_id) :]

    import modal
    from modal_app import app, run_project

    print(f"将在 Modal T4 上依次运行: {', '.join(ids)}")
    with modal.enable_output():
        with app.run():
            for pid in ids:
                print("\n" + "=" * 72)
                print(f"RUN {pid}")
                print("=" * 72)
                data = load_data()
                for p in data["projects"]:
                    if p["id"] == pid:
                        p["status"] = "running"
                save_data(data)

                try:
                    result = run_project.remote(BY_ID[pid])
                except Exception as exc:
                    result = {
                        "id": pid,
                        "ok": False,
                        "returncode": 1,
                        "elapsed_sec": None,
                        "gpu": None,
                        "log": f"modal call failed: {exc}",
                        "files": [],
                        "archive": b"",
                    }

                # archive 很大，ingest 后再丢掉
                rec = ingest_result(result)
                print(
                    f"[{pid}] status={rec['status']} elapsed={rec['elapsed_sec']} "
                    f"images={rec['images']}"
                )
                if rec.get("log"):
                    print(rec["log"][-1500:])
                if not args.no_push:
                    commit_and_push(pid, rec)

    data = load_data()
    print(
        f"\n完成: ok={data['completed']} fail={data['failed']} pending={data['pending']}"
    )
    return 0 if data["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
