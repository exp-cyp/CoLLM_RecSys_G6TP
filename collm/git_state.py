"""Require the exact clean source commit to be pushed before a real experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_output(*arguments, root=ROOT):
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def require_published(root=ROOT):
    """Use local remote-tracking refs; no network or credential prompt at Train."""
    if git_output("status", "--porcelain", "--untracked-files=normal", root=root):
        raise RuntimeError("โค้ดยังมีไฟล์ที่ไม่ได้ commit: บันทึกและ push โค้ดก่อนเริ่ม Train")
    try:
        branch = git_output("symbolic-ref", "--short", "HEAD", root=root)
        upstream = git_output("rev-parse", "--abbrev-ref", "@{upstream}", root=root)
        commit = git_output("rev-parse", "HEAD", root=root)
        published = git_output("rev-parse", "@{upstream}", root=root)
    except RuntimeError as error:
        raise RuntimeError("Git branch ยังไม่มี upstream ที่ใช้ยืนยันการ push") from error
    if commit != published:
        raise RuntimeError("โค้ดเวอร์ชันปัจจุบันยังไม่ตรงกับ upstream: push/sync Git ก่อนเริ่ม Train")
    return {"commit": commit, "branch": branch, "upstream": upstream}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify clean, published source only")
    parser.parse_args()
    try:
        print(json.dumps(require_published(), ensure_ascii=False))
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"{error}\n")
