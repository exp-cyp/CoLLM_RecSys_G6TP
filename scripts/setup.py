"""Create the local environment without changing PowerShell execution policy."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["xpu", "cpu"], default="xpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(root / ".cache/uv")
    env["UV_PYTHON_INSTALL_DIR"] = str(root / ".tools/python")
    env["UV_PYTHON_BIN_DIR"] = str(root / ".tools/bin")
    uv = root / ".tools/uv-package/bin/uv.exe"

    def run(*command):
        subprocess.run([str(value) for value in command], cwd=root, env=env, check=True)

    if not uv.exists():
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            ".tools/uv-package",
            "uv==0.13.6",
        )
    python = root / ".venv/Scripts/python.exe"
    if not python.exists():
        run(uv, "python", "install", "3.11")
        run(uv, "venv", "--python", "3.11", ".venv")
    # The local version suffix matters: torch==2.8.0 alone also accepts an already
    # installed 2.8.0+xpu build when the user requests the CPU wheel (and vice versa).
    run(
        uv,
        "pip",
        "install",
        "--python",
        python,
        f"torch==2.8.0+{args.device}",
        "--index-url",
        f"https://download.pytorch.org/whl/{args.device}",
    )
    lock_path = root / "requirements-lock.txt"
    install_args = [uv, "pip", "install", "--python", python, "-r", "requirements.txt"]
    if lock_path.exists():
        # Constraints pin dependencies that are actually required; they do not
        # install the XPU-only runtimes on a CPU setup. Keep the chosen torch build.
        constraints = root / ".cache/setup-constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        lines = lock_path.read_text(encoding="utf-8-sig").splitlines()
        lines = [
            line
            for line in lines
            if line.strip()
            and not line.startswith(("#", "-e "))
            and line.partition("==")[0].strip().lower() != "torch"
        ]
        constraints.write_text("\n".join(lines) + "\n", encoding="utf-8")
        install_args.extend(["--constraint", constraints])
    run(*install_args)
    run(uv, "pip", "install", "--python", python, "--no-deps", "-e", ".")
    run(python, "-m", "collm.assets", "probe")


if __name__ == "__main__":
    main()
