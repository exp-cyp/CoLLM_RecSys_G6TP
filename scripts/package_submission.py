"""Package source, configs, documentation, and measured results (no model/data blobs)."""

import argparse
import zipfile
from pathlib import Path
from uuid import uuid4

ROOT_FILES = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "requirements-lock.txt",
    "pyproject.toml",
    "NOTICE.md",
    "START_TRAIN.cmd",
)
SOURCE_FOLDERS = ("collm", "configs", "scripts", "tests", "docs")
EXCLUDED_SUFFIXES = {".pyc", ".zip", ".pt", ".pth", ".bin", ".safetensors"}


def package_project(root, target=None, result_dir=None):
    """Package source and results, excluding downloads, weights, caches and ZIPs.

    A per-run target is never replaced. The default standalone source archive may
    be refreshed deliberately by running this script again.
    """
    root = Path(root).resolve()
    default_target = target is None
    target = (root / (target or "submission/CoLLM_RecSys_G6TP_source.zip")).resolve()
    if not target.is_relative_to(root / "submission") and not target.is_relative_to(
        root / "results"
    ):
        raise ValueError("Archive target must stay inside project submission/ or results/")
    if target.exists() and not default_target:
        raise FileExistsError(f"Refusing to replace an existing run archive: {target}")

    include = [root / name for name in ROOT_FILES]
    for folder in SOURCE_FOLDERS:
        include.extend((root / folder).rglob("*"))
    if result_dir is None:
        include.extend((root / "results").rglob("*"))
    else:
        result_dir = (root / result_dir).resolve()
        if not result_dir.is_relative_to(root / "results") or result_dir == root / "results":
            raise ValueError("Choose one run directory inside project results/")
        include.extend((root / "results").glob("*.json"))
        include.extend(result_dir.rglob("*"))

    files = sorted(
        path
        for path in set(include)
        if path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root)
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
        and path != target
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="New archive under submission/ or results/")
    parser.add_argument(
        "--result-dir", help="Include this run's results instead of every result run"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(package_project(root, args.target, args.result_dir))


if __name__ == "__main__":
    main()
