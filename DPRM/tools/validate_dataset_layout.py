"""Validate an SCD dataset directory without copying or changing its files."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_SUBDIRECTORIES = ("T1", "T2", "GT_T1", "GT_T2")
OPTIONAL_SUBDIRECTORIES = ("GT_CD",)


def relative_files(directory: Path) -> set[Path]:
    """Return all files below a directory as paths relative to it."""
    return {item.relative_to(directory) for item in directory.rglob("*") if item.is_file()}


def validate_split(split: Path) -> list[str]:
    """Return human-readable validation errors for one split."""
    errors: list[str] = []
    directories = {name: split / name for name in REQUIRED_SUBDIRECTORIES + OPTIONAL_SUBDIRECTORIES}
    for name in REQUIRED_SUBDIRECTORIES:
        if not directories[name].is_dir():
            errors.append(f"missing required directory: {split / name}")
    if errors:
        return errors

    reference = relative_files(directories["T1"])
    if not reference:
        errors.append(f"no files found in {directories['T1']}")
    for name in REQUIRED_SUBDIRECTORIES[1:]:
        current = relative_files(directories[name])
        missing, extra = reference - current, current - reference
        if missing:
            errors.append(f"{split / name}: missing {len(missing)} file(s) matched from T1")
        if extra:
            errors.append(f"{split / name}: has {len(extra)} unmatched file(s)")
    optional = directories["GT_CD"]
    if optional.is_dir():
        current = relative_files(optional)
        missing, extra = reference - current, current - reference
        if missing:
            errors.append(f"{optional}: missing {len(missing)} file(s) matched from T1")
        if extra:
            errors.append(f"{optional}: has {len(extra)} unmatched file(s)")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the directory layout of a local SCD dataset.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    args = parser.parse_args()
    root = args.dataset_root.expanduser()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")

    checked = 0
    failed = False
    for name in args.splits:
        split = root / name
        if not split.exists():
            print(f"SKIP {split} (not present)")
            continue
        checked += 1
        errors = validate_split(split)
        if errors:
            failed = True
            print(f"FAIL {split}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {split}")
    if not checked:
        raise SystemExit("No requested split directory was found.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
