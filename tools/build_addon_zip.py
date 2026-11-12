#!/usr/bin/env python3
"""Build an installable zip for the ChatGPT TMD2 add-on.

Usage
-----
    python tools/build_addon_zip.py [--output dist]

The resulting archive contains a single top level directory ``chatgpt_tmd2/``
with ``__init__.py`` and ``blender_manifest.toml`` inside, which is what both
``Edit > Preferences > Add-ons > Install from Disk`` and
``Get Extensions > Install from Disk`` expect.
"""

import argparse
import os
import re
import sys
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "chatgpt_tmd2"

EXCLUDE_DIRS = {"__pycache__", ".git", "tests", ".pytest_cache", ".mypy_cache"}
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".zip")
EXCLUDE_FILES = {".DS_Store", "hashes.json.bak"}


def read_version(package_dir):
    manifest = os.path.join(package_dir, "blender_manifest.toml")
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as handle:
            match = re.search(r'^version\s*=\s*"([^"]+)"', handle.read(), re.M)
            if match:
                return match.group(1)
    init = os.path.join(package_dir, "__init__.py")
    with open(init, "r", encoding="utf-8") as handle:
        match = re.search(r'"version"\s*:\s*\(([^)]+)\)', handle.read())
        if match:
            return ".".join(part.strip() for part in match.group(1).split(","))
    return "0.0.0"


def iter_files(package_dir):
    for current, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        for name in sorted(files):
            if name in EXCLUDE_FILES or name.endswith(EXCLUDE_SUFFIX):
                continue
            path = os.path.join(current, name)
            yield path, os.path.relpath(path, os.path.dirname(package_dir))


def check_python(files):
    """Compile every module in memory (no bytecode is written to disk)."""
    errors = []
    for path, _ in files:
        if not path.endswith(".py"):
            continue
        try:
            with open(path, "rb") as handle:
                compile(handle.read(), path, "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def build(output_dir):
    package_dir = os.path.join(ROOT, PACKAGE)
    version = read_version(package_dir)
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{PACKAGE}-{version}.zip")

    files = list(iter_files(package_dir))
    errors = check_python(files)
    if errors:
        print("Syntax errors, aborting:")
        for error in errors:
            print("  ", error)
        return 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in files:
            archive.write(path, arcname.replace(os.sep, "/"))

    size = os.path.getsize(zip_path)
    print(f"built {zip_path} ({size / 1048576:.2f} MiB, {len(files)} files, v{version})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(ROOT, "release"),
                        help="directory that receives the zip")
    args = parser.parse_args()
    return build(os.path.abspath(args.output))


if __name__ == "__main__":
    sys.exit(main())
