# Roboflow -> local YOLOv8 layout - Python 3.11+, Windows 11
# usage: python fetch_dataset.py --key KEY   |   python fetch_dataset.py --zip arsenal.zip

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

WORKSPACE = "rihardss-workspace"
PROJECT = "arsenal-4lwze"
DEST = Path("dataset")


def normalize(dest: Path) -> None:
    """roboflow zip and export folder differ slightly -> force dataset/data.yaml layout"""
    yaml_file = dest / "data.yaml"
    if not yaml_file.exists():
        found = list(dest.rglob("data.yaml"))
        if not found:
            sys.exit("no data.yaml in download")
        if found[0].parent != dest:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(found[0].parent, dest)
            yaml_file = dest / "data.yaml"

    for split in ("train", "valid"):
        if not (dest / split / "images").exists():
            sys.exit(f"missing {dest / split / 'images'}")
    if (dest / "test").exists() and not (dest / "test" / "images").exists():
        shutil.rmtree(dest / "test")

    print(f"dataset ready -> {yaml_file}")
    print(yaml_file.read_text(encoding="utf-8").strip())


def fetch_key(key: str) -> None:
    from roboflow import Roboflow

    workspace = Roboflow(api_key=key).workspace(WORKSPACE)
    project = workspace.project(PROJECT)
    version = sorted(project.versions(), key=lambda v: v.version)[-1]
    print(f"downloading {PROJECT} v{version.version} classes={version.classes}")
    if DEST.exists():
        shutil.rmtree(DEST)
    version.download("yolov8", location=str(DEST))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="", help="roboflow private api key")
    parser.add_argument("--zip", default="", help="local roboflow/yolo zip")
    args = parser.parse_args()

    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall("_dl")
        src = next(Path("_dl").rglob("data.yaml")).parent
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.move(str(src), str(DEST))
        shutil.rmtree("_dl", ignore_errors=True)
    elif args.key:
        fetch_key(args.key)
    else:
        sys.exit("give --key (https://roboflow.com/keys) or --zip path/to/export.zip")

    normalize(DEST)


if __name__ == "__main__":
    main()
