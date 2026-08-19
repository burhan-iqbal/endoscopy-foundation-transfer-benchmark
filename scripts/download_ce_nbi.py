#!/usr/bin/env python3
"""Download and extract the public CE-NBI dataset from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests

# Allow running as a script from dissertation_project/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path

logger = setup_logger("download_ce_nbi")

DEFAULT_URL = (
    "https://zenodo.org/records/6674034/files/Larynx_CE-NBI_Dataset.zip?download=1"
)
DEFAULT_MD5 = "17425958e554782ee2ccbaa066b44258"


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100.0 * written / total
                    logger.info("  progress: %.1f%% (%d / %d bytes)", pct, written, total)
    tmp.replace(dest)
    logger.info("Download complete: %s (%.2f GB)", dest, dest.stat().st_size / 1e9)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    logger.info("Extracting %s -> %s", zip_path, out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    logger.info("Extraction complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data.yaml (relative to project root)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if zip exists")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config if not Path(args.config).is_absolute() else args.config)
    url = cfg.get("zenodo_url", DEFAULT_URL)
    expected_md5 = cfg.get("zenodo_md5", DEFAULT_MD5)
    zip_path = resolve_path(cfg["paths"]["raw_zip"], root)
    interim_dir = resolve_path(cfg["paths"]["interim_dir"], root)

    if zip_path.exists() and not args.force:
        logger.info("Zip already present: %s", zip_path)
    else:
        download_file(url, zip_path)

    digest = md5_file(zip_path)
    if digest.lower() != expected_md5.lower():
        logger.error(
            "MD5 mismatch for %s: got %s, expected %s",
            zip_path,
            digest,
            expected_md5,
        )
        return 1
    logger.info("MD5 OK: %s", digest)

    if not args.skip_extract:
        # If interim already has content, still allow re-extract with --force
        marker = interim_dir / ".extracted"
        if marker.exists() and not args.force:
            logger.info("Already extracted (found %s). Use --force to re-extract.", marker)
        else:
            extract_zip(zip_path, interim_dir)
            marker.write_text("ok\n", encoding="utf-8")

    logger.info("Done. Next: python -m src.data.build_metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
