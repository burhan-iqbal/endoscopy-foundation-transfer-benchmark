"""Build cleaned CE-NBI metadata CSV and a data-audit report."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Headless-safe matplotlib (sandbox / CI may lack a writable cache dir)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cenbi")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path

logger = setup_logger("build_metadata")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_dataset_root(interim_dir: Path) -> Path:
    """Locate the folder that contains benign/malignant (or Excel + images)."""
    if not interim_dir.exists():
        raise FileNotFoundError(f"Interim directory missing: {interim_dir}")

    candidates = [interim_dir, *interim_dir.rglob("*")]
    for cand in candidates:
        if not cand.is_dir():
            continue
        names = {p.name.lower() for p in cand.iterdir()}
        if "benign" in names and "malignant" in names:
            return cand
    # Fallback: directory that contains an xlsx and many images
    for cand in candidates:
        if not cand.is_dir():
            continue
        xlsx = list(cand.glob("*.xlsx")) + list(cand.glob("*.xls"))
        if xlsx:
            return cand
    raise FileNotFoundError(
        f"Could not find CE-NBI benign/malignant folders under {interim_dir}"
    )


def find_excel(dataset_root: Path) -> Path | None:
    matches = list(dataset_root.rglob("*.xlsx")) + list(dataset_root.rglob("*.xls"))
    # Prefer files with mapping-like names
    preferred = [
        m
        for m in matches
        if any(k in m.name.lower() for k in ("label", "map", "meta", "patient", "ce-nbi", "ce_nbi"))
    ]
    if preferred:
        return preferred[0]
    return matches[0] if matches else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Excel column names to a common schema."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        rename[col] = key
    out = df.rename(columns=rename)

    aliases = {
        "image_id": [
            "image_id",
            "image",
            "image_name",
            "filename",
            "file_name",
            "img_name",
            "image_file",
        ],
        "patient_id": [
            "patient_id",
            "patient",
            "patientid",
            "subject_id",
            "id_patient",
        ],
        "histopathology": [
            "histopathology",
            "histo",
            "histopathology_label",
            "diagnosed_laryngeal_histopathology",
            "pathology",
            "class",
        ],
        "lesion_type": [
            "lesion_type",
            "lesion",
            "type_of_lesion",
            "benign_malignant",
            "lesion_type_benign_malignant",
            "label",
            "binary_label",
        ],
        "leukoplakia_label": [
            "leukoplakia_label",
            "leukoplakia",
            "leukoplakia_diagnosis",
            "leukoplakia_diagnosis_label",
        ],
        "width": ["width", "image_width", "w"],
        "height": ["height", "image_height", "h"],
        "image_size": ["image_size", "imagesize", "size"],
        "n_images": ["number_of_images", "n_images", "num_images"],
    }

    resolved: dict[str, str] = {}
    cols = set(out.columns)
    for target, options in aliases.items():
        for opt in options:
            if opt in cols:
                resolved[target] = opt
                break

    standardized = pd.DataFrame()
    for target, src in resolved.items():
        standardized[target] = out[src]
    return standardized


def patient_id_from_path(img_path: Path, dataset_root: Path) -> str | None:
    """Extract PatientXXX folder name from the CE-NBI directory layout."""
    for part in img_path.relative_to(dataset_root).parts:
        if re.fullmatch(r"Patient\d+", part, flags=re.IGNORECASE):
            # Normalize to P055-style IDs used in filenames when possible
            digits = "".join(ch for ch in part if ch.isdigit())
            return f"P{int(digits):03d}" if digits else part
    return None


def collect_images(dataset_root: Path) -> pd.DataFrame:
    """Walk folder structure: {benign|malignant}/{histopathology}/PatientXXX/*.jpg."""
    rows = []
    for lesion_dir in sorted(dataset_root.iterdir()):
        if not lesion_dir.is_dir():
            continue
        lesion_name = lesion_dir.name.lower()
        if lesion_name not in {"benign", "malignant"}:
            # Some zips nest one extra level
            continue
        binary_label = 1 if lesion_name == "malignant" else 0
        for histo_dir in sorted(lesion_dir.iterdir()):
            if not histo_dir.is_dir():
                continue
            histopathology = histo_dir.name
            for img_path in sorted(histo_dir.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append(
                    {
                        "image_path": img_path.resolve().relative_to(ROOT).as_posix(),
                        "image_id": img_path.name,
                        "image_stem": img_path.stem,
                        "lesion_type": lesion_name,
                        "binary_label": binary_label,
                        "histopathology": histopathology,
                        "patient_id_folder": patient_id_from_path(img_path, dataset_root),
                    }
                )
    if not rows:
        # Flat fallback: any images under dataset_root
        for img_path in sorted(dataset_root.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            parts = [p.lower() for p in img_path.relative_to(dataset_root).parts]
            lesion = "malignant" if "malignant" in parts else (
                "benign" if "benign" in parts else "unknown"
            )
            binary_label = 1 if lesion == "malignant" else (0 if lesion == "benign" else -1)
            histo = "unknown"
            for p in img_path.relative_to(dataset_root).parts[:-1]:
                if p.lower() not in {"benign", "malignant"}:
                    histo = p
            rows.append(
                {
                    "image_path": img_path.resolve().relative_to(ROOT).as_posix(),
                    "image_id": img_path.name,
                    "image_stem": img_path.stem,
                    "lesion_type": lesion,
                    "binary_label": binary_label,
                    "histopathology": histo,
                }
            )
    return pd.DataFrame(rows)


def infer_patient_id(image_id: str, image_stem: str) -> str:
    """
    Infer patient ID from filename when Excel lacks an explicit mapping.

    CE-NBI images are typically grouped per patient; common patterns use a
    patient prefix before an underscore or hyphen (e.g. P001_001.jpg).
    """
    stem = image_stem or Path(image_id).stem
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep)[0]
    # Digits-only prefix
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits
    return stem


def merge_excel(images: pd.DataFrame, excel_path: Path | None) -> pd.DataFrame:
    meta = images.copy()
    if excel_path is None:
        logger.warning("No Excel metadata found; inferring patient IDs from filenames")
        meta["patient_id"] = [
            infer_patient_id(i, s) for i, s in zip(meta["image_id"], meta["image_stem"])
        ]
        meta["leukoplakia_label"] = "unknown"
        meta["width"] = pd.NA
        meta["height"] = pd.NA
        return meta

    logger.info("Reading Excel metadata: %s", excel_path)
    raw = pd.read_excel(excel_path)
    excel = normalize_columns(raw)
    if "patient_id" in excel.columns:
        excel = excel.dropna(subset=["patient_id"])
        excel = excel[excel["patient_id"].astype(str).str.strip().ne("")]
        excel = excel[excel["patient_id"].astype(str).str.lower().ne("nan")]
    logger.info(
        "Excel columns normalized to: %s (%d patient rows)",
        list(excel.columns),
        len(excel),
    )

    # Normalize patient IDs in excel to PXXX
    if "patient_id" in excel.columns:
        def _norm_pid(v: object) -> str:
            s = str(v)
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                return f"P{int(digits):03d}"
            return s

        excel["patient_id"] = excel["patient_id"].map(_norm_pid)

    # Parse "1280*1008 pixels" style sizes if present
    if "image_size" in excel.columns and ("width" not in excel.columns or excel["width"].isna().all()):
        def _parse_size(v: object) -> tuple[object, object]:
            s = str(v)
            m = re.search(r"(\d+)\s*[*xX]\s*(\d+)", s)
            if m:
                return int(m.group(1)), int(m.group(2))
            return pd.NA, pd.NA

        parsed = excel["image_size"].map(_parse_size)
        excel["width"] = parsed.map(lambda t: t[0])
        excel["height"] = parsed.map(lambda t: t[1])

    if "image_id" in excel.columns:
        excel["image_id"] = excel["image_id"].astype(str).map(lambda x: Path(x).name)
        excel["image_stem"] = excel["image_id"].map(lambda x: Path(x).stem)
        # Prefer join on filename
        merged = meta.merge(excel, on="image_id", how="left", suffixes=("", "_excel"))
        if "patient_id" not in merged.columns or merged["patient_id"].isna().all():
            merged = meta.merge(excel, on="image_stem", how="left", suffixes=("", "_excel"))
    elif "patient_id" in excel.columns:
        # Patient-level Excel (CE-NBI official list): join leukoplakia / histo extras
        keep_cols = [
            c
            for c in (
                "patient_id",
                "leukoplakia_label",
                "histopathology",
                "lesion_type",
                "width",
                "height",
            )
            if c in excel.columns
        ]
        excel_p = excel[keep_cols].drop_duplicates("patient_id")
        # Prefer folder patient ids for the join key
        if "patient_id_folder" in meta.columns:
            meta = meta.copy()
            meta["patient_id"] = meta["patient_id_folder"]
        merged = meta.merge(excel_p, on="patient_id", how="left", suffixes=("", "_excel"))
    else:
        merged = meta.copy()
        for col in excel.columns:
            if col not in merged.columns:
                merged[col] = pd.NA

    # Resolve patient_id: folder > excel > filename inference
    if "patient_id_folder" in merged.columns and merged["patient_id_folder"].notna().any():
        merged["patient_id"] = merged["patient_id_folder"]
    elif "patient_id" not in merged.columns or merged["patient_id"].isna().all():
        if "patient_id_excel" in merged.columns:
            merged["patient_id"] = merged["patient_id_excel"]
        else:
            merged["patient_id"] = [
                infer_patient_id(i, s)
                for i, s in zip(merged["image_id"], merged["image_stem"])
            ]
    # Fill any remaining gaps from filename
    missing_pid = merged["patient_id"].isna() | (merged["patient_id"].astype(str) == "nan")
    if missing_pid.any():
        merged.loc[missing_pid, "patient_id"] = [
            infer_patient_id(i, s)
            for i, s in zip(
                merged.loc[missing_pid, "image_id"],
                merged.loc[missing_pid, "image_stem"],
            )
        ]
    merged["patient_id"] = merged["patient_id"].astype(str)

    # Prefer folder-derived binary_label; only fill unknown rows from excel
    if "lesion_type_excel" in merged.columns:
        mask = merged["binary_label"] < 0
        if mask.any():
            lt = merged.loc[mask, "lesion_type_excel"].astype(str).str.lower()
            mapped = lt.map(
                lambda x: 1 if "malig" in x else (0 if "benign" in x else -1)
            )
            merged.loc[mask, "binary_label"] = mapped.astype(int)
            merged.loc[mask, "lesion_type"] = lt.values

    if "leukoplakia_label" not in merged.columns or merged["leukoplakia_label"].isna().all():
        if "leukoplakia_label_excel" in merged.columns:
            merged["leukoplakia_label"] = merged["leukoplakia_label_excel"]
        elif "leukoplakia_label" not in merged.columns:
            merged["leukoplakia_label"] = "unknown"
    merged["leukoplakia_label"] = merged["leukoplakia_label"].fillna("unknown")

    for dim in ("width", "height"):
        if dim not in merged.columns:
            alt = f"{dim}_excel"
            merged[dim] = merged[alt] if alt in merged.columns else pd.NA
        elif merged[dim].isna().any() and f"{dim}_excel" in merged.columns:
            merged[dim] = merged[dim].fillna(merged[f"{dim}_excel"])

    return merged


def plot_class_balance(meta: pd.DataFrame, figures_dir: Path) -> None:
    ensure_dir(figures_dir)

    # Image-level
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = meta["lesion_type"].value_counts()
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("CE-NBI class balance (image-level)")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Lesion type")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_balance_image_level.png", dpi=150)
    plt.close(fig)

    # Patient-level
    patient = (
        meta.groupby("patient_id", as_index=False)["binary_label"]
        .max()
        .assign(lesion_type=lambda d: d["binary_label"].map({0: "benign", 1: "malignant"}))
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    pcounts = patient["lesion_type"].value_counts()
    pcounts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("CE-NBI class balance (patient-level)")
    ax.set_ylabel("Number of patients")
    ax.set_xlabel("Lesion type")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_balance_patient_level.png", dpi=150)
    plt.close(fig)


def write_audit(
    meta: pd.DataFrame,
    audit_path: Path,
    expected_n_images: int,
    expected_n_patients: int,
    dataset_root: Path,
    excel_path: Path | None,
) -> None:
    n_images = len(meta)
    n_patients = meta["patient_id"].nunique()
    img_balance = meta["lesion_type"].value_counts().to_dict()
    patient = meta.groupby("patient_id")["binary_label"].max()
    pat_balance = {
        "benign": int((patient == 0).sum()),
        "malignant": int((patient == 1).sum()),
    }
    conflicts = (
        meta.groupby("patient_id")["binary_label"].nunique().loc[lambda s: s > 1]
    )
    histo = meta.groupby(["lesion_type", "histopathology"]).size().reset_index(name="n")
    try:
        histo_table = histo.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        histo_table = histo.to_string(index=False)

    lines = [
        "# CE-NBI Data Audit",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Excel metadata: `{excel_path}`" if excel_path else "- Excel metadata: **not found**",
        f"- Images found: **{n_images}** (expected {expected_n_images})",
        f"- Patients found: **{n_patients}** (expected {expected_n_patients})",
        f"- Image-level class counts: {img_balance}",
        f"- Patient-level class counts: {pat_balance}",
        f"- Patients with conflicting binary labels: **{len(conflicts)}**",
        "",
        "## Histopathology breakdown (image counts)",
        "",
        histo_table,
        "",
        "## Notes",
        "",
        "- Raw zip is never modified; interim extract is used for analysis.",
        "- Binary target is benign (0) vs malignant (1).",
        "- Patient IDs are taken from `PatientXXX` folders (normalized to `PXXX`), with Excel leukoplakia/histopathology joined when available.",
        "",
    ]
    if n_images != expected_n_images or n_patients != expected_n_patients:
        lines.append(
            f"**WARNING:** Counts differ from the dataset paper "
            f"({expected_n_images} images / {expected_n_patients} patients). "
            "Investigate before claiming exact reproduction."
        )
        lines.append("")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote audit report: %s", audit_path)


def build_metadata(cfg_path: str = "configs/data.yaml") -> pd.DataFrame:
    root = project_root()
    cfg = load_yaml(root / cfg_path)
    interim_dir = resolve_path(cfg["paths"]["interim_dir"], root)
    metadata_csv = resolve_path(cfg["paths"]["metadata_csv"], root)
    figures_dir = resolve_path(cfg["paths"]["figures_dir"], root)
    audit_path = resolve_path(cfg["paths"]["audit_report"], root)
    expected_n_images = int(cfg.get("expected_n_images", 11144))
    expected_n_patients = int(cfg.get("expected_n_patients", 210))

    dataset_root = find_dataset_root(interim_dir)
    logger.info("Using dataset root: %s", dataset_root)
    excel_path = find_excel(dataset_root)

    images = collect_images(dataset_root)
    if images.empty:
        raise RuntimeError(f"No images found under {dataset_root}")
    logger.info("Collected %d images from folders", len(images))

    meta = merge_excel(images, excel_path)

    # Fill missing dimensions via Pillow when needed
    missing_dims = meta["width"].isna() | meta["height"].isna()
    if missing_dims.any():
        try:
            from PIL import Image

            for idx in meta.index[missing_dims]:
                with Image.open(root / meta.at[idx, "image_path"]) as im:
                    meta.at[idx, "width"], meta.at[idx, "height"] = im.size
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fill image dimensions: %s", exc)

    keep = [
        "image_path",
        "image_id",
        "patient_id",
        "lesion_type",
        "binary_label",
        "histopathology",
        "leukoplakia_label",
        "width",
        "height",
    ]
    for col in keep:
        if col not in meta.columns:
            meta[col] = pd.NA
    out = meta[keep].copy()
    out["binary_label"] = out["binary_label"].astype(int)
    out["patient_id"] = out["patient_id"].astype(str)

    # Drop unusable rows
    out = out[out["binary_label"].isin([0, 1])].reset_index(drop=True)

    ensure_dir(metadata_csv.parent)
    out.to_csv(metadata_csv, index=False)
    logger.info("Wrote metadata: %s (%d rows)", metadata_csv, len(out))

    plot_class_balance(out, figures_dir)
    write_audit(
        out,
        audit_path,
        expected_n_images,
        expected_n_patients,
        dataset_root,
        excel_path,
    )

    if abs(len(out) - expected_n_images) > 50:
        raise RuntimeError(
            f"Image count {len(out)} diverges too far from expected {expected_n_images}"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    build_metadata(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
