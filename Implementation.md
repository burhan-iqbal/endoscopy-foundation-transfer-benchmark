# CE-NBI Benchmark — Detailed Implementation Log

Updated: **2026-08-07**

Living record of what we did, in order, from chat decisions and code actions.
Working results stay **in this file** and under `reports/` — not in `main.tex` until the final matrix is ready.

**Forward plan (GPU rental → submission):** see  
[`../03_Drafts_and_Planning/FUTURE_NEXT_STEPS.md`](../03_Drafts_and_Planning/FUTURE_NEXT_STEPS.md)

**Next-agent brief (no chat history):** see  
[`../03_Drafts_and_Planning/AGENT_HANDOFF.md`](../03_Drafts_and_Planning/AGENT_HANDOFF.md)

---

## 0. Project goal (locked)

**Working title:** Do Endoscopy Foundation Models Improve Label-Efficient Laryngeal Lesion Classification? A Patient-Level Benchmark on Public CE-NBI Images

**Question:** On public CE-NBI throat images, do endoscopy foundation models beat ImageNet / DINOv2 when labelled data is scarce?

| Item | Choice |
|---|---|
| Task | Benign vs malignant |
| Data | Public CE-NBI (Zenodo) |
| Split rule | **Patient-level** (no same-patient leakage) |
| Models | ResNet-50 (ImageNet), DINOv2 ViT-S/14, Endo-FM |
| Not this project | Clinical deployment / hospital product |

Student: Burhan Khan (ec25164). Supervisor: Tayyaba Irum. Final deadline ~**19 Aug 2026**.

---

## 0.1 Dataset facts (CE-NBI)

| Field | Value |
|---|---|
| Source | Zenodo record **6674034** (`Larynx_CE-NBI_Dataset.zip`) |
| MD5 (verified) | `17425958e554782ee2ccbaa066b44258` |
| Images | **11,144** (expected 11,144) |
| Patients | **210** (expected 210) |
| Image-level classes | benign **7,657** / malignant **3,487** |
| Patient-level classes | benign **150** / malignant **60** |
| Positive class | malignant |
| Conflicting patient labels | **0** |

Histopathology breakdown (image counts) is in `reports/data_audit.md` — benign types include Reinke's edema (2,661), Low-grade dysplasia (1,428), Papillomatosis (1,103); malignant types are SCC (1,906), High-grade dysplasia (1,039), Carcinoma in situ (542).

---

## 0.2 Experimental protocol (locked, `configs/experiments.yaml`)

**Split:** patient-level, stratified by patient binary label, **70 / 15 / 15** → ~**147 / 31 / 32** patients. Test set = **1,600** images, val = **1,466** images. A cautionary image-level split exists only to quantify leakage.

**Label-efficiency regimes:** train on **5% / 10% / 25% / 50% / 100%** of the *training patients'* labels. Seeds **0, 1, 2** defined (only seed 0 run so far).

**Image / augmentation:**
- Input size 224 (ViT variant size 384 available), ImageNet mean/std normalisation
- Train aug: horizontal flip, ±10° rotation, mild color jitter (b0.1/c0.1/s0.05/h0.02), scale jitter [0.9, 1.1]

**Training hyperparameters (all models):**

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| LR (fine-tune backbone) | 1e-4 |
| LR (head) | 1e-3 |
| Warmup epochs | 2 |
| Max epochs | 50 |
| Early stopping patience | 10 (monitor `val_auroc`) |
| Batch size | 32 |
| Class weights | on (handles benign/malignant imbalance) |
| Mixed precision (AMP) | on |
| Model selection | best `val_auroc` |

**Per-model recipe:**

| Model | Family | Backbone | Mode | Head |
|---|---|---|---|---|
| ResNet-50 | ImageNet supervised | `resnet50` (timm) | fine-tune | built-in |
| DINOv2 ViT-S/14 | Generic SSL | `dinov2_vits14` (torch.hub) | frozen | linear |
| Endo-FM | Endoscopy FM | ViT-B/16 spatial trunk | frozen | linear |

**Evaluation:** metrics at image and patient level (AUROC, AUPRC, accuracy, F1, precision, sensitivity, specificity, confusion matrix, Brier, ECE). Patient-level = mean predicted malignant probability per patient. Bootstrap: **1,000** resamples, 95% CI. Calibration: temperature scaling fit on validation.

---

## 1. Starting point (before code)

Already present under the Project tree:

- QMUL guidance / templates (`00_`)
- Literature + related-work notes (`01_`, `02_`)
- Draft dissertation paper `03_Drafts_and_Planning/Draft_Dissertation_Paper/main.tex` (methods locked; results still “Under progress”)
- Presentations / phase plan / `PROJECT_STRUCTURE.md`
- Codex skills under `04_`

**Missing:** no `dissertation_project/` code, no downloaded CE-NBI, no splits, no runs.

---

## 2. Step-by-step what we did (chronological)

### Step 1 — Inventory the whole project (9 Jul ~11:13)

**User:** Go through everything in `PROJECT_STRUCTURE.md`, then implement.

**Actions:**
1. Walked the full folder tree (guidance, literature, drafts, presentations).
2. Confirmed draft paper exists; results placeholders only.
3. Confirmed **no implementation code** yet.
4. Asked user to choose scope for the next phase.

**User decision:** **B** — build full code + download CE-NBI + prepare splits locally; train later on GPU.

---

### Step 2 — Create `dissertation_project/` scaffold (9 Jul ~11:23)

Built the repo layout:

```text
dissertation_project/
  configs/data.yaml
  configs/experiments.yaml
  requirements.txt
  README.md
  pytest.ini
  .gitignore
  scripts/
  src/utils/     (paths, seed, logging)
  src/data/
  src/models/
  src/training/
  src/analysis/
  tests/
  data/{raw,interim,processed,splits}/
  models/external_weights/
  reports/{logs,figures,tables,predictions}/
```

Also wrote:
- `scripts/download_ce_nbi.py`
- `scripts/download_endo_fm_weights.md`
- Related-work matrix CSV + model access notes skeleton

---

### Step 3 — Download CE-NBI from Zenodo

**Actions:**
1. Created Python venv + installed `requirements.txt`.
2. Ran `python scripts/download_ce_nbi.py` (needed `python3` on Mac).
3. Downloaded zip from Zenodo record **6674034**, verified MD5, extracted to `data/interim/ce_nbi/`.

**Result:** Dataset on disk (~1.4 GB zip), ready for metadata.

---

### Step 4 — Metadata + data audit (with fixes)

**Actions:**
1. Wrote / ran `python -m src.data.build_metadata`.
2. Hit issues and fixed them one by one:
   - Matplotlib crash in sandbox → headless backend + `MPLCONFIGDIR`.
   - Patient IDs taken from `PatientXXX` folders → normalised to `PXXX` (Excel alone was unreliable).
   - Excel file has many empty rows / column aliases → hardened reader.
   - Folder labels kept authoritative for benign/malignant; Excel used for histopathology/leukoplakia join.
3. Wrote `reports/data_audit.md` + class-balance figures.

**Verified counts:**
- **11,144** images (expected 11,144)
- **210** patients (expected 210)
- Image classes: benign 7657 / malignant 3487
- Patient classes: benign 150 / malignant 60
- Conflicting patient labels: **0**

Output: `data/processed/ce_nbi_metadata.csv`

---

### Step 5 — Patient-level + low-label splits

**Actions:**
1. Wrote / ran `python -m src.data.make_splits`.
2. Created:
   - `data/splits/patient_level_main_split.csv` (~147 / 31 / 32 train/val/test patients)
   - Low-label CSVs: `low_label_{05,10,25,50,100}_seed_{0,1,2}.csv`
   - Cautionary `image_level_main_split.csv` (leakage comparison only)
3. Wrote tests: `tests/test_splits.py`, `tests/test_metrics.py`.
4. Ran pytest → **9/9 passed**.

Also wrote dataset loader + transforms (`src/data/dataset.py`, `transforms.py`).

---

### Step 6 — Models, train/eval/calibrate, analysis code

**Models (`src/models/`):**
1. **ResNet-50** — `timm` ImageNet, fine-tune
2. **DINOv2 ViT-S/14** — `torch.hub`, frozen + linear/MLP head
3. **Endo-FM** — load weights into ViT-B/16 spatial trunk, frozen + head

**Training stack:**
- `src/training/train.py` — train one run
- `src/training/evaluate.py` — test metrics + bootstrap
- `src/training/calibrate.py` — temperature scaling
- `src/analysis/` — metrics, bootstrap, plots, saliency

At this point (original Scope B), **full training was still deferred to GPU**.

---

### Step 7 — User said “Go ahead” → smoke train + Endo-FM + H100 script (9 Jul ~12:00+)

**Actions:**
1. **CPU smoke train:** ResNet-50, 5% labels, 1 epoch  
   - val AUROC ≈ **0.69**, test AUROC ≈ **0.81** (pipeline sanity only)
2. **Downloaded Endo-FM weights** via `gdown` (~**2.2 GB**) → `models/external_weights/endo_fm.pth`
3. Fixed Endo-FM loader for PyTorch 2.6+ (`weights_only=False`), then improved mapping:
   - Prefer `teacher` / `student` TimeSformer backbone keys
   - Map spatial ViT-B/16 keys into `timm` `vit_base_patch16_224`
   - Skip temporal / mismatched keys
   - Final: **~150 keys mapped**, 0 missing for the spatial trunk
4. Fixed AMP deprecations → `torch.amp.*`
5. Added `scripts/run_experiments.sh` for full H100 matrix
6. Updated README + `reports/model_access_notes.md`

---

### Step 8 — Checkpoint / resume hardening (user concern before SSH)

**User worry:** SSH GPU session might drop; need checkpoints every epoch + resume.

**Before:** only `checkpoint_best.pt` on val improvement — **no resume**.

**After (updated `train.py`):**

| File | When | Purpose |
|---|---|---|
| `checkpoint_last.pt` | **Every epoch** (atomic write) | Resume: model + optimizer + scheduler + scaler + epoch + early-stop wait + curve |
| `checkpoint_best.pt` | Val AUROC improves | Weights used for final eval |
| `metrics.json` | Run finished | Marks complete → re-run skips |

CLI: `--resume` (default), `--no-resume`, `--save-every-epoch`  
`run_experiments.sh` safe to re-run after drops.

---

### Step 9 — Run locally on Mac instead of SSH (user: check CUDA → MPS → CPU)

**Actions:**
1. Added `src/utils/device.py` — prefer **CUDA → MPS → CPU**.
2. Device check on this machine: CUDA no; **MPS yes** (Apple M1 Max).
3. MPS tweaks: `num_workers=0`; periodic `torch.mps.empty_cache()` in predict; evaluate hardened.
4. Wrote `scripts/run_local_mps.sh`.
5. Fixed **CRLF** line endings that broke bash `pipefail`.
6. Cleared short smoke/debug runs so they would not be mistaken for real results (`seed_0_cpu_smoke_1epoch`, `seed_0_mps_debug_2epoch` archived).
7. `nohup` / `setsid` unstable on Mac outside Cursor → launched matrix as a **managed Cursor background shell**.

**Full seed-0 matrix ran on MPS** (~**3.8 hours**, exit 0):  
3 models × 5 label fractions = **15 runs**, each with train → evaluate → calibrate.

Log: `reports/logs/local_mps_matrix.log`

---

### Step 10 — Summaries & decisions (13 Jul / 22 Jul)

1. Wrote a short summary into this file (13 Jul).
2. Confirmed status: paper results still placeholders; seeds 1–2 not run.
3. **User decision (22 Jul):** do **not** put intermediate seed-0 numbers into `main.tex`; keep working results **here** and in `reports/`.
4. Expanded this file into the detailed step-by-step log (this version).

---

## 3. Code map (what each part does)

### Configs
| Path | Role |
|---|---|
| `configs/data.yaml` | Zenodo URL/MD5, expected counts, split fractions, image size |
| `configs/experiments.yaml` | Full protocol: seeds, label fractions, augmentation, hyperparameters, per-model recipes, calibration, bootstrap, paths |

### Scripts
| Path | Role |
|---|---|
| `scripts/download_ce_nbi.py` | Download zip from Zenodo, verify MD5, extract to `data/interim/` |
| `scripts/download_endo_fm_weights.md` | Instructions for the Endo-FM Google Drive weights |
| `scripts/run_local_mps.sh` | Full local matrix on Mac MPS (train → eval → calibrate) |
| `scripts/run_experiments.sh` | Same matrix for a rented H100 (edit `SEEDS=(0 1 2)`) |

### Data pipeline (`src/data/`)
| Path | Role |
|---|---|
| `build_metadata.py` | Scan `PatientXXX` folders, derive `PXXX` IDs + binary label, join Excel histopathology, write `ce_nbi_metadata.csv`, `data_audit.md`, class-balance figures |
| `make_splits.py` | Patient-level stratified 70/15/15 split + low-label subset CSVs per seed; cautionary image-level split |
| `dataset.py` | PyTorch `Dataset` reading image paths + labels from split CSVs |
| `transforms.py` | Train/eval torchvision transforms (resize, aug, ImageNet normalisation) |

### Models (`src/models/`)
| Path | Role |
|---|---|
| `encoders.py` | Build ResNet-50 (timm), DINOv2 (torch.hub), Endo-FM (teacher/student TimeSformer → ViT-B/16 spatial trunk, temporal keys skipped) |
| `heads.py` | Linear / MLP classification head on top of frozen features |
| `registry.py` | Map model name → (encoder, head, train mode) from config |

### Training (`src/training/`)
| Path | Role |
|---|---|
| `train.py` | One run: build model, AdamW + warmup, class weights, AMP, early stopping on val AUROC, per-epoch `checkpoint_last.pt` + best on improvement, resume, write `metrics.json` + `training_curve.csv` |
| `evaluate.py` | Load best checkpoint, predict, compute image + patient metrics, bootstrap CIs |
| `calibrate.py` | Temperature scaling fit on val, apply to test, write `calibration.json` + reliability diagram |

### Analysis + utils
| Path | Role |
|---|---|
| `src/analysis/metrics.py` | AUROC/AUPRC/F1/sens/spec/Brier/ECE/confusion |
| `src/analysis/bootstrap.py` | Resample patients/images for 95% CIs |
| `src/analysis/plots.py` | Label-efficiency curve, class balance, aggregation |
| `src/analysis/saliency.py` | Saliency/attention maps for explainability |
| `src/utils/paths.py` | Central path resolver |
| `src/utils/seed.py` | Global seeding (torch/numpy/random) |
| `src/utils/logging.py` | Run logging setup |
| `src/utils/device.py` | Device selection **CUDA → MPS → CPU** |
| `tests/test_splits.py` | No patient leakage across splits; subset containment |
| `tests/test_metrics.py` | Metric correctness on toy inputs |

---

## 3.1 Environment

| Item | Value |
|---|---|
| Machine | Apple **M1 Max**, macOS |
| Python | 3.14 in local `.venv` |
| Compute used | **MPS** (CUDA unavailable locally) |
| Key libs | PyTorch (+`torch.amp`), timm, torchvision, pandas, scikit-learn, matplotlib, gdown, pytest |
| Endo-FM weights | `models/external_weights/endo_fm.pth` (~2.2 GB) |

---

## 3.2 How to reproduce (commands)

```bash
cd dissertation_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# data
python3 scripts/download_ce_nbi.py
python3 -m src.data.build_metadata
python3 -m src.data.make_splits
pytest -q                          # 9 tests

# full matrix (Mac MPS)
bash scripts/run_local_mps.sh
#   or a single run:
python3 -m src.training.train    --model endo_fm --label-frac 0.10 --seed 0
python3 -m src.training.evaluate --model endo_fm --label-frac 0.10 --seed 0 --bootstrap
python3 -m src.training.calibrate --model endo_fm --label-frac 0.10 --seed 0

# aggregate
python3 -m src.analysis.plots --aggregate
```

On a rented H100, use `scripts/run_experiments.sh` (set `SEEDS=(0 1 2)`); runs are resume-safe and skip finished work.

---

## 4. Working results (seed 0, local MPS) — keep here only

All **15 runs finished** (3 models × 5 label fractions), each with train → evaluate → calibrate. **Do not** treat as final paper claim until seeds 1–2 (optional) are done. Test set = 1,600 images / 32 patients.

### 4.1 Test AUROC — image level (headline)

| Labels used | ResNet-50 | DINOv2 | Endo-FM |
|---:|---:|---:|---:|
| 5% | 0.776 | **0.879** | 0.845 |
| 10% | 0.768 | 0.857 | **0.895** |
| 25% | **0.787** | 0.769 | 0.761 |
| 50% | 0.858 | **0.885** | 0.804 |
| 100% | **0.881** | 0.788 | 0.763 |

### 4.2 Test AUROC — patient level (main split is patient-level)

| Labels used | ResNet-50 | DINOv2 | Endo-FM |
|---:|---:|---:|---:|
| 5% | 0.749 | **0.850** | 0.773 |
| 10% | 0.647 | 0.816 | **0.855** |
| 25% | **0.749** | 0.734 | 0.729 |
| 50% | 0.831 | 0.879 | **0.860** |
| 100% | 0.850 | 0.816 | 0.783 |

### 4.3 Full per-run metrics (image level, test set)

| Model | Labels | AUROC | AUPRC | Acc | F1 | Sens | Spec | Brier | ECE | Epochs | Best val AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 5% | 0.776 | 0.662 | 0.728 | 0.436 | 0.309 | 0.944 | 0.217 | 0.185 | 11 | 0.677 |
| ResNet-50 | 10% | 0.768 | 0.517 | 0.641 | 0.014 | 0.007 | 0.967 | 0.261 | 0.241 | 15 | 0.713 |
| ResNet-50 | 25% | 0.787 | 0.533 | 0.668 | 0.306 | 0.215 | 0.902 | 0.246 | 0.225 | 13 | 0.844 |
| ResNet-50 | 50% | 0.858 | 0.711 | 0.724 | 0.400 | 0.270 | 0.958 | 0.174 | 0.132 | 11 | 0.810 |
| ResNet-50 | 100% | 0.881 | 0.764 | 0.763 | 0.563 | 0.449 | 0.925 | 0.145 | 0.091 | 11 | 0.861 |
| DINOv2 | 5% | 0.879 | 0.742 | 0.799 | 0.650 | 0.548 | 0.929 | 0.158 | 0.145 | 21 | 0.887 |
| DINOv2 | 10% | 0.857 | 0.692 | 0.778 | 0.721 | 0.846 | 0.742 | 0.165 | 0.150 | 12 | 0.845 |
| DINOv2 | 25% | 0.769 | 0.533 | 0.746 | 0.603 | 0.568 | 0.837 | 0.186 | 0.110 | 11 | 0.867 |
| DINOv2 | 50% | 0.885 | 0.675 | 0.810 | 0.756 | 0.866 | 0.781 | 0.137 | 0.110 | 11 | 0.905 |
| DINOv2 | 100% | 0.788 | 0.600 | 0.735 | 0.581 | 0.540 | 0.835 | 0.180 | 0.066 | 12 | 0.883 |
| Endo-FM | 5% | 0.845 | 0.724 | 0.740 | 0.481 | 0.355 | 0.938 | 0.177 | 0.148 | 30 | 0.747 |
| Endo-FM | 10% | 0.895 | 0.757 | 0.837 | 0.770 | 0.805 | 0.853 | 0.128 | 0.101 | 12 | 0.761 |
| Endo-FM | 25% | 0.761 | 0.555 | 0.709 | 0.561 | 0.546 | 0.794 | 0.209 | 0.148 | 14 | 0.793 |
| Endo-FM | 50% | 0.804 | 0.589 | 0.720 | 0.548 | 0.500 | 0.833 | 0.192 | 0.131 | 21 | 0.794 |
| Endo-FM | 100% | 0.763 | 0.581 | 0.691 | 0.521 | 0.494 | 0.792 | 0.209 | 0.150 | 20 | 0.827 |

Threshold fixed at 0.5. Sensitivity is low at very small label fractions (few malignant examples), which is expected and worth flagging in the discussion.

### 4.4 Calibration (temperature scaling example)

Temperature scaling is fit on validation and applied to test. It leaves AUROC/accuracy unchanged (monotonic) and nudges probability quality. Example — ResNet-50, 100% labels:

| | ECE | Brier | Temperature |
|---|---:|---:|---:|
| Before | 0.091 | 0.1445 | — |
| After | 0.084 | 0.1405 | 1.38 |

Per-run `calibration.json` + reliability diagrams under `reports/figures/calibration/`.

### 4.5 Plain-English reading

- **Very scarce labels (5–10%):** the frozen foundation encoders (DINOv2, Endo-FM) beat ResNet-50 at both image and patient level.
- **Full labels (100%):** ResNet-50 fine-tuning is strongest here (image AUROC 0.881).
- **Endo-FM** wins clearly only at 10% (image 0.895 / patient 0.855); it does **not** dominate overall on this seed — an honest, reportable finding.
- **Sensitivity vs specificity:** most models are high-specificity / low-sensitivity at 0.5 threshold, especially at low labels; threshold tuning / patient aggregation matters.
- The 25% column is noisy (ResNet-50 oddly best) — single seed; seeds 1–2 would smooth this.
- Seed-0 only → working evidence, not final submission numbers.

### 4.6 Artefacts on disk

- Per run: `reports/logs/{resnet50,dinov2_vits14,endo_fm}/label_{05,10,25,50,100}/seed_0/`
  - `config.yaml`, `metrics.json`, `metrics_eval.json`, `training_curve.csv`
  - `predictions.csv`, `predictions_val.csv`, `predictions_calibrated.csv`
  - `calibration.json`, `checkpoint_best.pt`, `checkpoint_last.pt`
- Aggregate figure: `reports/figures/label_efficiency_auroc.png`
- Aggregate table: `reports/tables/main_metrics_image_level.csv`
- Class-balance figures: `reports/figures/class_balance_{image,patient}_level.png`
- Prediction copies: `reports/predictions/{model}_label_{frac}_seed_0.csv`
- Full run log: `reports/logs/local_mps_matrix.log`

---

## 5. Every error we faced and how we fixed it (detailed)

Full troubleshooting log, in the order the problems appeared. Each entry: what broke, why, the fix, and how we confirmed it.

### 5.1 `python: command not found` on macOS
- **When:** first attempt to run `scripts/download_ce_nbi.py`.
- **Symptom:** `python scripts/download_ce_nbi.py` failed — macOS has no bare `python`.
- **Cause:** macOS ships `python3` only; there is no `python` shim.
- **Fix:** use `python3` everywhere; created a venv (`python3 -m venv .venv`) and always `source .venv/bin/activate` before running.
- **Confirmed:** download script then started and ran.

### 5.2 Dependency install / environment setup
- **When:** before first data build.
- **Symptom:** imports (`torch`, `timm`, `pandas`, `matplotlib`, `gdown`) not available.
- **Cause:** fresh venv, nothing installed.
- **Fix:** `pip install -r requirements.txt`; later added `gdown` to requirements for the Endo-FM download.
- **Confirmed:** module import dry-run succeeded; pytest ran.

### 5.3 Matplotlib crash in headless sandbox
- **When:** `python -m src.data.build_metadata` (metadata wrote 11,144 rows, then crashed on plotting).
- **Symptom:** process crashed while drawing class-balance figures.
- **Cause:** no display / non-writable Matplotlib cache dir in the sandbox; default interactive backend.
- **Fix:** force headless Agg backend and set `MPLCONFIGDIR=/tmp/matplotlib-cenbi` before running; hardened the plotting/audit code paths.
- **Confirmed:** re-run produced `reports/data_audit.md` + class-balance figures without crashing.

### 5.4 Patient IDs unreliable from Excel
- **When:** building metadata.
- **Symptom:** patient grouping was inconsistent when driven from the Excel sheet.
- **Cause:** the released Excel metadata does not expose clean, complete patient identifiers.
- **Fix:** take patient IDs from the `PatientXXX` **folder names**, normalised to `PXXX`; treat folder as the authoritative source of the patient grouping.
- **Confirmed:** **210** patients detected (expected 210), **0** patients with conflicting binary labels.

### 5.5 Excel file has many empty rows / inconsistent columns
- **When:** joining histopathology / leukoplakia info from Excel.
- **Symptom:** merge produced wrong / empty values; column names didn't match expectations.
- **Cause:** `Patients_List_Updated_Final.xlsx` contains many blank rows and column-name variants.
- **Fix:** hardened the Excel reader (drop empty rows, add column aliases); joined Excel **only** for histopathology/leukoplakia, keyed by patient ID.
- **Confirmed:** histopathology breakdown table in `data_audit.md` populated correctly.

### 5.6 Excel merge overwrote authoritative labels
- **When:** same metadata step.
- **Symptom:** benign/malignant labels risked being changed by the Excel merge.
- **Cause:** merge assignment let Excel values override the folder-derived label.
- **Fix:** folder-derived benign/malignant label kept authoritative; Excel used for supplementary fields only.
- **Confirmed:** image classes benign 7657 / malignant 3487; patient classes benign 150 / malignant 60.

### 5.7 Endo-FM checkpoint won't load on PyTorch 2.6+
- **When:** first `torch.load` of `endo_fm.pth`.
- **Symptom:** load failed due to the new `weights_only=True` default in PyTorch 2.6+.
- **Cause:** the checkpoint is a full pickled DINO object, not a plain tensor dict.
- **Fix:** load with `weights_only=False`.
- **Confirmed:** checkpoint loaded so we could inspect its keys.

### 5.8 Endo-FM: most weights reported missing
- **When:** after the checkpoint loaded.
- **Symptom:** mapping into `vit_base_patch16_224` left most keys missing — encoder essentially untrained.
- **Cause:** Endo-FM is a **TimeSformer** (video) DINO student/teacher; keys are nested and include temporal-attention / `time_embed` tensors that don't exist in an image ViT-B/16.
- **Fix:** extract the `teacher` (preferred) / `student` backbone; map only the **spatial** ViT-B/16 trunk keys; skip temporal-attention, `time_embed`, and shape-mismatched tensors (e.g. video `pos_embed`), logging skips on the module.
- **Confirmed:** **~150 keys mapped, 0 missing** for the spatial trunk; documented in `reports/model_access_notes.md`.

### 5.9 AMP deprecation warnings
- **When:** wiring mixed precision in training.
- **Symptom:** deprecation warnings from `torch.cuda.amp.*`.
- **Cause:** old AMP API deprecated in favour of device-generic `torch.amp.*`.
- **Fix:** switched to `torch.amp.autocast(...)` / `torch.amp.GradScaler(...)`.
- **Confirmed:** warnings gone; training ran.

### 5.10 No resume support (risk before SSH GPU)
- **When:** user raised concern about SSH sessions dropping.
- **Symptom:** only `checkpoint_best.pt` was saved (on val improvement); a dropped run would lose all progress.
- **Cause:** original `train.py` had no per-epoch state and no resume path.
- **Fix:** added atomic `checkpoint_last.pt` **every epoch** (model + optimizer + scheduler + AMP scaler + epoch + early-stop wait + curve); kept `checkpoint_best.pt` on improvement; `metrics.json` marks a run complete so re-runs skip it. Added CLI `--resume` (default), `--no-resume`, `--save-every-epoch`; made `run_experiments.sh` re-run-safe.
- **Confirmed:** checkpoints appeared after epoch 1; re-running a finished run skips it.

### 5.11 MPS reported unavailable inside the sandbox
- **When:** first CUDA/MPS device check.
- **Symptom:** MPS "built" but "not available".
- **Cause:** the sandboxed shell blocked MPS access.
- **Fix:** re-ran the device check outside the sandbox; added `src/utils/device.py` selecting **CUDA → MPS → CPU**.
- **Confirmed:** device resolved to `mps:0` on the Apple M1 Max.

### 5.12 MPS DataLoader instability
- **When:** starting ResNet-50 training on MPS.
- **Symptom:** worker processes caused instability with the MPS backend.
- **Cause:** multi-worker DataLoader does not play well with MPS here.
- **Fix:** set `num_workers=0` for MPS runs.
- **Confirmed:** training ran steadily (~1.8 it/s).

### 5.13 MPS memory pressure during prediction/eval
- **When:** evaluate / predict passes.
- **Symptom:** memory build-up on the MPS device.
- **Cause:** cached allocations not released between batches.
- **Fix:** call `torch.mps.empty_cache()` periodically in the predict loop; hardened evaluate memory use.
- **Confirmed:** evaluation completed without running out of memory.

### 5.14 CRLF line endings broke the bash scripts
- **When:** running `run_local_mps.sh`.
- **Symptom:** bash `set -o pipefail` / script parsing failed.
- **Cause:** the script had Windows CRLF line endings.
- **Fix:** converted the script to LF.
- **Confirmed:** script parsed and executed.

### 5.15 Smoke/debug runs mistaken for real results
- **When:** the matrix "skipped" the 5% ResNet-50 run.
- **Symptom:** the 5% slot was filled by the **1-epoch CPU smoke** (and a 2-epoch MPS debug) run, not a real result, and later the matrix died during evaluate on that stale run.
- **Cause:** leftover smoke/debug output under `label_05/seed_0` counted as "complete".
- **Fix:** archived the throwaway runs (`seed_0_cpu_smoke_1epoch`, `seed_0_mps_debug_2epoch`) and re-ran a clean full matrix.
- **Confirmed:** all 15 real seed-0 runs completed with proper epochs.

### 5.16 Background jobs killed outside Cursor (`nohup` / `setsid`)
- **When:** trying to run the long matrix in the background on macOS.
- **Symptom:** `setsid` not available on macOS; `nohup` + `disown` jobs got killed when launched outside Cursor's terminal.
- **Cause:** macOS process/session handling for detached background jobs.
- **Fix:** ran the full matrix as a **managed Cursor background shell**, which stayed alive for the whole run.
- **Confirmed:** matrix ran ~3.8 h to completion (exit 0); log in `reports/logs/local_mps_matrix.log`.

### Quick reference table

| # | Error / symptom | Root cause | Fix |
|---|---|---|---|
| 5.1 | `python` not found | macOS has only `python3` | `python3` + venv |
| 5.2 | Missing imports | empty venv | `pip install -r requirements.txt` (+`gdown`) |
| 5.3 | Matplotlib crash | headless sandbox | Agg backend + `MPLCONFIGDIR` |
| 5.4 | Bad patient grouping | Excel IDs unreliable | patient ID from `PatientXXX` folders |
| 5.5 | Wrong/empty Excel merge | blank rows / column variants | harden reader + aliases |
| 5.6 | Labels overwritten | Excel merge override | folder label authoritative |
| 5.7 | Endo-FM load fails | PyTorch 2.6 `weights_only` | `weights_only=False` |
| 5.8 | Endo-FM keys missing | TimeSformer video weights | map spatial trunk, skip temporal |
| 5.9 | AMP warnings | old `torch.cuda.amp` | `torch.amp.*` |
| 5.10 | No resume | best-only checkpoint | per-epoch `checkpoint_last.pt` + resume |
| 5.11 | MPS unavailable | sandbox block | run outside sandbox; `device.py` |
| 5.12 | DataLoader unstable on MPS | multi-worker + MPS | `num_workers=0` |
| 5.13 | MPS memory pressure | cached allocations | `torch.mps.empty_cache()` |
| 5.14 | Bash script fails | CRLF endings | convert to LF |
| 5.15 | Fake 5% result | leftover smoke/debug run | archive + re-run clean |
| 5.16 | Background job killed | macOS `nohup`/`setsid` | managed Cursor shell |

---

## 6. GPU rental — readiness (re-checked 2026-08-07)

**Yes — ready to rent 1× H100 and run on the fly.**

Local gates all green (re-verified today): metadata OK, Endo-FM weights present (2.2 GB), 15 low-label split CSVs, **9/9 tests passed**, seed-0 MPS matrix complete, resume-safe H100 script ready.

Full pre-flight, seed strategy (A: seeds 1–2 only / B: re-run 0–1–2), day-of runbook, and submission timeline are in:

→ **[`../03_Drafts_and_Planning/FUTURE_NEXT_STEPS.md`](../03_Drafts_and_Planning/FUTURE_NEXT_STEPS.md)**

Also see compute estimates: `../03_Drafts_and_Planning/compute_and_training_time_h100_h200.md`.

**Decision (still in force):** keep intermediate seed-0 numbers in this file / `reports/` only; fill `main.tex` **after** the multi-seed GPU matrix.

---

## 7. Next steps (ordered) — summary

1. **Rent 1× H100 (~10–12 h)** → sync code/data/weights → CUDA smoke → `SEEDS=(0 1 2)` (Option B, preferred as of 2026-08-07; `1 2` = Option A fallback) → `./scripts/run_experiments.sh`.
2. Pull `reports/` back; aggregate plots/tables.
3. Fill `main.tex` results + honest discussion once.
4. Reflective essay, supporting zip, presentation video → **19 Aug**.

Details and checklists: `FUTURE_NEXT_STEPS.md`.

---

## 8. Status checklist

| Item | Status |
|---|---|
| Project inventory / RQ locked | Done |
| `dissertation_project/` scaffold | Done |
| CE-NBI download + MD5 | Done |
| Metadata + audit (11,144 / 210) | Done |
| Patient + low-label splits | Done |
| Tests (9/9) | Done (re-checked 2026-08-07) |
| Train / eval / calibrate / plots code | Done |
| Endo-FM weights + loader | Done |
| CPU smoke train | Done |
| Per-epoch checkpoint + resume | Done |
| Device: CUDA → MPS → CPU | Done |
| Seed-0 full MPS matrix (15 runs) | Done |
| Calibration for seed-0 | Done |
| Working results in this MD | Done (keep here) |
| **GPU-ready (rent now)** | **Yes** |
| Seeds 1–2 on H100 | **Next action** |
| `main.tex` numeric results | Wait for final matrix |
| Reflective essay + submission pack | Todo (~19 Aug) |
