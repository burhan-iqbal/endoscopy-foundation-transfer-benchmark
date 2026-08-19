# Downloading Endo-FM Pretrained Weights

Endo-FM weights are hosted by the authors on Google Drive / SharePoint.

## Official sources

- Paper: https://arxiv.org/abs/2306.16741
- Code: https://github.com/med-air/Endo-FM
- Mirror: https://github.com/openmedlab/Endo-FM

## Automated download (preferred)

From `dissertation_project/` with the venv active:

```bash
pip install gdown
python - <<'PY'
import gdown
from pathlib import Path
out = Path("models/external_weights/endo_fm.pth")
out.parent.mkdir(parents=True, exist_ok=True)
# Official Endo-FM checkpoint (Google Drive, from med-air/Endo-FM README)
url = "https://drive.google.com/uc?id=1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl"
gdown.download(url, str(out), quiet=False)
print("saved", out, "bytes", out.stat().st_size)
PY
```

Expected size: about **2.2 GB**. Save path:

```text
dissertation_project/models/external_weights/endo_fm.pth
```

## Manual download

1. Open the **Pre-trained Weights** section on https://github.com/med-air/Endo-FM
2. Download via the Google Drive or SharePoint link
3. Place the file at `models/external_weights/endo_fm.pth`

## Status in this workspace

As of the local prep session, `endo_fm.pth` was successfully downloaded via `gdown` (~2.2 GB).

## If weights cannot be obtained

1. Document the access failure in `reports/model_access_notes.md`
2. Run ResNet-50 and DINOv2 under the full protocol
3. In the dissertation, adjust the claim to feasibility/limitations of released endoscopy foundation models

## Note on architecture mapping

`src/models/encoders.py` reads the Endo-FM DINO `teacher` (fallback `student`) TimeSformer checkpoint, strips `module.backbone.` / `backbone.` prefixes, drops temporal attention / `time_embed` parameters, and loads matching spatial ViT-B/16 tensors into `timm` `vit_base_patch16_224`. Verified locally: **150 spatial keys loaded, 0 missing**. Report the video→image transfer protocol honestly in the paper.
