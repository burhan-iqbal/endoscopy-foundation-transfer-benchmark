# Model access notes

## Endo-FM

- Source: https://github.com/med-air/Endo-FM (Google Drive id `1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl`)
- Local path: `models/external_weights/endo_fm.pth` (~2.2 GB) — **downloaded**
- Checkpoint format: DINO student/teacher TimeSformer (`teacher` preferred)
- Loader maps spatial ViT-B/16 trunk weights into `timm` `vit_base_patch16_224`, skipping temporal attention / `time_embed`
- Shape-mismatched tensors (e.g. video `pos_embed`) are skipped and logged on the module

## DINOv2

- `torch.hub` (`facebookresearch/dinov2`, `dinov2_vits14`); network needed on first load

## ResNet-50

- `timm` ImageNet pretrained; network needed on first load

## Smoke validation (CPU)

- `python -m src.training.train --model resnet50 --label-frac 0.05 --seed 0 --max-epochs 1`
- Result: val AUROC ≈ 0.69, test AUROC ≈ 0.81 after 1 epoch (pipeline sanity only, not a final result)
