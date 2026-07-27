# Test assets for the Tier2-Mobile System Cost benchmark

`masked_video.mp4` and `mask.mp4` are a real Tier-1 export pair (grey-filled
person video + white=person segmentation mask video), produced by:

```
python scripts/run.py "data/dataset/Movement of Faces/Bystander_Movement/2_bystander_movement.mp4" \
  --anonymizer yoloseg11ncnn --export-dir <tmp> --benchmark --no-save --headless --export-people 1
```

Source clip: `2_bystander_movement.mp4` (MovementBystander category, static/
near-static camera -- matches the STATIC/JITTER path this benchmark's
BackgroundReconstructor implements, not the DYNAMIC/windowed path).

**Scope note (2026-07-27):** originally exported as a true 300-frame/10s
clip at native 1264x1264 (30fps), but this crashed on-device (iPhone 15 Pro
Max) at frame-loading time -- holding 300 native-resolution frames as
Float32 RGB buffers simultaneously (BackgroundReconstructor.reconstruct()
keeps the full aligned color+valid stack in memory for the trimmed-mean
pass) is roughly 1.5-2GB+ per video, times two videos (masked+mask),
exceeding what the app can hold before being killed. Interim fix (per
explicit user decision): re-sampled down to 50 frames at ~10fps (5s of real
footage, every 3rd frame of the original 300), same 1264x1264 resolution.
If 50 frames still crashes, the real fix is reworking
BackgroundReconstructor to keep frames as UInt8 (not Float32) until the
final aggregation step and/or process in batches, not just shrinking the
clip further -- flagged as the next thing to try if this interim size still
fails.

These feed `BenchmarkView.runTier2MobileSystemCostBenchmark()`, which runs
the on-device Swift port of:
1. Background Reconstruction (alignment pyramid + trimmed-mean + LaMa core-fill)
2. Illumination Extraction (lightmap)
3. Final Compositing (placeholder character alpha-blend + relight)

RIFE is deliberately excluded from this benchmark, per the project's RIFE
scoping decision (see memory: project_rife_decision_off.md).

Final Compositing uses a synthetic PLACEHOLDER character (a soft-edged
ellipse cutout, generated in Compositor.placeholderCharacter) rather than a
real cloud-returned WanAnimate render -- this measures compositing/relight
MATH cost only, not end-to-end visual fidelity with a real character.
