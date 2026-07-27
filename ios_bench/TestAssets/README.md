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
