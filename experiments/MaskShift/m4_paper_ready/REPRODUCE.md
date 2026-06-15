# MaskShift Reproduction

Run from repository root:

```bash
python3 -m experiments.MaskShift.m0_mask_suite
python3 -m experiments.MaskShift.m1_mechanism_audit
python3 -m experiments.MaskShift.m2_typed_head
python3 -m experiments.MaskShift.m3_statistical_tests
python3 -m experiments.MaskShift.m6_deep_backbone_sweep
python3 -m experiments.MaskShift.m7_severity_curves
python3 -m experiments.MaskShift.m8_mechanism_decomposition
python3 -m experiments.MaskShift.m9_official_tslib_reproduction
python3 -m experiments.MaskShift.m4_paper_ready
python3 -m experiments.MaskShift.m5_main_track_audit
```

Outputs are written under `experiments/MaskShift/`.
