# AAAI-27 Reproducibility Checklist Draft

This is a draft mapping for the AAAI reproducibility checklist requirement. It should be copied into the official AAAI form after the target author kit and submission system fields are available.

| Checklist item | Current answer | Evidence/action |
| --- | --- | --- |
| Research question and scope | Yes | PAPER.md and main.tex state benchmark/theory scope and exclude method/SOTA claims. |
| Datasets | Yes | Weather, Electricity, Traffic, AirConvection with paths and natural missing rates recorded by M0/M16/M17. |
| Train/test split | Yes | Chronological splits and encoder-input-only masks are recorded in scripts and M18. |
| Randomness/seeds | Yes with limits | Seed offsets are recorded in M10-M14; three-seed summaries are disclosed as sprint-time evidence. |
| Baselines | Yes with scope | Official TSLib PatchTST/TimeXer, ChannelTokenFormer_missing, and S4M are adaptations under MaskShift protocol. |
| Hyperparameters | Draft-ready | Script configs are serialized in JSON summaries; final AAAI appendix should copy exact values. |
| Compute/runtime | Draft-ready | Local reduced protocols are described; final artifact should add measured wall-clock budget if available. |
| Code availability | Draft-ready | M18 statement and reproduction commands exist; anonymous artifact bundle still needs packaging. |
| Data availability | Draft-ready | Public benchmark input paths are recorded; raw license/source metadata should be checked before public release. |
| Limitations | Yes | main.tex discloses reduced protocols, mixed datasets, three-seed summaries, and failed typed head. |
