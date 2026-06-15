# MaskShift Final Integrity Report

- Verdict: `PASS_FINAL_INTEGRITY`
- Generated: `2026-06-15T08:24:05.363887+00:00`
- Blocking issues: `0`

## Gate Summary

| Gate | Pass | Notes |
| --- | --- | --- |
| references | `True` | 14 references; 0 dangling; 0 orphan |
| artifacts | `True` | 0 missing; 0 undersized |
| code | `True` | 0 compile errors; device issues 0 |
| numeric_trace | `True` | 25/25 main-text snippets |
| text_hygiene | `True` | 0 stale markers |
| submission_supplement | `True` |  |
| submission_policy_pack | `True` | 9 statements; template pending-target-selection |
| aaai27_target_dossier | `True` | upload ready False; blockers ['Submission system and official form'] |
| aaai27_preflight_conversion | `True` | pages 5; official-kit upload ready True |
| aaai27_reproducibility_checklist | `True` | 31 answers; placeholders 0 |

## AI Research Failure Mode Audit

| Mode | Status | Evidence |
| --- | --- | --- |
| Mode 1: implementation bug passing self-review | CLEAR | M15 py_compile over MaskShift scripts and S4M Bank.py passed; M5 remains STRONG_CONFERENCE_READY with zero blocking items. |
| Mode 2: hallucinated citation | CLEAR | No dangling/orphan citations; every BibTeX entry has a DOI or URL and a recorded external audit source. |
| Mode 3: hallucinated experimental result | CLEAR | Core numerical claims in main.tex and M14/M16 tables are traced to JSON summaries; M5/M12/M13/M14/M16/M17 summaries are parseable. |
| Mode 4: shortcut reliance | CLEAR_WITH_SCOPE_NOTE | M8 non-retirement decomposition and mixed Traffic/AirConvection reporting reduce shortcut risk; full causal shortcut ablation remains out of scope for a benchmark paper. |
| Mode 5: implementation bug reframed as insight | CLEAR | Typed-head surprise is reported as a negative diagnostic, and S4M negative evidence is disclosed as contrastive rather than converted into a universal failure claim. |
| Mode 6: methodology fabrication | CLEAR | Method scope in main.tex, the M17 supplement, and M18 statements matches run summaries: official-architecture adaptations, reduced channels/samples, three seed offsets, encoder-mask-only protocol, and target-agnostic disclosure boundaries are stated. |
| Mode 7: early frame-lock | CLEAR | The paper frame has been revised to benchmark/theory; method/SOTA claims are explicitly excluded, M18 marks target-template compliance as pending-target-selection, M19 records AAAI-27 upload blockers instead of hiding them, M20 separates page-pressure preflight from official-kit readiness, and M5 method_claim_ready remains false. |

## Issues

No integrity issues detected by M15.

## Reference Audit Sources

| Key | Source |
| --- | --- |
| `cao2018brits` | https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html |
| `che2018grud` | https://www.nature.com/articles/s41598-018-24271-9 |
| `islam2025sadi` | https://ojs.aaai.org/index.php/AAAI/article/view/33931 |
| `jang2026channeltokenformer` | https://openreview.net/forum?id=r4ZamwBE8P |
| `jing2025s4m` | https://openreview.net/forum?id=BkftcwIVmR |
| `liu2024itransformer` | https://arxiv.org/abs/2310.06625 |
| `nie2023patchtst` | https://openreview.net/forum?id=Jbdc0vTOcol |
| `rockenschaub2024missingness` | https://arxiv.org/abs/2406.16484 |
| `sunesh2026blackouts` | https://arxiv.org/abs/2601.01480 |
| `tashiro2021csdi` | https://proceedings.neurips.cc/paper_files/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html |
| `wang2024timexer` | https://arxiv.org/abs/2402.19072 |
| `yalavarthi2024grafiti` | https://ojs.aaai.org/index.php/AAAI/article/view/29560 |
| `yang2025crib` | https://arxiv.org/abs/2509.23494 |
| `zeng2023dlinear` | https://ojs.aaai.org/index.php/AAAI/article/view/26317 |

## Scope Note

M15 is a final local integrity gate. It verifies citation graph hygiene, BibTeX hygiene, internal numerical traceability, artifact presence, code compilation, device-selection rules, and ARS AI-research failure-mode coverage. It does not replace professional plagiarism software or a full external reproducibility replication.
