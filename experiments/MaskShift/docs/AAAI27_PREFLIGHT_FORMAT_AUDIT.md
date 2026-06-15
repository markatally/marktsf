# AAAI-27 Preflight Conversion Audit

M20 converts the current generic article manuscript into both a two-column, US-letter, anonymous preflight draft and an official `aaai2027` anonymous submission-template build from the official AAAI author kit. The preflight remains a page-pressure audit; `paper/aaai27_official.pdf` is the current official-template submission candidate.

| Check | Value | Evidence |
| --- | --- | --- |
| Two-column preflight | True | Generated from paper/main.tex into paper/aaai27_preflight.tex. |
| US Letter preflight | True | Preflight wrapper uses letterpaper. |
| Two-column table fit | True | Result tables are promoted to table* floats to avoid single-column overflow. |
| Anonymous title block | True | Author line is Anonymous Submission. |
| Page-count preflight | 5 | Must be <=7 pages under the preflight wrapper. |
| Official style files | {'aaai2027.sty': True, 'aaai2027.bst': True, 'aaai.sty': False} | Official aaai2027.sty/aaai2027.bst are present when both required files are true. |
| Official AAAI build | True | paper/aaai27_official.pdf pages=5, fresh=True. |

## Upload Boundary

Passing M20 means the manuscript body fits a conservative AAAI-style preflight and the official `aaai2027` anonymous submission-template build passes. The remaining upload boundary is the OpenReview submission workflow and official reproducibility checklist fields.
