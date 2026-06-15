# MaskShift AAAI-27 Target-Readiness Audit

This document is target-specific for the AAAI-27 Main Technical Track. It complements M18 by checking the selected venue's dates, review process, page limit, reproducibility requirement, and AI-assistance policy against the current MaskShift package.

## Official Venue Facts

| Field | Value |
| --- | --- |
| venue | AAAI-27 Main Technical Track |
| conference_dates | 2027-02-16 to 2027-02-23 |
| submission_site_author_registration | 2026-06-17 |
| submission_site_paper_open | 2026-06-24 |
| abstract_deadline | 2026-07-21 23:59 UTC-12 |
| full_paper_deadline | 2026-07-28 23:59 UTC-12 |
| supplement_code_deadline | 2026-07-31 23:59 UTC-12 |
| phase1_rejection_notification | 2026-09-24 |
| author_feedback_window | 2026-10-19 to 2026-10-25 |
| final_decision | 2026-11-30 |
| camera_ready | 2026-12-14 |
| technical_content_limit | up to 7 pages plus references |
| supplement_policy | supplement allowed, but reviewers are not required to review it; critical material belongs in the main body |
| reproducibility | all authors must complete a reproducibility checklist |
| generative_ai_policy | authors may judiciously use generative AI tools, but remain fully responsible for all submitted material |
| review_process | two-phase reviewing with an AI-generated non-decisional review supplementing Phase 1 |
| review_criteria | significance, novelty, theoretical/empirical soundness, relevance, clarity, responsible practices, and reproducibility |
| source | https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/ |
| accessed | 2026-06-15 |

## Current Package State

| Field | Value |
| --- | --- |
| main_pdf_pages_generic | 8 |
| supplement_pdf_pages_generic | 3 |
| policy_pdf_pages_generic | 3 |
| preflight_pdf_pages | 5 |
| preflight_page_limit_pass | True |
| preflight_twocolumn | True |
| preflight_table_fit | True |
| preflight_pdf_fresh | True |
| preflight_pdf_size | 256786 |
| official_pdf_pages | 5 |
| official_page_limit_pass | True |
| official_template_build_pass | True |
| official_pdf_fresh | True |
| official_pdf_size | 400593 |
| documentclass_line | \documentclass[10pt]{article} |
| uses_aaai_style | False |
| aaai_style_present | True |
| anonymous_authors | True |
| m5 | STRONG_CONFERENCE_READY |
| m15 | PASS_FINAL_INTEGRITY |
| m17 | PASS_SUBMISSION_SUPPLEMENT |
| m18 | PASS_SUBMISSION_POLICY_PACK |
| m20 | PASS_AAAI27_PREFLIGHT_CONVERSION |
| m21 | PASS_AAAI27_REPRODUCIBILITY_CHECKLIST |
| official_checklist_pdf_size | 99818 |
| official_checklist_remaining_placeholders | 0 |
| aaai_official_kit_upload_ready | True |

## Gap Table

| Item | Status | Evidence | Required action |
| --- | --- | --- | --- |
| AAAI-27 timing | ready-for-open-site | Submission site opens 2026-06-24; abstract/full deadlines are 2026-07-21 23:59 UTC-12 and 2026-07-28 23:59 UTC-12. | Register authors when the site opens; prepare abstract by 2026-07-21. |
| Target style | pass | \documentclass[10pt]{article}; M20=PASS_AAAI27_PREFLIGHT_CONVERSION; official build=True. | Use paper/aaai27_official.pdf for AAAI-style review upload; do not upload the generic article-class PDF. |
| Page limit | pass-official-template | Generic main.pdf pages: 8; preflight pages: 5; official pages: 5; AAAI limit is up to 7 pages plus references. | Keep official aaai2027 build under <=7 technical-content pages plus references. |
| AAAI-27 preflight conversion | pass | M20=PASS_AAAI27_PREFLIGHT_CONVERSION; preflight pages=5; official pages=5; official PDF fresh=True; official size=400593 bytes. | Use paper/aaai27_official.pdf as the current official-template submission candidate. |
| Supplement policy | pass-with-main-body-caveat | M17=PASS_SUBMISSION_SUPPLEMENT; supplement pages: 3. | Keep all critical claims in the main body because AAAI reviewers are not required to review supplements. |
| Reproducibility checklist and artifact instructions | filled-local-ready | M15=PASS_FINAL_INTEGRITY; M18=PASS_SUBMISSION_POLICY_PACK; M21=PASS_AAAI27_REPRODUCIBILITY_CHECKLIST; local official-checklist PDF size=99818 bytes. | Copy the filled local checklist answers into AAAI's official OpenReview fields when the site opens. |
| Submission system and official form | calendar-blocking-before-site-open | Paper submission site opens 2026-06-24; current audit date 2026-06-15; M21=PASS_AAAI27_REPRODUCIBILITY_CHECKLIST. | When OpenReview paper submission opens, upload the official PDF and complete the official reproducibility checklist fields. |
| AI-assisted authorship disclosure | draft-ready | M18 includes AI Assistance Disclosure; AAAI says authors remain fully responsible for submitted material. | Keep provenance notes and disclose assistance only in the target system or statement field if requested. |
| AI-generated first-stage review readiness | prepared | M19 response plan maps likely human and AI-review objections to evidence artifacts. | During author feedback, answer both human and AI-generated reviews with the same evidence discipline. |
| Submission-quality science gate | pass | M5=STRONG_CONFERENCE_READY; M15=PASS_FINAL_INTEGRITY; M17=PASS_SUBMISSION_SUPPLEMENT; M18=PASS_SUBMISSION_POLICY_PACK; M20=PASS_AAAI27_PREFLIGHT_CONVERSION. | Preserve benchmark/theory scope; do not convert the paper into a method/SOTA claim. |

## Upload Verdict

MaskShift is scientifically ready for a strong-conference benchmark/theory submission, and M20 now provides both a two-column US-letter preflight and an official aaai2027 anonymous submission-template PDF within the seven-page technical-content pressure. The remaining upload boundary is operational: the OpenReview paper submission site is not yet open on the audit date, and the official reproducibility checklist fields must still be completed in the venue system.
