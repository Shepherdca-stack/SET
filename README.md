![verify](https://github.com/Shepherdca-stack/SET/actions/workflows/verify.yml/badge.svg)
# SET v1.8: a machine-checked mathematical framework with two registered, falsifiable predictions

**Author:** Corey Shepherd, independent researcher, Olive Hill, Kentucky
**Version:** 1.8 (merged August 11, 2026; registry-hygiene correction to one Predictions cell applied August 12, 2026, documented below)
**Verification status at issue:** 94 machine checks, 0 failures

## What this is

This repository contains a body of mathematical work built from a single postulated object: the horn torus, the surface of revolution with tube radius equal to ring radius. The radial metric of that surface is exactly the Fisher information metric of a single Bernoulli random variable, so the object can be read as the statistical geometry of one bit. Everything in the work is geometry, spectra, or dynamics of that one shape, plus one experimental calibrant (N = 19,776, from the baryon-to-photon ratio) and a small, explicitly listed set of named assumptions.

From that starting point the work derives a large set of quantities and compares them to measured physics. Nearly all of those comparisons are **postdictions**: the measured values were known before the derivations were built, and the record labels them that way. The framework's evidential exposure lives in two **registered predictions** (P6 and P7, described below), each carrying an explicit kill condition and no retreat clause.

In plain terms: this is a mathematical structure that reproduces a surprising number of measured constants from very little input, together with an honest accounting of which parts are proven, which are conditional, which are candidates, and exactly what future measurement would kill it.

## What this is not

This work does not claim to be established physics. Its dark-matter proposal (derivation chain 31) is labeled **CANDIDATE** throughout, resting on a named two-clause postulate called **P-RING** that has not been derived from dynamics. Its quantization step (chain 32) rests on a named conditional called **Q-LAW**. The two registered predictions, **P6** and **P7**, inherit those labels: P6's geometric ratio is proven within the framework, but the carrier story that makes it a physical prediction is a candidate; P7 rides on the same candidate plus a derived coupling ceiling. The repository's own documents state, for every claim, whether it is PROVEN, THEOREM, CONDITIONAL, CANDIDATE, OBSERVED, or SUPERSEDED, and on what assumptions it rests.

## What 94 green checks mean, and what they do not mean

Running the verification suite and seeing 94 PASS lines tells you the following, and only the following: **the mathematical claims labeled as machine-checked in the workbook are true as mathematics.** Symbolic identities simplify to zero in a computer algebra system. Numerical spectra come out as claimed. Statistical exclusions compute to the stated sigmas. The suite is a proof that the internal record is honest, that the derivations do what the documents say they do, and that nothing labeled PROVEN depends on an unexecuted calculation.

The checks do **not** mean the framework describes the physical world. A structure can be internally exact and physically wrong. Whether this one is right is the job of the two registered predictions: P6 dies if next-generation cosmology excludes the ratio 2pi - 1 at more than 3 sigma, and P7 dies the day any direct-detection experiment confirms a non-gravitational dark-matter signal. The suite verifies the mathematics; the sky verifies the physics.

## The three-file set

1. **SET_Unified_Master_State_v1-8.md** — the complete narrative record. Read this first. Sections cover the foundations, the derivation chains, the assumption ledger, the predictions register, the superseded register, the open problems, and a plain-language summary.
2. **SET_Master_Workbook_v1-8.xlsx** — the full registry: Overview, Foundations, Derivation Chains 1-32, Equations E1-E117, Predictions, Novel Equations, Assumptions & Open Problems, Superseded & Withdrawn, Lay Summary. Every equation row carries its status and error. This copy carries one post-merge correction relative to the workbook as issued at the v1.8 merge: the Predictions tab's P6 clock cell (F13) now records the July 9, 2025 cancellation of CMB-S4 and the successor experiment set. No other cell differs; the edit was verified by full-workbook cell comparison.
3. **SET_verification_suite_v5.py** — the ground-up machine checks, foundations through chain 32. 94 checks. Runs in a few minutes on a laptop.

Supporting numerics live in nine versioned scripts, `scripts/SET_time_T4_wilson.py` through `scripts/SET_time_T12_landau.py`, which carry the full per-check detail behind derivation chains 26-32. See `scripts/README.md` for the manifest.

## How to run the suite

```
pip install -r requirements.txt
python SET_verification_suite_v5.py
```

Expected output: section-by-section PASS lines, 94 in total, zero FAIL lines, ending with `Suite v5 complete.` The pinned dependency versions in `requirements.txt` (numpy 2.4.4, scipy 1.17.1, sympy 1.14.0, Python 3.12) are the versions the suite was verified against on August 12, 2026. The mathematics does not depend on these exact versions, but pinning them removes one source of irreproducibility.

Note for automation: the suite reports through printed PASS/FAIL lines rather than its process exit code. The included GitHub Actions workflow (`.github/workflows/verify.yml`) accounts for this by parsing the output; it fails the build on any FAIL line or on a PASS count other than 94.

## The two registered predictions

**P6** (E46, chain 31 — CANDIDATE carrier on PROVEN geometry): the cosmological dark-matter-to-baryon density ratio converges to exactly 2pi - 1 = 5.2832, zero adjustable parameters. The current Planck-era value, 5.357 at roughly 1.2% precision, sits 1.1 sigma inside. Kill condition: a greater-than-3-sigma exclusion of 5.2832 kills equation E46 and chain 31 together. The originally cited decisive instrument, CMB-S4, was cancelled by DOE and NSF on July 9, 2025; the successor clock (Simons Observatory enhanced plus SPT-3G extended surveys, then survey combinations in the early-to-mid 2030s) is recorded in the workbook and in the pre-registration text under `registration/`.

**P7** (E112, E113, chain 31 — rides the same CANDIDATE): the eternal null. Within the framework, dark matter couples to ordinary matter at the proton's gravitational fine-structure constant and no stronger, so every direct-detection experiment stays empty forever, at every mass, in every channel. Kill condition: any confirmed non-gravitational dark-matter signal, anywhere, falsifies the framework outright.

Both predictions were registered at the v1.8 merge on August 11, 2026, before the decisive measurements exist. The pre-registration text prepared for OSF is in `registration/OSF_prereg_P6_P7.md`.

## Reading order for a skeptic

Start with the master state's section 10 (plain language), then section 6 (the predictions register), then run the suite. If the suite passes, open the workbook's Equations tab and pick any row labeled PROVEN: the master state or the chain rows will point you to its check. The Assumptions & Open Problems tab lists everything the framework imports or postulates, including the two items doing the heaviest lifting in the current candidate: P-RING (a postulate, chain 31) and Q-LAW (a conditional, chain 32). The Superseded & Withdrawn tab records every claim the work itself killed, including its own earlier predictions; the kill record is part of the evidence that the labels mean something.

## Licensing

Code (the verification suite and the scripts under `scripts/`) is MIT licensed; see `LICENSE`. Documents (the master state, the workbook, and the registration texts) are CC BY 4.0; see `LICENSE-DOCS`. Cite via `CITATION.cff`.
