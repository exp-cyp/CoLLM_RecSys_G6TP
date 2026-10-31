# Project decisions

## Accepted scope — 2026-10-30

- Work locally on the user's HP OmniBook 7; no paid cloud service.
- Start with `Qwen/Qwen2-0.5B`; consider a larger backbone only after measuring feasibility.
- Compare MF, TALLRec-style text-only LoRA, and CoLLM-MF on MovieLens-1M using ROC AUC.
- Preserve text-only LoRA tuning followed by mapping-only CIE tuning.
- Limit each training/evaluation command to six hours. This is a ceiling, not a requested runtime
  and not a global limit shared by all pipeline stages.
- **Do not start the real training run yet.** Preparation, preflight, unit tests, short benchmarks,
  and small smoke checks are authorized. Real training awaits the user's start instruction.
- Report changes from the paper: backbone, sampled data, input length, training budget,
  selected MF dimension, and constant learning rates.

## Hardware and historical measurements

The earlier hardware inspection found Core Ultra 7 255H (16 cores/threads), about 32 GB RAM
(31.4 GiB visible), Intel Arc 140T integrated graphics, Windows 11, and about 625 GB free on C:.
Available RAM was about 13 GiB at that inspection and can change with other applications.
System Python is 3.14.6; the project uses isolated Python 3.11 under `.venv`.

Historical short checks with PyTorch 2.8.0+xpu and the actual Qwen2-0.5B measured roughly:

| Device / batch | LoRA microbatch | Mapper microbatch |
|---|---:|---:|
| XPU / 1 | 0.33 s | 0.27 s |
| XPU / 4 | 0.53 s | 0.52 s |
| CPU / 1 | 1.77 s | 1.75 s |

The batch-4 check recorded about 2.20 GiB peak PyTorch XPU allocation. These are historical
short measurements from preparation, not post-cleanup training results or total system memory
requirements. They exclude model loading, validation, checkpoint I/O, and sustained thermal effects.

## Proposed real experiment — not executed

`configs/local.yaml` proposes 4,096 / 512 / 1,024 train/validation/test interactions, seed 42,
64-dimensional MF factors, input length 256, batch 4, accumulation 2, and 512 optimizer updates
per LLM stage. Effective batch is 8; one complete stage processes approximately one pass
over the selected training rows. Checkpoints are selected only by complete validation AUC.

## Readiness cleanup — 2026-10-31

- Pin model ID/revision in YAML and verify all local model inputs against `collm_source.json`.
- Verify the official dataset archive and fingerprint full selected record content, not IDs alone.
- Keep only the Yes output-head row; remove unused No scoring, prompt attributes, and history IDs.
- Keep labels separate from `CoLLM.forward()` inputs.
- Bind run compatibility to semantic config, data content, model assets, and implementation.
- Save every selected checkpoint in an immutable directory; atomically publish `stage/best.json`.
- Bind Stage 2 to the exact MF and Stage 1 adapter artifacts, then verify those parents at evaluation.
- Refuse populated stage/result directories instead of silently mixing old and new runs.
- Preserve `last/` snapshots and stop/error status; no automatic resume is implemented.
- Provide `collm.run --check` for readiness checks without loading LLM weights or training.

Latest verification outcomes are recorded in `results/verification.json`; this document does not
freeze a test count while verification is still being updated. Test-set scores must not influence
training or configuration selection. No exact reproduction or comparative improvement is claimed.

## One-click handoff and Git — 2026-11-01

- User requests a final Train button as the only remaining step and asks to push Git before real training.
- START_TRAIN.cmd opens a local Tkinter window; opening it does not train.
- A clean HEAD matching the fetched/pushed upstream is required before an experiment can start.
- Each click creates a new run/config snapshot; a project lock prevents concurrent jobs.
- Stop is cooperative. Completed compatible stages can be reused; an incomplete stage restarts.
- Evaluation is followed by Thai/HTML summaries, training curves, CSVs, Git provenance, and a source/results ZIP.
- Results are reported as measured, including worse/tied CoLLM AUC or time-limited training.
- Real training remains unstarted until the user presses Train.
