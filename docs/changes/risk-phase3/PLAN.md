# Repository audit and Phase 3 risk benchmark

## Scope and plan

Inspect duplicate paths, tracked runtime files, Phase 2 data and results; build a research-only risk benchmark using the verified Phase 2 extracted OHLCV cache; test temporal alignment and run a chronological validation and later holdout. No production source, deployment repository, or API contract changes.

## Audit before cleanup (2026-11-13)

- Current HEAD is `0b04265e5740b0f9ee412233605c10b6522ca79d`, while the supplied reviewed commit `d2867a166b874b9b3d9e921360437d81f64860d8` is unavailable locally. Thus the audit applies to the current checkout, not that unavailable tree.
- `backend/backend/` and `.github/.github/workflows/verify.yml` do not exist in the current checkout or current HEAD tree. The valid workflow is `.github/workflows/verify.yml`; no duplicate CI job exists here. No deletion is warranted. If these paths appear in another checkout, compare file hashes and unique paths there before cleanup.
- `experiments/forecast-benchmark/requirements.lock.txt` is empty and untracked. It cannot reproduce an environment. Preserve this existing user file; recommend removing the placeholder after review or replacing it with a generated, platform-specific Conda explicit specification when environment freezing is needed. Phase 3 declares direct dependencies separately.
- Initial `git status --short` showed only `?? experiments/`; `git diff --check` passed. `git ls-files` showed no tracked runtime databases, Parquet files, `.pyc`, or live `.env` files; tracked `.env.example` files are templates. A filename-only secret-pattern scan found `.env.example` and `backend/README.md` as example/documentation hits. No live credential file was found; this heuristic is not a complete secret audit. The repository's sensitive-file script was inspected but not run because it prints matching lines, potentially exposing values.
- Phase 2's eight tests passed. Its manifest reports four 2,011-session extracted datasets from 2018-01-02 through 2025-12-31. The saved `results.json` and `summary.csv` contain 80 model cells, 44 paired runs each; all 80 summary MAEs were recomputed from the saved predictions and matched.

## Tasks

- [x] Inspect current tracked paths and preserve all unique content.
- [x] Run Phase 2 tests and inspect manifest/results.
- [x] Implement isolated risk targets, baselines, LightGBM, walk-forward scoring and tests.
- [x] Run data verification, eight Phase 3 tests, benchmark and document measured outcomes.

## Outcome and limits

No directory or placeholder was deleted. No production or deployment source was modified. Phase 3 checked all four extracted hashes and source audits, scored 240 summary rows, and wrote paired predictions and final holdout comparisons. The existing `experiments/` tree contains extensive pre-existing untracked caches; review the intended additions selectively before staging. Current `git diff --check` passes, but Git does not include untracked additions in that check until staged.
