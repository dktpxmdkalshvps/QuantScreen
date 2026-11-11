---
name: quantscreen-change
description: Plan and verify QuantScreen frontend, FastAPI, API contract, or repository sync changes.
---

# QuantScreen changes

Read `docs/CODE_SYNC_POLICY.md`, `docs/CI_CD_NOTICE.md`, `SYNC_MANIFEST.md`, and the relevant API contract before editing. Record a scoped requirement, plan, and task list under `docs/changes/<id>/` for substantive changes.

This repository is the review surface; live Vercel and Render services deploy from `QS_front` and `QS_render`. Preserve that split and update sync records when application source is synchronized. Preserve existing API response shape and Alembic history. `backend/start.sh` applies migrations at boot, so schema changes need an explicit migration and rollout review.

Verify the affected frontend build and backend import or focused tests. Use `backend/docs/QA_CHECKLIST.md` for deployment checks. Inspect scripts before execution, particularly `scripts/verify_no_sensitive_files.sh`, which may print matching lines. Never expose environment variable values or secrets in logs. Report source/sync status, contract impact, verification, and whether operational repositories were touched.
