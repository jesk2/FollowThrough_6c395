# TODO — Cross-team integrations

Outstanding work where one teammate's component needs another teammate's
to land. Organized by the seam, not by owner. Each item names both sides
so it's clear who needs to talk to whom.

People:

- **Nicole** — BOCD detector, recommender, response_logic, scheduler jobs
- **Kaitlyn** — `ml/` pipeline (CF model, linear probe, inference API,
  synthetic data, training)
- **Audrey** — Frontend (React, hooks, pages, components, styling)
- **Jessica** — FastAPI routers, ORM, dependencies, notifications,
  migrations, CI

---

## 1. Kaitlyn's `ml/` pipeline → Backend routers (unblocks Nicole's recommender)

**Status:** Pretrained artifacts and inference API delivered; wiring in
backend routers still pending. Kaitlyn finished her side and the
simulation notebook now runs against the real probe end-to-end.

**Kaitlyn-delivered (Apr 2026):**

- [`ml/inference/inference_api.py`](ml/inference/inference_api.py) exposes
  the four entry points the backend wiring expects:
  `initialize_new_user() -> int`, `get_user_state(embedding_id) -> float`,
  `predict_task(embedding_id, features) -> float`,
  `incremental_update_api(embedding_id, features, completed) -> None`.
  All take/return primitives — no PyTorch tensors at the boundary.
- `TaskFeatures` Pydantic model with the exact field names called out
  below: `difficulty`, `category_index`, `planned_duration_minutes`,
  `days_until_planned_start`, `deadline_pressure_index`.
- Pretrained artifacts under
  [`ml/artifacts/`](ml/artifacts/): `cf_model.pt`, `user_state.pt`,
  `ridge_probe.joblib`, `feature_stats.json` (the last two are not
  optional — without them probe inference and feature normalization
  fail).
- Optional turnkey adapters under
  [`backend/ml/`](backend/ml/) — `cf_model.py`, `probe.py`, `train.py`,
  `features.py` are now thin wrappers over the `ml.inference` API,
  preserving the existing router call signatures. Backend lead can keep
  them or replace with direct `from ml.inference.inference_api ...`
  imports — both work; adapters mean less router code changes, direct
  imports mean one fewer indirection.

**Owners:** Backend lead writes the wiring. Kaitlyn confirms the API
shape (done above). Nicole's code doesn't change.

**Action items remaining (Backend lead):**

- [ ] Decide: keep `backend/ml/` adapters or delete and import directly
      from `ml.inference.inference_api`. (Adapters are wired up and
      smoke-tested; choice is purely stylistic.)
- [ ] Rewrite `_refresh_cf_and_beta` in
      [`backend/routers/checkins.py`](backend/routers/checkins.py) to:
  1. Allocate `user.embedding_id` via `initialize_new_user()` if missing
     (defensive — production should allocate at signup, but this catches
     pre-existing rows from the dev DB)
  2. Build a `TaskFeatures` instance from the `Task` row
     (`difficulty=(task.difficulty - 1)/4`, `category_index` from a
     str→int map, etc. — see the patched cell 1 of
     [`07_pipeline_simulation.ipynb`](notebooks/07_pipeline_simulation.ipynb)
     for a working translation)
  3. `incremental_update_api(user.embedding_id, features, completed)`
  4. `user.beta_proxy = get_user_state(user.embedding_id)`
- [ ] In [`backend/routers/auth.py:signup`](backend/routers/auth.py),
      call `initialize_new_user()` after creating the User row, store the
      returned int in `user.embedding_id` before commit. (Idempotent if
      called twice — embedding ids monotonically grow, so a duplicate
      call just allocates an unused slot; the more important property
      is "every user has one before their first check-in.")
- [x] Re-run
      [`notebooks/07_pipeline_simulation.ipynb`](notebooks/07_pipeline_simulation.ipynb)
      with the real probe — **done by Kaitlyn**. Notebook now imports
      `ml.inference.inference_api`, isolates user-state in a temp dir
      so production embeddings aren't mutated, and replaces the mock-EMA
      β block with `incremental_update_api` + `get_user_state`. All five
      scenarios execute end-to-end; plots and audit-log queries populate.

**Finding from the rerun (worth knowing before launch):** with the real
Ridge probe (α=1.0) and only 40–60 events per user, β_proxy stays
within ~±0.02 of the population mean (0.70) regardless of true β. The
mock-EMA tracked rolling completion more aggressively. Net effect on the
recommender: the β-baseline branch will resolve to L1 for almost
everyone in the first ~2 months; escalations to L3/L4 will be driven by
the failure-streak and BOCD-drift branches instead. Probably fine —
that's the conservative behavior we want for new users — but if the
product team wants β to move faster, options are (a) lower
`ridge_alpha` in pretrain, (b) raise the per-event learning rate in
`IncrementalUpdateConfig`, or (c) both. Kaitlyn happy to discuss.

**Estimated remaining size:** ~15 lines in routers, ~20 min. Risk: see
§7 below — should be low since the current pin is `numpy>=1.26,<2.0`
and Kaitlyn confirmed her artifacts load on that combo.

---

## 2. Nicole's `NotificationDescriptor` → notifications layer (Backend)

**Status:** Nicole's `process_checkin` produces structured descriptors
with `kind` ∈ {`reminder`, `streak_milestone`, `level_3_failure`,
`device_change`, `drift_nudge`} and rich payloads. The router currently
calls `send_reminder(email, payload.get('task_name', kind), now())` —
the `kind` and most of the payload are dropped on the floor. A Level-3
user who fails a task gets the same generic email a Level-0 user gets
for a normal reminder. The behavioral-economics nuance (loss framing,
earned-reward tone, identity priming) is wasted.

**Owners:** Backend lead extends notifications.py. Nicole's descriptors
already carry everything the renderer needs.

**Action items:**

- [ ] Add `send_notification(email: str, descriptor: NotificationDescriptor) -> None`
      in [`backend/notifications.py`](backend/notifications.py)
- [ ] Switch on `descriptor.kind`, render an appropriate template per kind:
  - `reminder` — current template; if `impl_where`/`impl_what_first`
    are in the payload, echo them ("You said you'd do this _at the
    gym, starting with putting on your shoes_.")
  - `streak_milestone` — celebratory ("You hit a {milestone}-day streak!")
  - `level_3_failure` — loss-framed ("You lost {streak_lost} streak
    points; you're now at {new_streak}.")
  - `device_change` — branch on payload `tone` (neutral for escalation,
    positive for de-escalation), use `from_label`/`to_label` from
    payload
  - `drift_nudge` — supportive ("We noticed a brief dip but you're back
    on track — keep going.")
- [ ] Update [`backend/routers/checkins.py`](backend/routers/checkins.py)
      `_apply_checkin_actions` to call `send_notification` for each
      `NotificationDescriptor` in `actions.notifications`
- [ ] Update the device-change notification fire-site in the same router
- [ ] Update [`backend/scheduler.py:checkin_reminder_job`](backend/scheduler.py)
      to build a `reminder` `NotificationDescriptor` from the
      `(user, task, payload)` tuples that Nicole's `get_pending_reminders`
      returns, then call `send_notification` (instead of the current
      `send_reminder` shortcut)

**Estimated size:** ~2 hours including templates.

---

## 3. Backend `drift_status` vocabulary → Frontend remap

**Status:** Migration 002 widened `ProfileResponse.drift_status` from
`["stable", "shifting", "improving"]` to
`["stable", "potential", "confirmed_decline", "confirmed_improvement"]`.
Anywhere the frontend renders this string needs a remap; old labels
won't appear in API responses anymore.

**Owners:** Audrey. Backend lead can confirm the Pydantic Literal in
[`backend/models/schemas.py`](backend/models/schemas.py).

**Action items:**

- [ ] Update string mapping in [`Profile.jsx`](frontend/src/pages/Profile.jsx)
      and any badge/status component
- [ ] Decide UI treatment for `potential` (it's a "watching" state —
      distinct from `confirmed_decline`/`confirmed_improvement`)
- [ ] Optional: a small `formatDriftStatus(s)` helper in
      [`frontend/src/utils/formatters.js`](frontend/src/utils/formatters.js)

---

## 4. Backend `tasks.locked` + reschedule confirmation → Frontend UI

**Status:** Migration 002 added `tasks.locked` (now in `TaskResponse`).
Nicole built a new `PUT /tasks/{task_id}/reschedule` endpoint and
`process_reschedule_attempt` with a 24-hour-window confirmation token
flow. Frontend doesn't know about either yet — no padlock UI on locked
tasks, no confirmation modal on reschedule.

**Owners:** Audrey, with the contract documented by Nicole.

**Frontend flow Nicole's backend expects:**

1. User clicks "reschedule" on a task card → frontend calls
   `PUT /tasks/{id}/reschedule` with `{new_start}`
2. If response is **409** with `detail.requires_confirmation: true`,
   show modal carrying `detail.confirmation_token`
3. On user confirm, call again with `{new_start, confirmation_token}`
   (must be the _same_ `new_start` — the token is bound to it)
4. Token TTL is 10 min; if expired, retry from step 1

**Action items:**

- [ ] Add padlock icon to [`TaskCard.jsx`](frontend/src/components/TaskCard.jsx)
      when `task.locked === true`
- [ ] Add `useRescheduleTask` hook in
      [`frontend/src/hooks/useTasks.js`](frontend/src/hooks/useTasks.js)
- [ ] Add corresponding API call in
      [`frontend/src/api/tasks.js`](frontend/src/api/tasks.js)
- [ ] Confirmation modal component
- [ ] Wire modal into wherever reschedule is initiated

---

## 5. Frontend hooks → empty pages (Audrey internal)

**Status:** Audrey's hooks ([`useTasks`](frontend/src/hooks/useTasks.js),
[`useCheckins`](frontend/src/hooks/useCheckins.js),
[`useProfile`](frontend/src/hooks/useProfile.js)) are fleshed out but four
of the five pages are still placeholders.

**Owner:** Audrey.

**Action items:**

- [ ] Implement [`Dashboard.jsx`](frontend/src/pages/Dashboard.jsx) (uses
      `useProfile`, `usePendingTasks`)
- [ ] Implement [`AddTask.jsx`](frontend/src/pages/AddTask.jsx) (uses
      `useCreateTask`; conditionally requires impl_where/impl_what_first
      based on `profile.current_device >= 1`)
- [ ] Implement [`Checkin.jsx`](frontend/src/pages/Checkin.jsx) (uses
      `useSubmitCheckin`)
- [ ] Implement [`Profile.jsx`](frontend/src/pages/Profile.jsx) (uses
      `useProfile`, `useEmbedding`)
- [ ] Implement [`DeviceBadge.jsx`](frontend/src/components/DeviceBadge.jsx)
      (currently `// TODO`); takes `current_device: int` and `device_label: str`
      from profile
- [ ] Add DM Sans / DM Mono font imports in
      [`index.css`](frontend/src/index.css) or
      [`index.html`](frontend/index.html) — Tailwind config references
      them but nothing actually loads them

---

## 6. Migration 002 → everyone with a local DB

**Status:** Nicole's migration adds `users.detector_state` (JSONB),
`tasks.locked` (Bool), and the `reschedule_events` table. New columns
are nullable / have defaults — safe upgrade. CI workflow already runs it.

**Owners:** Anyone running the backend locally.

**Action items:**

- [ ] Run `alembic upgrade head` in `backend/` after pulling
- [ ] Confirm `users.detector_state` column exists; new BOCD writes will
      fail without it
- [ ] Confirm `reschedule_events` table exists; the new
      `PUT /tasks/{id}/reschedule` endpoint writes to it

---

## 7. Dependency pinning — `numpy` × `torch 2.3.0` (Backend lead)

**Status:** Likely no longer broken.
[`backend/requirements.txt`](backend/requirements.txt) currently pins
`numpy>=1.26,<2.0`, which is compatible with `torch 2.3.0`. The original
note here predated that pin — leaving the section in place so the CI
test idea isn't lost.

Kaitlyn confirmed (May 2026): training and inference were done against
`torch 2.3.0 + numpy 1.26.4`; artifacts in `ml/artifacts/` load cleanly
under that combo. If anyone bumps `numpy` past 2.0 again, also bump
`torch` to ≥2.4 (where the numpy-2 ABI is fixed) and re-run
[`07_pipeline_simulation.ipynb`](notebooks/07_pipeline_simulation.ipynb)
to verify.

**Owners:** Backend lead.

**Action items:**

- [x] Confirm current `requirements.txt` keeps `numpy<2` while torch is
      pinned at 2.3.0 — done; pin is `numpy>=1.26,<2.0`.
- [ ] Add a CI test that imports `ml.inference.inference_api` so future
      ABI regressions are caught at PR time, not at first-request time.
