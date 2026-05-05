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

**Status:** Not wired. Backend imports legacy stubs that raise
`NotImplementedError`; the calls are wrapped in try/except so check-ins
succeed but `user.beta_proxy` never moves off `0.70`. Nicole's recommender
β-baseline branch is therefore degenerate — every user looks the same to it.

**Owners:** Backend lead writes the wiring. Kaitlyn confirms the API
shape ([`ml/inference/inference_api.py`](ml/inference/inference_api.py)).
Nicole's code doesn't change.

**Action items:**

- [ ] In [`backend/routers/checkins.py`](backend/routers/checkins.py),
      replace 4 imports with
      `from ml.inference.inference_api import initialize_new_user, get_user_state, incremental_update_api`
- [ ] Rewrite `_refresh_cf_and_beta` to:
  1. Allocate `user.embedding_id` via `initialize_new_user()` if missing
  2. Build a `TaskFeatures` Pydantic instance from the `Task` row
     (Kaitlyn: confirm field names — difficulty, category_index,
     planned_duration_minutes, days_until_planned_start,
     deadline_pressure_index)
  3. `incremental_update_api(embedding_id, features, completed)`
  4. `user.beta_proxy = get_user_state(user.embedding_id)`
- [ ] In [`backend/routers/auth.py:signup`](backend/routers/auth.py),
      call `initialize_new_user()` after creating the User row, store the
      returned int in `user.embedding_id` before commit
- [ ] Delete the four legacy stubs: `backend/ml/cf_model.py`, `features.py`,
      `probe.py`, `train.py` (no callers will remain)
- [ ] Once wired, re-run
      [`notebooks/07_pipeline_simulation.ipynb`](notebooks/07_pipeline_simulation.ipynb)
      with the real probe instead of the mock-EMA β estimator

**Estimated size:** ~20 lines, 30 min — assuming Kaitlyn's API works as
documented. Risk: `numpy 2.4.4 / torch 2.3.0` ABI compatibility (see
section 7) could break the import.

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

## 7. Dependency pinning — `numpy 2.4.4` × `torch 2.3.0` (Backend lead)

**Status:** Latest backend-deps bump in
[`backend/requirements.txt`](backend/requirements.txt) put `numpy` at
`2.4.4` while `torch` is still `2.3.0`. torch 2.3.0 was built against
numpy 1.x and there are documented import-time failures on numpy 2.x.
CI passed because nothing in the test suite actually exercises the
torch-heavy paths — but Kaitlyn's `ml/inference/inference_api.py` does
load torch at first call. This will likely blow up the moment item 1
(wiring the ML pipeline) lands.

**Owners:** Backend lead, in consultation with Kaitlyn.

**Action items:**

- [ ] Either bump `torch>=2.4` (which supports numpy 2.x) or pin
      `numpy<2`. Coordinate with Kaitlyn since her training was likely
      done against the older combination
- [ ] Re-run `pip install -r backend/requirements.txt` cleanly and
      verify `python -c "import torch, numpy; torch.zeros(3)"` works
      end-to-end
- [ ] Add a CI test that imports `ml.inference.inference_api` so future
      regressions are caught
