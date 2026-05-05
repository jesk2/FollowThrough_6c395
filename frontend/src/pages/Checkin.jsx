// Audrey implements this screen.
// Route param: taskId. Fetches task details, submits POST /checkins.
// export default function Checkin() {
//   return <div>Check-in — TODO (Audrey)</div>;
// }

import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "react-query";
import { useSubmitCheckin } from "../hooks/useCheckins";
import { CATEGORY_COLORS, FAILURE_REASONS } from "../utils/constants";
import { formatDuration, formatDateTime } from "../utils/formatters";
import client from "../api/client";

// fetch a single task by id
const getTask = (taskId) => client.get(`/tasks/${taskId}`).then((r) => r.data);

export default function Checkin() {
  const { taskId } = useParams();
  const navigate = useNavigate();
  const { mutate: submitCheckin, isLoading: submitting } = useSubmitCheckin();

  const { data: task, isLoading, isError } = useQuery(
    ["task", taskId],
    () => getTask(taskId),
    { enabled: !!taskId }
  );

  const [completed, setCompleted] = useState(null);   // 0.0 | 0.5 | 1.0
  const [actualDuration, setActualDuration] = useState("");
  const [failureReason, setFailureReason] = useState("");

  const canSubmit =
    completed !== null &&
    (completed === 0.0 || actualDuration) &&
    (completed !== 0.0 || failureReason);

  const handleSubmit = () => {
    const payload = {
      task_id: taskId,
      completed,
      actual_duration: actualDuration ? parseInt(actualDuration) : undefined,
      failure_reason: completed === 0.0 ? failureReason : undefined,
    };
    submitCheckin(payload, {
      onSuccess: () => navigate("/dashboard"),
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-ink-secondary text-sm">Loading…</p>
      </div>
    );
  }

  if (isError || !task) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-red-500 text-sm">Couldn't load task. Try again from the dashboard.</p>
      </div>
    );
  }

  const color = CATEGORY_COLORS[task.category] ?? "#6b6860";
  const duration = task.corrected_duration ?? task.planned_duration;

  return (
    <div className="min-h-screen bg-surface">
      {/* ── Nav ─────────────────────────────────────────────────────── */}
      <nav className="bg-card border-b border-border px-6 py-4 flex items-center gap-3">
        <button
          onClick={() => navigate("/dashboard")}
          className="text-ink-tertiary hover:text-ink transition-colors text-sm"
        >
          ← Back
        </button>
        <span className="text-sm text-ink-tertiary">Check-in</span>
      </nav>

      <main className="max-w-lg mx-auto px-6 py-10 flex flex-col gap-6">

        {/* ── Task summary card ───────────────────────────────────── */}
        <div
          className="bg-card border border-border rounded-2xl p-5"
          style={{ borderLeftColor: color, borderLeftWidth: "3px" }}
        >
          <p className="font-semibold text-base">{task.name}</p>
          <div className="flex items-center gap-3 mt-2 flex-wrap">
            <span className="text-xs capitalize font-medium" style={{ color }}>
              {task.category}
            </span>
            <span className="text-xs text-ink-tertiary">
              {formatDateTime(task.planned_start)}
            </span>
            <span className="text-xs text-ink-tertiary">
              {formatDuration(duration)}
            </span>
          </div>
        </div>

        {/* ── Q1: Did you complete it? ────────────────────────────── */}
        <section className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-3">
          <p className="text-sm font-medium">Did you complete it?</p>
          <div className="flex gap-2">
            {[
              { value: 1.0, label: "Yes" },
              { value: 0.5, label: "Partially" },
              { value: 0.0, label: "No" },
            ].map(({ value, label }) => (
              <button
                key={value}
                onClick={() => {
                  setCompleted(value);
                  // reset dependent fields when switching
                  setActualDuration("");
                  setFailureReason("");
                }}
                className={`flex-1 py-2 rounded-xl text-sm font-medium border transition-all
                  ${completed === value
                    ? "bg-indigo-50 border-indigo-300 text-indigo-700"
                    : "border-border text-ink-secondary hover:border-ink-tertiary"}`}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        {/* ── Q2: How long did it take? (yes or partially) ────────── */}
        {(completed === 1.0 || completed === 0.5) && (
          <section className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-3">
            <p className="text-sm font-medium">How long did it take? <span className="text-ink-tertiary font-normal">(minutes)</span></p>
            <input
              type="number"
              min={1}
              placeholder={`Planned: ${duration}m`}
              value={actualDuration}
              onChange={(e) => setActualDuration(e.target.value)}
              className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </section>
        )}

        {/* ── Q3: Why not? (no only) ──────────────────────────────── */}
        {completed === 0.0 && (
          <section className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-3">
            <p className="text-sm font-medium">Why not?</p>
            <div className="flex flex-col gap-2">
              {FAILURE_REASONS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setFailureReason(value)}
                  className={`text-left px-4 py-2.5 rounded-xl text-sm border transition-all
                    ${failureReason === value
                      ? "bg-indigo-50 border-indigo-300 text-indigo-700"
                      : "border-border text-ink-secondary hover:border-ink-tertiary"}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </section>
        )}

        {/* ── Submit ──────────────────────────────────────────────── */}
        <button
          onClick={handleSubmit}
          disabled={!canSubmit || submitting}
          className="w-full py-3 bg-indigo-600 text-white rounded-xl text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 transition-colors"
        >
          {submitting ? "Submitting…" : "Submit check-in"}
        </button>

      </main>
    </div>
  );
}
