// Audrey implements this screen.
// Shows beta_proxy, proj_bias_score, EmbeddingPlot, device history, checkin history
// export default function Profile() {
//   return <div>Profile — TODO (Audrey)</div>;
// }

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useProfile } from "../hooks/useProfile";
import { useCheckinHistory } from "../hooks/useCheckins";
import DeviceBadge from "../components/DeviceBadge";
import BetaGauge from "../components/BetaGauge";
import EmbeddingPlot from "../components/EmbeddingPlot";
import { formatBeta, formatPercent, formatDateTime, formatDuration } from "../utils/formatters";
import { CATEGORY_COLORS, FAILURE_REASONS } from "../utils/constants";

const COMPLETED_LABELS = {
  1.0: { label: "Completed",  color: "text-emerald-600" },
  0.5: { label: "Partial",    color: "text-amber-600"   },
  0.0: { label: "Missed",     color: "text-red-500"     },
};

export default function Profile() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);

  const { data: profile, isLoading: profileLoading } = useProfile();
  const { data: historyData, isLoading: historyLoading } = useCheckinHistory(page);

  if (profileLoading) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-ink-secondary text-sm">Loading…</p>
      </div>
    );
  }

  const checkins = historyData?.checkins ?? [];
  const totalPages = historyData ? Math.ceil(historyData.total / historyData.page_size) : 1;

  return (
    <div className="min-h-screen bg-surface">

      {/* ── Nav ───────────────────────────────────────────────────────── */}
      <nav className="bg-card border-b border-border px-6 py-4 flex items-center gap-3">
        <button
          onClick={() => navigate("/dashboard")}
          className="text-ink-tertiary hover:text-ink transition-colors text-sm"
        >
          ← Back
        </button>
        <span className="text-sm text-ink-tertiary">Profile</span>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-8">

        {/* ── Header ──────────────────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl font-semibold">{profile?.email}</h1>
          <p className="text-ink-secondary text-sm mt-1">
            Member since {new Date(profile?.created_at).toLocaleDateString(undefined, {
              month: "long", year: "numeric"
            })}
          </p>
        </div>

        {/* ── Behavioral stats ────────────────────────────────────────── */}
        <section className="bg-card border border-border rounded-2xl p-6 flex flex-col gap-6">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Behavioral profile
          </h2>

          {/* stat row */}
          <div className="grid grid-cols-3 gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-ink-tertiary">14-day completion</span>
              <span className="font-mono text-xl font-medium">
                {formatPercent(profile?.completion_rate_14d ?? 0)}
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-ink-tertiary">Streak</span>
              <span className="font-mono text-xl font-medium">
                {profile?.streak ?? 0}
                <span className="text-sm text-ink-secondary ml-1">days</span>
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-ink-tertiary">Projection bias</span>
              <span className="font-mono text-xl font-medium">
                {formatPercent(profile?.proj_bias_score ?? 0)}
              </span>
            </div>
          </div>

          {/* beta gauge */}
          <BetaGauge beta={profile?.beta_proxy} />
        </section>

        {/* ── Current device ──────────────────────────────────────────── */}
        <section className="bg-card border border-border rounded-2xl p-6 flex flex-col gap-4">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Commitment device
          </h2>
          <DeviceBadge level={profile?.current_device} showDescription={true} />
        </section>

        {/* ── Embedding plot ──────────────────────────────────────────── */}
        <section className="bg-card border border-border rounded-2xl p-6 flex flex-col gap-4">
          <div>
            <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
              Your position
            </h2>
            <p className="text-xs text-ink-secondary mt-1">
              Where you sit relative to other users in behavioral space.
            </p>
          </div>
          <EmbeddingPlot />
        </section>

        {/* ── Check-in history ────────────────────────────────────────── */}
        <section className="flex flex-col gap-3">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Check-in history
          </h2>

          {historyLoading ? (
            <div className="bg-card border border-border rounded-2xl p-8 text-center">
              <p className="text-xs text-ink-tertiary">Loading…</p>
            </div>
          ) : checkins.length === 0 ? (
            <div className="bg-card border border-border rounded-2xl p-8 text-center">
              <p className="text-ink-secondary text-sm">No check-ins yet.</p>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {checkins.map((checkin) => {
                const status = COMPLETED_LABELS[checkin.completed] ?? COMPLETED_LABELS[0.0];
                const reason = FAILURE_REASONS.find((r) => r.value === checkin.failure_reason);
                return (
                  <div
                    key={checkin.id}
                    className="bg-card border border-border rounded-2xl p-4 flex items-start justify-between gap-4"
                  >
                    <div className="flex flex-col gap-1">
                      <span className={`text-xs font-medium ${status.color}`}>
                        {status.label}
                      </span>
                      <span className="text-xs text-ink-tertiary">
                        {formatDateTime(checkin.checked_in_at)}
                      </span>
                      {reason && (
                        <span className="text-xs text-ink-tertiary">
                          Reason: {reason.label}
                        </span>
                      )}
                    </div>
                    {checkin.actual_duration && (
                      <span className="text-xs text-ink-secondary font-mono flex-shrink-0">
                        {formatDuration(checkin.actual_duration)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="text-xs text-ink-secondary hover:text-ink disabled:opacity-40 transition-colors"
              >
                ← Previous
              </button>
              <span className="text-xs text-ink-tertiary">
                {page} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="text-xs text-ink-secondary hover:text-ink disabled:opacity-40 transition-colors"
              >
                Next →
              </button>
            </div>
          )}
        </section>

      </main>
    </div>
  );
}
