// Audrey implements this screen.
// Sections: StatusBar, BetaGauge, CompletionChart (14-day), pending TaskCards

// export default function Dashboard() {
//   return <div>Dashboard — TODO (Audrey)</div>;
// }

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePendingTasks } from "../hooks/useTasks";
import { useProfile } from "../hooks/useProfile";
import DeviceBadge from "../components/DeviceBadge";
import BetaGauge from "../components/BetaGauge";
import CompletionChart from "../components/CompletionChart";
import TaskCard from "../components/TaskCard";
import AddTaskModal from "../components/AddTaskModal";

// ── flip to false once the backend is running ──────────────────────────────
const USE_MOCK = true;

const MOCK_PROFILE = {
  email: "audrey@test.com",
  beta_proxy: 0.72,
  proj_bias_score: 0.28,
  current_device: 1,
  device_label: "Implementation Intention",
  streak: 5,
  completion_rate_14d: 0.68,
  drift_status: "stable",
};

const MOCK_PENDING = [
  {
    id: "1",
    name: "Read chapter 4",
    category: "academic",
    difficulty: 3,
    deadline_pressure: "today",
    planned_start: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    planned_duration: 45,
    corrected_duration: 60,
  },
  {
    id: "2",
    name: "Morning run",
    category: "exercise",
    difficulty: 2,
    deadline_pressure: "today",
    planned_start: new Date(Date.now() - 1000 * 60 * 90).toISOString(),
    planned_duration: 30,
    corrected_duration: null,
  },
];
// ──────────────────────────────────────────────────────────────────────────

const DRIFT_STYLES = {
  stable:                 { label: "Stable",     color: "text-emerald-600" },
  potential:              { label: "Shifting",   color: "text-amber-600"   },
  confirmed_decline:      { label: "Declining",  color: "text-red-600"     },
  confirmed_improvement:  { label: "Improving",  color: "text-blue-600"    },
};

function parseJwt(token) {
  try { return JSON.parse(atob(token.split(".")[1])); }
  catch { return null; }
}

export default function Dashboard() {
  const navigate = useNavigate();

  const { data: liveProfile, isLoading: profileLoading } = useProfile();
  const { data: livePending, isLoading: tasksLoading }   = usePendingTasks();

  const profile = USE_MOCK ? MOCK_PROFILE : liveProfile;
  const pending = USE_MOCK ? MOCK_PENDING : (livePending?.tasks ?? []);
  const isLoading = USE_MOCK ? false : (profileLoading || tasksLoading);

  const [showAddTask, setShowAddTask] = useState(false);

  // derive email from JWT as fallback if profile doesn't load yet
  const token = localStorage.getItem("access_token");
  const jwtEmail = parseJwt(token)?.email;
  const email = profile?.email ?? jwtEmail ?? "";
  const firstName = email.split("@")[0];

  const drift = DRIFT_STYLES[profile?.drift_status] ?? DRIFT_STYLES.stable;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-ink-secondary text-sm">Loading…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface">
      {/* ── Top nav ────────────────────────────────────────────────────── */}
      <nav className="bg-card border-b border-border px-6 py-4 flex items-center justify-between">
        <span className="font-semibold tracking-tight">FollowThrough</span>
        <div className="flex items-center gap-4">
          <span className="text-sm text-ink-secondary">{email}</span>
          <button
            onClick={() => navigate("/profile")}
            className="text-sm text-ink-secondary hover:text-ink transition-colors"
          >
            Profile
          </button>
          <button
            onClick={() => {
              localStorage.removeItem("access_token");
              navigate("/login");
            }}
            className="text-sm text-ink-secondary hover:text-ink transition-colors"
          >
            Sign out
          </button>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-8">

        {/* ── Greeting ───────────────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl font-semibold">
            Hey{firstName ? `, ${firstName}` : ""}.
          </h1>
          <p className="text-ink-secondary text-sm mt-1">
            Here's where you stand today.
          </p>
        </div>

        {/* ── Status bar ─────────────────────────────────────────────── */}
        <section className="bg-card rounded-2xl border border-border p-6 flex flex-col gap-4">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Status
          </h2>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <DeviceBadge level={profile?.current_device} />
            <div className="flex items-center gap-6">
              <div className="flex flex-col items-end">
                <span className="text-xs text-ink-tertiary">Streak</span>
                <span className="font-mono text-lg font-medium">
                  {profile?.streak ?? 0}
                  <span className="text-sm text-ink-secondary ml-1">days</span>
                </span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-xs text-ink-tertiary">Trend</span>
                <span className={`text-sm font-medium ${drift.color}`}>
                  {drift.label}
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* ── Beta gauge ─────────────────────────────────────────────── */}
        <section className="bg-card rounded-2xl border border-border p-6 flex flex-col gap-4">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Your behavioral profile
          </h2>
          <BetaGauge beta={profile?.beta_proxy} />
        </section>

        {/* ── 14-day completion chart ────────────────────────────────── */}
        <section className="bg-card rounded-2xl border border-border p-6 flex flex-col gap-4">
          <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
            Last 14 days
          </h2>
          <CompletionChart />
        </section>

        {/* ── Pending check-ins ──────────────────────────────────────── */}
        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
              Needs check-in
            </h2>
            <button
              onClick={() => setShowAddTask(true)} // ← opens modal instead of navigating
              className="text-xs text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
            >
              + Add task
            </button>
          </div>

          {pending.length === 0 ? (
            <div className="bg-card rounded-2xl border border-border p-8 text-center">
              <p className="text-ink-secondary text-sm">
                No pending check-ins. You're all caught up.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {pending.map((task) => (
                <TaskCard key={task.id} task={task} />
              ))}
            </div>
          )}
        </section>

      </main>
    <AddTaskModal
        isOpen={showAddTask}
        onClose={() => setShowAddTask(false)}
        deviceLevel={profile?.current_device}
      />
    </div>
  );
}
