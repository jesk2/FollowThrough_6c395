import { useState } from "react";
import { useTasks } from "../hooks/useTasks";
import { CATEGORY_COLORS } from "../utils/constants";
import { formatDateTime } from "../utils/formatters";

const CATEGORY_ORDER = ["academic", "work", "exercise", "personal"];

const SHOW_OPTIONS = [
  { value: "none",  label: "Hide done" },
  { value: "day",   label: "Past day"  },
  { value: "week",  label: "Past week" },
];

function groupByCategory(tasks) {
  const groups = {};
  for (const task of tasks) {
    if (!groups[task.category]) groups[task.category] = [];
    groups[task.category].push(task);
  }
  return groups;
}

function isWithin(dateStr, window) {
  if (window === "none") return false;
  const ms = window === "day" ? 24 * 60 * 60 * 1000 : 7 * 24 * 60 * 60 * 1000;
  return Date.now() - new Date(dateStr).getTime() < ms;
}

function SidebarTask({ task }) {
  const color = CATEGORY_COLORS[task.category] ?? "#6b6860";
  const isPast = new Date(task.planned_start) < new Date();
  const done = task.has_checkin;

  return (
    <div className="flex items-start gap-2.5 py-2.5 border-b border-border last:border-0">
      <span
        className="w-1.5 h-1.5 rounded-full flex-shrink-0 mt-1.5"
        style={{ backgroundColor: color }}
      />
      <div className="flex flex-col gap-0.5 min-w-0 flex-1">
        <span className={`text-xs font-medium truncate leading-snug
          ${done ? "line-through text-ink-tertiary" : "text-ink"}`}>
          {task.name}
        </span>
        <span className={`text-xs ${isPast && !done ? "text-red-400" : "text-ink-tertiary"}`}>
          {formatDateTime(task.planned_start)}
        </span>
      </div>
      {done && (
        <span className="text-emerald-500 text-xs flex-shrink-0">✓</span>
      )}
    </div>
  );
}

export default function TaskSidebar() {
  const { data, isLoading } = useTasks();
  const [showDone, setShowDone] = useState("none");

  const allTasks = data?.tasks ?? [];

  // filter: always show incomplete tasks, show completed only if they fall within the chosen window
  const visibleTasks = allTasks.filter((task) => {
    if (!task.has_checkin) return true;
    return isWithin(task.planned_start, showDone);
  });

  const groups = groupByCategory(visibleTasks);
  const hasAny = visibleTasks.length > 0;

  return (
    <aside className="w-56 flex-shrink-0 self-start sticky top-6 max-h-[calc(100vh-5rem)] overflow-y-auto flex flex-col gap-4 bg-card border border-border rounded-2xl p-4">

      {/* ── header + filter ── */}
      <div className="flex flex-col gap-2">
        <h2 className="text-xs font-medium text-ink-tertiary uppercase tracking-widest">
          All tasks
        </h2>
        <div className="flex rounded-lg border border-border overflow-hidden">
          {SHOW_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setShowDone(value)}
              className={`flex-1 py-1 text-xs font-medium transition-colors
                ${showDone === value
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-ink-tertiary hover:text-ink bg-card"}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && (
        <p className="text-xs text-ink-tertiary">Loading…</p>
      )}

      {!isLoading && !hasAny && (
        <p className="text-xs text-ink-tertiary leading-relaxed">
          {allTasks.length === 0
            ? "No tasks yet. Hit \"+ Add task\" to get started."
            : "No tasks to show for this filter."}
        </p>
      )}

      {CATEGORY_ORDER
        .filter((cat) => groups[cat]?.length > 0)
        .map((cat) => (
          <div key={cat} className="flex flex-col">
            <span
              className="text-xs font-semibold capitalize mb-1"
              style={{ color: CATEGORY_COLORS[cat] }}
            >
              {cat}
            </span>
            {groups[cat]
              .sort((a, b) => new Date(a.planned_start) - new Date(b.planned_start))
              .map((task) => (
                <SidebarTask key={task.id} task={task} />
              ))}
          </div>
        ))}
    </aside>
  );
}
