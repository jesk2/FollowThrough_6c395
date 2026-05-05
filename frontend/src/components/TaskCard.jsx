import { useNavigate } from "react-router-dom";
import { CATEGORY_COLORS } from "../utils/constants";
import { formatDuration, formatDateTime } from "../utils/formatters";

function getOverdueLabel(plannedStart) {
  const diffMs = Date.now() - new Date(plannedStart).getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 60) return `${diffMin}m overdue`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h overdue`;
  return `${Math.floor(diffHr / 24)}d overdue`;
}

const DIFFICULTY_LABELS = ["", "Easy", "Moderate", "Medium", "Challenging", "Hard"];

export default function TaskCard({ task }) {
  const navigate = useNavigate();
  const color = CATEGORY_COLORS[task.category] ?? "#6b6860";
  const isOverdue = new Date(task.planned_start) < new Date();
  const duration = task.corrected_duration ?? task.planned_duration;
  const wasCorrected = task.corrected_duration && task.corrected_duration !== task.planned_duration;

  return (
    <div
      className="bg-card border border-border rounded-2xl p-5 flex flex-col gap-3 hover:border-ink-tertiary transition-colors"
      style={{ borderLeftColor: color, borderLeftWidth: "3px" }}
    >
      {/* ── Top row: name + check-in button ── */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="font-medium text-sm leading-snug">{task.name}</span>
          <span
            className="text-xs capitalize font-medium"
            style={{ color }}
          >
            {task.category}
          </span>
        </div>
        <button
          onClick={() => navigate(`/checkin/${task.id}`)}
          className="flex-shrink-0 px-3 py-1.5 bg-indigo-600 text-white text-xs font-medium rounded-lg hover:bg-indigo-700 transition-colors"
        >
          Check in
        </button>
      </div>

      {/* ── Meta row: difficulty, duration, time ── */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* difficulty pill */}
        <span className="text-xs text-ink-secondary bg-surface px-2 py-0.5 rounded-full border border-border">
          {DIFFICULTY_LABELS[task.difficulty] ?? `Difficulty ${task.difficulty}`}
        </span>

        {/* duration — show corrected with strikethrough if adjusted */}
        <span className="text-xs text-ink-secondary flex items-center gap-1">
          {wasCorrected ? (
            <>
              <span className="line-through text-ink-tertiary">
                {formatDuration(task.planned_duration)}
              </span>
              <span className="text-amber-600 font-medium">
                {formatDuration(task.corrected_duration)}
              </span>
            </>
          ) : (
            formatDuration(duration)
          )}
        </span>

        {/* deadline pressure */}
        {task.deadline_pressure !== "none" && (
          <span className="text-xs text-ink-secondary capitalize">
            Due {task.deadline_pressure === "this_week" ? "this week" : "today"}
          </span>
        )}

        {/* overdue label */}
        {isOverdue && (
          <span className="text-xs text-red-500 font-medium ml-auto">
            {getOverdueLabel(task.planned_start)}
          </span>
        )}
      </div>

      {/* ── Planned time ── */}
      <div className="text-xs text-ink-tertiary">
        Planned for {formatDateTime(task.planned_start)}
      </div>
    </div>
  );
}
