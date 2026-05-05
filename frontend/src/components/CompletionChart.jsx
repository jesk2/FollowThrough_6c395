import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useCompletionHistory } from "../hooks/useProfile";

function formatDay(dateStr) {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const { rate, count } = payload[0].payload;
  return (
    <div className="bg-card border border-border rounded-xl px-3 py-2 text-xs shadow-sm">
      <p className="font-medium text-ink">{label}</p>
      {rate == null ? (
        <p className="text-ink-tertiary">No check-ins</p>
      ) : (
        <>
          <p className="text-ink-secondary">{Math.round(rate * 100)}% completion</p>
          <p className="text-ink-tertiary">{count} task{count !== 1 ? "s" : ""}</p>
        </>
      )}
    </div>
  );
}

export default function CompletionChart() {
  const { data, isLoading } = useCompletionHistory();

  if (isLoading) {
    return (
      <div className="h-32 flex items-center justify-center">
        <p className="text-xs text-ink-tertiary">Loading…</p>
      </div>
    );
  }

  // fill null rates with 0 for rendering, but track which days had no data
  const chartData = (data ?? []).map((d) => ({
    ...d,
    displayRate: d.rate ?? 0,
    label: formatDay(d.date),
  }));

  const hasAnyData = chartData.some((d) => d.count > 0);

  return (
    <div className="flex flex-col gap-3">
      {!hasAnyData && (
        <p className="text-xs text-ink-tertiary text-center py-4">
          No check-ins in the last 14 days yet. Complete some tasks to see your history.
        </p>
      )}

      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={chartData} barSize={14} margin={{ top: 4, right: 0, left: -28, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: "#9c9890" }}
            tickLine={false}
            axisLine={false}
            interval={1}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 10, fill: "#9c9890" }}
            tickLine={false}
            axisLine={false}
            ticks={[0, 0.5, 1]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#f7f6f3" }} />
          <Bar dataKey="displayRate" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={
                  entry.count === 0
                    ? "#e8e6e1"           // no data — light gray
                    : entry.rate >= 0.8
                    ? "#22c55e"           // great — green
                    : entry.rate >= 0.5
                    ? "#f59e0b"           // ok — amber
                    : "#ef4444"           // poor — red
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex items-center gap-4 justify-end">
        {[
          { color: "#22c55e", label: "≥ 80%" },
          { color: "#f59e0b", label: "50–79%" },
          { color: "#ef4444", label: "< 50%" },
          { color: "#e8e6e1", label: "No data" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="text-xs text-ink-tertiary">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Each bar is color-coded by performance: green for ≥80% completion,
// amber for 50–79%, red for below 50%, and light gray for days with
// no check-ins at all. The tooltip shows the exact percentage and
// task count when you hover. Days with no data still render as a
// short gray bar so the 14-day timeline always looks complete rather
// than having gaps.
