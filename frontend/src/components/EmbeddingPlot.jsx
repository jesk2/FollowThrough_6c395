import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useEmbedding } from "../hooks/useProfile";

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { isUser } = payload[0].payload;
  if (!isUser) return null;
  return (
    <div className="bg-card border border-border rounded-xl px-3 py-2 text-xs shadow-sm">
      <p className="font-medium text-indigo-600">You are here</p>
    </div>
  );
}

export default function EmbeddingPlot() {
  const { data, isLoading } = useEmbedding();

  if (isLoading) {
    return (
      <div className="h-48 flex items-center justify-center">
        <p className="text-xs text-ink-tertiary">Loading…</p>
      </div>
    );
  }

  if (!data || !data.user) {
    return (
      <div className="h-48 flex items-center justify-center">
        <p className="text-xs text-ink-tertiary">
          Not enough data yet — complete more check-ins to see your position.
        </p>
      </div>
    );
  }

  // combine population + user into one dataset, flag the user point
  const points = [
    ...(data.population ?? []).map((p) => ({ ...p, isUser: false })),
    { ...data.user, isUser: true },
  ];

  return (
    <div className="flex flex-col gap-3">
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <XAxis
            dataKey="x"
            type="number"
            name="Consistency"
            tick={{ fontSize: 10, fill: "#9c9890" }}
            tickLine={false}
            axisLine={false}
            label={{
              value: "Consistency →",
              position: "insideBottomRight",
              offset: -4,
              fontSize: 10,
              fill: "#9c9890",
            }}
          />
          <YAxis
            dataKey="y"
            type="number"
            name="Time sensitivity"
            tick={{ fontSize: 10, fill: "#9c9890" }}
            tickLine={false}
            axisLine={false}
            label={{
              value: "Time sensitivity →",
              angle: -90,
              position: "insideTopLeft",
              offset: 12,
              fontSize: 10,
              fill: "#9c9890",
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Scatter data={points}>
            {points.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.isUser ? "#4f46e5" : "#e8e6e1"}
                r={entry.isUser ? 7 : 4}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 justify-end">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 flex-shrink-0" />
          <span className="text-xs text-ink-tertiary">You</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-border flex-shrink-0" />
          <span className="text-xs text-ink-tertiary">Other users (anonymized)</span>
        </div>
      </div>
    </div>
  );
}
