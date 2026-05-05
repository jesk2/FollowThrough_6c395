import { formatBeta } from "../utils/formatters";

// Maps beta value to a human-readable label and color
function getBetaDescription(beta) {
  if (beta < 0.5)  return { label: "Low present bias",      color: "#22c55e" };
  if (beta < 0.65) return { label: "Mild present bias",     color: "#84cc16" };
  if (beta < 0.75) return { label: "Moderate present bias", color: "#f59e0b" };
  if (beta < 0.85) return { label: "High present bias",     color: "#f97316" };
  return                  { label: "Severe present bias",   color: "#ef4444" };
}

export default function BetaGauge({ beta }) {
  if (beta == null) return null;

  const percent = Math.round(beta * 100);
  const { label, color } = getBetaDescription(beta);

  return (
    <div className="flex flex-col gap-3">

      {/* ── top row: β value + label ── */}
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-2xl font-medium" style={{ color }}>
          {formatBeta(beta)}
        </span>
        <span className="text-xs text-ink-secondary">{label}</span>
      </div>

      {/* ── track ── */}
      <div className="relative h-2 rounded-full bg-surface border border-border overflow-hidden">
        {/* gradient fill */}
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all duration-700"
          style={{
            width: `${percent}%`,
            background: `linear-gradient(to right, #22c55e, #f59e0b, #ef4444)`,
          }}
        />
      </div>

      {/* ── axis labels ── */}
      <div className="flex justify-between text-xs text-ink-tertiary">
        <span>Low bias (β = 0)</span>
        <span>High bias (β = 1)</span>
      </div>

      {/* ── explainer ── */}
      <p className="text-xs text-ink-secondary leading-relaxed pt-1 border-t border-border">
        Your β score reflects how much you discount future rewards relative to present ones.
        A lower score means you follow through more consistently.
        This updates as you complete more check-ins.
      </p>

    </div>
  );
}
