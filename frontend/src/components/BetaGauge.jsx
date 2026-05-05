// TODO (Audrey)
// BetaGauge.jsx
export default function BetaGauge({ beta }) {
  return <div className="text-sm font-mono">β = {beta?.toFixed(2) ?? "—"}</div>;
}
