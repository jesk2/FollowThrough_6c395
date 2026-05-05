import { DEVICE_LABELS, DEVICE_DESCRIPTIONS } from "../utils/constants";

// Each level gets a color that encodes escalation intensity —
// green (gentle) → amber → orange → red (strict)
const LEVEL_STYLES = [
  { badge: "bg-emerald-50 text-emerald-700 border-emerald-200",  dot: "bg-emerald-400" },
  { badge: "bg-blue-50 text-blue-700 border-blue-200",           dot: "bg-blue-400"    },
  { badge: "bg-amber-50 text-amber-700 border-amber-200",        dot: "bg-amber-400"   },
  { badge: "bg-orange-50 text-orange-700 border-orange-200",     dot: "bg-orange-400"  },
  { badge: "bg-red-50 text-red-700 border-red-200",              dot: "bg-red-400"     },
];

export default function DeviceBadge({ level, showDescription = true }) {
  if (level == null || level < 0 || level > 4) return null;

  const label = DEVICE_LABELS[level];
  const description = DEVICE_DESCRIPTIONS[level];
  const { badge, dot } = LEVEL_STYLES[level];

  return (
    <div className="flex flex-col gap-1.5">
      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium w-fit ${badge}`}>
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
        Level {level} — {label}
      </div>
      {showDescription && (
        <p className="text-xs text-ink-secondary leading-relaxed">{description}</p>
      )}
    </div>
  );
}

// usage from Dashboard or Profile
// full badge with description (for dashboard status bar)
{/* <DeviceBadge level={profile.current_device} />

// compact badge only, no description (for tight spaces like a card header)
<DeviceBadge level={profile.current_device} showDescription={false} /> */}
