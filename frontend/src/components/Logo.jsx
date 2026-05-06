export default function Logo({ size = "md" }) {
  const textSize = size === "lg" ? "text-2xl" : "text-base";
  return (
    <div className={`flex items-center gap-2 ${textSize} font-semibold tracking-tight select-none`}>
      {/* Checkmark-in-circle mark */}
      <svg
        width={size === "lg" ? 32 : 22}
        height={size === "lg" ? 32 : 22}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="16" cy="16" r="15" stroke="#6366f1" strokeWidth="2" />
        {/* Arrow curving right — "follow through" motion */}
        <path
          d="M9 17 L14 22 L23 11"
          stroke="#6366f1"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>
        <span className="text-indigo-500">Follow</span>
        <span className="text-gray-800">Through</span>
      </span>
    </div>
  );
}
