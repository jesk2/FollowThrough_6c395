import { useState } from "react";
import { useCreateTask } from "../hooks/useTasks";
import { CATEGORY_COLORS } from "../utils/constants";

const CATEGORIES = ["academic", "exercise", "work", "personal"];
const DEADLINE_OPTIONS = [
  { value: "today",     label: "Today"     },
  { value: "this_week", label: "This week" },
  { value: "none",      label: "No deadline" },
];

// ── flip to false once backend is running ──────────────────────────────────
const USE_MOCK = false;
const MOCK_DEVICE_LEVEL = 1; // change to 0 to hide impl intention fields
// ──────────────────────────────────────────────────────────────────────────

export default function AddTaskModal({ isOpen, onClose, deviceLevel: liveDeviceLevel }) {
  const deviceLevel = USE_MOCK ? MOCK_DEVICE_LEVEL : liveDeviceLevel;
  const { mutate: createTask, isLoading, isError, error } = useCreateTask();

  const [form, setForm] = useState({
    name: "",
    category: "academic",
    difficulty: 3,
    deadline_pressure: "none",
    planned_start: "",
    planned_duration: "",
    impl_where: "",
    impl_what_first: "",
  });

  const [correctedDuration, setCorrectedDuration] = useState(null);

  const set = (field, value) =>
    setForm((prev) => ({ ...prev, [field]: value }));

  const handleSubmit = () => {
    if (!form.name.trim() || !form.planned_start || !form.planned_duration) return;

    const payload = {
      ...form,
      planned_duration: parseInt(form.planned_duration),
      planned_start: new Date(form.planned_start).toISOString(),
      impl_where:       deviceLevel >= 1 ? form.impl_where       : undefined,
      impl_what_first:  deviceLevel >= 1 ? form.impl_what_first  : undefined,
    };

    if (USE_MOCK) {
      // simulate a corrected duration banner
      const mocked = { ...payload, corrected_duration: parseInt(form.planned_duration) + 15 };
      setCorrectedDuration(mocked.corrected_duration);
      setTimeout(() => { setCorrectedDuration(null); onClose(); }, 2000);
      return;
    }

    createTask(payload, {
      onSuccess: (data) => {
        if (data.corrected_duration && data.corrected_duration !== payload.planned_duration) {
          setCorrectedDuration(data.corrected_duration);
          setTimeout(() => { setCorrectedDuration(null); onClose(); }, 2500);
        } else {
          onClose();
        }
      },
    });
  };

  const handleClose = () => {
    setForm({
      name: "", category: "academic", difficulty: 3,
      deadline_pressure: "none", planned_start: "",
      planned_duration: "", impl_where: "", impl_what_first: "",
    });
    setCorrectedDuration(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    // backdrop
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm px-4"
      onClick={(e) => { if (e.target === e.currentTarget) handleClose(); }}
    >
      {/* modal card */}
      <div className="bg-card w-full max-w-md rounded-2xl border border-border shadow-xl flex flex-col max-h-[90vh] overflow-y-auto">

        {/* header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4 border-b border-border">
          <h2 className="font-semibold text-base">Add a task</h2>
          <button
            onClick={handleClose}
            className="text-ink-tertiary hover:text-ink transition-colors text-xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="px-6 py-5 flex flex-col gap-5">

          {/* corrected duration banner */}
          {correctedDuration && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
              Based on your history, we've scheduled{" "}
              <span className="font-medium">{correctedDuration} minutes</span> instead of{" "}
              {form.planned_duration}.
            </div>
          )}

          {/* error banner */}
          {isError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              {error?.response?.data?.detail ?? "Something went wrong."}
            </div>
          )}

          {/* task name */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-secondary">Task name</label>
            <input
              type="text"
              placeholder="e.g. Read chapter 4"
              value={form.name}
              onChange={(e) => set("name", e.target.value)}
              className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>

          {/* category */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-secondary">Category</label>
            <div className="flex gap-2 flex-wrap">
              {CATEGORIES.map((cat) => {
                const selected = form.category === cat;
                const color = CATEGORY_COLORS[cat];
                return (
                  <button
                    key={cat}
                    onClick={() => set("category", cat)}
                    style={selected ? { backgroundColor: color + "18", borderColor: color, color } : {}}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all capitalize
                      ${selected ? "" : "border-border text-ink-secondary hover:border-ink-tertiary"}`}
                  >
                    {cat}
                  </button>
                );
              })}
            </div>
          </div>

          {/* difficulty */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-ink-secondary">Difficulty</label>
              <span className="text-xs font-mono text-ink-tertiary">{form.difficulty} / 5</span>
            </div>
            <input
              type="range" min={1} max={5} step={1}
              value={form.difficulty}
              onChange={(e) => set("difficulty", parseInt(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-ink-tertiary">
              <span>Easy</span><span>Hard</span>
            </div>
          </div>

          {/* deadline pressure */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-secondary">Deadline</label>
            <div className="flex gap-2">
              {DEADLINE_OPTIONS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => set("deadline_pressure", value)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-all
                    ${form.deadline_pressure === value
                      ? "bg-indigo-50 border-indigo-300 text-indigo-700"
                      : "border-border text-ink-secondary hover:border-ink-tertiary"}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* planned start */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-secondary">Planned start</label>
            <input
              type="datetime-local"
              value={form.planned_start}
              onChange={(e) => set("planned_start", e.target.value)}
              className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>

          {/* planned duration */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-secondary">Planned duration (minutes)</label>
            <input
              type="number" min={1} placeholder="e.g. 45"
              value={form.planned_duration}
              onChange={(e) => set("planned_duration", e.target.value)}
              className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>

          {/* implementation intention fields — only shown for device level >= 1 */}
          {deviceLevel >= 1 && (
            <div className="flex flex-col gap-4 pt-1 border-t border-border">
              <div>
                <p className="text-xs font-medium text-indigo-600 mb-3">
                  Implementation intention — where and how will you do this?
                </p>
                <div className="flex flex-col gap-3">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-ink-secondary">Where will you do this?</label>
                    <input
                      type="text" placeholder="e.g. Library, 3rd floor"
                      value={form.impl_where}
                      onChange={(e) => set("impl_where", e.target.value)}
                      className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-medium text-ink-secondary">What will you do first?</label>
                    <input
                      type="text" placeholder="e.g. Open the PDF and read the abstract"
                      value={form.impl_what_first}
                      onChange={(e) => set("impl_what_first", e.target.value)}
                      className="border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* footer */}
        <div className="px-6 pb-6 pt-2 flex justify-end gap-3">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-ink-secondary hover:text-ink transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !form.name.trim() || !form.planned_start || !form.planned_duration}
            className="px-5 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 transition-colors"
          >
            {isLoading ? "Saving…" : "Add task"}
          </button>
        </div>

      </div>
    </div>
  );
}
